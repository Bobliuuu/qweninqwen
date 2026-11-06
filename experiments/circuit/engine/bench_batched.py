"""GPU-batched circuit MAC engine: B independent signed INT8 x INT4 tile circuits (own weights, own activations) executed in ONE forward
of the stock Qwen3_5 model that qwentpu's compiler emits (all arithmetic done by the model's weights / GDN state / SwiGLU).
Host: builds token ids (weight/activation/read tokens), reads the output-bit hidden dims at the read positions, decodes bits->int
(transport of the emitted code only) and compares with x@w for VALIDATION only.
usage: python bench_batched.py ROWS COLS NVEC B1,B2,... [--dtype bf16] [--out res.json]"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import signed_tile
from qwentpu.compiler.tiny import build as compile_model

def build_model(rows, cols, dtype, device):
    g, d = signed_tile.build(rows, cols)
    one = g.one
    outs = [[g.ffn([(1, one, b)], f"out{c}.{i}") for i, b in enumerate(d["outputs"][c])] for c in range(cols)]
    dims = [[next(iter(e.t)) for e in col] for col in outs]
    vocab = list(range(d["read0"] + cols))
    model = compile_model(g, vocab, [x for col in dims for x in col], dtype=dtype)
    odim = torch.tensor([[model.layout.dim(x) for x in col] for col in dims])   # [cols, width]
    return g, d, model.to(device), odim

def make_batch(d, rows, cols, nvec, B, rng):
    Ws = rng.integers(-8, 8, (B, rows, cols)); Xs = rng.integers(-128, 128, (B, nvec, rows))
    toks = []; marks = None
    for b in range(B):
        t, m = signed_tile.trace(d, Ws[b], Xs[b]); toks.append(t); marks = m
    return Ws, Xs, torch.tensor(toks), marks

@torch.inference_mode()
def run(model, ids, odim, marks, dev):
    h = model.model(ids.to(dev), use_cache=False).last_hidden_state          # [B,T,H]
    pos = torch.tensor([t for t, _ in marks], device=dev); col = torch.tensor([c for _, c in marks], device=dev)
    sel = h[:, pos]                                                          # [B,M,H]
    idx = odim.to(dev)[col]                                                  # [M,width]
    bits = (torch.gather(sel, 2, idx[None].expand(sel.shape[0], -1, -1)) > 0).long()
    return bits

def decode(bits, nvec, cols):
    w = bits.shape[-1]
    val = (bits * (2 ** torch.arange(w, device=bits.device))).sum(-1)
    val = torch.where(bits[..., -1].bool(), val - (1 << w), val)
    return val.reshape(bits.shape[0], nvec, cols).cpu().numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rows", type=int); ap.add_argument("cols", type=int); ap.add_argument("nvec", type=int)
    ap.add_argument("batches"); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--out", default=None); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; dt = dict(bf16=torch.bfloat16, fp32=torch.float32)[a.dtype]
    t0 = time.time(); g, d, model, odim = build_model(a.rows, a.cols, dt, dev); build_s = time.time() - t0
    nl = len(model.model.layers); print(f"built {a.rows}x{a.cols}: layers={nl} hidden={model.config.hidden_size} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M stats={g.stats()} build_s={build_s:.1f}", flush=True)
    rng = np.random.default_rng(1); res = dict(rows=a.rows, cols=a.cols, nvec=a.nvec, dtype=a.dtype, layers=nl, hidden=model.config.hidden_size,
        params=sum(p.numel() for p in model.parameters()), device=torch.cuda.get_device_name() if dev == "cuda" else "cpu", runs=[])
    for B in [int(x) for x in a.batches.split(",")]:
        try:
            Ws, Xs, ids, marks = make_batch(d, a.rows, a.cols, a.nvec, B, rng)
            bits = run(model, ids, odim, marks, dev)                          # warmup + correctness
            got = decode(bits, a.nvec, a.cols); want = np.einsum("bvr,brc->bvc", Xs, Ws)
            exact = bool((got == want).all()); nbad = int((got != want).sum())
            ts = []
            for _ in range(a.reps):
                if dev == "cuda": torch.cuda.synchronize()
                t = time.time(); run(model, ids, odim, marks, dev)
                if dev == "cuda": torch.cuda.synchronize()
                ts.append(time.time() - t)
            sec = min(ts); macs = B * a.rows * a.cols * a.nvec
            r = dict(B=B, tokens_per_seq=ids.shape[1], macs=macs, seconds=sec, mac_per_s=macs / sec, tok_per_s=B * ids.shape[1] / sec,
                     exact=exact, n_bad=nbad, n_outputs=int(want.size), peak_gb=(torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0))
        except torch.cuda.OutOfMemoryError as e:
            r = dict(B=B, error="OOM"); torch.cuda.empty_cache()
        print(json.dumps(r), flush=True); res["runs"].append(r)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
main()
