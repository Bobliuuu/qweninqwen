"""Step time of the appended ROM+MAC pipelines (experiments/circuit/memory/mac_layer.py, MEMORY_DESIGN.md s.8) on the fused-executor style
sparse GPU treatment, on top of the measured chip-circuit step (stories260k, tuned B=1).  Structure is exactly mac_layer.run(K=230, H=2560, I=9216):
ROM (9216 units, gate nnz 29, up 1, down 230 scattered only for active units), PRODUCT (1840 units, gate 2, up 1, down 1), DIGITIZE (5152 units, gate 2, up 1, down 2),
LOSUM (368 units, gate 1, up 1, down 1).  TIMING-ONLY: random table values / inputs with the right sparsity pattern (exactly one ROM unit matches the address),
kernel arithmetic is relu-gated (a*relu(b)) like the chip executor, not the bf16-silu of mac_layer, so values are not checked here (mac_layer.py checks values).
Also times the dense HF treatment (stock Qwen3_5DecoderLayer at 4B width, 65 layers, bf16) and reports memory.
usage: python mac_exec_bench.py ELF [--P 13] [--dense]"""
import argparse, json, math, time
import numpy as np, torch, triton, triton.language as tl

@triton.jit
def _ffn_mo(r, yn, D, U, rpa, ixa, wa, rpb, ixb, wb, rpd, ixd, wd, BLOCK: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr, KD: tl.constexpr):
    pid = tl.program_id(0); pl = tl.program_id(1)
    rb, yb = r + pl * D, yn + pl * D
    u = pid * BLOCK + tl.arange(0, BLOCK); m = u < U
    ka = tl.arange(0, KA); a0 = tl.load(rpa + u, mask=m, other=0); na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    ma = m[:, None] & (ka[None, :] < na[:, None])
    a = tl.sum(tl.load(wa + a0[:, None] + ka[None, :], mask=ma, other=0.0) * tl.load(rb + tl.load(ixa + a0[:, None] + ka[None, :], mask=ma, other=0), mask=ma, other=0.0), axis=1)
    kb = tl.arange(0, KB); b0 = tl.load(rpb + u, mask=m, other=0); nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    mb = m[:, None] & (kb[None, :] < nb[:, None])
    b = tl.sum(tl.load(wb + b0[:, None] + kb[None, :], mask=mb, other=0.0) * tl.load(rb + tl.load(ixb + b0[:, None] + kb[None, :], mask=mb, other=0), mask=mb, other=0.0), axis=1)
    val = a * tl.maximum(b, 0.0)
    kd = tl.arange(0, KD); d0 = tl.load(rpd + u, mask=m, other=0); nd = tl.load(rpd + u + 1, mask=m, other=0) - d0
    md = (m & (val != 0.0))[:, None] & (kd[None, :] < nd[:, None])
    di = tl.load(ixd + d0[:, None] + kd[None, :], mask=md, other=0); dw = tl.load(wd + d0[:, None] + kd[None, :], mask=md, other=0.0)
    tl.atomic_add(yb + di, dw * val[:, None], mask=md)

def csr(rows, dev):
    rp = np.concatenate([[0], np.cumsum([len(i) for i, _ in rows])]).astype(np.int32)
    ix = np.concatenate([i for i, _ in rows] + [np.zeros(1, np.int32)]).astype(np.int32); w = np.concatenate([x for _, x in rows] + [np.zeros(1, np.float32)]).astype(np.float32)
    return [torch.tensor(t, device=dev) for t in (rp, ix, w)]

def build_stages(K, H, I, dev, rng):
    n = math.ceil(math.log2(I)); dims = {}; cur = 0
    def new(nm, c=1):
        nonlocal cur; dims[nm] = cur; cur += c; return dims[nm]
    ab = new("addr", 2 * n); car = new("car"); W = new("W", K); A = new("A", 8 * K); G = math.ceil(K / 5); C = new("C", G * 8); HS = new("HS", 8 * 2); LS = new("LS", 8 * 4); assert cur <= H
    one = lambda i, w=1.0: (np.array([i], np.int32), np.array([w], np.float32))
    rom_g, rom_u, rom_d = [], [], []
    for j in range(I):
        gi = np.array([ab + b if (j >> b) & 1 else ab + n + b for b in range(n)] + [car], np.int32); rom_g.append((gi, np.array([1.0] * n + [-(n - 0.5)], np.float32)))
        rom_u.append(one(car)); rom_d.append((np.arange(W, W + K, dtype=np.int32), rng.integers(0, 16, K).astype(np.float32) / 16))
    pr_g, pr_u, pr_d = [], [], []
    for k in range(K):
        for p in range(8):
            pr_g.append((np.array([W + k, car], np.int32), np.array([1.0, 0.5], np.float32))); pr_u.append(one(A + p * K + k)); pr_d.append(one(C + (k // 5) * 8 + p))
    dg_g, dg_u, dg_d = [], [], []
    for g in range(G):
        for p in range(8):
            for t in range(16, 128, 16):
                for s in (1, -1):
                    ci = C + g * 8 + p; dg_g.append((np.array([ci, car], np.int32), np.array([1.0, -(t - 1.0)], np.float32))); dg_u.append(one(car))
                    dg_d.append((np.array([HS + p * 2 + g // 18, ci], np.int32), np.array([s / 16, -s], np.float32)))
    ls_g, ls_u, ls_d = [], [], []
    for g in range(G):
        for p in range(8):
            ls_g.append(one(car, 1.0)); ls_u.append(one(C + g * 8 + p)); ls_d.append(one(LS + p * 4 + g // 8, 1 / 16))
    st = []
    for nm, g_, u_, d_, BL, KD in (("ROM", rom_g, rom_u, rom_d, 8, 256), ("PRODUCT", pr_g, pr_u, pr_d, 32, 2), ("DIGITIZE", dg_g, dg_u, dg_d, 32, 2), ("LOSUM", ls_g, ls_u, ls_d, 32, 2)):
        st.append(dict(name=nm, U=len(g_), BLOCK=BL, KA=2, KB=max(2, 1 << (max(len(i) for i, _ in g_) - 1).bit_length()), KD=KD, a=csr(u_, dev), b=csr(g_, dev), d=csr(d_, dev)))
    r = torch.zeros(H, device=dev); r[car] = 1.0; addr = int(rng.integers(0, I))
    for b in range(n): r[ab + b] = float((addr >> b) & 1); r[ab + n + b] = float(1 - ((addr >> b) & 1))
    r[A:A + 8 * K] = torch.tensor(rng.integers(0, 2, 8 * K), dtype=torch.float32, device=dev); r[W:W + K] = 1.0; r[C:C + G * 8] = torch.tensor(rng.integers(0, 120, G * 8), dtype=torch.float32, device=dev)
    return st, r

def launch(s, r, y, H, P):
    _ffn_mo[(triton.cdiv(s["U"], s["BLOCK"]), P)](r, y, H, s["U"], *s["a"], *s["b"], *s["d"], BLOCK=s["BLOCK"], KA=s["KA"], KB=s["KB"], KD=s["KD"], num_warps=s.get("nw", 4))

def time_graph(fn, reps=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): g.replay()
    torch.cuda.synchronize(); return (time.time() - t) / reps * 1e3

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--P", type=int, default=13); ap.add_argument("--K", type=int, default=230); ap.add_argument("--base_ms", type=float, default=0.2015)
    ap.add_argument("--dense", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = "cuda"; H, I = 2560, 9216; rng = np.random.default_rng(0); st, r = build_stages(a.K, H, I, dev, rng)
    res = dict(gpu=torch.cuda.get_device_name(), K=a.K, H=H, I=I, P=a.P, base_chip_step_ms=a.base_ms, stages={s["name"]: dict(U=s["U"], nnz_down=int(s["d"][0][-1])) for s in st})
    # per-stage tuning of num_warps (one pipeline)
    y = torch.zeros(a.P, H, device=dev); rr = r[None].repeat(a.P, 1).contiguous()
    for s in st:
        best = None
        for nw in (1, 2, 4, 8):
            s["nw"] = nw; ms = time_graph(lambda: launch(s, rr, y, H, 1)) 
            if best is None or ms < best[0]: best = (ms, nw)
        s["nw"] = best[1]; res["stages"][s["name"]].update(single_pipeline_ms=best[0], num_warps=best[1]); print(s["name"], best, flush=True)
    seq = time_graph(lambda: [launch(s, rr[p:p + 1], y[p:p + 1], H, 1) for p in range(a.P) for s in st])
    bat = time_graph(lambda: [launch(s, rr, y, H, a.P) for s in st])
    res.update(sparse_sequential_all_pipelines_ms=seq, sparse_stage_batched_over_pipelines_ms=bat)
    for nm, extra in (("sequential", seq), ("stage_batched", bat)):
        tot = a.base_ms + extra + a.P * 0.011      # + 13 DeltaNet-read layers at the measured ~11 us per head kernel
        res[f"total_ms_per_step_{nm}"] = tot; res[f"tok_per_s_{nm}"] = 1e3 / tot
    res["sparse_state_mb"] = sum(sum(t.numel() * t.element_size() for t in s[k]) for s in st for k in ("a", "b", "d")) / 2**20 * a.P
    print(json.dumps(res), flush=True)
    if a.dense:   # dense HF treatment of the 65 appended sublayers: stock Qwen3_5MLP (bf16) per layer, DeltaNet-read layers NOT included (lower bound)
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
        torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated(); n = 5 * a.P - a.P
        cfg = Qwen3_5TextConfig(hidden_size=H, intermediate_size=I, vocab_size=8)
        try:
            with torch.device(dev): layers = [Qwen3_5MLP(cfg, I).to(torch.bfloat16).eval() for _ in range(n)]
            x = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16)
            @torch.inference_mode()
            def f():
                h = x
                for l in layers: h = h + l(h)
                return h
            for _ in range(3): f()
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20): f()
            torch.cuda.synchronize(); dms = (time.time() - t) / 20 * 1e3
            res.update(dense_hf_mlp_layers=n, dense_hf_ms_per_step=dms, dense_hf_mem_gb=(torch.cuda.memory_allocated() - m0) / 2**30, dense_hf_params_b=sum(p.numel() for l in layers for p in l.parameters()) / 1e9)
        except torch.cuda.OutOfMemoryError: res["dense_hf"] = "OOM"
        print(json.dumps({k: v for k, v in res.items() if k.startswith("dense")}), flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
main()
