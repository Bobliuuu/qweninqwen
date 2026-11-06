"""Per-sublayer autotune of triton num_warps for the fused executor (B=1): kernel durations from CUPTI under a grid of
(ffn warps, head warps); pick the best warps per sublayer; verify exactness token-for-token and report ms/step.
usage: python autotune_layers.py ELF --want want.npy --steps 20000 --out tuned.json"""
import argparse, json, time, itertools
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity
from chipformer.rv import core, elf, ref, fusedexec
from chipformer.rv import spec as S

CUR = {"i": 0, "wf": {}, "wh": {}, "dwf": 4, "dwh": 4}
class PL:
    """kernel wrapper: num_warps looked up per sublayer index."""
    def __init__(self, k, tab, dflt): self.k, self.tab, self.dflt = k, tab, dflt
    def __getitem__(self, grid):
        f = self.k[grid]
        return lambda *a, **kw: f(*a, num_warps=CUR[self.tab].get(CUR["i"], CUR[self.dflt]), **kw)

def setup(fr):
    of, oh = fr._ffn, fr._head
    fr._ffn = lambda i, *a, **k: (CUR.__setitem__("i", i), of(i, *a, **k))[1]
    fr._head = lambda i, *a, **k: (CUR.__setitem__("i", i), oh(i, *a, **k))[1]

def durations(fr, n_pre=200):
    for _ in range(n_pre): fr.graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): fr.graph.replay()
        torch.cuda.synchronize()
    ev = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
    n = len(ev) // 5
    return np.array([np.median([ev[r * n + k].time_range.elapsed_us() for r in range(5)]) for k in range(n)])

def build(c, prog):
    core.snapshot(c, ref.Machine.from_program(prog))
    fr = fusedexec.FreeRun3(c, 1, "cuda"); fr.load(0, S.tok("GO")); setup(fr); fr.capture(); return fr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--want", default="want.npy"); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--out", default=None)
    a = ap.parse_args(); prog = elf.program(a.elf); want = np.load(a.want)[1:a.steps + 1]
    for n, tab in (("_ffn3", "wf"), ("_ffn4", "wf"), ("_head3", "wh"), ("_head4", "wh")):
        setattr(fusedexec, n, PL(getattr(fusedexec, n), tab, "d" + tab))
    c = core.build(); res = {}
    for wf, wh in itertools.product((2, 4, 8), (4, 8, 16, 32)):
        CUR.update(wf={}, wh={}, dwf=wf, dwh=wh); fr = build(c, prog)
        res[(wf, wh)] = durations(fr); print(wf, wh, round(res[(wf, wh)].sum(), 1), flush=True); del fr; torch.cuda.empty_cache()
    steps = None
    fr = build(c, prog); steps = fr.ex.steps; del fr
    best_f, best_h = {}, {}
    for i, st in enumerate(steps):
        k = i   # kernel k == sublayer i (decode kernels follow)
        if st["kind"] == "ffn": best_f[i] = min({w for w, _ in res}, key=lambda w: min(res[(w, h)][k] for h in {h for _, h in res}))
        else: best_h[i] = min({h for _, h in res}, key=lambda h: min(res[(f, h)][k] for f in {f for f, _ in res}))
    CUR.update(wf=best_f, wh=best_h)
    from tune_warps import W
    orig_e = fusedexec._decode_embed; best_wd, bd = 4, 1e9
    for wd in (1, 2, 4, 8, 16):
        fusedexec._decode_embed = W(orig_e, num_warps=wd); fr = build(c, prog); d = durations(fr)
        print("decode_embed warps", wd, "us", round(float(d[-1]), 1), flush=True)
        if d[-1] < bd: bd, best_wd = d[-1], wd
        del fr; torch.cuda.empty_cache()
    fusedexec._decode_embed = W(orig_e, num_warps=best_wd); fr = build(c, prog)
    d = durations(fr); print("tuned per-step kernel sum us", round(d.sum(), 1), "ffn warps", best_f, "head warps", best_h, flush=True)
    del fr; torch.cuda.empty_cache(); fr = build(c, prog)      # profiling replays advanced the machine: restart from the snapshot
    out = torch.zeros(1, a.steps, dtype=torch.long, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for s in range(a.steps): fr.graph.replay(); out[:, s].copy_(fr.tok)
    torch.cuda.synchronize(); dt = time.time() - t
    r = dict(ms_per_step=dt / a.steps * 1e3, tok_per_s=a.steps / dt, exact=bool((out.cpu().numpy() == want[None]).all()), decode_embed_warps=best_wd, ffn_warps=best_f, head_warps=best_h,
             grid_kernel_sum_us={f"{k[0]},{k[1]}": float(v.sum()) for k, v in res.items()})
    print(json.dumps(r)); 
    if a.out: json.dump(r, open(a.out, "w"), indent=1)
main()
