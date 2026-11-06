"""Sweep triton num_warps / num_stages for the fused executor's kernels (B=1 step time, CUDA graph), checking exactness vs the reference prefix.
usage: python tune_warps.py ELF --steps N --want want.npy"""
import argparse, json, time, itertools
import numpy as np, torch
from chipformer.rv import core, elf, ref, fusedexec
from chipformer.rv import spec as S

class W:
    def __init__(self, k, **kw): self.k, self.kw = k, kw
    def __getitem__(self, grid):
        f = self.k[grid]
        return lambda *a, **kw: f(*a, **{**self.kw, **kw})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--want", default="want.npy")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    prog = elf.program(a.elf); want = np.load(a.want)[1:a.steps + 1]
    c = core.build(); orig = {n: getattr(fusedexec, n) for n in ("_ffn3", "_ffn4", "_head3", "_head4")}
    res = []
    for wf, wh in itertools.product((1, 2, 4, 8), (1, 2, 4, 8)):
        for n in ("_ffn3", "_ffn4"): setattr(fusedexec, n, W(orig[n], num_warps=wf))
        for n in ("_head3", "_head4"): setattr(fusedexec, n, W(orig[n], num_warps=wh))
        core.snapshot(c, ref.Machine.from_program(prog))
        fr = fusedexec.FreeRun3(c, 1, "cuda"); fr.load(0, S.tok("GO")); fr.capture(); torch.cuda.synchronize()
        out = torch.zeros(1, a.steps, dtype=torch.long, device="cuda"); t = time.time()
        for s in range(a.steps): fr.graph.replay(); out[:, s].copy_(fr.tok)
        torch.cuda.synchronize(); dt = time.time() - t
        r = dict(warps_ffn=wf, warps_head=wh, ms_per_step=dt / a.steps * 1e3, tok_per_s=a.steps / dt, exact=bool((out.cpu().numpy() == want[None]).all()))
        print(json.dumps(r), flush=True); res.append(r); del fr; torch.cuda.empty_cache()
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
if __name__ == '__main__':
    main()
