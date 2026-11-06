"""Per-sublayer kernel time of the fused executor (scratch buffers, CUDA events). usage: python prof_steps.py ELF [--wf 4 --wh 32]"""
import argparse, json, torch
from chipformer.rv import core, elf, ref, fusedexec
from chipformer.rv import spec as S
from tune_warps import W
ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--wf", type=int, default=4); ap.add_argument("--wh", type=int, default=32)
ap.add_argument("--reps", type=int, default=50); ap.add_argument("--out", default=None); a = ap.parse_args()
for n in ("_ffn3", "_ffn4"): setattr(fusedexec, n, W(getattr(fusedexec, n), num_warps=a.wf))
for n in ("_head3", "_head4"): setattr(fusedexec, n, W(getattr(fusedexec, n), num_warps=a.wh))
c = core.build(); core.snapshot(c, ref.Machine.from_program(elf.program(a.elf)))
fr = fusedexec.FreeRun3(c, 1, "cuda"); fr.load(0, S.tok("GO")); fr.tune()
scratch = [torch.zeros_like(fr.X[0]) for _ in range(5)] + [torch.zeros_like(fr.dummy)]
rows = []
for i, st in enumerate(fr.ex.steps):
    f = (lambda: fr._ffn(i, st, *scratch, False, fr.use4.get(i, False))) if st["kind"] == "ffn" else (lambda: fr._head(i, st, *scratch, False, fr.use_h4.get(i, False)))
    if st["kind"] == "head": S0, H0 = st["S"].clone(), st["hist"].clone()
    for _ in range(3): f()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(a.reps): f()
    e1.record(); torch.cuda.synchronize()
    if st["kind"] == "head": st["S"].copy_(S0); st["hist"].copy_(H0)
    d = dict(i=i, kind=st["kind"], us=e0.elapsed_time(e1) / a.reps * 1e3)
    d.update(U=st["U"], variant4=fr.use4.get(i)) if st["kind"] == "ffn" else d.update(V=st["V"], KQ=st["KQ"], KK=st["KK"], KV=st["KV"], KZ=st["KZ"], QHOT=st["QHOT"], variant4=fr.use_h4.get(i))
    rows.append(d); print(json.dumps(d), flush=True)
print("total us", sum(r["us"] for r in rows), "ffn", sum(r["us"] for r in rows if r["kind"] == "ffn"), "head", sum(r["us"] for r in rows if r["kind"] == "head"))
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
