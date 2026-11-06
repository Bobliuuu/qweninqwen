"""Kernel timeline of one graph-replayed executor step: per kernel duration and gap to the previous kernel (CUPTI via torch.profiler)."""
import argparse, json
import torch
from torch.profiler import profile, ProfilerActivity
from chipformer.rv import core, elf, ref, fusedexec
from chipformer.rv import spec as S
from tune_warps import W
ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--wf", type=int, default=4); ap.add_argument("--wh", type=int, default=32); ap.add_argument("--steps", type=int, default=3000)
a = ap.parse_args()
for n in ("_ffn3", "_ffn4"): setattr(fusedexec, n, W(getattr(fusedexec, n), num_warps=a.wf))
for n in ("_head3", "_head4"): setattr(fusedexec, n, W(getattr(fusedexec, n), num_warps=a.wh))
c = core.build(); core.snapshot(c, ref.Machine.from_program(elf.program(a.elf)))
fr = fusedexec.FreeRun3(c, 1, "cuda"); fr.load(0, S.tok("GO")); fr.capture()
for _ in range(a.steps): fr.graph.replay()   # reach a typical state
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(3): fr.graph.replay()
    torch.cuda.synchronize()
ev = sorted([e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda e: e.time_range.start)
n = len(ev) // 3; ev = ev[-n:]
prev = None; tot = 0
for i, (e, st) in enumerate(zip(ev, list(fr.ex.steps) + [None, None])):
    gap = 0 if prev is None else (e.time_range.start - prev) / 1e3
    prev = e.time_range.end
    print(json.dumps(dict(k=i, name=e.name[:14], dur_us=round(e.time_range.elapsed_us(), 1), gap_us=round(gap, 1), kind=(st or {}).get("kind"), U=(st or {}).get("U"), V=(st or {}).get("V"))))
print("kernels/step", n, "span us", round((ev[-1].time_range.end - ev[0].time_range.start) / 1e3, 1), "sum dur us", round(sum(e.time_range.elapsed_us() for e in ev), 1))
