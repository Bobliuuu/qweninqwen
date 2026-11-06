"""Run an ELF on the fused executor with the per-sublayer tuned num_warps (results/autotune.json), free-running, checked token-for-token vs ref.Machine.
usage: python run_tuned.py ELF [--steps N] [--out res.json]"""
import argparse, json, time
import numpy as np, torch
from chipformer.rv import core, elf, ref, fusedexec
from chipformer.rv import spec as S
import autotune_layers as AL
from tune_warps import W
ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--steps", type=int, default=0); ap.add_argument("--tuned", default="autotune.json"); ap.add_argument("--out", default=None)
a = ap.parse_args(); cfg = json.load(open(a.tuned))
for n, tab in (("_ffn3", "wf"), ("_ffn4", "wf"), ("_head3", "wh"), ("_head4", "wh")):
    setattr(fusedexec, n, AL.PL(getattr(fusedexec, n), tab, "d" + tab))
fusedexec._decode_embed = W(fusedexec._decode_embed, num_warps=cfg.get("decode_embed_warps", 4))
AL.CUR.update(wf={int(k): v for k, v in cfg["ffn_warps"].items()}, wh={int(k): v for k, v in cfg["head_warps"].items()})
prog = elf.program(a.elf); m = ref.Machine.from_program(prog); t = time.time(); want = np.asarray(m.run(10**9))
st = ref.trace_stats(list(want)); print(f"reference: {st['tokens']:,} tokens, {st['instructions']:,} instructions ({time.time()-t:.0f} s): {m.out!r}", flush=True)
n = a.steps or len(want) - 1
c = core.build(); fr = AL.build(c, prog)
torch.cuda.synchronize(); t = time.time(); out = torch.zeros(1, n, dtype=torch.long, device="cuda")
for s in range(n): fr.graph.replay(); out[:, s].copy_(fr.tok)
torch.cuda.synchronize(); dt = time.time() - t; got = out.cpu().numpy()[0]
bad = next((i for i in range(n) if got[i] != want[i + 1]), None)
r = dict(elf=a.elf, steps=n, instructions=st["instructions"], seconds=dt, ms_per_step=dt / n * 1e3, tok_per_s=n / dt, exact=bad is None, first_mismatch=bad, gpu=torch.cuda.get_device_name())
print(json.dumps(r), flush=True)
if a.out: json.dump(r, open(a.out, "w"), indent=1)
