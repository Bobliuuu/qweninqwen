"""Fused executor (FreeRun3, A100) on the weight-fetch circuit: free-running exactness of N-VDOT chunks
(ROW tokens embedded by table gather, every generated token compared with the reference trace) and ms/step vs
the 11-layer baseline. usage: python bench_wf_exec.py [--steps 3000] ; needs PYTHONPATH=<riscv> and tests/ on path."""
import argparse, json, os, random, sys, time
os.environ.setdefault("WF_ROW_SIZE", "64")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from chipformer.rv import core, fusedexec, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble
import test_rv_wfetch as T

ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--out", default=None)
a = ap.parse_args()
res = dict(gpu=torch.cuda.get_device_name(), runs=[])

def run(name, c, prog, mach_kw, steps):
    m0 = ref.Machine.from_program(prog)
    for k, v in mach_kw.items(): setattr(m0, k, v)
    want = np.asarray(m0.run())
    m1 = ref.Machine.from_program(prog)
    for k, v in mach_kw.items(): setattr(m1, k, v)
    core.snapshot(c, m1)
    fr = fusedexec.FreeRun3(c, 1, "cuda"); fr.load(0, S.tok("GO")); fr.capture()
    out = torch.zeros(1, steps, dtype=torch.long, device="cuda")
    torch.cuda.synchronize(); t = time.time()
    for s in range(steps): fr.graph.replay(); out[:, s].copy_(fr.tok)
    torch.cuda.synchronize(); dt = time.time() - t
    got = out.cpu().numpy()[0]; n = len(want) - 1
    ok = bool((got[:n] == want[1:n + 1]).all())
    r = dict(name=name, layers=c.stats()["layers"], dims=c.stats()["dims"], trace_tokens=len(want), steps_timed=steps,
             ms_per_step=dt / steps * 1e3, tok_per_s=steps / dt, free_run_exact_over_trace=ok)
    print(json.dumps(r), flush=True); res["runs"].append(r); del fr; torch.cuda.empty_cache()

# baseline: 11-layer circuit, the same chunk on the WROM
S.ROW_TABLE = None
for W, n in ((1, 300), (8, 300), (16, 600)):
    rng = random.Random(100 + W)
    _, tab0, _, exp, _ = T.chunk(rng, W, n)
    bs, _, wrom, exp2, _ = T.chunk(random.Random(100 + W), W, n, rows=False)
    prog = assemble(bs + "\n.wrom\n" + "\n".join(f".byte {', '.join(str(b) for b in wrom[i:i + 64])}" for i in range(0, len(wrom), 64)))
    c0 = core.build()
    run(f"baseline11_wrom_chunk_N{n}", c0, prog, {}, a.steps); del c0
    S.ROW_TABLE = tab0
    c = core.build(wrow=W)
    src, tab, _, _, _ = T.chunk(random.Random(100 + W), W, n)
    run(f"wf_W{W}_chunk_N{n}", c, assemble(src), dict(rowtab=tab, wrow_w=W), a.steps); del c
    S.ROW_TABLE = None
if a.out: json.dump(res, open(a.out, "w"), indent=1)
