"""Audit of the layer-major executor assumption: 'write frames are idle (exactly zero contribution, no state change) in every layer
except their bank layer'.  Runs vs2.Runner twice on the same program: (a) as shipped (write jobs only to their bank layer),
(b) write jobs fed to EVERY layer (what a single stock forward over the whole frame sequence would do). Compares control-frame
final outputs and the trace of every layer.   usage: equiv_idle.py d256|d512 [ntok]"""
import sys, os, re, inspect, time, torch
torch.set_grad_enabled(False)
sys.path.insert(0, os.environ.get('QIQ_TRUEQIQ', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qweninqwen')))
import test_vs2 as T
import vs2
from vs2 import *
scale = sys.argv[1]; ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 1
c = {'d256': Cfg(256, 4, 2, 64, 512, 2, 512, I_MLP=4096), 'd512': Cfg(512, 8, 4, 64, 1024, 2, 1024, I_MLP=4096)}[scale]
toy = T.make_toy(c); Wd, P = T.weights(toy, c); S = build_layout(c); plan = make_plan(S, c)
src = inspect.getsource(Runner.run)
assert "or pr.jobs[it[1]]['li'] == li" in src
src_all = src.replace("or pr.jobs[it[1]]['li'] == li", "or True")
src_all = "\n".join(l[4:] if l.startswith("    ") else l for l in src_all.splitlines()).replace("def run(", "def run_all(", 1)
ns = dict(vars(vs2)); exec("import torch\n" + src_all.replace("@torch.no_grad()", ""), ns); run_all = ns['run_all']
emb = lambda t: toy.model.embed_tokens.weight[t].float()
prompt = [5, 17, 301, 42]
def go(mode):
    rn = Runner(S, c, plan, P, dev='cpu'); outs = []
    pr = Prog(S, c, plan, 'cpu'); init_steps(pr)
    (rn.run if mode == 'ship' else (lambda p, **k: run_all(rn, p, **k)))(pr)
    for step in range(ntok):
        pr = Prog(S, c, plan, 'cpu'); token_program(pr, prompt[step], step, Wd, emb)
        f = rn.run if mode == 'ship' else (lambda p, **k: run_all(rn, p, **k))
        Hs, tr = f(pr, trace_rows={'all': torch.arange(len(pr.ctrl))})
        outs.append((Hs, tr))
    return outs
t0 = time.time(); A = go('ship'); print('shipped executor', time.time() - t0, flush=True)
t0 = time.time(); Bv = go('all'); print('all-layers executor', time.time() - t0, flush=True)
worst = 0; worst_layer = None; worst_abs = 0
for (Ha, ta), (Hb, tb) in zip(A, Bv):
    d = (Ha - Hb).abs(); scale_ = Ha.abs().max().item()
    worst = max(worst, d.max().item() / scale_); worst_abs = max(worst_abs, d.max().item())
    for k in ta:
        dd = (ta[k] - tb[k]).abs().max().item()
        if dd > 0 and worst_layer is None: worst_layer = (k, dd)
print(f'final control outputs: max|diff|={worst_abs:.3e}  rel to max|H|={worst:.3e}; first layer with any difference: {worst_layer}')
print('EQUIVALENT (bitwise)' if worst_abs == 0 else ('EQUIVALENT (<=1e-6 rel)' if worst < 1e-6 else 'NOT EQUIVALENT'))
