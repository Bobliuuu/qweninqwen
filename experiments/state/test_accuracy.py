"""Accuracy suite for the state matvec (real Qwen2.5-0.5B weights).  usage: python test_accuracy.py [cpu|cuda]"""
import sys, json, os, torch
from run_block import load_w
from tiled import *
import gdn_impl; gdn_impl.set_impl('torch', 'torch')
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
torch.manual_seed(0)
res = {}
def mk(nL, H, K, B):
    lay = Layout(); spec = StateMV(lay, nL=nL, H=H, K=K, B=B)
    m = make_outer(lay, ['linear_attention'] * nL, 8, gdn=dict(hk=H, hv=H, dk=128, dv=128), vocab=8); spec.compile(m); return spec, m.to(dev)
def err(y, ref): return dict(rel_l2=((y - ref).norm() / ref.norm()).item(), max_over_max=((y - ref).abs().max() / ref.abs().max()).item(),
                             max_elem_rel=((y - ref).abs() / ref.abs().clamp(min=1e-2 * ref.abs().max())).max().item())
W = load_w('model.layers.0.self_attn.q_proj.weight')
# (a) single real tile: 126 outputs x 63 inputs, 1 head (the 'one 128x128 state block')
spec, m = mk(1, 1, 63, 1e6); X = torch.randn(4, 63) * 3
y = project(spec, m, W[:126, :63], X); res['single_tile_126x63_1head'] = err(y, X.double() @ W[:126, :63].double().T)
# (b) full real projections
for nm, nL in [('model.layers.0.self_attn.q_proj.weight', 1), ('model.layers.0.self_attn.k_proj.weight', 1), ('model.layers.0.mlp.gate_proj.weight', 1),
               ('model.layers.0.mlp.down_proj.weight', 5), ('model.layers.5.mlp.up_proj.weight', 1)]:
    Wm = load_w(nm); K = Wm.shape[1]; spec, m = mk(nL, 16, K, 1e6); X = torch.randn(3, K) * 3
    y = project(spec, m, Wm, X); res[nm] = err(y, X.double() @ Wm.double().T)
# (c) ballast / activation-scale sweep
Wm = load_w('model.layers.0.self_attn.q_proj.weight')
for B in (1e4, 1e5, 1e6, 1e7):
    spec, m = mk(1, 16, 896, B)
    for sc in (1, 30, 300):
        X = torch.randn(2, 896) * sc; y = project(spec, m, Wm, X); res[f'B{B:g}_xscale{sc}'] = err(y, X.double() @ Wm.double().T)
# (d) bf16-rounded activation (inherent input rounding, engine is exact on whatever x it is given)
X = torch.randn(2, 896) * 3; spec, m = mk(1, 16, 896, 1e6)
y = project(spec, m, Wm, X.bfloat16().float()); ref32 = X.double() @ Wm.double().T
res['bf16_rounded_x_vs_fp32_x_exact'] = err(y, ref32); res['bf16_rounded_x_vs_bf16_x_exact'] = err(y, X.bfloat16().double() @ Wm.double().T)
# (e) reads are non-destructive & state resident: 3 reads of same state give same answers
X = torch.randn(3, 896); y = project(spec, m, Wm, X); res['resident_multi_read'] = err(y, X.double() @ Wm.double().T)
for k, v in res.items(): print(f'{k:55s}', {a: float(f'{b:.2e}') for a, b in v.items()})
json.dump(res, open(f'results_accuracy_{dev}.json', 'w'), indent=1)
