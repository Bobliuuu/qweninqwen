import sys, torch
from run_block import load_w
from tiled import *
B = 1e6; torch.manual_seed(1)
W = load_w('model.layers.0.self_attn.q_proj.weight')       # real 896x896 Qwen2.5-0.5B
X = torch.randn(2, 896) * 4
lay = Layout(); spec = StateMV(lay, nL=1, H=16, K=896, B=B)
model = make_outer(lay, ['linear_attention'], 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8); spec.compile(model)
y, st = project(spec, model, W, X, return_stats=True)
ref = X.double() @ W.double().T
print(st); e = (y - ref).abs().max(1).values / ref.abs().max(1).values
print('full q_proj 896x896 maxrel vs max|ref|', e.tolist(), ' mean rel', ((y-ref).norm(dim=1)/ref.norm(dim=1)).tolist())
