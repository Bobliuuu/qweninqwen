import sys, torch
from run_block import load_w
from tiled import *
torch.manual_seed(2)
nm = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
W = load_w(nm)       # real
Oall, Kall = W.shape
nL = -(-Kall // (63 * 16))
X = torch.randn(1, Kall) * 3
lay = Layout(); spec = StateMV(lay, nL=nL, H=16, K=Kall, B=1e6)
model = make_outer(lay, ['linear_attention'] * nL, 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8); spec.compile(model); model.to(dev)
y, st = project(spec, model, W, X, return_stats=True)
ref = X.double() @ W.double().T
print(nm, W.shape, 'nL', nL, st)
print('maxrel vs max|ref|', ((y - ref).abs().max() / ref.abs().max()).item(), 'rel L2', ((y-ref).norm()/ref.norm()).item())
