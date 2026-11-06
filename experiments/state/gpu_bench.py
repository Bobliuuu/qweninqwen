import sys, time, torch
from run_block import load_w
from tiled import *
import gdn_impl as kernels
nm = sys.argv[1]; nL_arg = int(sys.argv[2]); seg = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
import os
import gdn_impl as kernels
kernels.set_impl(os.environ.get('CHUNK','torch'), os.environ.get('REC','torch'))
dev = 'cuda'
torch.manual_seed(2)
W = load_w(nm); Oall, Kall = W.shape
X = torch.randn(1, Kall) * 3
G = int(os.environ.get('G', 1)); T = -(-Kall // 63); KL = -(-T // 16); nL = KL * G
lay = Layout(); spec = StateMV(lay, nL=nL, H=16, K=Kall, B=1e6, G=G)
model = make_outer(lay, ['linear_attention'] * nL, 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8); spec.compile(model); model.to(dev)
for rep in range(2):
    y, st = project(spec, model, W, X, seg=seg, return_stats=True)
ref = X.double() @ W.double().T
print(nm, tuple(W.shape), 'nL', nL, 'G', G, 'seg', seg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()})
print('rel L2', ((y - ref).norm() / ref.norm()).item(), 'maxrel', ((y - ref).abs().max() / ref.abs().max()).item())
