import sys, time, torch
sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '.'))
from mlpmul import *
torch.manual_seed(0)
name = sys.argv[1] if len(sys.argv)>1 else 'model.layers.3.self_attn.q_proj.weight'
Wt = load_w(name)               # [O,K]
O,K = Wt.shape; Kc = int(sys.argv[2]) if len(sys.argv)>2 else K
x = torch.randn(Kc)*3+0.5; x[5]=300.
mm = MulMLP(Kc, 1)
n = min(O, 64)
F = mm.frames(Wt[:n,:Kc].reshape(n,1,Kc), x)
y = mm.run(F)[:,0]
ref = (Wt[:n,:Kc].double() @ x.double())
rel = ((y.double()-ref).abs()/ref.abs().clamp_min(1e-9))
print(name, 'C',Kc,'max rel',rel.max().item(),'median',rel.median().item(),
      'max abs/|ref|max', ((y.double()-ref).abs().max()/ref.abs().max()).item())
ref32 = (Wt[:n,:Kc] @ x)
print('torch fp32 matmul vs fp64: max rel', ((ref32.double()-ref).abs()/ref.abs()).max().item())
