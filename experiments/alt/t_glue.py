import sys, torch, time
sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '.'))
from glue import *
C,O,R,nt = [int(a) for a in sys.argv[1:5]]
dev = sys.argv[5] if len(sys.argv)>5 else 'cpu'
torch.manual_seed(0)
K=C*nt
if len(sys.argv)>6:
    from mlpmul import load_w
    Wfull = load_w(sys.argv[6]); Wm = Wfull[:O,:K].contiguous()
else: Wm = torch.randn(O,K)*0.03
x = torch.randn(K)*2+.5; x[3]=300.
mv = Matvec(C,O,R,nt,device=dev)
t=time.time(); yb,xg = mv.run(Wm,x); print('D',mv.D,'hd',mv.hd,'T',mv.T,'time',time.time()-t)
ref = (Wm.double()@x.double()).cpu()
ymult = yb.reshape(-1).double().cpu()  # partials of last tile only (skip)
mv_skip=True
print('gathered vector  max err/|ref|max:', ((xg.double().cpu()-ref).abs().max()/ref.abs().max()).item(), '')
