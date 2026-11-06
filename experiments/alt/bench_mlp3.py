import sys, time, torch
sys.path.insert(0,'.')
from mlpmul import MulMLP
torch.backends.cuda.matmul.allow_tf32=False
def tm(f,n=5):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n
for (C,R,T) in [(896,1,4096),(224,4,4096),(128,7,8192),(64,14,16384)]:
    mm=MulMLP(C,R,device='cuda',ltype='full_attention'); F=mm.frames(torch.randn(T,R,C,device='cuda')*.03,torch.randn(C,device='cuda'))
    y=mm.run(F); ref=(F[:,mm.L['W']].reshape(T,R,C).double()@F[0,mm.L['X']].double())
    err=((y.double()-ref).abs().max()/ref.abs().max()).item()
    t=tm(lambda: mm.run(F))
    print(f"FA-layer C{C} R{R} T{T}: {t*1e3:.1f}ms MACps {T*R*C/t/1e9:.3f}G s/tok494M {494e6/(T*R*C/t):.2f} err {err:.1e}",flush=True)
