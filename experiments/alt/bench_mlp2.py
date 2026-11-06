import sys, time, torch, json
sys.path.insert(0,'.')
from mlpmul import MulMLP
torch.backends.cuda.matmul.allow_tf32=False
def tm(f,n=5):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n
for (C,R,T) in [(896,1,1024),(896,1,4096),(224,4,4096),(128,7,8192),(64,14,16384)]:
    mm=MulMLP(C,R,device='cuda'); F=mm.frames(torch.randn(T,R,C,device='cuda')*.03,torch.randn(C,device='cuda'))
    lay=mm.m.model.layers[0]
    tfull=tm(lambda: mm.run(F)); tmlp=tm(lambda: lay.mlp(F[None]))
    fl=2*3*T*mm.D*mm.I
    print(f"C{C} R{R} T{T}: full {tfull*1e3:.1f}ms  mlp-only {tmlp*1e3:.1f}ms  mlp TFLOPs {fl/tmlp/1e12:.1f}  MACps(mlp) {T*R*C/tmlp/1e9:.2f}G  s/tok494M {494e6/(T*R*C/tmlp):.2f}",flush=True)
