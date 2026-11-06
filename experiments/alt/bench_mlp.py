import sys, time, torch, json
sys.path.insert(0,'.')
from mlpmul import MulMLP
dev='cuda'
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
res=[]
for (C,R,T) in [(896,1,1024),(448,2,1024),(224,4,2048),(128,7,4096),(64,14,4096),(64,1,8192),(32,1,8192),(4864,1,256),(973,1,1024)]:
    mm=MulMLP(C,R,device=dev)
    W=torch.randn(T,R,C,device=dev)*0.03; x=torch.randn(C,device=dev)
    F=mm.frames(W,x)
    y=mm.run(F); torch.cuda.synchronize()
    ref=(W.double()@x.double())
    err=((y.double()-ref).abs().max()/ref.abs().max()).item()
    t0=time.time(); n=5
    for _ in range(n): y=mm.run(F)
    torch.cuda.synchronize(); dt=(time.time()-t0)/n
    mac=T*R*C/dt
    r=dict(C=C,R=R,T=T,D=mm.D,I=mm.I,err_vs_fp64=err,s_per_fwd=dt,steps_per_s=T/dt,MAC_per_step=R*C,MACps=mac,
           sec_per_token_494M=494e6/mac,mem_GB=torch.cuda.max_memory_allocated()/1e9)
    print(json.dumps(r),flush=True); res.append(r)
    del mm,F,W; torch.cuda.empty_cache()
json.dump(res,open('bench_mlp_a100.json','w'),indent=1)
