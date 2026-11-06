import sys, time, torch, json
sys.path.insert(0,'.')
from glue import Matvec
torch.backends.cuda.matmul.allow_tf32=False
cfgs=[('q_proj 896x896',896,896,64,14,14),('q_proj 896x896 R16',896,896,64,16,14),('gate 4864x896',4864,896,64,16,14),('down 896x4864',896,4864,64,16,76),
      ('q_proj Ct128',896,896,128,16,7)]
res=[]
for name,O,K,Ct,R,nt in cfgs:
    if O%R: continue
    torch.manual_seed(0)
    W=torch.randn(O,K,device='cuda')*0.03; x=torch.randn(K,device='cuda'); x[3]=300
    try:
        mv=Matvec(Ct,O,R,nt,device='cuda')
        FB=mv.frames_B(W)
        yb,xg=mv.run(W,x,FB); torch.cuda.synchronize()
        ref=W.double()@x.double(); err=((xg.double()-ref).abs().max()/ref.abs().max()).item()
        t0=time.time(); n=3
        for _ in range(n): mv.run(W,x,FB)
        torch.cuda.synchronize(); dt=(time.time()-t0)/n
        r=dict(name=name,O=O,K=K,Ct=Ct,R=R,nt=nt,D=mv.D,hd=mv.hd,steps=mv.T*nt,err=err,sec_per_op=dt,MACps=O*K/dt,
               steps_per_s=mv.T*nt/dt,gpu_GB=torch.cuda.max_memory_allocated()/1e9)
    except Exception as e:
        r=dict(name=name,error=repr(e)[:300])
    print(json.dumps(r),flush=True); res.append(r)
    try: del mv,FB
    except: pass
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
json.dump(res,open('bench_glue_a100.json','w'),indent=1)
