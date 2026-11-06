import sys,time,torch
sys.path.insert(0,'.')
from glue import Matvec
from transformers import DynamicCache
torch.backends.cuda.matmul.allow_tf32=False
O=K=896; Ct,R,nt=64,14,14
mv=Matvec(Ct,O,R,nt,device='cuda'); W=torch.randn(O,K,device='cuda')*.03; x=torch.randn(K,device='cuda')
FB=mv.frames_B(W); S=mv.L.slots
def sync(): torch.cuda.synchronize(); return time.time()
for rep in range(3):
    cache=DynamicCache()
    def fwd(F):
        pos=cache.get_seq_length()+torch.arange(F.shape[0],device='cuda')
        return mv.m(inputs_embeds=F[None],past_key_values=cache,use_cache=True,position_ids=pos[None])
    FA=torch.zeros(nt,mv.D,device='cuda'); FA[:,0]=mv.B; FA[:,S['X']]=x.reshape(nt,Ct); FA[:,S['XS'].start]=1
    FA[:,S['XADDR']]=mv.code(torch.arange(nt,device='cuda'),mv.na)
    t0=sync(); fwd(FA); t1=sync(); fwd(FB); t2=sync()
    FC=torch.zeros(1,mv.D,device='cuda'); FC[0,0]=mv.B; fwd(FC); t3=sync()
    print(f'roundA {1e3*(t1-t0):.1f}ms  roundB({FB.shape[0]} steps) {1e3*(t2-t1):.1f}ms  roundC {1e3*(t3-t2):.1f}ms')
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,torch.profiler.ProfilerActivity.CPU]) as p:
    cache=DynamicCache(); fwd(FA); fwd(FB); fwd(FC); torch.cuda.synchronize()
print(p.key_averages().table(sort_by='cuda_time_total',row_limit=6,max_name_column_width=50))
