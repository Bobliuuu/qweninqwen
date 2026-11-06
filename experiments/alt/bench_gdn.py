import sys,time,torch

from machine import Layout, make_outer
torch.backends.cuda.matmul.allow_tf32=False
def tm(f,n=3):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n
for hv,T in [(16,2048),(16,8192),(32,8192)]:
    L=Layout(); L.alloc('p',2048)
    m=make_outer(L,['linear_attention'],64,gdn=dict(hk=16,hv=hv,dk=128,dv=128),vocab=8).cuda()
    # random weights so the kernel does real work
    for p in m.parameters(): torch.nn.init.normal_(p,0,0.02)
    F=torch.randn(1,T,L.n,device='cuda')
    t=tm(lambda: m.model(inputs_embeds=F,use_cache=False))
    print(f'GDN layer (torch-ref chunk kernel) hv={hv} T={T}: {t*1e3:.1f} ms = {T/t:.0f} steps/s ; state/write capacity per step: {hv*128} rows? floats/step written (k one-hot row x 128 v) = {hv*128}')
