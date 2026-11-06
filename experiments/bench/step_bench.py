import sys,time,torch,json
sys.path.insert(0,'.')
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
dev=sys.argv[1]; D=int(sys.argv[2]); I=int(sys.argv[3]); L=int(sys.argv[4]); dt=getattr(torch,sys.argv[5]); N=int(sys.argv[6]); H=int(sys.argv[7]) if len(sys.argv)>7 else 16
cfg=Qwen3_5TextConfig(hidden_size=D,intermediate_size=I,num_hidden_layers=L,layer_types=['linear_attention']*(L-1)+['full_attention'],vocab_size=1000,
  linear_num_key_heads=H,linear_num_value_heads=H,linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4,
  num_attention_heads=2,num_key_value_heads=1,head_dim=64)
m=Qwen3_5ForCausalLM(cfg).to(dev).to(dt).eval()
npar=sum(p.numel() for p in m.parameters())
x=torch.randn(1,N,D,device=dev,dtype=dt)*0.1
res={}
with torch.no_grad():
    for name,n in (('prefill_chunk',N),('decode1',1)):
        c=DynamicCache(config=cfg)
        m(inputs_embeds=x[:,:n],past_key_values=c,use_cache=True,position_ids=torch.arange(n,device=dev)[None],cache_position=torch.arange(n,device=dev))  # warm
        if dev=='cuda': torch.cuda.synchronize()
        reps=3 if n>1 else 20
        t=time.time()
        for _ in range(reps): m(inputs_embeds=x[:,:n],past_key_values=c,use_cache=True,position_ids=torch.arange(n,device=dev)[None],cache_position=torch.arange(n,device=dev))
        if dev=='cuda': torch.cuda.synchronize()
        s=(time.time()-t)/reps; res[name]=dict(n=n,sec=s,steps_per_s=n/s)
print(json.dumps(dict(dev=dev,D=D,I=I,L=L,dtype=sys.argv[5],H=H,params=npar,**res)))
