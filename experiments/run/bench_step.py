# stand-in outer stack throughput: random-weight Qwen3_5 with exact outer config. usage: bench_step.py D I L dtype tf32(0/1) layers_kind(gdn|mixed) Ns...
import sys,time,json,torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
D,I,L=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); dt=getattr(torch,sys.argv[4]); tf=int(sys.argv[5]); kind=sys.argv[6]; Ns=[int(x) for x in sys.argv[7:]]
torch.backends.cuda.matmul.allow_tf32=bool(tf)
lt=['linear_attention']*L if kind=='gdn' else [('full_attention' if (i+1)%4==0 else 'linear_attention') for i in range(L)]
cfg=Qwen3_5TextConfig(hidden_size=D,intermediate_size=I,num_hidden_layers=L,layer_types=lt,vocab_size=1000,
  linear_num_key_heads=16,linear_num_value_heads=16,linear_key_head_dim=128,linear_value_head_dim=128,linear_conv_kernel_dim=4,
  num_attention_heads=8,num_key_value_heads=2,head_dim=64)
import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
if kind=='gdn': MQ.create_causal_mask=lambda **k: None
dev='cuda' if torch.cuda.is_available() else 'cpu'
m=Qwen3_5ForCausalLM(cfg).to(dev).to(dt).eval()
try:
    import fla; fl='fla'
except Exception: fl='torch'
out=dict(D=D,I=I,L=L,dtype=sys.argv[4],tf32=tf,kind=kind,dev=dev,impl=fl,params=sum(p.numel() for p in m.parameters()))
def sync():
    if dev=='cuda': torch.cuda.synchronize()
with torch.no_grad():
  for n in Ns:
    x=torch.randn(1,n,D,device=dev,dtype=dt)*0.1
    c=DynamicCache(config=cfg)
    pos=lambda: dict(position_ids=torch.arange(n,device=dev)[None],cache_position=torch.arange(n,device=dev))
    m(inputs_embeds=x,past_key_values=c,use_cache=True,**pos()); sync()   # warm
    c=DynamicCache(config=cfg) if n>1 else c
    reps=30 if n==1 else 6
    ts=[]
    for _ in range(reps):
        cc=DynamicCache(config=cfg) if n>1 else c
        sync(); t=time.time()
        m(inputs_embeds=x,past_key_values=cc,use_cache=True,**pos())
        sync(); ts.append(time.time()-t)
    s=min(ts)
    out[f'n{n}']=dict(sec=s,steps_per_s=n/s)
if dev=='cuda': out['peak_gb']=torch.cuda.max_memory_allocated()/1e9
print(json.dumps(out))
