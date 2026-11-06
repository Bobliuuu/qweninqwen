"""Complete 32x31 integer projection via Qwen GDN + compiled Qwen reduction.
Bulk state/input encoding is host I/O. Query formation and signed summation
execute in BF16 neural operations. No pretrained outer language weights.
"""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from signed_tile import compress
from qwentpu.compiler.gadgets import Circ
from qwentpu.compiler.tiny import build as compile_model
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,Qwen3_5RMSNormGated)

def reduction():
    g=Circ();one=g.one;width=17
    counts=[g.token_feature(f'count{i}',lambda t:0) for i in range(32)]
    buckets=[[] for _ in range(width)];correction=0
    for a in range(8):
        for b in range(4):
            negative=(a==7)!=(b==3)
            for bit in range(6):
                terms=[]
                for m in range(1,(32>>bit)+1):
                    terms+=g.terms_ge(counts[a*4+b],m<<bit,0,32,1 if m%2 else -1)
                cb=g.node(terms,f'countbits{a}.{b}.{bit}')
                if a+b+bit<width:buckets[a+b+bit].append(one-cb if negative else cb)
            if negative:correction+=(1-64)<<(a+b)
    correction%=1<<width
    for i in range(width):
        if (correction>>i)&1:buckets[i].append(one)
    bits=compress(g,buckets,width,'signed')
    outs=[g.ffn([(1,one,b)],f'out{i}') for i,b in enumerate(bits)]
    dims=[next(iter(e.t)) for e in outs]
    model=compile_model(g,[0],dims)
    ins=[model.layout.dim(next(iter(e.t))) for e in counts]
    outs=[model.layout.dim(d) for d in dims]
    return model,ins,outs,g.stats()

class Engine(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(257,1024,bias=False,dtype=torch.bfloat16)
        self.query.weight.data.zero_()
        for a in range(8):
            for r in range(32):
                self.query.weight.data[a*128+r,r*8+a]=128
                self.query.weight.data[a*128+32+r,r*8+a]=-128
                self.query.weight.data[a*128+32+r,256]=128
        self.norm=Qwen3_5RMSNormGated(128).bfloat16()
        self.norm.weight.data.fill_(1)
        self.count_proj=nn.Linear(128,124,bias=False,dtype=torch.bfloat16)
        self.count_proj.weight.data.zero_()
        for i in range(124):self.count_proj.weight.data[i,i+2]=4096
        self.reducer,self.ins,self.outs,self.stats=reduction()

    def load(self,w):
        if w.shape!=(32,31) or np.any(w < -8) or np.any(w>7):raise ValueError('INT4 32x31 required')
        self.state=torch.zeros(1,8,128,128)
        self.state[:,:,:64,:2]=2**24
        for c in range(31):
            for b in range(4):self.state[:,:,0:32,2+4*c+b]=torch.tensor((w[:,c]>>b)&1)*128

    def encode(self,x):
        if x.shape!=(32,) or np.any(x < -128) or np.any(x>127):raise ValueError('INT8 vector required')
        return torch.tensor([*((x[:,None]>>np.arange(8))&1).ravel(),1],dtype=torch.bfloat16)

    @torch.inference_mode()
    def forward(self,frame):
        q=self.query(frame).reshape(1,1,8,128);z=torch.zeros_like(q)
        y,_=torch_recurrent_gated_delta_rule(q,z,z,torch.zeros(1,1,8),torch.zeros(1,1,8),
                initial_state=self.state,output_final_state=False,use_qk_l2norm_in_kernel=True)
        y=self.norm(y,torch.full_like(y,128))
        counts=self.count_proj(y).reshape(8,31,4).permute(1,0,2).reshape(31,32)
        # Transfer count vectors into compiled reducer residual frames.
        frames=self.reducer.model.embed_tokens(torch.zeros(31,1,dtype=torch.long)).clone()
        frames[:,0,self.ins]=counts
        out=self.reducer.model(inputs_embeds=frames,use_cache=False).last_hidden_state
        return (out[:,0,self.outs]>0).long()

def decode(bits):
    # Output representation decoding only; all MACs/reduction already executed.
    vals=(bits*(2**torch.arange(17))).sum(-1)
    return torch.where(bits[:,-1].bool(),vals-2**17,vals).numpy()

def main():
    torch.set_num_threads(2);start=time.perf_counter();engine=Engine();compile_s=time.perf_counter()-start
    rng=np.random.default_rng(119);runs=[]
    checkpoint=sys.argv[1] if len(sys.argv)>1 else None
    for case in range(6 if checkpoint else 5):
        x=rng.integers(-128,128,32);w=rng.integers(-8,8,(32,31))
        if case==0:x[:]=-128;w[:]=-8
        if case==1:x[:]=127;w[:]=7
        if case==5:
            from safetensors import safe_open
            with safe_open(checkpoint,framework='pt') as sf:
                real=sf.get_slice('model.layers.0.mlp.gate_proj.weight')[:31,:].float().numpy()
            scales=np.max(np.abs(real),axis=1)/7
            w=np.clip(np.rint(real[:,:32]/scales[:,None]),-7,7).astype(np.int64).T
        tick=time.perf_counter();engine.load(w);frame=engine.encode(x);load_s=time.perf_counter()-tick
        tick=time.perf_counter();got=decode(engine(frame));run_s=time.perf_counter()-tick
        np.testing.assert_array_equal(got,x@w)
        runs.append(dict(case=case,load_encode_seconds=load_s,compute_decode_seconds=run_s,
                         total_seconds=load_s+run_s,macs_per_second=992/(load_s+run_s)))
        print(json.dumps(runs[-1]),flush=True)
    # Mutate the neural query projection; arithmetic must change.
    engine.query.weight.data.zero_()
    corrupted=decode(engine(frame))
    assert not np.array_equal(corrupted,x@w)
    report=dict(mutation_detected=True,real_checkpoint_tile_checked=bool(checkpoint),evidence='Qwen_GDN_gated_norm_and_compiled_Qwen_layers_CPU_BF16',
                shape=[32,31],outputs_checked=31*len(runs),compile_seconds=compile_s,runs=runs,
                reduction_stats=engine.stats,parameters=sum(p.numel() for p in engine.parameters()),
                state_bytes=engine.state.numel()*4,
                exclusions=['group_scales','cross_tile_accumulation','sequencer','pretrained_outer_weights'])
    Path('parallel_complete_results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
