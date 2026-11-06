"""Projection through actual Qwen3.5 layers with persistent DynamicCache.
Circuit-only models; host transports values and schedules ops, never adds MACs.
"""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
import signed_tile,persistent
from qwentpu.compiler.tiny import build as compile_model

def model_for(g,bits,basebit,vocab):
    code=[]
    for i in range(basebit+1):
        e=g.token_feature(f'code{i}',lambda t,i=i:128*(2*((t>>i)&1)-1))
        bit=bits[i] if i<len(bits) else (g.one if i==basebit else 0*g.one)
        g.ffn([(8192,g.one,bit),(-4096,g.one,g.one),(-1,g.one,e),(1,g.one,-e)],
              f'codeout{i}',into=e)
        code.append(next(iter(e.t)))
    model=compile_model(g,list(range(vocab)),code)
    model.chip_code_indices=[model.layout.dim(d) for d in code]
    return model

class Session:
    def __init__(self,model):self.model=model;self.cache=None;self.tokens=0;self.readout_checks=0
    def feed(self,tokens):
        with torch.inference_mode():
            result=self.model.model(torch.tensor([tokens]),past_key_values=self.cache,use_cache=True)
            bits=(result.last_hidden_state[0,:,self.model.chip_code_indices]>0).long()
            ids=(bits*(2**torch.arange(bits.shape[-1]))).sum(-1)
            if self.readout_checks<2:
                reference=self.model.lm_head(result.last_hidden_state)[0].argmax(-1)
                assert torch.equal(ids,reference), (ids,reference)
                self.readout_checks+=1
        self.cache=result.past_key_values;self.tokens+=len(tokens)
        return ids.tolist()

def main():
    torch.set_num_threads(2);start=time.perf_counter()
    rows=int(sys.argv[1]) if len(sys.argv)>1 else 2
    g,d=signed_tile.build(rows,1);width=d['width'];base=1<<width
    tile=model_for(g,d['outputs'][0],width,2*base)
    g,bits=persistent.build();adder=model_for(g,bits,12,4352)
    rng=np.random.default_rng(225);runs=[]
    real=None
    if len(sys.argv)>2:
        from safetensors import safe_open
        with safe_open(sys.argv[2],framework='pt') as sf:
            real=sf.get_slice('model.layers.0.mlp.gate_proj.weight')[0,:].float().numpy()
        scale=float(np.max(np.abs(real))/7)
        real=np.clip(np.rint(real/scale),-7,7).astype(np.int64)
    for k in ([len(real)] if real is not None else [2*rows,8*rows,16*rows]):
        x=rng.integers(-128,128,k);w=rng.integers(-8,8,k)
        if real is not None:w=real
        if k==4:x[:]=127;w[:]=[7,7,-7,-7]
        ts=Session(tile);ac=Session(adder);ac.feed(persistent.encode(0,1))
        tick=time.perf_counter();partials=[]
        for off in range(0,k,rows):
            toks,_=signed_tile.trace(d,w[off:off+rows,None],x[None,off:off+rows])
            token=ts.feed(toks)[-1];assert base<=token<2*base,token
            partial=token-base
            if partial>>(width-1):partial-=1<<width
            partials.append(partial)
            outputs=ac.feed(persistent.encode(partial,0)+list(range(persistent.QUERY,persistent.QUERY+4)))[-4:]
            assert all(4096<=v<4352 for v in outputs),outputs
            bs=[v-4096 for v in outputs]
            ac.feed([1024+i*256+b for i,b in enumerate(bs)])
        got=sum(b<<(8*i) for i,b in enumerate(bs))
        if got>>31:got-=1<<32
        assert got==int(x@w),(got,int(x@w))
        elapsed=time.perf_counter()-tick
        runs.append(dict(k=k,macs=k,seconds=elapsed,macs_per_second=k/elapsed,
                         tile_tokens=ts.tokens,accumulator_tokens=ac.tokens,result=got,
                         partials=partials))
        print(json.dumps(runs[-1]),flush=True)
    report=dict(evidence='actual_Qwen3_5_CPU_BF16_cached_execution',
                pretrained_language_weights=False,runs=runs,seconds=time.perf_counter()-start,
                scope='host_scheduled_two_component_projection_not_full_Qwen_inference')
    if real is not None:report['real_weight_test']=dict(tensor='model.layers.0.mlp.gate_proj.weight',output_row=0,scale=scale,quantization='symmetric_INT4_per_output_row',activation='synthetic_INT8',checkpoint_revision='060db6499f32faf8b98477b0a26969ef7d8b9987')
    Path('real_weight_fast_results.json' if real is not None else f'streaming_fast_results_r{rows}.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
