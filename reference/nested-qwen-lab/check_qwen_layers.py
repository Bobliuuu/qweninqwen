"""Compile persistent accumulator into stock Qwen3.5 architecture, CPU BF16.
Circuit-only random/zero language layers; NOT a pretrained Qwen checkpoint.
"""
import json,time
from pathlib import Path
import torch
from persistent import build,encode,QUERY
from qwentpu.compiler.tiny import build as compile_model

def main():
    torch.set_num_threads(2)
    start=time.perf_counter();g,out=build();code=[]
    for i in range(13):
        e=g.token_feature(f'code{i}',lambda t,i=i:128*(2*((t>>i)&1)-1))
        bit=out[i] if i<8 else (g.one if i==12 else 0*g.one)
        g.ffn([(8192,g.one,bit),(-4096,g.one,g.one),
               (-1,g.one,e),(1,g.one,-e)],f'codeout{i}',into=e)
        code.append(next(iter(e.t)))
    model=compile_model(g,list(range(4352)),code)
    history=encode(0,1);results=[];expected=0
    for value in [1778,-1778,3145728,-3145728]:
        history+=encode(value,0)+list(range(QUERY,QUERY+4))
        with torch.inference_mode():
            logits=model(torch.tensor([history]),use_cache=False).logits
        tokens=logits[0,-4:].argmax(-1).tolist()
        assert all(4096<=t<4352 for t in tokens),tokens
        bs=[t-4096 for t in tokens]
        history += [1024+i*256+b for i,b in enumerate(bs)]
        got=sum(b<<(8*i) for i,b in enumerate(bs))
        if got>>31:got-=1<<32
        expected+=value
        assert got==expected,(got,expected)
        results.append(got)
    report=dict(evidence='stock_Qwen3_5_architecture_CPU_BF16_circuit_only',
                pretrained_language_weights=False,results=results,
                parameters=sum(p.numel() for p in model.parameters()),
                layers=model.config.num_hidden_layers,seconds=time.perf_counter()-start,
                torch=torch.__version__)
    Path('qwen_layer_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
