"""Signed tile in real Qwen3.5 layers, CPU BF16; circuit-only model."""
import json,time
from pathlib import Path
import numpy as np
import torch
from signed_tile import build,trace
from qwentpu.compiler.tiny import build as compile_model

def main():
    torch.set_num_threads(2);start=time.perf_counter()
    g,d=build(2,1);width=d['width'];base=1<<width;code=[]
    for i in range(width+1):
        e=g.token_feature(f'code{i}',lambda t,i=i:128*(2*((t>>i)&1)-1))
        bit=d['outputs'][0][i] if i<width else g.one
        g.ffn([(8192,g.one,bit),(-4096,g.one,g.one),
               (-1,g.one,e),(1,g.one,-e)],f'codeout{i}',into=e)
        code.append(next(iter(e.t)))
    model=compile_model(g,list(range(2*base)),code)
    xs=np.array([[-128,-128],[127,127],[-128,127],[0,0],[17,-31]])
    checked=0
    for w in (np.array([[-8],[-8]]),np.array([[7],[-7]])):
        toks,marks=trace(d,w,xs)
        with torch.inference_mode():logits=model(torch.tensor([toks]),use_cache=False).logits
        got=[]
        for t,c in marks:
            token=int(logits[0,t].argmax())
            assert base<=token<2*base,token
            v=token-base
            if v>>(width-1):v-=1<<width
            got.append(v)
        np.testing.assert_array_equal(got,(xs@w).ravel());checked+=len(xs)
    report=dict(evidence='stock_Qwen3_5_layers_CPU_BF16_circuit_only',
                pretrained_language_weights=False,readouts=checked,
                layers=model.config.num_hidden_layers,
                parameters=sum(p.numel() for p in model.parameters()),
                seconds=time.perf_counter()-start)
    Path('tile_layer_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
