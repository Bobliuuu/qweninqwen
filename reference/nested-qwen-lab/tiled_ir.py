"""Tiled projection: tile AND cross-tile addition execute in compiler IR.
Host schedules circuits and transports integer encodings. Separate accumulator
sessions per output column; not one merged checkpoint or autonomous program.
"""
import json,time
from pathlib import Path
import numpy as np
from signed_tile import build,trace,decode
from persistent import execute
from qwentpu.compiler.evaluate import evaluate

def main():
    start=time.perf_counter();rng=np.random.default_rng(111)
    k,n,tile=64,4,8
    x=rng.integers(-128,128,k);w=rng.integers(-8,8,(k,n))
    g,d=build(tile,n)
    partials=[]
    for offset in range(0,k,tile):
        toks,marks=trace(d,w[offset:offset+tile],x[None,offset:offset+tile])
        partials.append(decode(d,evaluate(g,toks,check=True),marks)[0])
    got=[]
    for col in range(n):
        # execute adds these values through the persistent GDN circuit.
        values=[int(p[col]) for p in partials]
        accumulated,_,_=execute(values)
        got.append(accumulated[-1])
    expected=x@w # validation only
    np.testing.assert_array_equal(got,expected)
    result=dict(shape=[k,n],tile_rows=tile,output=got,expected=expected.tolist(),
                tile_stats=g.stats(),seconds=time.perf_counter()-start,
                evidence='tile_and_accumulation_in_circuit_IR',
                limitation='host_scheduling_and_encoding_not_merged_checkpoint')
    Path('tiled_ir_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
