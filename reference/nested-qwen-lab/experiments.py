"""Run reproducible CPU circuit-IR tests and projection planning. Requires NumPy.

No merged checkpoint or physical GPU execution is performed here.
"""
import json
from pathlib import Path
import time
import numpy as np
from signed_tile import build, trace, decode, add_bits
from qwentpu import circuit as C
from qwentpu.compiler.gadgets import Circ
from qwentpu.compiler.evaluate import evaluate, ev, ExactnessError


def measure(name, fn):
    t=time.perf_counter()
    result=fn()
    return dict(name=name,seconds=time.perf_counter()-t,**result)


def baseline():
    runs=[]
    rng=np.random.default_rng(5)
    for rows,cols,ab,wb in [(8,8,3,3),(8,8,5,5),(16,16,5,5),(32,32,3,3)]:
        C.configure(rows,cols,ab,wb)
        g,d=C.build()
        w=rng.integers(C.wlo(),C.whi()+1,(rows,cols))
        w[:,0]=C.whi();w[:,1]=C.wlo()
        xs=rng.integers(0,C.AMAX+1,(8,rows));xs[0]=C.AMAX;xs[1]=0
        toks=C.load_trace(w);marks=[]
        for x in xs:
            base=len(toks);toks+=C.matvec_trace(x)
            marks += [(base+C.KS+k,want) for k,want in enumerate(C.expected_outs(x,w))]
        m=evaluate(g,toks,check=True)
        for t,want in marks:
            got=sum((1<<i)*int(ev(e,m[t:t+1])[0]>0) for i,e in enumerate(d['code']))
            assert got==want,(rows,cols,got,want)
        runs.append(dict(config=[rows,cols,ab,wb],readouts=len(marks),stats=g.stats()))
    C.configure()
    return dict(runs=runs,readouts=sum(r['readouts'] for r in runs))


def signed_tiles():
    rng=np.random.default_rng(20261105)
    runs=[]
    for rows in (2,4,8,16,32,64):
        g,d=build(rows,2)
        w=rng.integers(-8,8,(rows,2))
        xs=rng.integers(-128,128,(20,rows))
        xs[0]=-128;xs[1]=127;xs[2]=0
        # Half random runs, half adversarial all-min/max weights.
        for extreme in (False,True):
            ww=w.copy()
            if extreme: ww[:,0]=-8;ww[:,1]=7
            toks,marks=trace(d,ww,xs)
            got=decode(d,evaluate(g,toks,check=True),marks)
            np.testing.assert_array_equal(got,xs@ww)
            runs.append(dict(rows=rows,columns=2,extreme=extreme,
                             readouts=int(got.size),stats=g.stats()))
    return dict(runs=runs,readouts=sum(r['readouts'] for r in runs),max_integer_error=0)


def exhaustive():
    # All 4096 signed operand pairs pass through the actual circuit graph.
    g,d=build(2,1)
    total=0
    xs=np.zeros((256,2),dtype=np.int64);xs[:,0]=np.arange(-128,128)
    for w0 in range(-8,8):
        w=np.array([[w0],[0]])
        toks,marks=trace(d,w,xs)
        got=decode(d,evaluate(g,toks,check=True),marks)
        np.testing.assert_array_equal(got,xs@w)
        total+=len(xs)
    return dict(cases=total,max_integer_error=0,stats=g.stats())


def wide_adder():
    # Gadget-level IR input bits. Not a realizable 2^46-token embedding/ISA.
    rng=np.random.default_rng(91);width=23;mask=(1<<width)-1
    pairs=[(-3145728,3145728),(-1,1),(3144704,1024),(-3144704,-1024)]
    pairs += [tuple(map(int,rng.integers(-1500000,1500000,2))) for _ in range(96)]
    g=Circ()
    left=[g.token_feature(f'a{i}',lambda t,i=i:(t>>i)&1) for i in range(width)]
    right=[g.token_feature(f'b{i}',lambda t,i=i:(t>>(width+i))&1) for i in range(width)]
    out=add_bits(g,left,right,'wide')
    toks=[(a&mask)|((b&mask)<<width) for a,b in pairs]
    m=evaluate(g,toks,check=True)
    for t,(a,b) in enumerate(pairs):
        n=sum(int(ev(e,m[t:t+1])[0])<<i for i,e in enumerate(out))
        if n>>(width-1):n-=1<<width
        assert n==a+b,(a,b,n)
    return dict(cases=len(pairs),width=width,stats=g.stats(),
                scope='arithmetic_gadget_not_persistent_state_or_token_protocol')


def mutation():
    g,d=build(8,1)
    w=np.full((8,1),7);xs=np.full((1,8),127)
    toks,marks=trace(d,w,xs)
    original=decode(d,evaluate(g,toks),marks)
    # Delete all carry-save and final adder gates; expected result must break.
    g.nodes=[n for n in g.nodes if not n.name.startswith('sum0.')]
    try:
        changed=decode(d,evaluate(g,toks),marks)
        assert not np.array_equal(original,changed)
        corrupted=int(changed[0,0])
    except (ExactnessError,AssertionError) as error:
        # Invalid output bits/exactness failures also detect corrupted arithmetic.
        # Do not catch our mismatch assertion: equality must fail this test.
        if 'readout is not bits' not in str(error) and not isinstance(error,ExactnessError):raise
        corrupted='invalid_arithmetic_readout'
    return dict(expected=int(original[0,0]),corrupted=corrupted,detected=True)


def cancellation():
    # Tile arithmetic is circuit IR; combining tile results below is reference
    # validation only, not a completed persistent compiled accumulator.
    g,d=build(2,1)
    xs=np.full((1,2),127)
    parts=[]
    for sign in (1,-1):
        w=np.full((2,1),7*sign)
        toks,marks=trace(d,w,xs)
        parts.append(int(decode(d,evaluate(g,toks),marks)[0,0]))
    assert sum(parts)==0
    bad=sum(max(0,min(31,v)) for v in parts)
    assert bad!=0
    return dict(raw_tile_results=parts,correct_sum=sum(parts),old_clamped_sum=bad,
                scope='cross_tile_sum_is_host_reference_only')


def full_projection():
    # Independent vectorized bit-plane oracle at actual Qwen3 FFN dimensions.
    # This is CPU arithmetic; it is NOT an accelerator throughput benchmark.
    rng=np.random.default_rng(73)
    k,n=1024,3072
    x=rng.integers(-128,128,k,dtype=np.int64)
    w=rng.integers(-8,8,(k,n),dtype=np.int64)
    expected=x@w
    result=np.zeros(n,dtype=np.int64)
    for start in range(0,k,32):
        xx=x[start:start+32];ww=w[start:start+32]
        for a in range(8):
            ca=-(1<<a) if a==7 else 1<<a
            for b in range(4):
                cb=-(1<<b) if b==3 else 1<<b
                result+=ca*cb*(((xx>>a)&1)@((ww>>b)&1))
    np.testing.assert_array_equal(result,expected)
    return dict(shape=[k,n],macs=k*n,outputs=n,max_integer_error=0,
                evidence='numpy_bitplane_oracle_only_not_full_projection_circuit')


def plans():
    configs={
        'Qwen3-0.6B':dict(h=1024,ff=3072,l=28,q=2048,kv=1024,vocab=151936),
        'Qwen2.5-0.5B':dict(h=896,ff=4864,l=24,q=896,kv=128,vocab=151936),
    }
    out={}
    for name,c in configs.items():
        h,ff,l,q,kv,v=(c[k] for k in ('h','ff','l','q','kv','vocab'))
        dims=[(h,q,l),(h,kv,l),(h,kv,l),(q,h,l),
              (h,ff,l),(h,ff,l),(ff,h,l),(h,v,1)]
        macs=sum(k*n*copies for k,n,copies in dims)
        tiles=sum(((k+31)//32)*((n+31)//32)*copies for k,n,copies in dims)
        protocols={}
        for packing in (1,8,32,256):
            tokens=tiles*((1024+packing-1)//packing+32+64)
            protocols[str(packing)]=dict(trace_tokens=tokens,
                seconds_at_assumed_100k_trace_tokens_per_s=tokens/100000)
        out[name]=dict(config=c,macs_per_token=macs,packed_int4_bytes=macs//2,
            kv_bf16_bytes_context128=2*l*128*kv*2,protocols=protocols,
            evidence='analytical_hypothetical_protocol_not_measured',
            exclusions=['scales','attention','nonlinearities','control','limb arithmetic'])
    assert out['Qwen3-0.6B']['macs_per_token']==595984384
    return out


def main():
    results=dict(environment='CPU NumPy; no torch, checkpoint, or CUDA',
                 circuit_evidence='upstream exactness evaluator with check=True',
                 experiments=[])
    for name,fn in [('upstream_baseline',baseline),('signed_tiles',signed_tiles),
                    ('exhaustive_int8_int4',exhaustive),('wide_adder',wide_adder),
                    ('mutation',mutation),('cancellation',cancellation),
                    ('full_projection_oracle',full_projection)]:
        r=measure(name,fn);results['experiments'].append(r)
        print(json.dumps(r),flush=True)
    results['plans']=plans()
    Path('results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ALL EXPERIMENTS PASSED',flush=True)


if __name__=='__main__':main()
