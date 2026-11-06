"""32-bit persistent GDN accumulator, circuit IR; host transports result bytes.
No merged checkpoint execution. QUERY produces eight output bits. All four
result bytes must be read before commits overwrite the accumulator.
"""
import json
from pathlib import Path
import numpy as np
from signed_tile import prefix_add
from qwentpu.compiler.gadgets import Circ
from qwentpu.compiler.ir import A_BIT,A_REF,K
from qwentpu.compiler.evaluate import evaluate,ev

QUERY=2048

def build():
    g=Circ();one=g.one;f=g.token_feature
    # 0..1023: incoming operand; 1024..2047: committed accumulator.
    read=f('read',lambda t:int(QUERY<=t<QUERY+4))
    operands=[]
    for bank in range(2):
        bits=[]
        for byte in range(4):
            base=bank*1024+byte*256
            write=f(f'w{bank}.{byte}',lambda t,base=base:int(base<=t<base+256))
            payload=[f(f'b{bank}.{byte}.{i}',lambda t,i=i,base=base:
                       ((t-base)>>i)&1 if base<=t<base+256 else 0) for i in range(8)]
            bits+=g.head(f'state{bank}.{byte}',{0:K*read},{0:K*write},
                         {0:A_REF*one,1:A_REF*one,
                          **{2+i:A_BIT*b for i,b in enumerate(payload)}},0,
                         {2+i:K*one for i in range(8)},[(2+i,128) for i in range(8)])
        operands.append(bits)
    summed=prefix_add(g,*operands,'sum')
    selectors=[f(f'q{byte}',lambda t,byte=byte:int(t==QUERY+byte)) for byte in range(4)]
    out=[]
    for i in range(8):
        terms=[(1,one,summed[byte*8+i]+selectors[byte]-one) for byte in range(4)]
        out.append(g.ffn(terms,f'output{i}'))
    return g,out

def encode(value,bank):
    return [bank*1024+i*256+((value>>(8*i))&255) for i in range(4)]

def execute(values):
    g,out=build();history=encode(0,1);observed=[]
    for value in values:
        history+=encode(value,0)+list(range(QUERY,QUERY+4))
        m=evaluate(g,history,check=True)
        # Only transport emitted bits into byte tokens; no host sum.
        bs=[sum(int(ev(e,m[t:t+1])[0])<<i for i,e in enumerate(out))
            for t in range(len(history)-4,len(history))]
        history += [1024+i*256+b for i,b in enumerate(bs)]
        unsigned=sum(b<<(8*i) for i,b in enumerate(bs))
        observed.append(unsigned-(1<<32) if unsigned>>31 else unsigned)
    return observed,g.stats(),len(history)

def main():
    rng=np.random.default_rng(771)
    values=[1778,-1778,3145728,-3145728,2147483647,1,-1,-2147483647]
    values+=list(map(int,rng.integers(-5000000,5000000,24)))
    got,stats,tokens=execute(values)
    want=[];acc=0
    for v in values:
        acc=(acc+v)&0xffffffff
        want.append(acc-(1<<32) if acc>>31 else acc)
    assert got==want,(got,want)
    r=dict(evidence='GDN_SwiGLU_IR_check_true_host_byte_transport',
           additions=len(values),values=values,results=got,stats=stats,
           tokens=tokens,overflow='twos_complement_wrap',
           limitation='not_merged_checkpoint_or_autonomous_sequencer')
    Path('persistent_results.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))

if __name__=='__main__':main()
