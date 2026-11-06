"""Signed INT8 x INT4, unclamped dot products in qwenTPU's circuit IR.

This evaluates compiled GDN/SwiGLU circuit nodes, not a merged Qwen checkpoint.
Readout is a vector of bits, not yet an autonomous emitted-token protocol.
"""
import numpy as np
from qwentpu.compiler.gadgets import Circ
from qwentpu.compiler.ir import K, A_BIT, A_REF


def band(g, a, b, name):
    return g.ffn([(1, g.one, a + b - g.one)], name)


def add_bits(g, left, right, name):
    """Modulo-2^width ripple addition. Every numerical intermediate is <=3."""
    assert len(left) == len(right)
    carry = 0 * g.one
    out = []
    for i, (a, b) in enumerate(zip(left, right)):
        total = a + b + carry
        # majority(total): ReLU(total-1)-ReLU(total-2)
        nxt = g.ffn([(1, g.one, total - g.one),
                     (-1, g.one, total - 2 * g.one)], f"{name}.carry{i}")
        bit = g.ffn([(1, g.one, total - 2 * nxt)], f"{name}.bit{i}")
        out.append(bit)
        carry = nxt
    return out


def neg_bits(g, bits, name):
    inv = [g.one - b for b in bits]
    return add_bits(g, inv, [g.one] + [0 * g.one] * (len(bits)-1), name)


def prefix_add(g, left, right, name):
    """Parallel-prefix carry addition: logarithmic carry depth, 0/1 intermediates."""
    assert len(left)==len(right)
    gens=[band(g,a,b,f'{name}.gen{i}') for i,(a,b) in enumerate(zip(left,right))]
    props=[a+b-2*gg for a,b,gg in zip(left,right,gens)]
    original=list(props)
    distance=1
    while distance<len(left):
        gg=list(gens);pp=list(props)
        for i in range(distance,len(left)):
            carried=band(g,props[i],gens[i-distance],f'{name}.g{distance}.{i}')
            # Generate and propagate-and-generate events are disjoint.
            gg[i]=gens[i]+carried
            pp[i]=band(g,props[i],props[i-distance],f'{name}.p{distance}.{i}')
        gens,props=gg,pp
        distance*=2
    carries=[0*g.one]+gens[:-1]
    out=[]
    for i,(p,c) in enumerate(zip(original,carries)):
        both=band(g,p,c,f'{name}.outand{i}')
        out.append(p+c-2*both)
    return out


def compress(g, buckets, width, name, final_adder='prefix'):
    """Wallace-style carry-save reduction, then one final ripple addition."""
    buckets = [list(b) for b in buckets]
    level = 0
    while any(len(b) > 2 for b in buckets):
        nxt = [[] for _ in range(width)]
        for bit, items in enumerate(buckets):
            i = 0
            while i + 2 < len(items):
                total = items[i] + items[i+1] + items[i+2]
                carry = g.ffn([(1,g.one,total-g.one),(-1,g.one,total-2*g.one)],
                              f"{name}.cs{level}.{bit}.{i}")
                nxt[bit].append(total - 2*carry)
                if bit+1 < width: nxt[bit+1].append(carry)
                i += 3
            nxt[bit] += items[i:]
        buckets=nxt
        level+=1
    left=[b[0] if b else 0*g.one for b in buckets]
    right=[b[1] if len(b)>1 else 0*g.one for b in buckets]
    add = prefix_add if final_adder=='prefix' else add_bits
    return add(g,left,right,name+'.final')


def build(rows=8, cols=2, final_adder='auto'):
    if rows not in (2, 4, 8, 16, 32, 64) or cols < 1:
        raise ValueError("rows must be 2,4,8,16,32,64; cols must be positive")
    if final_adder not in ('auto','prefix','ripple'):
        raise ValueError('unknown final adder')
    if final_adder=='auto':final_adder='ripple' if rows==8 else 'prefix'
    g = Circ()
    one, f = g.one, g.token_feature
    cell0, act0 = 0, rows * cols * 16
    read0 = act0 + rows * 256
    iscell = lambda t: cell0 <= t < act0
    isact = lambda t: act0 <= t < read0
    rf = [f(f"row{r}", lambda t, r=r: int(iscell(t) and t // (cols*16) == r)) for r in range(rows)]
    cf = [f(f"col{c}", lambda t, c=c: int(iscell(t) and t//16 % cols == c)) for c in range(cols)]
    wb = [f(f"wb{b}", lambda t,b=b: (t >> b)&1 if iscell(t) else 0) for b in range(4)]
    ar = [f(f"ar{r}", lambda t,r=r: int(isact(t) and (t-act0)//256 == r)) for r in range(rows)]
    ab = [f(f"ab{a}", lambda t,a=a: ((t-act0)>>a)&1 if isact(t) else 0) for a in range(8)]
    rd = [f(f"rd{c}", lambda t,c=c: int(t == read0+c)) for c in range(cols)]
    ref = {0: A_REF*one, 1: A_REF*one}
    qrows = rows if (rows.bit_length()-1) % 2 else 2*rows
    # Represent enough signed bits for every raw tile sum.
    width = (rows * 1024).bit_length() + 1
    outputs = []
    for c in range(cols):
        w = g.head(f"W{c}", {r:K*ar[r] for r in range(rows)},
                   {r:K*(rf[r]+cf[c]-one) for r in range(rows)},
                   {**ref, **{2+b:A_BIT*wb[b] for b in range(4)}}, 0,
                   {2+b:K*one for b in range(4)}, [(2+b,128) for b in range(4)])
        p = [band(g, ab[a], w[b], f"p{c}.{a}.{b}") for a in range(8) for b in range(4)]
        counts = g.head(f"A{c}", {r:K*rd[c] for r in range(qrows)},
                        {r:K*ar[r] for r in range(rows)},
                        {**ref, **{2+i:A_BIT*v for i,v in enumerate(p)}}, 0,
                        {2+i:K*one for i in range(32)},
                        [(2+i,128*rows) for i in range(32)])
        buckets = [[] for _ in range(width)]
        correction = 0
        for a in range(8):
            for b in range(4):
                count = counts[a*4+b]
                cb = []
                for bit in range(rows.bit_length()):
                    terms=[]
                    for m in range(1,(rows >> bit)+1):
                        terms += g.terms_ge(count, m << bit, 0, rows, 1 if m%2 else -1)
                    cb.append(g.node(terms, f"count{c}.{a}.{b}.{bit}"))
                negative = (a == 7) != (b == 3)
                for i, bit in enumerate(cb):
                    if a+b+i < width:
                        buckets[a+b+i].append(one-bit if negative else bit)
                if negative:
                    correction += (1-(1<<len(cb))) << (a+b)
        correction %= 1<<width
        for i in range(width):
            if (correction>>i)&1: buckets[i].append(one)
        acc=compress(g,buckets,width,f"sum{c}",final_adder)
        outputs.append(acc)
    return g, dict(rows=rows, cols=cols, width=width, outputs=outputs,
                   act0=act0, read0=read0)


def trace(d, weights, vectors):
    w, xs = np.asarray(weights), np.asarray(vectors)
    rows, cols=d['rows'],d['cols']
    if w.shape != (rows,cols) or xs.ndim != 2 or xs.shape[1] != rows:
        raise ValueError("shape mismatch")
    if not np.issubdtype(w.dtype,np.integer) or not np.issubdtype(xs.dtype,np.integer):
        raise ValueError("integer operands required")
    if np.any(w < -8) or np.any(w > 7) or np.any(xs < -128) or np.any(xs > 127):
        raise ValueError("operands out of range")
    toks=[(r*cols+c)*16+(int(w[r,c])&15) for r in range(rows) for c in range(cols)]
    marks=[]
    for x in xs:
        toks += [d['act0']+r*256+(int(v)&255) for r,v in enumerate(x)]
        for c in range(cols):
            marks.append((len(toks),c))
            toks.append(d['read0']+c)
    return toks,marks


def decode(d,m,marks):
    from qwentpu.compiler.evaluate import ev
    values=[]
    for t,c in marks:
        bits=[int(ev(e,m[t:t+1])[0]) for e in d['outputs'][c]]
        if any(b not in (0,1) for b in bits):
            raise AssertionError("readout is not bits")
        n=sum(b<<i for i,b in enumerate(bits))
        if bits[-1]: n-=1<<len(bits)
        values.append(n)
    return np.asarray(values,dtype=np.int64).reshape(-1,d['cols'])
