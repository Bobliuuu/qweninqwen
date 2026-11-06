"""Gate-level integer / fp32-like arithmetic on a Net.  Buses are LSB-first wire lists.
fp32 format: bit31 sign, bits30..23 exponent, bits22..0 mantissa; implicit leading 1; exponent 0 == zero (flush-to-zero, canonical zero = all 0 bits);
NO inf/nan; results truncate (round toward zero); exponent overflow is not detected (callers keep values in range).  Stated error vs IEEE fp32 is measured in test_fp.py."""
from netlist import *

def const(n, v, w=None): return [1 if (v >> i) & 1 else 0 for i in range(w if w else n)]
def fa(n, a, b, c):
    t = n.XOR(a, b); return n.XOR(t, c), n.OR(n.AND(a, b), n.AND(c, t))
def add(n, a, b, cin=0, width=None):
    """ripple adder; returns (sum bits of len width, cout).  a,b zero-extended."""
    w = width or max(len(a), len(b)); s = []; c = cin
    for i in range(w):
        x = a[i] if i < len(a) else 0; y = b[i] if i < len(b) else 0
        si, c = fa(n, x, y, c); s.append(si)
    return s, c
def sub(n, a, b, width=None):
    """a-b two's complement, returns (diff, borrow_free_carry) ; carry=1 iff a>=b (unsigned)"""
    w = width or max(len(a), len(b))
    nb = [n.NOT(b[i]) if i < len(b) else 1 for i in range(w)]
    return add(n, a + [0] * (w - len(a)), nb, 1, w)
def ge(n, a, b):
    w = max(len(a), len(b)); _, c = sub(n, a, b, w); return c
def mux_bus(n, s, a, b):
    w = max(len(a), len(b)); return [n.MUX(s, a[i] if i < len(a) else 0, b[i] if i < len(b) else 0) for i in range(w)]
def mul(n, a, b, out_lo=0):
    """unsigned a*b; returns full len(a)+len(b) bits (array multiplier, ripple rows).  out_lo>0: drop partial-product columns below out_lo
    (truncated multiplier: result bits < out_lo are unreliable/zero)."""
    la, lb = len(a), len(b); acc = [0] * (la + lb)
    for j in range(lb):
        row = [n.AND(a[i], b[j]) if i + j >= out_lo else 0 for i in range(la)]
        # add row << j into acc[j : j+la+1]
        c = 0
        for i in range(la):
            if i + j < out_lo and acc[i + j] == 0 and row[i] == 0: continue
            acc[i + j], c = fa(n, acc[i + j], row[i], c)
        k = j + la
        while c != 0 and k < la + lb:
            acc[k], c = n.XOR(acc[k], c), n.AND(acc[k], c); k += 1
    return acc
def shr(n, a, amt, fill=0):
    """barrel logical right shift by bus amt (LSB first); shifts >= len(a) give all fill"""
    cur = list(a); L = len(a)
    for k, s in enumerate(amt):
        d = 1 << k
        cur = [n.MUX(s, cur[i + d] if i + d < L else fill, cur[i]) for i in range(L)]
    return cur
def shl(n, a, amt):
    cur = list(a); L = len(a)
    for k, s in enumerate(amt):
        d = 1 << k
        cur = [n.MUX(s, cur[i - d] if i - d >= 0 else 0, cur[i]) for i in range(L)]
    return cur
def or_all(n, bits):
    r = 0
    for b in bits: r = n.OR(r, b)
    return r
def neg_bus(n, a, width=None):
    w = width or len(a); s, _ = add(n, [n.NOT(a[i]) if i < len(a) else 1 for i in range(w)], [], 1, w); return s

# ----------------------------------------------------------------------------- fp32-like
def parts(x): return x[31], x[23:31], x[:23]
def mant24(x):
    e = x[23:31]; return x[:23] + [or_all(None if False else _N, e)] if False else None
_N = None
def fp_unpack(n, x):
    e = x[23:31]; nz = or_all(n, e); return x[31], e, x[:23] + [nz], nz
def fmul(n, a, b):
    sa, ea, ma, za = fp_unpack(n, a); sb, eb, mb, zb = fp_unpack(n, b)
    p = mul(n, ma, mb, out_lo=21)                      # 48 bits; low 21 columns dropped (truncation error < 2^-24 rel, measured)
    top = p[47]
    mant = mux_bus(n, top, p[24:47], p[23:46])[:23]
    s10, _ = add(n, ea + [0, 0], eb + [0, 0], top, 10)
    r, _ = add(n, s10, const(10, (-127) % 1024), 0, 10)            # r = ea+eb+top-127 (10 bit two's complement)
    neg = r[9]; rz = n.NOT(or_all(n, r))
    ok = n.AND(n.AND(za, zb), n.AND(n.NOT(neg), n.NOT(rz)))
    out = [n.AND(ok, m) for m in mant] + [n.AND(ok, r[i]) for i in range(8)] + [n.AND(ok, n.XOR(sa, sb))]
    return out
def fneg(n, a): return a[:31] + [n.AND(or_all(n, a[23:31]), n.NOT(a[31]))]
def fadd(n, a, b):
    sa, ea, ma, za = fp_unpack(n, a); sb, eb, mb, zb = fp_unpack(n, b)
    agb = ge(n, a[:31], b[:31])                                  # |a| >= |b|
    big_m = mux_bus(n, agb, ma, mb); sml_m = mux_bus(n, agb, mb, ma)
    big_e = mux_bus(n, agb, ea, eb); sml_e = mux_bus(n, agb, eb, ea); sg = n.MUX(agb, sa, sb)
    d, _ = sub(n, big_e, sml_e, 8)
    big = [0, 0, 0] + big_m                                      # 27 bits, guard bits at 0..2
    sm = [0, 0, 0] + sml_m
    sm = shr(n, sm, d[:5]); far = or_all(n, d[5:8]); sm = [n.AND(n.NOT(far), x) for x in sm]
    sb_ = n.XOR(sa, sb)
    s, c = add(n, big, [n.XOR(x, sb_) for x in sm], sb_, 27)
    R = s + [n.AND(c, n.NOT(sb_))]                               # 28 bits
    # normalise: shift left by lz so that msb at bit 27
    cur = R; lz = []
    for k in (4, 3, 2, 1, 0):
        d_ = 1 << k; z = n.NOT(or_all(n, cur[28 - d_:]))
        cur = [n.MUX(z, cur[i - d_] if i - d_ >= 0 else 0, cur[i]) for i in range(28)]; lz.append(z)
    lzb = lz[::-1]                                               # lz bit0..4
    nzr = cur[27]
    # exp_new = big_e + 1 - lz   (10-bit)
    t, _ = add(n, big_e + [0, 0], [], 1, 10)
    en, _ = sub(n, t, lzb + [0] * 5, 10)
    neg = en[9]; ez = n.NOT(or_all(n, en))
    ok = n.AND(nzr, n.AND(n.NOT(neg), n.NOT(ez)))
    return [n.AND(ok, cur[i]) for i in range(4, 27)] + [n.AND(ok, en[i]) for i in range(8)] + [n.AND(ok, sg)]
def fsub(n, a, b): return fadd(n, a, fneg(n, b))
def fhalf(n, a):
    """a * 0.5: exponent - 1 (flush to zero if it hits 0)"""
    e, _ = sub(n, a[23:31] + [0], const(9, 1), 9); ok = n.AND(n.NOT(e[8]), n.NOT(n.NOT(or_all(n, e[:8]))))
    ok = n.AND(ok, or_all(n, a[23:31]))
    return [n.AND(ok, x) for x in a[:23]] + [n.AND(ok, e[i]) for i in range(8)] + [n.AND(ok, a[31])]
def fconst(v):
    import struct
    u = struct.unpack('<I', struct.pack('<f', v))[0]; return [(u >> i) & 1 for i in range(32)]
def fcmp_key(n, a):
    """monotone unsigned key of a float (sign-magnitude -> ordered): negative: ~bits ; positive: bits | sign"""
    s = a[31]; return [n.XOR(a[i], s) for i in range(31)] + [n.NOT(s)]
def fmax(n, a, b):
    ka, kb = fcmp_key(n, a), fcmp_key(n, b); g = ge(n, ka, kb); return mux_bus(n, g, a, b), g
