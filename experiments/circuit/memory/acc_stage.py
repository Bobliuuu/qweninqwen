"""Digit accumulation: reduce the digits of Pg MAC pipelines (72 digits each, from mac_layer.py: 8 planes x (3 hi-chunk
+ 6 lo-chunk) digits <=126/120) into binary rails of  T = sum_pipelines sum_p 2^p (16*sum HS + sum LS)  ( = sum (8+w')a' ).
Every layer is a real stock-shaped SwiGLU sublayer run in bf16 (same engine as mac_layer.py) and is checked exact.
Layers: PRESPLIT (digit -> nibbles) -> COMPRESS levels (group sum <=255 -> two nibbles, one column up, in radix 16)
-> F1 (carry generate/propagate) -> F2 (carry lookahead, AND units) -> F3 (final nibble -> 4 bit rails).
python acc_stage.py [Pg]"""
import sys, json, math, torch
from mac_layer import SC, DEV, DT, Res, fwd, mk

GD = 16.0

class Ent:
    __slots__ = ('dim', 'mult', 'cap', 'col')
    def __init__(s, dim, mult, cap, col): s.dim, s.mult, s.cap, s.col = dim, mult, cap, col

def group_rows(car, inputs, capS, out_lo, out_hi, out_mult=1.0):
    """inputs [(dim, coef)], S = sum coef*x <= capS (<=255); lo=S-16hi -> out_lo dim, hi=floor(S/16) -> out_hi dim."""
    rows = []
    nh = capS // 16
    for t in range(16, 16 * nh + 1, 16):
        for sgn, off in ((1, 1.0), (-1, 0.0)):
            g = {d: GD * c * SC for d, c in inputs}; g[car] = GD * (off - t)
            rows.append((g, {car: 1.0}, {out_hi: sgn * out_mult / GD, out_lo: -sgn * out_mult}))
    rows.append(({car: 16.0}, {d: c * SC for d, c in inputs}, {out_lo: out_mult / 16.0}))     # identity: lo gets S
    for d, c in inputs:                                                                         # clear inputs
        rows.append(({car: 16.0}, {d: SC}, {d: -1.0 / 16.0}))
    return rows, nh

def run(Pg=14, P=32, H=2560, I=9216):
    R = Res(H); car = R.new('car'); R.new('cst', H // 256)
    nH, nL = 3, 6
    ents = {}                                    # col -> list of Ent
    digits = []                                  # (dim, kind, p, chunk)
    free = []
    for pi in range(Pg):
        for p in range(8):
            for kind, nn, cap in (('HS', nH, 126), ('LS', nL, 120)):
                for ch in range(nn):
                    digits.append((R.new('d'), kind, p, cap))
    used = R.n
    free = list(range(used, H))
    X = torch.zeros(P, H); X[:, car] = SC; X[:, R.d['cst']:R.d['cst'] + H // 256] = 2.0 ** 18
    vals = torch.zeros(P, len(digits)); ref = torch.zeros(P, dtype=torch.long)
    for i, (d, kind, p, cap) in enumerate(digits):
        v = torch.randint(0, cap + 1, (P,)); X[:, d] = v.float(); vals[:, i] = v.float()
        ref += (v.long() << (p + (4 if kind == 'HS' else 0)))
    X = X.to(DEV, DT); layers = []
    def apply(rows_all, name):
        nonlocal X
        # pack groups into layers of <= I units
        cur = []; packs = []
        for grp in rows_all:
            if len(cur) + len(grp) > I: packs.append(cur); cur = []
            cur += grp
        if cur: packs.append(cur)
        for pk in packs:
            X, _ = fwd(X, *mk(I, H, pk)); layers.append((name, len(pk)))
    # PRESPLIT: each digit -> lo nibble (col k, mult m) and hi (col k+1, mult m); logical mult m=2^r kept as coefficient later
    pending_free = []
    cols = {}
    alloc = lambda: free.pop()
    CH = 450
    for c0 in range(0, len(digits), CH):
        rows_all = []; newfree = []
        for (d, kind, p, cap) in digits[c0:c0 + CH]:
            base = p + (4 if kind == 'HS' else 0); k, r = divmod(base, 4); m = float(2 ** r)
            lo = alloc(); hi = alloc()
            rows, nh = group_rows(car, [(d, 1.0)], cap, lo, hi)
            rows_all.append(rows); newfree.append(d)
            cols.setdefault(k, []).append(Ent(lo, m, 15, k)); cols.setdefault(k + 1, []).append(Ent(hi, m, nh, k + 1))
        apply(rows_all, 'PRESPLIT'); free += newfree
    level = 0
    while True:
        tot = {k: sum(e.cap * e.mult for e in v) for k, v in cols.items()}
        if all(t <= 31 for t in tot.values()): break
        level += 1; newcols = {}; todo = []
        for k, v in sorted(cols.items()):
            v = sorted(v, key=lambda e: -e.cap * e.mult); gl = []; cur = []; cs = 0
            for e in v:
                c = e.cap * e.mult
                if cs + c > 255: gl.append((cur, cs)); cur, cs = [], 0
                cur.append(e); cs += c
            if cur: gl.append((cur, cs))
            todo += [(k, g, cs) for (g, cs) in gl]
        while todo:
            rows_all = []; newfree = []; nu = 0
            while todo and len(free) >= 2:
                k, g, cs = todo[0]
                nh_ = int(cs) // 16
                need = 2 * nh_ + 1 + len(g)
                if nu + need > I: break
                todo.pop(0)
                lo = alloc(); hi = alloc()
                rows, nh = group_rows(car, [(e.dim, e.mult) for e in g], int(cs), lo, hi)
                rows_all.append(rows); nu += len(rows); newfree += [e.dim for e in g]
                newcols.setdefault(k, []).append(Ent(lo, 1.0, min(15, int(cs)), k))
                newcols.setdefault(k + 1, []).append(Ent(hi, 1.0, nh, k + 1))
            assert rows_all, 'dim pool exhausted'
            apply(rows_all, f'COMPRESS{level}'); free += newfree
        cols = newcols
        if level > 12: raise RuntimeError('no convergence')
    K = max(cols) + 1
    # F1: s_k = sum of entries in column k (<=31): g_k=[s>=16], p_k=[s==15]  (as 0/1 rails)
    gdim, pdim = {}, {}; rows_all = []; rows = []
    for k in range(K):
        es = cols.get(k, [])
        if not es: continue
        gdim[k] = alloc(); pdim[k] = alloc()
        for (t, tgt, sg0) in ((16, gdim[k], 1), (15, pdim[k], 1), (16, pdim[k], -1)):
            for sgn, off in ((1, 1.0), (-1, 0.0)):
                g = {e.dim: GD * e.mult * SC for e in es}; g[car] = GD * (off - t)
                rows.append((g, {car: 1.0}, {tgt: sg0 * sgn / GD}))
    apply([rows], 'F1')
    # F2: carries c_j (into column j) = sum_{i<j} g_i prod_{i<l<j} p_l   (AND units; terms mutually exclusive)
    cdim = {}; rows = []
    for j in range(1, K + 1):
        cdim[j] = alloc()
        for i in range(j):
            if i not in gdim: continue
            fac = [gdim[i]] + [pdim[l] for l in range(i + 1, j) if l in pdim]
            if any(l not in pdim for l in range(i + 1, j)): continue
            g = {d: GD * SC for d in fac}; g[car] = GD * (1 - len(fac))          # sum - (n-1) in units of x'=rail value
            g = {d: GD for d in fac}; g[car] = GD * (1 - len(fac))
            rows.append((g, {car: 1.0}, {cdim[j]: 1.0 / GD}))
    # rails here are 0/1 stored raw (not SC-scaled): coefficients above act on x' = x/SC... fix by scaling
    rows = [({d: (c * SC if d != car else c) for d, c in g.items()}, u, dn) for (g, u, dn) in rows]
    apply([rows], 'F2')
    # F3: nibble_k = (s_k + c_k) mod 16 as 4 bit rails, using telescoping steps t=1..31
    bit = {}; rows = []
    for k in range(K):
        es = cols.get(k, [])
        bit[k] = [alloc() for _ in range(4)]
        for t in range(1, 32):
            for sgn, off in ((1, 1.0), (-1, 0.0)):
                g = {e.dim: GD * e.mult * SC for e in es}
                if k in cdim: g[cdim[k]] = GD * SC
                g[car] = GD * (off - t)
                dn = {}
                for j in range(4):
                    dlt = ((t >> j) & 1) - (((t - 1) >> j) & 1)
                    if dlt: dn[bit[k][j]] = sgn * dlt / GD
                rows.append((g, {car: 1.0}, dn))
    apply([rows], 'F3')
    Xf = X.float().cpu()
    got = torch.zeros(P, dtype=torch.long); raw_ok = True; dev = 0.0
    for k in range(K):
        for j in range(4):
            v = Xf[:, bit[k][j]]
            dev = max(dev, (v - v.round()).abs().max().item()); raw_ok &= bool(((v.round() == 0) | (v.round() == 1)).all())
            got += (v.round().long() << (4 * k + j))
    return dict(Pg=Pg, digits=len(digits), positions=P, layers=len(layers), layer_units=layers,
                accumulate_exact=bool((got == ref).all()), rails_are_bits=raw_ok, max_rail_dev=dev, result_bits=4 * K,
                max_T=int(ref.max()), live_dims_peak_bound=H - len(free))

if __name__ == '__main__':
    Pg = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    print(json.dumps(run(Pg)), flush=True)
