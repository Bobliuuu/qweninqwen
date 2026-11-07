#!/usr/bin/env python3
"""Reduce the real 96-chunk reference trace (qfr-data cfram_q/refQ/tok_NNNN.npy) to the small JSON the visuals read.
Inputs (downloaded with `modal volume get qfr-data cfram_q/refQ/tok_NNNN.npy <TOKDIR>` and cf_layout.json):  python3 prep_data.py TOKDIR LAYOUT_JSON OUT_JSON
Token decoding = the fixed family layout of chipformer.rv.spec with the weight-fetch patch (V,F0,F1,F2,J,R,BYTE,FLT,HEX,QS,QD,BD,WB,ROW(2^18), then NT,ST0,ST1,SS0,SS1,HALT,GO,SC)."""
import sys, json, os, numpy as np
tokdir, layout_p, out = sys.argv[1:4]
SIZES = [("V", 1 << 15), ("F0", 1 << 15), ("F1", 1 << 15), ("F2", 4), ("J", 1 << 16), ("R", 1 << 16), ("BYTE", 512), ("FLT", 8), ("HEX", 16), ("QS", 16), ("QD", 128), ("BD", 256), ("WB", 1 << 15), ("ROW", 1 << 18)]
BASE = {}; b = 0
for n, s in SIZES: BASE[n] = (b, s); b += s
CTRL = b; CNAMES = ["NT", "ST0", "ST1", "SS0", "SS1", "HALT", "GO", "SC"]
assert CTRL == 525228 and CTRL + 5 == 525233
def fam(t):
    if t >= CTRL: return CNAMES[t - CTRL], 0
    for n, (b0, s) in BASE.items():
        if b0 <= t < b0 + s: return n, t - b0
    raise ValueError(t)
FAMS = [n for n, _ in SIZES] + CNAMES
edges = np.array([BASE[n][0] for n, _ in SIZES] + [CTRL + i for i in range(len(CNAMES))])
def famidx(a): return np.searchsorted(edges, a, side='right') - 1
L = json.load(open(layout_p)); WR = L['layer_weight_rows']; KL = L['KL']; KH = L['KH']
starts = [KH]
for w in WR: starts.append(starts[-1] + w + KL)
LM0 = L['LM_ROW0']; assert starts[-1] + 3 == LM0, (starts[-1], LM0)     # 3 final-norm rows, then the LM-head weight rows
def row_layer(r):                                   # layer index 0..27, 28 = final norm/lm head weights, 29 = lm scales, -1 = head (embedding row, rope)
    r = np.asarray(r)
    lay = np.full(r.shape, -1)
    for l in range(len(WR)): lay[(r >= starts[l]) & (r < starts[l + 1])] = l
    lay[(r >= starts[-1]) & (r < LM0)] = 28; lay[(r >= LM0) & (r < L['LS_ROW'])] = 28; lay[r >= L['LS_ROW']] = 29
    return lay
NCH = 96; chunks = []; glob = 0; layer_first = {}; layer_last = {}
INS = {n: i for i, n in enumerate(FAMS)}
bins0 = None; win0 = None; tail95 = None
for c in range(NCH):
    t = np.load(f'{tokdir}/tok_{c:04d}.npy').astype(np.int64)
    n = len(t); fi = famidx(t)
    cnt = np.bincount(fi, minlength=len(FAMS)).tolist()
    rowm = fi == INS['ROW']; rid = t[rowm] - BASE['ROW'][0]
    lay = row_layer(rid); hist = np.bincount(lay + 1, minlength=31).tolist()   # index 0: head(-1), 1..28 layers 0..27, 29 lm, 30 lm scales
    pos = np.nonzero(rowm)[0]
    for l in np.unique(lay):
        p = pos[lay == l]; g0, g1 = glob + int(p.min()), glob + int(p.max())
        layer_first[int(l)] = min(layer_first.get(int(l), g0), g0); layer_last[int(l)] = max(layer_last.get(int(l), g1), g1)
    chunks.append(dict(chunk=c, tokens=n, start=glob, fam_counts=cnt, row_layer_hist=hist, row_min=int(rid.min()) if len(rid) else None, row_max=int(rid.max()) if len(rid) else None))
    if c == 0:
        nb = 160; edges_b = np.linspace(0, n, nb + 1).astype(int)
        bins0 = [np.bincount(fi[edges_b[i]:edges_b[i + 1]], minlength=len(FAMS)).tolist() for i in range(nb)]
        first_row = int(np.argmax(rowm)); s0 = max(1, first_row - 6)
        Wn = 18; rp = np.nonzero(rowm)[0]; rp = rp[rp > 20000]; rid_all = t[rp] - BASE['ROW'][0]; rich = None
        for p_, r_ in zip(rp, rid_all):
            if not (3 <= r_ < 4099): continue            # a layer-0 WEIGHT row (not a constant row)
            i0 = int(p_) - (Wn - 4); names_ = set(FAMS[k] for k in fi[i0:i0 + Wn])
            if 'WB' in names_ and ('J' in names_ or 'V' in names_) and len(names_) >= 4: rich = i0; break
        assert rich is not None
        rich0 = dict(start=int(rich), tokens=[int(x) for x in t[rich:rich + Wn]], names=[(f'{fam(int(x))[0]}({fam(int(x))[1]})' if fam(int(x))[0] not in CNAMES else fam(int(x))[0]) for x in t[rich:rich + Wn]])
        win0 = dict(start=s0, tokens=[int(x) for x in t[s0:s0 + 22]], names=[(f'{fam(int(x))[0]}({fam(int(x))[1]})' if fam(int(x))[0] not in CNAMES else fam(int(x))[0]) for x in t[s0:s0 + 22]])
    if c == 95:
        h = [i for i, x in enumerate(t) if x == CTRL + 5]; assert h == [n - 1] or len(h) == 1, h
        tail95 = dict(n=n, halt_index=int(h[0]), tail_names=[(f'{fam(int(x))[0]}({fam(int(x))[1]})' if fam(int(x))[0] not in CNAMES else fam(int(x))[0]) for x in t[-24:]], tail_tokens=[int(x) for x in t[-24:]])
    glob += n
lay_ranges = {str(k): [layer_first[k], layer_last[k]] for k in sorted(layer_first)}
json.dump(dict(total_tokens=glob, fams=FAMS, chunks=chunks, layer_ranges=lay_ranges, layer_starts=starts, layer_weight_rows=WR, lm_row0=LM0, bins0=bins0, win0=win0, rich0=rich0, tail95=tail95), open(out, 'w'))
print('total trace tokens', glob, 'chunks', len(chunks), 'layers', len(lay_ranges), 'out', out, os.path.getsize(out))
