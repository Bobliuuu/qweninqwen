#!/usr/bin/env python3
"""Render the README animations (white background) from REAL run data. See ../PROVENANCE.md.
  circuit_hero        10 s  the whole idea (schematic layout; layer widths, counts and the answer are real)
  circuit_one_token   24 s  how the single token is generated: whole run -> chunk 0 -> 18 real trace tokens -> 28 layers -> HALT -> 15846
  circuit_seam_grid   11 s  96 chunk jobs, 95 seams (replay of real per-job durations as if all started together)
  circuit_precision   10 s  mixed precision of the 28 layers (layers 2/26/27 at 8-bit weights + 16-bit activations)
usage: python3 make_anim.py [hero|one_token|seam_grid|precision ...] [--stills]   (needs Pillow + ffmpeg)"""
import json, os, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from anim_lib import *
OUT = os.path.abspath(os.path.join(HERE, '..')); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..')); FR = os.path.join(REPO, 'experiments', 'circuit', 'fullrun')
def J(p): return json.load(open(p))
D = J(os.path.join(HERE, 'one_token_data.json')); T = D['total_tokens']; LR = D['layer_ranges']
seams = J(os.path.join(FR, 'logs', 'cfram_q_vast', 'seams_0-95_20261107_004845.json')); rows = {r['chunk']: r for r in seams['rows']}
assert seams['ok'] and T == 199287158
bounds = [LR[str(l)][0] for l in range(28)] + [LR['28'][0], T]; bounds[0] = 0
SEG = [(bounds[i], bounds[i + 1]) for i in range(29)]            # 28 layers + LM head (+ final norm), real shares of the trace
SPAN = [b - a for a, b in SEG]; BIG = (2, 26, 27)
plain = statistics.median(SPAN[l] for l in range(28) if l not in BIG)
FOOT = 'Visualization of a real run  ·  one token at position 0  ·  96 chunks  ·  95/95 exact seams  ·  0 mismatches  ·  0 host injections'

def seg_bar(d, x0, y0, x1, h, g, labels=False, lab_size=14, alpha=1.0):
    """segments proportional to the real trace-token share of each layer; g = fraction of the run completed (0..1)"""
    wtot = x1 - x0
    for i, (a, b) in enumerate(SEG):
        sx0 = x0 + a / T * wtot; sx1 = x0 + b / T * wtot - 2; done = clamp((g * T - a) / (b - a))
        base = ACCL if i < 28 else G4
        rr(d, sx0, y0, sx1, y0 + h, 7, base, None, alpha=alpha)
        if done > 0:
            col = ACCD if i in BIG else (ACC if i < 28 else ACC2)
            rr(d, sx0, y0, sx0 + (sx1 - sx0) * done, y0 + h, 7, col, None, alpha=alpha)
            if done < 1: d.rectangle((sx0 + (sx1 - sx0) * done - 3, y0, sx0 + (sx1 - sx0) * done, y0 + h), fill=ACC)
        if labels:
            lab = str(i) if i < 28 else 'LM head'
            text(d, (sx0 + sx1) / 2, y0 + h + 14, lab, lab_size if i < 28 else lab_size + 4, ACC if i in BIG else G1, 'Semibold' if i in BIG else 'Regular', 'ma', alpha=alpha)

def style(name):
    f = name.split('(')[0]
    if f in ('ROW', 'WB'): return ACCD, ACCD, WHITE
    if f in ('V', 'F0', 'F1', 'F2'): return ACCL, ACCM, ACCD
    if f == 'HALT': return ACC, ACC, WHITE
    return G4, G3, G1
def chip(d, x, y, w, h, name, alpha=1.0, big=False):
    fill, out, tc = style(name); rr(d, x, y, x + w, y + h, 14, fill, out, 2, alpha)
    f, _, p = name.partition('('); p = p.rstrip(')')
    if p: text(d, x + w / 2, y + h * .30, f, 28 if big else 22, tc, 'Bold', 'mm', alpha=alpha); text(d, x + w / 2, y + h * .68, p, 22 if big else 18, tc, 'Regular', 'mm', True, alpha)
    else: text(d, x + w / 2, y + h / 2, f, 28 if big else 22, tc, 'Bold', 'mm', alpha=alpha)
def foot(d, a=1.0): text(d, W / 2, 1030, FOOT, 22, G2, 'Regular', 'mm', alpha=a)

# ============================ HERO ============================
RICH = D['rich0']['names']
def hero(t):
    img, d = new()
    a = smooth(P(t, 0, .9)); text(d, 120, 96, 'A real Qwen3-0.6B token,', 62, INK, 'Bold', alpha=a); text(d, 120, 172, 'computed inside another Qwen.', 62, ACC, 'Bold', alpha=a)
    text(d, 122, 262, "a RISC-V circuit in the weights of a Qwen3.5-style outer model  ·  weights fetched from the model's own embedding table", 26, G1, alpha=smooth(P(t, .8, 1.6)))
    bx0, by0, bx1, by1 = 480, 320, 1440, 800; ba = smooth(P(t, .5, 1.3))
    rr(d, bx0, by0, bx1, by1, 30, '#FAFAFF' if False else (250, 250, 255), ACCM, 3, ba)
    text(d, bx0 + 36, by0 + 30, 'OUTER MODEL', 20, ACC, 'Semibold', alpha=ba); text(d, bx0 + 36, by0 + 60, 'Qwen3_5 architecture · HF Qwen3_5ForCausalLM classes', 26, INK, 'Semibold', alpha=ba)
    ca = smooth(P(t, 1.2, 2.2))
    for i in range(44):                       # 12 chip layers + Qwen's 32
        y = by0 + 120 + i * 6.6; line(d, bx0 + 40, y, bx0 + 150, y, ACC if i < 12 else G2, 3, ca)
    text(d, bx0 + 40, by0 + 120 + 44 * 6.6 + 12, '12 chip layers', 16, ACC, 'Medium', alpha=ca); text(d, bx0 + 40, by0 + 120 + 44 * 6.6 + 32, "+ Qwen's 32", 16, G1, alpha=ca)
    rr(d, bx0 + 200, by0 + 110, bx0 + 560, by1 - 40, 20, WHITE, ACCM, 2, ca); text(d, bx0 + 224, by0 + 134, 'CIRCUIT', 18, ACC, 'Semibold', alpha=ca)
    run = ease_io(P(t, 2.5, 7.2)); pulse = .5 + .5 * math.sin(t * 9)
    for k, lab in enumerate(['registers · pc', 'RAM 32 KB', 'ALU · MUL · DIV', 'VDOT']):
        yy = by0 + 168 + k * 66; hot = (int(t * 6) % 4 == k) and 2.5 < t < 7.2
        rr(d, bx0 + 224, yy, bx0 + 536, yy + 52, 12, ACC if hot else ACCL, None, alpha=ca); text(d, bx0 + 244, yy + 26, lab, 22, WHITE if hot else ACCD, 'Medium', 'lm', alpha=ca)
    rr(d, bx0 + 600, by0 + 110, bx1 - 36, by1 - 40, 20, WHITE, ACCM, 2, ca); text(d, bx0 + 624, by0 + 134, 'EMBEDDING TABLE', 18, ACC, 'Semibold', alpha=ca)
    text(d, bx0 + 624, by0 + 160, 'embed_tokens.weight', 22, INK, 'Medium', alpha=ca)
    hi = int(t * 7) % 12 if 2.5 < t < 7.2 else -1
    for r in range(12): rr(d, bx0 + 624, by0 + 202 + r * 22, bx1 - 60, by0 + 214 + r * 22, 4, ACC if r == hi else ACCL, None, alpha=ca)
    text(d, bx0 + 624, by1 - 150, '168,684 rows × 3,840 int4 cells', 20, G1, alpha=ca); text(d, bx0 + 624, by1 - 120, 'ROW(r) → the weights', 20, ACC, alpha=ca)
    if 2.6 < t < 7.1: arrow(d, bx0 + 536, by0 + 330, bx0 + 600, by0 + 330 + 6 * math.sin(t * 9), ACC, 3)
    # input token
    ia = smooth(P(t, .6, 1.2)) * (1 - smooth(P(t, 2.1, 2.6)))
    ix = lerp(130, 560, ease_io(P(t, 1.3, 2.4)))
    if ia > 0: rr(d, ix, 548, ix + 210, 612, 16, ACCL, ACCM, 2, ia); text(d, ix + 105, 580, 'The · 785', 30, ACCD, 'Semibold', 'mm', alpha=ia)
    if t < 1.4: text(d, 130, 520, 'prompt token', 20, G2, alpha=ia)
    # trace token stream under the box
    if t > 2.4:
        sa = smooth(P(t, 2.4, 3.0)) * (1 - smooth(P(t, 7.2, 7.6)))
        sp = 34 + 8
        for k in range(-1, 45):
            xx = 120 + k * sp + (t * 340) % sp; nm = RICH[(k + int(t * 340 // sp)) % len(RICH)] if True else 'F0'
            if xx < 120 or xx > 1760: continue
            f, o, tc = style(nm); rr(d, xx, 836, xx + 32, 866, 6, f, o, 1, sa)
    g = run
    text(d, 120, 884, 'trace tokens', 22, G1, alpha=smooth(P(t, 2.4, 3)))
    text(d, 120, 908, commas(round(g * T)), 54, INK, 'Bold', alpha=smooth(P(t, 2.4, 3.0)))
    if 2.5 < t < 7.3: text(d, 120 + tw(commas(round(g * T)), 54, 'Bold') + 24, 944, 'sped up', 20, G2)
    seg_bar(d, 120, 990, 1800, 24, g, False, alpha=smooth(P(t, 2.4, 3)))
    if t > 3: text(d, 1800, 958, 'Qwen3-0.6B: layers 0 to 27, then the LM head (widths = real share of the 199,287,158 trace tokens)', 18, G2, 'Regular', 'ra', alpha=smooth(P(t, 2.6, 3.4)))
    # HALT and answer
    ha = smooth(P(t, 7.2, 7.8))
    if ha > 0:
        hx = lerp(1420, 1580, ease_out(P(t, 7.2, 8.2)))
        rr(d, hx, 380, hx + 120, 436, 14, ACC, None, 2, ha); text(d, hx + 60, 408, 'HALT', 28, WHITE, 'Bold', 'mm', alpha=ha)
    oa = smooth(P(t, 8.0, 8.9))
    if oa > 0:
        text(d, 1560, 560, '15846', 112, ACC, 'Bold', 'ls', alpha=oa) if False else None
        text(d, 1640, 470, '15846', 100, ACC, 'Bold', 'ma', alpha=oa); text(d, 1640, 596, "' Question'", 40, INK, 'Semibold', 'ma', alpha=smooth(P(t, 8.4, 9.2)))
        text(d, 1640, 656, "= Hugging Face's Qwen3-0.6B token", 22, G1, 'Regular', 'ma', alpha=smooth(P(t, 8.7, 9.4)))
    text(d, W / 2, 1052, FOOT, 20, G2, 'Regular', 'mm', alpha=smooth(P(t, 8.8, 9.6)))
    return img

# ============================ ONE TOKEN ============================
FAMGROUP = {'V': 'v', 'F0': 'v', 'F1': 'v', 'F2': 'v', 'BYTE': 'v', 'ROW': 'w', 'WB': 'w'}
FAMS = D['fams']
def chapter(d, n, title, sub, a=1.0):
    text(d, 120, 80, f'{n}', 22, ACC, 'Bold', alpha=a); text(d, 156, 76, title, 40, INK, 'Bold', alpha=a); text(d, 120, 138, sub, 24, G1, alpha=a)
    for k in range(5): d.ellipse((1700 + k * 26, 84, 1712 + k * 26, 96), fill=ACC if k + 1 == n else G3)
def span_text(a, b): return f'{commas(a)} to {commas(b)}'
def one_token(t):
    img, d = new()
    # chapter windows (seconds)
    if t < 5.0:      # 1 whole run
        n = 1; chapter(d, 1, 'The whole run', 'one token = 199,287,158 trace tokens, 96 chunks of 2,097,152 steps; one CPU instruction is about 1.12 trace tokens', smooth(P(t, 0, .5)))
        x0, x1, y0 = 160, 1760, 470; r = ease_io(P(t, .5, 2.8))
        text(d, W / 2, 380, f'{commas(round(T * r))}', 64, INK, 'Bold', 'mm', alpha=smooth(P(t, .3, 1)))
        text(d, W / 2, 428, 'trace tokens', 24, G1, 'Regular', 'mm', alpha=smooth(P(t, .3, 1)))
        seg_bar(d, x0, y0, x1, 76, r, True, 14)
        for c in range(96): xx = x0 + (c * 2097152) / T * (x1 - x0); line(d, xx, y0 + 100, xx, y0 + 112, G3, 1)
        text(d, x0, y0 + 136, '96 chunks', 20, G2, alpha=smooth(P(t, 2.6, 3.2)))
        text(d, x1, y0 + 136, 'widths = real share of the trace: layers 2, 26, 27 are ~3x wider (8-bit weights, 16-bit activations); the LM head is the last 26%', 20, G1, 'Regular', 'ra', alpha=smooth(P(t, 2.8, 3.6)))
        br = smooth(P(t, 3.2, 3.8)) * (1 - smooth(P(t, 4.4, 5.0)))
        cw = 2097152 / T * (x1 - x0); rr(d, x0 - 6, y0 - 12, x0 + cw + 6, y0 + 88, 8, None, ACC, 3, br); text(d, x0, y0 - 40, 'chunk 0', 22, ACC, 'Semibold', alpha=br)
        foot(d)
    elif t < 10.5:   # 2 chunk 0
        tt = t - 5.0; chapter(d, 2, 'Zoom into chunk 0', '2,097,153 steps (GO + 2,097,152), shown as 160 slices: what kind of token each step is', smooth(P(tt, 0, .5)))
        x0, x1, ybase, hmax = 160, 1760, 780, 360; B = D['bins0']; nb = len(B); bw = (x1 - x0) / nb
        grow = ease_io(P(tt, .4, 3.2)); idx = {f: i for i, f in enumerate(FAMS)}
        for i, bn in enumerate(B):
            if i / nb > grow: break
            tot = sum(bn); v = sum(bn[idx[f]] for f in ('V', 'F0', 'F1', 'F2', 'BYTE')) / tot; w_ = sum(bn[idx[f]] for f in ('ROW', 'WB')) / tot; c_ = max(0, 1 - v - w_)
            xx = x0 + i * bw; h1, h2, h3 = v * hmax, w_ * hmax, c_ * hmax
            d.rectangle((xx, ybase - h1, xx + bw - 1, ybase), fill=ACCM); d.rectangle((xx, ybase - h1 - h2, xx + bw - 1, ybase - h1), fill=ACCD); d.rectangle((xx, ybase - hmax, xx + bw - 1, ybase - h1 - h2), fill=G4)
        text(d, x0, ybase + 22, 'step 0', 20, G2); text(d, x1, ybase + 22, 'step 2,097,152', 20, G2, 'Regular', 'ra')
        text(d, x1, 190, f'step {commas(round(2097152 * clamp(grow)))}', 30, INK, 'Semibold', 'ra')
        lx = x0
        for lab, col in (('value tokens (V, F0, F1, F2)', ACCM), ('weight fetch (ROW, WB)', ACCD), ('control flow and the rest (J, R, ...)', G4)):
            rr(d, lx, ybase + 64, lx + 22, ybase + 86, 5, col, G3, 1); text(d, lx + 32, ybase + 75, lab, 22, G1, 'Regular', 'lm'); lx += 36 + tw(lab, 22) + 40
        ma = smooth(P(tt, 3.6, 4.2)) * (1 - smooth(P(tt, 5.0, 5.5))); sx = x0 + D['rich0']['start'] / 2097153 * (x1 - x0)
        rr(d, sx - 8, ybase - hmax - 10, sx + 8, ybase + 4, 4, None, ACC, 3, ma); text(d, sx + 24, ybase - hmax - 8, f'steps {commas(D["rich0"]["start"])} to {commas(D["rich0"]["start"] + 17)}', 22, ACC, 'Semibold', alpha=ma)
        foot(d)
    elif t < 16.0:   # 3 18 real tokens
        tt = t - 10.5; chapter(d, 3, '18 real trace tokens', f'chunk 0, steps {commas(D["rich0"]["start"])} to {commas(D["rich0"]["start"] + 17)}: the CPU computes values and fetches a weight row', smooth(P(tt, 0, .5)))
        names = D['rich0']['names']; cwid, gap = 88, 6; x0 = (W - (18 * (cwid + gap) - gap)) / 2; y = 400
        for i, nm in enumerate(names):
            ca = smooth(P(tt, .4 + i * .13, .65 + i * .13)); yy = y + (1 - ca) * 20
            chip(d, x0 + i * (cwid + gap), yy, cwid, 120, nm, ca, False)
        # annotations
        def bracket(i0, i1, lab, col, ta, yb=560):
            xa = x0 + i0 * (cwid + gap); xb = x0 + (i1 + 1) * (cwid + gap) - gap
            line(d, xa, yb, xb, yb, col, 3, ta); line(d, xa, yb - 8, xa, yb, col, 3, ta); line(d, xb, yb - 8, xb, yb, col, 3, ta); text(d, (xa + xb) / 2, yb + 14, lab, 22, col, 'Semibold', 'ma', alpha=ta)
        names_s = [n.split('(')[0] for n in names]
        wi = names_s.index('WB'); ri = names_s.index('ROW')
        ta = smooth(P(tt, 3.0, 3.6)); bracket(0, 12, 'values the CPU computes: 1 to 3 tokens each (V, F0, F1, F2)', ACC, ta)
        ta2 = smooth(P(tt, 3.4, 4.0)); bracket(wi, ri, 'WB: set the weight base · ROW: fetch a table row', ACCD, ta2)
        rid = int(names[ri].split('(')[1].rstrip(')'))
        pa = smooth(P(tt, 4.2, 4.9)); rr(d, 160, 700, 1760, 880, 20, ACCL, None, 2, pa)
        text(d, 200, 730, f'ROW({rid})', 40, ACCD, 'Bold', alpha=pa, mono=True)
        text(d, 200, 790, f"the next step embeds row {commas(rid)} of the outer model's embed_tokens.weight: its 3,840 int4 cells are the weights VDOT consumes.", 26, INK, alpha=pa)
        text(d, 200, 830, 'Row ids 3 to 4,098 are the weight rows of layer 0 of Qwen3-0.6B. No host supplies them.', 26, G1, alpha=pa)
        foot(d)
    elif t < 20.0:   # 4 layers
        tt = t - 16.0; chapter(d, 4, 'The 28 layers, then the LM head', 'each row id belongs to a layer: the trace sweeps through the model once', smooth(P(tt, 0, .5)))
        g = ease_io(P(tt, .4, 2.6)); seg_bar(d, 160, 400, 1760, 110, g, True, 16)
        text(d, 160, 330, f'{commas(round(g * T))} of {commas(T)} trace tokens', 30, INK, 'Semibold')
        la = smooth(P(tt, 2.5, 3.2))
        text(d, 160, 640, f'layer 2:  {span_text(0, 0).split(" ")[0] if False else commas(SPAN[2])} trace tokens', 28, ACCD, 'Semibold', alpha=la); text(d, 160, 684, f'a plain layer:  about {commas(round(plain))}', 28, G1, alpha=la)
        text(d, 1000, 640, f'layers 2, 26, 27 take about {SPAN[2] / plain:.1f}x the trace tokens of a plain layer', 28, INK, alpha=la); text(d, 1000, 684, '(8-bit weights, 16-bit activations)', 24, G1, alpha=la)
        text(d, 160, 760, f'LM head: {commas(SPAN[28])} trace tokens = {SPAN[28] / T * 100:.0f}% of the run', 28, INK, alpha=la)
        foot(d)
    else:            # 5 final
        tt = t - 20.0; chapter(d, 5, 'The last chunk: HALT', f'chunk 95 has {commas(D["tail95"]["n"])} tokens; the circuit halts at step {commas(D["tail95"]["halt_index"])}', smooth(P(tt, 0, .5)))
        names = D['tail95']['tail_names'][-14:]; cwid, gap = 100, 8; x0 = (W - (len(names) * (cwid + gap) - gap)) / 2
        for i, nm in enumerate(names): ca = smooth(P(tt, .3 + i * .08, .55 + i * .08)); chip(d, x0 + i * (cwid + gap), 330 + (1 - ca) * 16, cwid, 120, nm, ca)
        text(d, x0 + (len(names) - 1) * (cwid + gap) + cwid / 2, 480, f'step {commas(D["tail95"]["halt_index"])}', 22, ACC, 'Semibold', 'ma', alpha=smooth(P(tt, 1.6, 2.2)))
        by = seams['rows'][95]['readout']['circuit_bytes']; word = seams['rows'][95]['readout']['circuit_word']; assert word == 15846 and by[0] + by[1] * 256 == 15846
        ba = smooth(P(tt, 2.0, 2.7)); text(d, W / 2, 590, f'answer bytes {by}', 34, G1, 'Regular', 'mm', True, ba)
        text(d, W / 2, 650, f'{by[0]} + {by[1]} × 256 = {word}', 40, INK, 'Semibold', 'mm', alpha=smooth(P(tt, 2.5, 3.1)))
        oa = smooth(P(tt, 3.0, 3.8)); text(d, W / 2, 760, '15846', 130, ACC, 'Bold', 'mm', alpha=oa); text(d, W / 2, 860, "' Question'", 44, INK, 'Semibold', 'mm', alpha=smooth(P(tt, 3.4, 4.0)))
        text(d, W / 2, 920, "= the token Hugging Face's Qwen3-0.6B (fp32 and bf16) produces for this prompt", 26, G1, 'Regular', 'mm', alpha=smooth(P(tt, 3.6, 4.2)))
        foot(d)
    return img
ONE_TOKEN_S = 24.0

# ============================ SEAM GRID ============================
COLS, ROWS_, CS, CG = 12, 8, 80, 16; GX, GY = 120, 214
def cell_xy(k):                                  # snake order so consecutive chunks are always neighbours
    r, c = divmod(k, COLS); c = c if r % 2 == 0 else COLS - 1 - c
    return GX + c * (CS + CG), GY + r * (CS + CG)
SECS = [rows[c]['seconds'] for c in range(96)]; SMAX = max(SECS)
TFIN = [.9 + s / SMAX * 5.6 for s in SECS]
def seam_grid(t):
    img, d = new()
    text(d, 120, 70, 'Run the 96 chunk jobs in parallel', 46, INK, 'Bold'); text(d, 120, 132, "each job runs its own chunk and saves the circuit's own end state; the end state must equal the next job's start state", 24, G1)
    text(d, 120, 176, f'replay: all 96 jobs start together; each cell fills for its real job time ({min(SECS[:95]) / 60:.1f} to {SMAX / 60:.1f} min; 1 s here = {SMAX / 5.6:.0f} s)', 20, G2)
    done = [t >= TFIN[k] for k in range(96)]
    for k in range(95):                          # seam links
        (x0, y0), (x1, y1) = cell_xy(k), cell_xy(k + 1); both = done[k] and done[k + 1]
        col = ACC if both else G3
        if y0 == y1: xa, xb = sorted((x0, x1)); d.rectangle((xa + CS, y0 + CS / 2 - 3, xb, y0 + CS / 2 + 3), fill=col)
        else: d.rectangle((x0 + CS / 2 - 3, y0 + CS, x0 + CS / 2 + 3, y1), fill=col)
    for k in range(96):
        x, y = cell_xy(k); rr(d, x, y, x + CS, y + CS, 14, ACCL, ACCM if not done[k] else None, 2)
        f = clamp((t - .9) / (TFIN[k] - .9)) if TFIN[k] > .9 else 1.0
        if done[k]: rr(d, x, y, x + CS, y + CS, 14, ACC, None)
        elif f > 0.03: d.rectangle((x + 3, y + CS - 3 - (CS - 6) * f, x + CS - 3, y + CS - 3), fill=ACC2)
        text(d, x + CS / 2, y + CS / 2, str(k), 22 if done[k] else 20, WHITE if (done[k] or f > .5) else G2, 'Bold' if done[k] else 'Medium', 'mm')
    nd = sum(done); ns = sum(1 for k in range(95) if done[k] and done[k + 1])
    px = 1330
    for i, (big, lab) in enumerate([(f'{nd} / 96', 'chunk jobs clean'), (f'{ns} / 95', 'seams exact'), ('0', 'free-token mismatches'), ('0', 'forced tokens'), ('0', 'host injections')]):
        yy = 230 + i * 120; text(d, px, yy, big, 64, ACC if i < 2 else INK, 'Bold'); text(d, px, yy + 78, lab, 24, G1)
    a = smooth(P(t, 6.6, 7.4)); text(d, W / 2 + 200, 1000 - 0, 'PASS  ·  96/96 jobs clean · seams exact 95/95 · final word 15846', 30, ACC, 'Semibold', 'mm', alpha=a)
    text(d, W / 2 + 200, 1040, 'a seam is exact when registers, pc, anchor, n, wb, latch, all 8,192 RAM words and bn match', 20, G2, 'Regular', 'mm', alpha=a)
    return img

# ============================ PRECISION ============================
QR = os.path.join(FR, 'quantfix', 'results'); QB = J(os.path.join(QR, 'b8_apl2.json'))
INSTR_TOK = QB['instr'][0]; WROM_WORDS = QB['blob']['wrom_words']; assert QB['argmax_fw'] == 15846 and (INSTR_TOK, WROM_WORDS) == (189109345, 80396288)
assert J(os.path.join(FR, 'diag_quant', 'diag_quant.json'))['hf_argmax'] == 15846
def precision(t):
    img, d = new()
    text(d, 120, 70, 'Mixed precision, one answer', 46, INK, 'Bold'); text(d, 120, 132, 'Qwen3-0.6B runs as int4 weights; three layers carry more precision', 24, G1)
    x0, x1, y0, h = 130, 1790, 330, 150; bw = (x1 - x0) / 28
    text(d, x0, 290, '28 layers of Qwen3-0.6B', 22, G1, 'Medium')
    for l in range(28):
        a = smooth(P(t, .3 + l * .04, .6 + l * .04)); big = l in BIG; hl = smooth(P(t, 2.6 + (0.25 if big else 0), 3.4 + (0.25 if big else 0))) if big else 0
        xa, xb = x0 + l * bw + 3, x0 + (l + 1) * bw - 3; base = mix(ACCM, a)
        rr(d, xa, y0 + (1 - a) * 20, xb, y0 + h + (1 - a) * 20, 8, ACCM if not big else (int(lerp(ACCM[0], ACCD[0], hl)), int(lerp(ACCM[1], ACCD[1], hl)), int(lerp(ACCM[2], ACCD[2], hl))), None, alpha=a)
        text(d, (xa + xb) / 2, y0 + h + 30 + (1 - a) * 20, str(l), 16, ACC if big else G2, 'Semibold' if big else 'Regular', 'mm', alpha=a)
    text(d, x0, y0 + h + 76, 'int4 weights, int8 activations', 22, G1, alpha=smooth(P(t, 1.8, 2.4)))
    ca = smooth(P(t, 3.2, 4.0))
    for l in BIG:
        cx = x0 + (l + .5) * bw; line(d, cx, y0 + h + 52, cx, y0 + h + 130, ACC, 3, ca)
    rr(d, 560, y0 + h + 130, 1420, y0 + h + 214, 18, ACC, None, 2, ca)
    text(d, 990, y0 + h + 172, 'layers 2 · 26 · 27: 8-bit weights + 16-bit activations', 32, WHITE, 'Semibold', 'mm', alpha=ca)
    na = smooth(P(t, 5.0, 5.8))
    for i, (big_, lab) in enumerate([(commas(INSTR_TOK), 'instructions per token (ctx 1)'), (commas(WROM_WORDS), 'weight words (8 int4 cells each)')]):
        x = 130 + i * 560; y = 700
        rr(d, x, y, x + 520, y + 150, 20, WHITE, G3, 2, na); text(d, x + 30, y + 44, big_, 52, INK, 'Bold', alpha=na); text(d, x + 30, y + 106, lab, 24, G1, alpha=na)
    fa = smooth(P(t, 7.6, 8.6)); rr(d, 1260, 700, 1790, 850, 20, ACCL, ACCM, 2, fa)
    text(d, 1290, 748, '15846', 56, ACC, 'Bold', alpha=fa); text(d, 1290, 818, "matches Hugging Face's Qwen3-0.6B", 24, INK, 'Semibold', alpha=fa)
    ga = smooth(P(t, 8.2, 9.0)); text(d, 1700 if False else 1760, 748, '✓', 56, ACC, 'Bold', 'ra', alpha=ga) if False else None
    text(d, W / 2, 1030, 'Emulator figures for the final firmware (one prompt token). Sources: quantfix/results/b8_apl2.json, diag_quant.json', 20, G2, 'Regular', 'mm')
    return img

SCENES = {'hero': (hero, 10.0), 'one_token': (one_token, ONE_TOKEN_S), 'seam_grid': (seam_grid, 11.0), 'precision': (precision, 10.5)}
if __name__ == '__main__':
    want = [a for a in sys.argv[1:] if not a.startswith('--')] or list(SCENES)
    if '--stills' in sys.argv:
        for k in want:
            fn, dur = SCENES[k]
            for fr in (0.15, 0.4, 0.7, 0.95): print(still(f'{k}_{int(fr * 100)}', HERE, fn, fr * dur))
    else:
        for k in want: fn, dur = SCENES[k]; render('circuit_' + k, OUT, fn, dur)
