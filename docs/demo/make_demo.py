#!/usr/bin/env python3
"""Build docs/demo/demo.mp4, demo.gif and stills for qweninqwen (white, calm, one accent colour).

HARD RULE: every number/label drawn here is READ from a data file in this repo (see PROVENANCE.md).
Nothing is simulated. The only synthetic thing is animation (easing, cross-fades) and the chunk-grid
reveal order (chunk number order: per-chunk completion wall-clock times were not logged).

usage: python3 docs/demo/make_demo.py      (needs Pillow, ffmpeg on PATH)
"""
import json, os, shutil, statistics, subprocess, sys, collections
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FR = os.path.join(REPO, 'experiments', 'circuit', 'fullrun')
TMP = os.path.join(HERE, '_frames')
STILLS = os.path.join(HERE, 'stills')
W, H = 1280, 720
# palette: white, neutral greys, ONE accent
BG, FG, DIM, MUTE, LINE, SOFT = (255, 255, 255), (17, 24, 39), (107, 114, 128), (156, 163, 175), (229, 231, 235), (249, 250, 251)
ACC, ACC_L, ACC_M = (37, 99, 235), (219, 234, 254), (147, 180, 247)
DARK = (55, 65, 81)
FPS = 24

# ---------- data (all real) ----------
def jload(p): return json.load(open(p))
verdict = jload(os.path.join(HERE, 'data', 'FINAL_VERDICT.json'))
fleet = jload(os.path.join(HERE, 'data', 'fleet_result.json'))
seams = jload(os.path.join(FR, 'logs', 'cfram_q_vast', 'seams_0-95_20261107_004845.json'))
rows = {r['chunk']: r for r in seams['rows']}
assert len(rows) == 96 and seams['ok'] is True and 'PASS' in seams['verdict']
dq1 = jload(os.path.join(FR, 'diag_quant', 'diag_quant.json'))
dq2 = jload(os.path.join(FR, 'diag_quant', 'diag_quant2.json'))
QR = os.path.join(FR, 'quantfix', 'results')
qf = {n: jload(os.path.join(QR, n + '.json')) for n in ('ctrl_w4', 'b8_apl1', 'w4_apl2only', 'b8_apl2', 'b12_apl2')}
hf_pos = jload(os.path.join(QR, 'hfcheck_cfq_pos_130k.json')); hf_neg = jload(os.path.join(QR, 'hfcheck_cfq_neg.json'))
assert 'Question' in open(os.path.join(FR, 'logs', 'final_token', 'REPORT.md')).read()
assert dq1['hf_argmax'] == 15846 and qf['ctrl_w4']['argmax_fw'] == 13 and qf['b8_apl2']['argmax_fw'] == 15846
last = rows[95]; ro = last['readout']
assert ro['circuit_word'] == 15846 and ro['circuit_halt_step'] == 57705 and ro['tail_equals_reference'] is True

def src_of(c):
    if c in verdict['modal_chunks']: return 'Modal'
    g = (fleet['done'].get(str(c)) or {}).get('gpu') or ''
    if '4090' in g: return 'RTX 4090'
    if 'A100' in g: return 'A100 PCIE'
    if 'L40S' in g: return 'L40S'
    return 'unknown'
SRC = {c: src_of(c) for c in range(96)}
SRC_COL = {'RTX 4090': ACC, 'A100 PCIE': ACC_M, 'L40S': MUTE, 'Modal': DARK, 'unknown': LINE}
SRC_TXT = {'RTX 4090': (255, 255, 255), 'A100 PCIE': FG, 'L40S': FG, 'Modal': (255, 255, 255), 'unknown': FG}
by_src = collections.defaultdict(list)
for c in range(95): by_src[SRC[c]].append(rows[c]['seconds'])      # chunk 95 is the short HALT chunk (15 s)
total_steps = sum(r['steps_run'] for r in rows.values()); total_sec = sum(r['seconds'] for r in rows.values())
n_seam = sum(1 for r in rows.values() if r.get('seam', {}).get('exact'))
sum_mism = sum(r['free_mismatches'] for r in rows.values()); sum_forced = sum(r['forced_tokens'] for r in rows.values()); sum_inj = sum(r['injections'] for r in rows.values())
assert (n_seam, sum_mism, sum_forced, sum_inj) == (95, 0, 0, 0)

# ---------- drawing helpers ----------
_fc = {}
def font(sz, w='Regular'):
    k = (sz, w)
    if k not in _fc:
        f = ImageFont.truetype('/System/Library/Fonts/SFNS.ttf', sz)
        try: f.set_variation_by_name(w)
        except Exception: pass
        _fc[k] = f
    return _fc[k]
def mono(sz):
    k = ('m', sz)
    if k not in _fc: _fc[k] = ImageFont.truetype('/System/Library/Fonts/SFNSMono.ttf', sz)
    return _fc[k]
def menlo(sz):
    k = ('menlo', sz)
    if k not in _fc: _fc[k] = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', sz)
    return _fc[k]
_emo = {}
def emoji(ch, sz):
    k = (ch, sz)
    if k not in _emo:
        f = ImageFont.truetype('/System/Library/Fonts/Apple Color Emoji.ttc', 160)
        im = Image.new('RGBA', (220, 220), (0, 0, 0, 0)); ImageDraw.Draw(im).text((0, 0), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox()); r = sz / im.height
        _emo[k] = im.resize((max(1, int(im.width * r)), sz), Image.LANCZOS)
    return _emo[k]
def ease(t):                      # smooth in-out (cubic)
    t = max(0.0, min(1.0, t)); return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
def ease_out(t):
    t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3
def mix(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
def new():
    im = Image.new('RGB', (W, H), BG); return im, ImageDraw.Draw(im)
def tw(d, s, f): return d.textlength(s, font=f)
def text(d, xy, s, f, fill=FG, anchor='la'): d.text(xy, s, font=f, fill=fill, anchor=anchor)
def wrap(d, s, f, maxw):
    out, cur = [], ''
    for wd in s.split(' '):
        t = (cur + ' ' + wd).strip()
        if tw(d, t, f) <= maxw: cur = t
        else: out.append(cur); cur = wd
    if cur: out.append(cur)
    return out
def para(d, x, y, s, f, fill, maxw, lh):
    for ln in wrap(d, s, f, maxw): text(d, (x, y), ln, f, fill); y += lh
    return y
_sh = {}
def card(im, box, r=16, fill=BG, border=LINE, bw=1, shadow=True):
    x0, y0, x1, y1 = [int(v) for v in box]
    if shadow:
        k = (x1 - x0, y1 - y0, r)
        if k not in _sh:
            pad = 24; s = Image.new('RGBA', (k[0] + 2 * pad, k[1] + 2 * pad), (0, 0, 0, 0))
            ImageDraw.Draw(s).rounded_rectangle((pad, pad + 4, pad + k[0], pad + 4 + k[1]), r, fill=(17, 24, 39, 20))
            _sh[k] = s.filter(ImageFilter.GaussianBlur(9))
        im.paste(_sh[k], (x0 - 24, y0 - 24), _sh[k])
    ImageDraw.Draw(im).rounded_rectangle((x0, y0, x1, y1), r, fill=fill, outline=border, width=bw)
def title(im, d, s, emo=None, sub=None):
    x = 56
    if emo: e = emoji(emo, 40); im.paste(e, (x, 40), e); x += e.width + 14
    text(d, (x, 32), s, font(42, 'Semibold'), FG)
    if sub: text(d, (58, 92), sub, font(21), DIM)
FOOT = 'Visualization of a real run   ·   one token at position 0   ·   parallel jobs seeded from the reference, verified at 95 exact seams   ·   sources: PROVENANCE.md'
def footer(d):
    d.line((56, 678, W - 56, 678), fill=LINE, width=1); text(d, (56, 688), FOOT, font(14), MUTE)

# ---------- frame bookkeeping: segments (scenes) -> cross-fades ----------
segments = []          # list of lists of (png, dur)
_n = [0]
def begin(): segments.append([])
def save(im, dur):
    p = os.path.join(TMP, f'f{_n[0]:05d}.png'); _n[0] += 1; im.save(p); segments[-1].append((p, dur)); return p
def clamp(x): return max(0.0, min(1.0, x))

# ---------- scene 1: title ----------
def scene_title():
    begin(); im, d = new()
    f = font(80, 'Semibold'); label = 'Qwen inside Qwen'; e = emoji('🧠', 92)
    left = int((W - (tw(d, label, f) + e.width + 26)) // 2)
    im.paste(e, (left, 168), e); text(d, (left + e.width + 26, 148), label, f, FG)
    d.line((W // 2 - 40, 296, W // 2 + 40, 296), fill=ACC, width=3)
    text(d, (W // 2, 328), 'one Qwen3-0.6B token, computed by a circuit', font(30), DIM, 'ma')
    text(d, (W // 2, 372), 'living in the weights of Qwen3.5-4B', font(30), DIM, 'ma')
    card(im, (220, 470, W - 220, 590), 16, SOFT, LINE)
    text(d, (W // 2, 490), 'A visualization of a real run', font(24, 'Semibold'), FG, 'ma')
    para(d, 256, 530, 'Every number comes from the repo logs: one prompt token at position 0, 96 chunk-jobs, 95 exact state seams, 0 host injections.', font(19), DIM, W - 512, 26)
    footer(d); save(im, 5.0)

# ---------- scene 2: the problem ----------
def scene_problem():
    begin()
    cos = [dq2[f'W4+act ONLY layer {k}']['cos_final'] for k in range(28)]
    amx = [dq2[f'W4+act ONLY layer {k}']['argmax'] for k in range(28)]
    prof = {p['l']: p for p in dq1['fp32_profile']}
    x0, y0, x1, y1 = 84, 200, 760, 556
    N = 36
    for step in range(N + 1):
        t = ease_out(step / N)
        im, d = new(); title(im, d, 'Why the first answer was wrong', '🔍', 'Quantise one layer at a time (int4 weights, int8 activations) and watch where the answer flips')
        card(im, (56, 144, 790, 626), 16)
        text(d, (76, 160), 'Cosine of the final hidden state against fp32', font(17, 'Medium'), DIM)
        for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            yy = y1 - (v - 0.5) / 0.5 * (y1 - y0); d.line((x0, yy, x1, yy), fill=LINE, width=1); text(d, (x0 - 10, yy), f'{v:.1f}', font(13), MUTE, 'rm')
        bw = (x1 - x0) / 28
        for k in range(28):
            hh = (cos[k] - 0.5) / 0.5 * (y1 - y0) * t
            broken = amx[k] != 15846; col = ACC if broken else (209, 213, 219)
            bx = x0 + k * bw + 3
            d.rectangle((bx, y1 - hh, bx + bw - 6, y1), fill=col)
            if k % 3 == 0 or broken: text(d, (bx + (bw - 6) / 2, y1 + 8), str(k), font(13, 'Medium' if broken else 'Regular'), FG if broken else MUTE, 'ma')
            if broken and step == N:
                lab = Image.new('RGBA', (110, 22), (0, 0, 0, 0)); ImageDraw.Draw(lab).text((0, 0), f'answer {amx[k]}', font=font(14, 'Semibold'), fill=(255, 255, 255, 255))
                lab = lab.rotate(90, expand=True); im.paste(lab, (int(bx + (bw - 6) / 2 - lab.width / 2), int(y1 - hh + 10)), lab)
        text(d, (x0, y1 + 34), 'layer index   ·   blue = the answer flips when only this layer is quantised', font(14), MUTE)
        # right column
        card(im, (816, 144, W - 56, 334), 16)
        text(d, (840, 162), 'Our first firmware (int4)', font(18, 'Medium'), DIM)
        n13 = f"{qf['ctrl_w4']['argmax_fw']}"; text(d, (840, 192), n13, font(64, 'Semibold'), DARK); text(d, (840 + tw(d, n13, font(64, 'Semibold')) + 22, 214), "'.'", menlo(30), MUTE)
        text(d, (840, 274), 'the circuit reproduced this faithfully', font(17), DIM)
        text(d, (840, 298), '(same answer as the emulator)', font(17), MUTE)
        card(im, (816, 350, W - 56, 538), 16, BG, ACC, 2)
        text(d, (840, 368), 'Qwen3-0.6B itself (HF fp32)', font(18, 'Medium'), DIM)
        n15 = f"{dq1['hf_argmax']}"; text(d, (840, 398), n15, font(64, 'Semibold'), ACC); text(d, (840 + tw(d, n15, font(64, 'Semibold')) + 16, 430), "' Question'", menlo(21), ACC)
        text(d, (840, 492), 'prompt: one token [785] ("The")', font(16), DIM)
        text(d, (840, 514), f"rank of 15846 in the firmware: {qf['ctrl_w4']['rank_15846_in_fw']:,}", font(16), MUTE)
        card(im, (816, 554, W - 56, 626), 14, SOFT, LINE, 1, False)
        l2 = prof[2]; para(d, 834, 562, f"At position 0, layer 2 writes a ~{l2['dmax']:.0f} channel; everything else is O(1).", font(15), DIM, W - 56 - 834 - 14, 20)
        footer(d); save(im, 1 / FPS if step < N else 9.0)

# ---------- scene 3: the fix ----------
def scene_fix():
    begin()
    items = [('ctrl_w4', 'Current: int4 weights everywhere'), ('b8_apl1', 'Layers 2, 26, 27: 8-bit weights only'),
             ('w4_apl2only', 'Layers 2, 26, 27: 16-bit activations only'), ('b8_apl2', 'Layers 2, 26, 27: 8-bit weights + 16-bit activations'),
             ('b12_apl2', 'Layers 2, 26, 27: 12-bit weights + 16-bit activations')]
    N = 30
    for step in range(N + 1):
        t = ease_out(step / N)
        im, d = new(); title(im, d, 'The fix: three layers get more precision', '🔧', 'Real integer firmware in the emulator, prompt [785]; everything else stays int4')
        y = 142
        for i, (k, lab) in enumerate(items):
            r = qf[k]; ok = r['argmax_fw'] == 15846
            card(im, (56, y, W - 56, y + 76), 14, BG, ACC if ok else LINE, 2 if ok else 1)
            text(d, (80, y + 10), lab, font(21, 'Semibold'), FG)
            text(d, (80, y + 44), f"cos(logits, HF) {r['cos_logits_fw_vs_hf']:.3f}     rank of 15846: {r['rank_15846_in_fw']:,}     top-1 gap {r['gap_fw_top1_top2']:.2f}", font(15), DIM)
            bx0, bx1 = 780, 1000
            d.rounded_rectangle((bx0, y + 30, bx1, y + 42), 6, fill=(243, 244, 246))
            d.rounded_rectangle((bx0, y + 30, bx0 + max(8, (bx1 - bx0) * r['cos_logits_fw_vs_hf'] * t), y + 42), 6, fill=ACC if ok else MUTE)
            text(d, (1040, y + 14), f"{r['argmax_fw']}", font(38, 'Semibold'), ACC if ok else DARK)
            y += 88
        i0, i1 = qf['ctrl_w4']['instr'][0], qf['b8_apl2']['instr'][0]; w0, w1 = qf['ctrl_w4']['blob']['wrom_words'], qf['b8_apl2']['blob']['wrom_words']
        text(d, (56, 596), f"Cost of the chosen fix (8-bit + 16-bit): {100 * (i1 / i0 - 1):+.1f}% instructions per token ({i0:,} to {i1:,}), {100 * (w1 / w0 - 1):+.1f}% weight words ({w0:,} to {w1:,}).", font(17), FG)
        text(d, (56, 630), 'Both knobs are needed: finer weights alone give 9, finer activations alone give 408.', font(15), DIM)
        footer(d); save(im, 1 / FPS if step < N else 8.0)

# ---------- scene 4: the run ----------
def grid_frame(reveal, final=False):
    im, d = new(); title(im, d, 'The run: 96 chunk-jobs on rented GPUs', '🚀', 'Each cell is one chunk of 2,097,153 circuit steps (chunk 95: 57,707 steps), colored by the machine that ran it')
    cols = 16; cw, ch = 46, 46; gx, gy = 56, 142
    for c in range(96):
        cx, cy = gx + (c % cols) * (cw + 6), gy + (c // cols) * (ch + 6)
        t = 1.0 if final else max(0.0, min(1.0, (reveal - c) / 4.0))
        d.rounded_rectangle((cx, cy, cx + cw, cy + ch), 6, fill=(243, 244, 246))
        if t > 0:
            sc = 0.5 + 0.5 * ease_out(t); m = (1 - sc) * cw / 2; col = mix((243, 244, 246), SRC_COL[SRC[c]], ease_out(t))
            d.rounded_rectangle((cx + m, cy + m, cx + cw - m, cy + ch - m), 6, fill=col)
            if t > 0.7: text(d, (cx + cw / 2, cy + ch / 2), str(c), font(14, 'Medium'), mix(col, SRC_TXT[SRC[c]], (t - 0.7) / 0.3), 'mm')
    k = int(reveal); frac = reveal - k
    n = 96 if final else int(min(96, max(0, reveal - 1.5)))
    vis = [rows[c] for c in range(min(n, 96))]
    seam_ok = sum(1 for r in vis if r.get('seam', {}).get('exact'))
    cy0 = 480
    stats = [('chunks verified', f'{len(vis)} / 96', FG), ('seams exact', f'{seam_ok} / 95', ACC), ('free-token mismatches', str(sum(r['free_mismatches'] for r in vis)), ACC),
             ('forced tokens', str(sum(r['forced_tokens'] for r in vis)), ACC), ('host injections', str(sum(r['injections'] for r in vis)), ACC)]
    for i, (lab, val, col) in enumerate(stats):
        x = 56 + i * 232; card(im, (x, cy0, x + 220, cy0 + 90), 14)
        text(d, (x + 18, cy0 + 12), val, font(34, 'Semibold'), col); text(d, (x + 18, cy0 + 58), lab, font(15), DIM)
    ly = 590
    for i, s in enumerate(['RTX 4090', 'A100 PCIE', 'L40S', 'Modal']):
        if s not in by_src: continue
        xx = 56 + i * 305; d.rounded_rectangle((xx, ly + 3, xx + 16, ly + 19), 4, fill=SRC_COL[s])
        med = statistics.median(by_src[s]); nn = len(by_src[s])
        text(d, (xx + 26, ly), f"{s}: {nn} chunk{'s' if nn != 1 else ''}, median {med / 60:.1f} min", font(15), FG)
    if final: text(d, (56, 620), f'{total_steps:,} circuit steps in total   ·   {total_sec / 3600:.1f} GPU-hours of job time (sum of per-chunk seconds, not billed time)', font(15), DIM)
    text(d, (W - 56, 648), 'reveal order = chunk number (real completion times were not logged); medians exclude the short chunk 95', font(12), MUTE, 'ra')
    footer(d); return im

def scene_run():
    begin(); N = 150
    for s in range(N + 1):
        save(grid_frame(96 * ease(s / N) if s < N else 96), 1 / FPS)
    save(grid_frame(96, True), 6.0)

# ---------- scene 5: final chunk ----------
def scene_final():
    begin(); names = ro['tail_names_circuit']
    for step in range(len(names) + 1):
        im, d = new(); title(im, d, 'The last chunk: the circuit halts on its own', '🏁', f"chunk 95 reads out the answer   ·   {last['steps_run']:,} steps   ·   halt_step {ro['circuit_halt_step']:,}")
        text(d, (56, 146), "The circuit's own last tokens (steps %s to %s of the chunk)" % (f"{ro['from_step']:,}", f"{ro['to_step']:,}"), font(16), DIM)
        x, y = 56, 184
        for i, nm in enumerate(names[:step]):
            wpx = tw(d, nm, mono(16)) + 22
            if x + wpx > W - 56: x = 56; y += 42
            halt = nm == 'HALT'
            d.rounded_rectangle((x, y, x + wpx, y + 32), 8, fill=ACC if halt else SOFT, outline=ACC if halt else LINE, width=1)
            text(d, (x + wpx / 2, y + 16), nm, mono(16), (255, 255, 255) if halt else FG, 'mm'); x += wpx + 8
        if step >= len(names):
            b = ro['circuit_bytes']; word = ro['circuit_word']
            card(im, (56, 346, W - 56, 550), 18, BG, ACC, 2)
            text(d, (84, 364), 'Answer bytes written by the circuit', font(18, 'Medium'), DIM)
            text(d, (84, 398), f"{b}   =   {b[0]} + {b[1]} x 256   =   {word}", mono(32), FG)
            fw = font(52, 'Semibold'); text(d, (84, 454), f"{word}", fw, ACC); text(d, (84 + tw(d, str(word), fw) + 28, 466), "' Question'", menlo(26), ACC); text(d, (84 + tw(d, str(word), fw) + 28 + tw(d, "' Question'", menlo(26)) + 26, 470), 'what Qwen3-0.6B itself says', font(24, 'Medium'), FG)
            text(d, (84, 524), f"tail equals the reference trace: {ro['tail_equals_reference']}      free mismatches {last['free_mismatches']}      forced {last['forced_tokens']}      injections {last['injections']}", font(15), DIM)
            text(d, (56, 576), 'Collector verdict', font(15, 'Medium'), DIM)
            para(d, 56, 600, seams['verdict'].replace('; 0s to collect', ''), font(15), FG, W - 112, 22)
        footer(d); save(im, 0.16 if step < len(names) else 8.0)

# ---------- scene 6: what the host does (only its remaining roles) ----------
def scene_limits():
    begin()
    cards = [('Boot the program', 'allowed by the design', False), ('The GO token', 'allowed by the design', False),
             ('The embed-and-append loop', 'allowed by the design', False), ('ecall I/O', 'allowed by the design', False),
             ('Reference comparison', 'the test, not an input', False),
             ("Later chunks' start state", f'seeded from the reference, verified by exact end-state equality at {n_seam} seams', True)]
    N = 48; cw, chh = 560, 118; gx, gy = 56, 168
    for step in range(N + 1):
        im, d = new(); title(im, d, 'What the host does', '📋', f'Everything else runs in the circuit: {sum_inj} injections, {sum_forced} forced tokens, {sum_mism} mismatches in all 96 chunks')
        for i, (head, sub, key) in enumerate(cards):
            t = ease_out((step - i * 6) / 14)
            if t <= 0: continue
            x = gx + (i % 2) * (cw + 48); y = int(gy + (i // 2) * (chh + 28) + (1 - t) * 16)
            card(im, (x, y, x + cw, y + chh), 16, BG, ACC if key else LINE, 2 if key else 1)
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((x + 24, y + 26, x + 30, y + 58), 3, fill=mix(BG, ACC if key else MUTE, t))
            text(d, (x + 46, y + 22), head, font(25, 'Semibold'), mix(BG, FG, t))
            para(d, x + 46, y + 64, sub, font(17), mix(BG, DIM, t), cw - 72, 23)
        footer(d); save(im, 1 / FPS if step < N else 9.0)

# ---------- scene 7: end ----------
def scene_end():
    begin(); im, d = new()
    e = emoji('✅', 70); f = font(72, 'Semibold'); left = int((W - (tw(d, 'PASS', f) + e.width + 22)) // 2)
    im.paste(e, (left, 170), e); text(d, (left + e.width + 22, 148), 'PASS', f, FG)
    d.line((W // 2 - 40, 270, W // 2 + 40, 270), fill=ACC, width=3)
    text(d, (W // 2, 304), f"96 / 96 chunks clean   ·   {n_seam} / 95 seams exact   ·   final word {ro['circuit_word']}", font(28, 'Medium'), FG, 'ma')
    text(d, (W // 2, 380), 'logs: experiments/circuit/fullrun/logs/cfram_q_vast/', mono(18), DIM, 'ma')
    text(d, (W // 2, 414), 'every number in this video: docs/demo/PROVENANCE.md', mono(18), DIM, 'ma')
    footer(d); save(im, 5.0)

# ---------- assemble ----------
XF = 12   # cross-fade frames (0.5 s)
def assemble():
    order = []   # (png, dur)
    prev_last = None
    for si, seg in enumerate(segments):
        if prev_last is not None:
            a = Image.open(prev_last); b = Image.open(seg[0][0])
            for j in range(1, XF + 1):
                p = os.path.join(TMP, f'x{si:02d}_{j:02d}.png'); Image.blend(a, b, ease(j / (XF + 1))).save(p); order.append((p, 1 / FPS))
        order += seg; prev_last = seg[-1][0]
    lst = os.path.join(TMP, 'list.txt')
    with open(lst, 'w') as f:
        for p, dur in order: f.write(f"file '{p}'\nduration {dur:.4f}\n")
        f.write(f"file '{order[-1][0]}'\n")
    out = os.path.join(HERE, 'demo.mp4')
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-vf', f'fps={FPS},format=yuv420p', '-c:v', 'libx264', '-crf', '20', '-preset', 'slow', '-movflags', '+faststart', out])
    total = sum(d for _, d in order)
    # teaser gif: title (2 s), the grid reveal (every 2nd frame), the final answer card
    grid = segments[3]; fin = segments[4]
    sel = [(segments[0][0][0], 2.0)] + [(p, 2 / FPS) for p, _ in grid[:-1:2]] + [(grid[-1][0], 2.0), (fin[-1][0], 3.0)]
    gl = os.path.join(TMP, 'glist.txt')
    with open(gl, 'w') as f:
        for p, dur in sel: f.write(f"file '{p}'\nduration {dur:.4f}\n")
        f.write(f"file '{sel[-1][0]}'\n")
    gif = os.path.join(HERE, 'demo.gif'); pal = os.path.join(TMP, 'pal.png'); vf = 'fps=12,scale=800:-1:flags=lanczos'
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', gl, '-vf', vf + ',palettegen=max_colors=64', pal])
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', gl, '-i', pal, '-lavfi', vf + '[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5', gif])
    # stills (last frame of each scene)
    shutil.rmtree(STILLS, ignore_errors=True); os.makedirs(STILLS)
    for name, seg in zip(('01_title', '02_problem', '03_fix', '04_run', '05_final_chunk', '06_host_roles', '07_pass'), segments):
        Image.open(seg[-1][0]).save(os.path.join(STILLS, name + '.png'), optimize=True)
    return total

if __name__ == '__main__':
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    scene_title(); scene_problem(); scene_fix(); scene_run(); scene_final(); scene_limits(); scene_end()
    total = assemble(); print('frames', sum(len(s) for s in segments), 'duration %.1f s' % total)
    for fn in ('demo.mp4', 'demo.gif'): print(fn, os.path.getsize(os.path.join(HERE, fn)) // 1024, 'KB')
