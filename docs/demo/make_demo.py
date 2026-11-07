#!/usr/bin/env python3
"""Build docs/demo/demo.mp4 and demo.gif for qweninqwen.

HARD RULE: every number/label drawn here is READ from a data file in this repo (see PROVENANCE.md).
Nothing is simulated. The only synthetic thing is animation order (see the caption in scene 4).

usage: python3 docs/demo/make_demo.py      (needs Pillow, numpy, ffmpeg on PATH)
"""
import json, os, re, shutil, statistics, subprocess, sys, collections
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FR = os.path.join(REPO, 'experiments', 'circuit', 'fullrun')
TMP = os.path.join(HERE, '_frames')
W, H = 1280, 720
BG, PANEL, FG, DIM = (13, 17, 23), (22, 27, 34), (230, 237, 243), (139, 148, 158)
GREEN, RED, BLUE, AMBER, PURPLE, TEAL = (63, 185, 80), (255, 107, 107), (88, 166, 255), (255, 193, 7), (188, 140, 255), (57, 197, 187)

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
SRC_COL = {'RTX 4090': GREEN, 'A100 PCIE': BLUE, 'L40S': AMBER, 'Modal': PURPLE, 'unknown': DIM}
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
def new():
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im); return im, d
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
def rrect(d, box, r, fill, outline=None, w=1): d.rounded_rectangle(box, r, fill=fill, outline=outline, width=w)
def title(im, d, s, emo=None, sub=None):
    x = 48
    if emo: im.paste(emoji(emo, 44), (x, 34), emoji(emo, 44)); x += emoji(emo, 44).width + 14
    text(d, (x, 30), s, font(44, 'Bold'))
    if sub: text(d, (50, 92), sub, font(22), DIM)
FOOT = 'Visualization of a real run  |  one token at position 0  |  parallel jobs seeded from the reference, verified at 95 exact seams  |  sources: PROVENANCE.md'
def footer(d):
    d.line((48, 676, W - 48, 676), fill=(48, 54, 61), width=1); text(d, (48, 686), FOOT, font(15), DIM)

frames = []   # (path, duration seconds)
_n = [0]
def save(im, dur):
    p = os.path.join(TMP, f'f{_n[0]:05d}.png'); _n[0] += 1; im.save(p); frames.append((p, dur)); return p
def clamp(x): return max(0.0, min(1.0, x))

# ---------- scene 1: title ----------
def scene_title():
    im, d = new()
    f84 = font(84, 'Bold'); tw84 = tw(d, 'Qwen inside Qwen', f84); e = emoji('🧠', 100)
    left = int((W - (tw84 + e.width + 24)) // 2)
    im.paste(e, (left, 168), e); text(d, (left + e.width + 24, 150), 'Qwen inside Qwen', f84, FG)
    text(d, (W // 2, 290), 'one Qwen3-0.6B token, computed by a circuit', font(34), BLUE, 'ma')
    text(d, (W // 2, 336), "living in the weights of Qwen3.5-4B", font(34), BLUE, 'ma')
    rrect(d, (190, 440, W - 190, 590), 18, PANEL, (48, 54, 61), 2)
    text(d, (W // 2, 462), 'This is a visualization of a real run.', font(28, 'Semibold'), FG, 'ma')
    para(d, 230, 510, 'Every number is read from the repo logs. One prompt token, position 0. 96 chunk-jobs, 95 exact state seams, 0 host injections.', font(22), DIM, W - 460, 30)
    footer(d); save(im, 6.0)

# ---------- scene 2: the problem ----------
def scene_problem():
    cos = [dq2[f'W4+act ONLY layer {k}']['cos_final'] for k in range(28)]
    amx = [dq2[f'W4+act ONLY layer {k}']['argmax'] for k in range(28)]
    prof = {p['l']: p for p in dq1['fp32_profile']}
    x0, y0, x1, y1 = 70, 190, 760, 560
    for step in range(0, 31):
        t = clamp(step / 24)
        im, d = new(); title(im, d, 'Why the first answer was wrong', '🔍', 'Quantise ONE layer at a time (int4 weights, int8 activations) and see where the answer breaks')
        rrect(d, (48, 150, 790, 610), 14, PANEL)
        text(d, (66, 160), 'cosine of final hidden state vs fp32  (higher = closer)', font(18), DIM)
        # axis 0.5 .. 1.0
        for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            yy = y1 - (v - 0.5) / 0.5 * (y1 - y0); d.line((x0, yy, x1, yy), fill=(40, 46, 54)); text(d, (x0 - 8, yy), f'{v:.1f}', font(14), DIM, 'rm')
        bw = (x1 - x0) / 28
        for k in range(28):
            hh = (cos[k] - 0.5) / 0.5 * (y1 - y0) * t
            good = amx[k] == 15846; col = GREEN if good else RED
            bx = x0 + k * bw + 3
            d.rectangle((bx, y1 - hh, bx + bw - 6, y1), fill=col)
            if k % 3 == 0 or not good: text(d, (bx + (bw - 6) / 2, y1 + 6), str(k), font(13), FG if not good else DIM, 'ma')
            if not good and t >= 1:
                lab = Image.new('RGBA', (90, 22), (0, 0, 0, 0)); ImageDraw.Draw(lab).text((0, 0), f'answer {amx[k]}', font=font(14, 'Semibold'), fill=(10, 12, 16, 255))
                lab = lab.rotate(90, expand=True); im.paste(lab, (int(bx + (bw - 6) / 2 - lab.width / 2), int(y1 - hh + 8)), lab)
        text(d, (x0, y1 + 28), 'layer index   (red = answer flips; label shows the wrong token it gives)', font(14), DIM)
        # right panel
        rrect(d, (810, 150, W - 48, 330), 14, PANEL, RED, 2)
        text(d, (830, 162), 'Our first firmware (int4)', font(20), DIM); text(d, (830, 192), f"{qf['ctrl_w4']['argmax_fw']}   '.'", font(60, 'Bold'), RED)
        text(d, (830, 270), 'the circuit faithfully reproduced this', font(18), DIM)
        text(d, (830, 296), "(same answer as the emulator)", font(18), DIM)
        rrect(d, (810, 345, W - 48, 525), 14, PANEL, GREEN, 2)
        text(d, (830, 357), 'Qwen3-0.6B itself (HF fp32)', font(20), DIM); text(d, (830, 392), f"{dq1['hf_argmax']}  ' Question'", font(40, 'Bold'), GREEN)
        text(d, (830, 460), 'prompt: single token [785] ("The")', font(18), DIM)
        text(d, (830, 486), f"rank of 15846 in the firmware: {qf['ctrl_w4']['rank_15846_in_fw']:,}", font(18), DIM)
        l2 = prof[2]
        rrect(d, (810, 540, W - 48, 610), 14, PANEL)
        para(d, 826, 548, f"Position 0 has a massive activation: layer 2 writes a ~{l2['dmax']:.0f} channel (everything else is O(1)).", font(16), FG, W - 48 - 826 - 12, 21)
        footer(d); save(im, 0.1 if step < 30 else 9.0)

# ---------- scene 3: the fix ----------
def scene_fix():
    items = [('ctrl_w4', 'current: int4 weights everywhere'), ('b8_apl1', 'layers 2,26,27: 8-bit weights only'),
             ('w4_apl2only', 'layers 2,26,27: 16-bit activations only'), ('b8_apl2', 'layers 2,26,27: 8-bit weights + 16-bit activations'),
             ('b12_apl2', 'layers 2,26,27: 12-bit weights + 16-bit activations')]
    for step in range(0, 25):
        t = clamp(step / 12)
        im, d = new(); title(im, d, 'The fix: 3 layers get more precision', '🔧', 'Real integer firmware run in the emulator, prompt [785], everything else stays int4')
        y = 160
        for i, (k, lab) in enumerate(items):
            r = qf[k]; ok = r['argmax_fw'] == 15846; col = GREEN if ok else RED
            rrect(d, (48, y, W - 48, y + 76), 12, PANEL, col if ok else None, 2)
            text(d, (68, y + 8), lab, font(22, 'Semibold'))
            text(d, (68, y + 42), f"cos(logits, HF) = {r['cos_logits_fw_vs_hf']:.3f}    rank of 15846 = {r['rank_15846_in_fw']:,}    top-1 gap = {r['gap_fw_top1_top2']:.2f}", font(16), DIM)
            # cos bar
            bx0, bx1 = 760, 1000
            d.rectangle((bx0, y + 30, bx1, y + 46), outline=(60, 66, 74)); d.rectangle((bx0, y + 30, bx0 + (bx1 - bx0) * r['cos_logits_fw_vs_hf'] * t, y + 46), fill=col)
            text(d, (1030, y + 16), f"{r['argmax_fw']}" if t > 0.6 else '', font(40, 'Bold'), col)
            if ok and t >= 1: text(d, (1180, y + 22), 'ok', font(26, 'Bold'), GREEN)
            y += 90
        i0, i1 = qf['ctrl_w4']['instr'][0], qf['b8_apl2']['instr'][0]; w0, w1 = qf['ctrl_w4']['blob']['wrom_words'], qf['b8_apl2']['blob']['wrom_words']
        text(d, (48, 618), f"Cost of the chosen fix (8-bit + 16-bit): {100 * (i1 / i0 - 1):+.1f}% instructions per token ({i0:,} -> {i1:,}), {100 * (w1 / w0 - 1):+.1f}% weight words ({w0:,} -> {w1:,}).", font(18), FG)
        text(d, (48, 646), 'Both knobs are needed: weights alone give 9, activations alone give 408.', font(16), DIM)
        footer(d); save(im, 0.1 if step < 24 else 8.0)

# ---------- scene 4: the run ----------
def grid_frame(nfill, final=False):
    im, d = new(); title(im, d, 'The run: 96 chunk-jobs on rented GPUs', '🚀', 'Each cell = 1 chunk of 2,097,153 circuit steps (chunk 95: 57,707 steps), colored by the machine that ran it')
    cols, rws = 16, 6; cw, ch = 46, 46; gx, gy = 48, 140
    for c in range(96):
        cx, cy = gx + (c % cols) * (cw + 6), gy + (c // cols) * (ch + 6)
        if c < nfill:
            col = SRC_COL[SRC[c]]; d.rectangle((cx, cy, cx + cw, cy + ch), fill=col); text(d, (cx + cw / 2, cy + ch / 2), str(c), font(14, 'Semibold'), (10, 12, 16), 'mm')
        else:
            d.rectangle((cx, cy, cx + cw, cy + ch), outline=(48, 54, 61))
    # counters
    k = min(nfill, 96); vis = [rows[c] for c in range(k)]
    seam_ok = sum(1 for r in vis if r.get('seam', {}).get('exact'))
    cx0, cy0 = 48, 480
    stats = [('chunks verified', f'{k} / 96', FG), ('seams exact', f'{seam_ok} / 95', GREEN), ('free-token mismatches', str(sum(r['free_mismatches'] for r in vis)), GREEN),
             ('forced tokens', str(sum(r['forced_tokens'] for r in vis)), GREEN), ('host injections', str(sum(r['injections'] for r in vis)), GREEN)]
    for i, (lab, val, col) in enumerate(stats):
        x = cx0 + i * 232; rrect(d, (x, cy0, x + 220, cy0 + 92), 12, PANEL)
        text(d, (x + 14, cy0 + 8), val, font(36, 'Bold'), col); text(d, (x + 14, cy0 + 58), lab, font(16), DIM)
    # legend + timings
    lx, ly = 48, 592
    for i, s in enumerate(['RTX 4090', 'A100 PCIE', 'L40S', 'Modal']):
        if s not in by_src: continue
        xx = lx + i * 305; d.rectangle((xx, ly + 4, xx + 16, ly + 20), fill=SRC_COL[s])
        med = statistics.median(by_src[s]); n = len(by_src[s])
        text(d, (xx + 24, ly), f"{s}: {n} chunk{'s' if n != 1 else ''}, median {med / 60:.1f} min", font(16), FG)
    if nfill >= 96 or final:
        text(d, (48, 622), f'{total_steps:,} circuit steps in total, {total_sec / 3600:.1f} GPU-hours of job time (sum of per-chunk seconds).', font(17), DIM)
    text(d, (W - 48, 648), 'animation order = chunk number (real completion times were not logged); medians exclude the short chunk 95', font(13), DIM, 'ra')
    footer(d); return im

def scene_run():
    for k in range(0, 97, 2):
        save(grid_frame(k), 0.14)
    save(grid_frame(96, True), 6.0)

# ---------- scene 5: final chunk ----------
def scene_final():
    names = ro['tail_names_circuit']
    for step in range(0, len(names) + 1, 1):
        im, d = new(); title(im, d, 'The last chunk: the circuit halts on its own', '🏁', f"chunk 95 reads out the answer; {last['steps_run']:,} steps, halt_step = {ro['circuit_halt_step']:,}")
        text(d, (48, 150), "the circuit's own last tokens (steps %s..%s of the chunk)" % (f"{ro['from_step']:,}", f"{ro['to_step']:,}"), font(18), DIM)
        x, y = 48, 190
        for i, nm in enumerate(names[:step]):
            wpx = tw(d, nm, mono(18)) + 22
            if x + wpx > W - 48: x = 48; y += 44
            halt = nm == 'HALT'; col = GREEN if halt else (40, 46, 54)
            rrect(d, (x, y, x + wpx, y + 34), 8, col, GREEN if halt else (60, 66, 74), 1); text(d, (x + wpx / 2, y + 17), nm, mono(18), (10, 12, 16) if halt else FG, 'mm'); x += wpx + 8
        if step >= len(names):
            b = ro['circuit_bytes']; word = ro['circuit_word']
            rrect(d, (48, 360, W - 48, 560), 16, PANEL, GREEN, 2)
            text(d, (72, 376), 'answer bytes written by the circuit', font(20), DIM)
            text(d, (72, 408), f"{b}  ->  {b[0]} + {b[1]} x 256  =  {word}", mono(34), FG)
            text(d, (72, 468), f"{word}  =  ' Question'  =  what Qwen3-0.6B itself says", font(40, 'Bold'), GREEN)
            text(d, (72, 524), f"tail equals the reference trace: {ro['tail_equals_reference']}    free mismatches: {last['free_mismatches']}    forced: {last['forced_tokens']}    injections: {last['injections']}", font(18), DIM)
            text(d, (48, 590), 'collector verdict:', font(18), DIM)
            para(d, 48, 616, seams['verdict'].replace('; 0s to collect', ''), font(16), FG, W - 96, 22)
        footer(d); save(im, 0.18 if step < len(names) else 7.5)

# ---------- scene 6: limits + host roles ----------
def scene_limits():
    im, d = new(); title(im, d, 'What is (and is not) on the host', '🧾', 'Honest scope of this result')
    colw = (W - 96 - 24) // 2
    def card(x, y, w, h, head, headcol, lines):
        rrect(d, (x, y, x + w, y + h), 14, PANEL, headcol, 2); text(d, (x + 18, y + 12), head, font(24, 'Bold'), headcol); yy = y + 52
        for ln in lines: yy = para(d, x + 18, yy, ln, font(19), FG, w - 36, 26) + 8
    n_inj = sum_inj
    card(48, 140, colw, 500, 'Removed from the host', GREEN, [
        f'- Weight, RAM and blob injections: {n_inj} in all 96 chunks (collector, inj column).',
        f'- Forced tokens: {sum_forced}. Free-token mismatches: {sum_mism}.',
        '- Weights and blob constants come from rows of a table (%s rows) that the circuit fetches itself. The same table sits inside a merged Qwen3.5-4B checkpoint (hidden 7680): HF-module check on chunk 0, %s steps, %s mismatches, %s ROW tokens; table+1 control diverges at step %s. The 96 jobs loaded a bit-identical decoded copy.' % (f"{hf_pos['table_rows']:,}", f"{hf_pos['steps_checked']:,}", hf_pos['n_mismatch'], f"{hf_pos['n_row_tokens_checked']:,}", hf_neg['first_mismatch']['step']),
        '- Chunk 0 boots from the program alone; the real in_ram / in_drom address decode replaces the bring-up patch.',
        '- Allowed by the plan: boot, the GO token, the embed-and-append loop, ecall I/O. The reference comparison is the test, not an input.'])
    card(48 + colw + 24, 140, colw, 500, 'Honest limits', AMBER, [
        '- One prompt token at position 0 (context length 1). Not general generation.',
        f'- Chunks 1-95 were seeded from the reference start state and run in parallel; each end state matched the next seed exactly at {n_seam} of 95 seams. By induction that equals one unbroken run, but it is not a literal one.',
        '- The executor is a fused Triton kernel, not the stock HF forward. A full HF check of 199M steps is infeasible; HF-module checks cover chunk 0 only (130k steps).',
        '- The table makes the checkpoint bigger (hidden 7680, 4-bit entries stored as bf16).',
        f"- The model is Qwen3-0.6B quantised to int4 with 3 layers at higher precision; the answer is {ro['circuit_word']}, matching HF fp32 for this prompt only."])
    footer(d); save(im, 20.0)

# ---------- scene 7: end ----------
def scene_end():
    im, d = new(); e = emoji('✅', 110); im.paste(e, (W // 2 - e.width // 2, 120), e)
    text(d, (W // 2, 260), 'PASS', font(92, 'Bold'), GREEN, 'ma')
    text(d, (W // 2, 380), f"96/96 chunks clean  |  {n_seam}/95 seams exact  |  final word {ro['circuit_word']}", font(32, 'Semibold'), FG, 'ma')
    text(d, (W // 2, 440), 'logs: experiments/circuit/fullrun/logs/cfram_q_vast/', mono(20), DIM, 'ma')
    text(d, (W // 2, 480), 'every number in this video: docs/demo/PROVENANCE.md', mono(20), DIM, 'ma')
    footer(d); save(im, 6.0)

def encode():
    lst = os.path.join(TMP, 'list.txt')
    with open(lst, 'w') as f:
        for p, dur in frames: f.write(f"file '{p}'\nduration {dur:.3f}\n")
        f.write(f"file '{frames[-1][0]}'\n")
    out = os.path.join(HERE, 'demo.mp4')
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-vf', 'fps=24,format=yuv420p', '-c:v', 'libx264', '-crf', '24', '-preset', 'slow', '-movflags', '+faststart', out])
    # teaser gif: title (2 s) + run grid + final tail
    sel = [(frames[0][0], 2.0)]
    grid = [fr for fr in frames if os.path.basename(fr[0]) in GRID_NAMES]
    sel += [(p, 0.12) for p, _ in grid[:-1]] + [(grid[-1][0], 2.0), (FINAL_TAIL_FRAMES[-1], 3.0)]
    gl = os.path.join(TMP, 'glist.txt')
    with open(gl, 'w') as f:
        for p, dur in sel: f.write(f"file '{p}'\nduration {dur:.3f}\n")
        f.write(f"file '{sel[-1][0]}'\n")
    gif = os.path.join(HERE, 'demo.gif'); pal = os.path.join(TMP, 'pal.png')
    vf = 'fps=10,scale=720:-1:flags=lanczos'
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', gl, '-vf', vf + ',palettegen=max_colors=96', pal])
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', gl, '-i', pal, '-lavfi', vf + '[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4', gif])

if __name__ == '__main__':
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    scene_title(); scene_problem(); scene_fix()
    g0 = _n[0]; scene_run(); GRID_NAMES = {f'f{i:05d}.png' for i in range(g0, _n[0])}
    f0 = _n[0]; scene_final(); FINAL_TAIL_FRAMES = [os.path.join(TMP, f'f{_n[0] - 1:05d}.png')]
    scene_limits(); scene_end()
    total = sum(d for _, d in frames); print('frames', len(frames), 'duration %.1f s' % total)
    encode()
    for fn in ('demo.mp4', 'demo.gif'): print(fn, os.path.getsize(os.path.join(HERE, fn)) // 1024, 'KB')
    if '--keep' not in sys.argv: pass
