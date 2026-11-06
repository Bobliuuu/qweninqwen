"""Generates every launch visual (PNG, GIF, MP4) for the README and the tweet thread.

Run from the repo root:  python docs/launch/make_media.py [png] [gif] [video]   (default: all)
Needs: Pillow, numpy, matplotlib (only for its bundled DejaVu fonts), ffmpeg on PATH for mp4.

Honesty rules baked in:
  * every measured number is read from experiments/vslice/logs/*.json at render time (see docs/launch/PROVENANCE.md);
  * panels that show a mechanism (state memory, pipeline, tournament, hero) are drawn as ILLUSTRATIONS and say so on screen;
  * pending results live in STATUS below. Flip a value there, re-run, and every figure/video follows (see UPDATING.md).
"""
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ASSETS = os.path.join(ROOT, 'docs', 'assets')
LOGS = os.path.join(ROOT, 'experiments', 'vslice', 'logs')
os.makedirs(ASSETS, exist_ok=True)

# ----------------------------------------------------------------------------- status (flip these when results land)
# values: 'done' | 'in progress'.  Keep in sync with the README status table (docs/launch/UPDATING.md).
STATUS = {
    'free_running': 'in progress',   # free-running generation vs HF greedy
    'hostguard':    'in progress',   # HostGuard audit of the real driver
    'speedup':      'in progress',   # tuned executor / speedup work
}

# ----------------------------------------------------------------------------- measured data (from logs)
gen = json.load(open(os.path.join(LOGS, 'res_vs2_real151936_L28.json')))     # 6 teacher-forced positions
one = json.load(open(os.path.join(LOGS, 'res_vs2_real151936_L28_tok1.json')))  # headline one-token run
STEPS = gen['steps']
TOK = gen['tokens']                       # [785, 6722, 315, 9625, 374, 12095, 13]
HEAD = one['steps'][0]                    # 268.6 s, maxrel 5.8e-5, argmax 15846
LAYERS, DDIM, FRAMES = gen['layers'], gen['D'], STEPS[0]['frames']
assert all(s['outer_argmax'] == s['hf_argmax'] for s in STEPS)
MAXREL_MIN = min(s['maxrel'] for s in STEPS); MAXREL_MAX = max(s['maxrel'] for s in STEPS)
SEC_MIN = min(s['sec'] for s in STEPS); SEC_MAX = max(s['sec'] for s in STEPS)
CIRCUIT_S = 3.4e7   # README cost model: circuit route extrapolation, matvecs only (estimate, labelled as such)

# ----------------------------------------------------------------------------- drawing kit
BG = (13, 17, 23); PANEL = (21, 27, 35); FG = (230, 237, 243); MUT = (145, 152, 161); LINE = (61, 68, 77)
BLUE = (68, 147, 248); YEL = (212, 167, 44); GRN = (63, 185, 80); PUR = (188, 140, 255); RED = (255, 123, 114)
mix = lambda a, b, t: tuple(int(a[i] + (b[i] - a[i]) * max(0, min(1, t))) for i in range(3))
clamp = lambda x, a=0.0, b=1.0: max(a, min(b, x))
def seg(p, a, b): return clamp((p - a) / (b - a))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
FD = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
_fc = {}
def font(sz, bold=False, mono=False):
    key = (round(sz), bold, mono)
    if key not in _fc:
        n = 'DejaVuSansMono' if mono else 'DejaVuSans'
        _fc[key] = ImageFont.truetype(os.path.join(FD, n + ('-Bold' if bold else '') + '.ttf'), max(6, round(sz)))
    return _fc[key]

class Canvas:
    def __init__(s, W=1280, H=720, k=1.5):
        s.W, s.H, s.k = W, H, k
        s.im = Image.new('RGB', (int(W * k), int(H * k)), BG); s.d = ImageDraw.Draw(s.im)
    def rect(s, x, y, w, h, fill=None, line=None, r=8, lw=2):
        k = s.k; s.d.rounded_rectangle([x * k, y * k, (x + w) * k, (y + h) * k], radius=r * k, fill=fill, outline=line, width=max(1, int(lw * k)) if line else 0)
    def text(s, x, y, t, sz=20, fill=FG, bold=False, mono=False, anchor='la'):
        s.d.text((x * s.k, y * s.k), t, font=font(sz * s.k, bold, mono), fill=fill, anchor=anchor)
    def line(s, pts, fill=MUT, w=2):
        s.d.line([(x * s.k, y * s.k) for x, y in pts], fill=fill, width=max(1, int(w * s.k)), joint='curve')
    def arrow(s, pts, fill=MUT, w=2.5, head=11):
        s.line(pts, fill, w)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        tri = [(x1, y1), (x1 - head * math.cos(a - .4), y1 - head * math.sin(a - .4)), (x1 - head * math.cos(a + .4), y1 - head * math.sin(a + .4))]
        s.d.polygon([(x * s.k, y * s.k) for x, y in tri], fill=fill)
    def circle(s, cx, cy, r, fill=None, line=None, lw=2):
        k = s.k; s.d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k], fill=fill, outline=line, width=max(1, int(lw * k)) if line else 0)
    def out(s, size=None):
        return s.im if size is None else s.im.resize(size, Image.LANCZOS)

def foot(c, t='ILLUSTRATION of the mechanism (not a recording). Every number comes from experiments/vslice/logs/.'):
    c.rect(0, 692, 1280, 28, fill=PANEL, r=0); c.text(640, 706, t, 13, MUT, anchor='mm')
def head(c, t, sub=None):
    c.text(48, 30, t, 34, FG, True)
    if sub: c.text(48, 76, sub, 18, MUT)
def fade(col, a): return mix(BG, col, a)
def sci(x): m, e = f'{x:.1e}'.split('e'); return f'{m}e{int(e)}'

# ----------------------------------------------------------------------------- scenes: f(c, p) with p in [0, 1]
def sc_title(c, p):
    a = ease(seg(p, 0, .25)); b = ease(seg(p, .2, .5)); d = ease(seg(p, .45, .8))
    c.text(640, 250, 'Qwen inside Qwen', 84, fade(FG, a), True, anchor='mm')
    c.text(640, 330, 'a real Qwen3-0.6B token, executed by the weights', 28, fade(MUT, b), anchor='mm')
    c.text(640, 366, 'and recurrent state of a stock Qwen3.5 model', 28, fade(MUT, b), anchor='mm')
    c.text(640, 470, f'{LAYERS} outer layers  |  full {151936:,}-token head  |  argmax computed in-model', 22, fade(BLUE, d), anchor='mm')
    foot(c, 'explainer  |  mechanism panels are illustrations; numbers come from the repo logs')

def sc_idea(c, p):
    head(c, 'What "Qwen inside Qwen" means', 'the program is a language model; the machine that runs it is another language model')
    a = ease(seg(p, 0, .2)); b = ease(seg(p, .2, .4)); m = seg(p, .4, .65); h = ease(seg(p, .7, .9))
    c.rect(60, 140, 420, 330, fade(PANEL, a * 1), fade(YEL, a), lw=3)
    c.text(270, 168, 'INNER  (the program)', 24, fade(YEL, a), True, anchor='mm')
    for i, t in enumerate(['real Qwen3-0.6B weights', '28 layers, d = 1024', 'vocab 151,936', 'RMSNorm, RoPE, GQA, SwiGLU']):
        c.text(270, 225 + i * 44, t, 21, fade(FG, a), anchor='mm')
    c.rect(800, 140, 420, 330, fade(PANEL, b), fade(BLUE, b), lw=3)
    c.text(1010, 168, 'OUTER  (the machine)', 24, fade(BLUE, b), True, anchor='mm')
    for i, t in enumerate(['stock Qwen3_5ForCausalLM', f'fp32, {LAYERS} layers, D = {DDIM:,}', 'DeltaNet + MLP + attention', 'code unpatched, only weights written']):
        c.text(1010, 225 + i * 44, t, 21, fade(FG, b), anchor='mm')
    if m > 0:
        x1 = 490 + (790 - 490) * ease(m)
        c.arrow([(490, 305), (x1, 305)], fade(GRN, 1), 4)
        c.text(640, 270, 'compiled into', 20, fade(GRN, ease(m)), anchor='mm'); c.text(640, 335, 'weights + state', 20, fade(GRN, ease(m)), anchor='mm')
    c.rect(60, 520, 1160, 120, fade(PANEL, h), fade(LINE, h))
    c.text(640, 550, 'Host (allowed): embedding row, flags, one-hot addresses, position ids, weight payloads,', 20, fade(FG, h), anchor='mm')
    c.text(640, 582, 'KV scratch truncation, lm_head evaluation through 36 columns, outer argmax + append', 20, fade(FG, h), anchor='mm')
    c.text(640, 614, 'Host never computes an inner activation.', 20, fade(RED, h), True, anchor='mm')
    foot(c)

_rng = np.random.RandomState(7); _W = np.round(_rng.uniform(-1, 1, (4, 5)), 2); _q = np.round(_rng.uniform(-1, 1, 4), 2)
def sc_memory(c, p):
    head(c, 'DeltaNet state = memory / registers', 'a write at one-hot key k stores a row; a read with q returns S^T q, i.e. a matrix-vector product')
    cw, ch, x0, y0 = 92, 62, 90, 200
    wr = seg(p, .05, .55); rd = seg(p, .6, .9); n = wr * 4
    def grid(x0, y0, M, nrows, title, col, vis=None):
        c.text(x0 + 2.5 * cw, y0 - 44, title, 20, col, True, anchor='mm')
        for i in range(4):
            for j in range(5):
                on = (vis is None) or i < vis
                v = M[i][j] if on else 0
                fillc = mix(PANEL, BLUE if v > 0 else RED, abs(v) * .55) if on else PANEL
                c.rect(x0 + j * cw, y0 + i * ch, cw - 6, ch - 6, fillc, LINE, r=5, lw=1)
                if on: c.text(x0 + j * cw + (cw - 6) / 2, y0 + i * ch + (ch - 6) / 2, f'{v:+.2f}', 17, FG, mono=True, anchor='mm')
    grid(x0, y0, _W, 4, 'inner weight rows W (toy 4x5)', YEL)
    grid(740, y0, _W, 4, 'outer DeltaNet state S', BLUE, vis=int(n + 1e-9))
    for i in range(int(n + 1e-9) + (1 if n % 1 > 0 and n < 4 else 0)):
        if i < 4: c.arrow([(x0 + 5 * cw + 4, y0 + i * ch + 28), (730, y0 + i * ch + 28)], fade(GRN, 1 if i < int(n + 1e-9) else n % 1), 3)
    c.text(595, y0 + 270, 'WRITE  k_i = one-hot(i),  v_i = row i of W   ->   S += k_i v_i^T', 19, GRN, anchor='mm')
    if rd > 0:
        c.text(740 - 60, 500, 'q =', 22, FG, anchor='mm')
        for i in range(4): c.text(740 - 20 + i * 70, 500, f'{_q[i]:+.2f}', 18, PUR, mono=True, anchor='mm')
        res = _q @ _W
        c.text(740 - 60, 560, 'S^T q =', 22, FG, anchor='mm')
        for j in range(5):
            if rd > j / 5.0: c.text(740 - 20 + j * 70 + 0, 560, f'{res[j]:+.2f}', 18, FG, True, mono=True, anchor='mm')
        c.text(595, 620, 'READ  q (the activation)  ->  S^T q = sum_i q_i W_i = the matvec, computed by the layer', 19, PUR, anchor='mm')
    foot(c, 'ILLUSTRATION: toy 4x5 random matrix, not real weights. The real run pages 595,984,384 weights through state.')

BLOCKS = ['RMSNorm', 'W_qkv bank', 'register', 'RoPE + attn', 'W_o + resid', 'RMSNorm', 'gate / up', 'SwiGLU', 'down + resid']
BKIND = [YEL, BLUE, PUR, GRN, BLUE, YEL, BLUE, YEL, BLUE]
def sc_pipeline(c, p):
    head(c, 'How one token flows: an unrolled pipeline in depth', f'each inner op gets its own outer layers; {LAYERS} layers and {FRAMES:,} frames per token')
    bw, gap, x0, y0 = 118, 10, 48, 150
    c.text(48, 128, 'one inner layer (repeated x28)', 18, MUT)
    cyc = (p * 3.0) % 1.0 if p < 1 else 0.999
    pos = cyc * len(BLOCKS) if p < 1 else len(BLOCKS) - .01
    for i, (t, col) in enumerate(zip(BLOCKS, BKIND)):
        on = clamp(1 - abs(pos - i - .5) / 1.2)
        x = x0 + i * (bw + gap)
        c.rect(x, y0, bw, 92, mix(PANEL, col, .12 + .5 * on), mix(LINE, col, .4 + .6 * on), lw=2)
        c.text(x + bw / 2, y0 + 46, t, 16, FG, True, anchor='mm')
        if i < len(BLOCKS) - 1: c.arrow([(x + bw, y0 + 46), (x + bw + gap, y0 + 46)], MUT, 2, 6)
    # overall chain
    chain = [('embed (host row)', MUT), ('inner layers x28', YEL), ('final norm', YEL), ('LM head: 594 blocks', BLUE), ('block + cross-block argmax', GRN), ('outer lm_head', MUT), ('token (host append)', MUT)]
    cw, cy = 160, 360
    c.text(48, 330, 'whole token', 18, MUT)
    for i, (t, col) in enumerate(chain):
        x = 48 + i * (cw + 6 * 1.2) ; on = ease(seg(p, i / 7., (i + 1) / 7.))
        c.rect(x, cy, cw, 90, mix(PANEL, col, .1 + .5 * on), mix(LINE, col, .3 + .7 * on))
        c.text(x + cw / 2, cy + 45, t, 15, FG, True, anchor='mm')
        if i < len(chain) - 1: c.arrow([(x + cw, cy + 45), (x + cw + 7, cy + 45)], MUT, 2, 6)
    n = int(LAYERS * p) if p < 1 else LAYERS
    c.rect(48, 500, 1184, 14, PANEL, LINE, r=7, lw=1); c.rect(48, 500, max(14, 1184 * p), 14, BLUE, None, r=7)
    c.text(48, 535, f'outer layers passed: {n:,} / {LAYERS:,}', 24, FG, True, mono=True)
    c.text(48, 580, f'frames per token: {FRAMES:,}  (291,072 weight writes + 1,996 control)   |   outer width D = {DDIM:,}', 19, MUT)
    c.text(48, 615, 'state and KV only carry data deeper into the stack: no loops, no hidden host maths', 19, MUT)
    foot(c)

_tr = np.random.RandomState(3); _bars = _tr.uniform(.18, .75, 594); WIN_BLOCK = HEAD['outer_argmax'] // 256; WIN_LOCAL = HEAD['outer_argmax'] % 256
_bars[WIN_BLOCK] = 1.0
RAILS0 = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]  # logs/real_L28_gen.log step 0 'rails' line (first 8 of 36 rails shown there)
def sc_tournament(c, p):
    head(c, 'The argmax tournament, inside the model', 'no host argmax over the vocabulary: blocks reduce themselves, then hard-max attention picks the winner')
    x0, x1, yb, hmax = 60, 1220, 440, 200
    sw = seg(p, .02, .3); bl = ease(seg(p, .3, .5)); ha = ease(seg(p, .5, .72)); cd = ease(seg(p, .72, .95))
    nb = 594
    for i in range(nb):
        if i / nb > sw: break
        x = x0 + (x1 - x0) * i / nb; h = hmax * _bars[i]
        col = MUT
        if bl > 0: col = mix(MUT, YEL, bl * .8)
        if ha > 0 and i == WIN_BLOCK: col = mix(YEL, GRN, ha)
        elif ha > 0: col = mix(col, LINE, ha * .8)
        c.rect(x, yb - h, 1.6, h, col, None, r=0, lw=0)
    c.line([(x0, yb), (x1, yb)], LINE, 2)
    c.text(x0, yb + 14, 'vocab 151,936 logits = 594 blocks of 256', 17, MUT)
    if bl > 0: c.text(640, 168, 'per block: in-model max + 8-bit local index (block-best logit)', 19, fade(YEL, bl), anchor='mm')
    if ha > 0:
        xw = x0 + (x1 - x0) * WIN_BLOCK / nb
        c.arrow([(xw + 1, 190), (xw + 1, yb - hmax - 8)], fade(GRN, ha), 3)
        c.text(640, 205, 'hard-max attention over the 594 block winners selects the global best', 19, fade(GRN, ha), anchor='mm')
    if cd > 0:
        a = cd
        c.rect(60, 500, 1160, 170, fade(PANEL, a), fade(LINE, a))
        c.text(80, 512, f'winner code = block {WIN_BLOCK} (10 bits) + local {WIN_LOCAL} (8 bits) = 18 bits = 36 dual rails  (block*256 + local = {HEAD["outer_argmax"]})', 17, fade(FG, a), mono=True)
        c.text(80, 548, 'first 8 rails, real run (logs/real_L28_gen.log step 0):', 17, fade(MUT, a))
        for i, v in enumerate(RAILS0): c.rect(560 + i * 70, 538, 60, 36, mix(PANEL, GRN if v > .5 else LINE, a * .7), None, r=5); c.text(590 + i * 70, 556, f'{v:.0f}', 18, fade(FG, a), True, mono=True, anchor='mm')
        c.text(80, 596, 'top code score 18.0007 (ideal 18)  ->  outer lm_head decodes 36 rails', 19, fade(FG, a), anchor='lm')
        c.text(80, 636, f'outer argmax = {HEAD["outer_argmax"]}   Hugging Face argmax = {HEAD["hf_argmax"]}', 26, fade(GRN, a), True, anchor='lm')
    foot(c, 'ILLUSTRATION: bar heights are random (except the winner). Block/local split derived from token id 15846 and block size 256.')

# tiers -------------------------------------------------------------------------------------------------
def tier_rows():
    fr, hg, sp = STATUS['free_running'], STATUS['hostguard'], STATUS['speedup']
    real_done = fr == 'done' and hg == 'done'
    return [
        ('T0', 'gadget in isolation', 'MAC, norm, product', 'done', 'verified'),
        ('T1', 'full projection', 'all 197 matrices, real weights', 'done', 'verified'),
        ('T2', 'op chain at real dims', 'ops ok; chained in toy only', 'partial', 'partial'),
        ('T3-toy', 'toy model generates', 'random weights, d=128', 'done', 'achieved'),
        ('T3 real', 'real Qwen3-0.6B token', '1 token + 6 teacher-forced', 'done' if real_done else 'partial', 'achieved' if real_done else 'partial'),
        ('T4', 'measured cost model', 'measured: 1 token 268.6 s', 'done' if sp == 'done' else 'partial', 'measured' if sp == 'done' else 'partial'),
    ]
def sc_tiers(c, p, title=True):
    if title: head(c, 'The tier ladder', 'how far up the staircase the evidence goes today (statuses are read from STATUS in make_media.py)')
    rows = tier_rows(); n = len(rows); sw, sh, x0, yb = 186, 62, 50, 640
    climb = seg(p, .05, .8) * (4 + 1e-9)   # marker climbs to T3 real (index 4)
    for i, (t, ttl, sub, st, lab) in enumerate(rows):
        x = x0 + i * (sw + 6); h = 110 + i * sh * .9; y = yb - h
        col = {'done': GRN, 'partial': YEL}[st]
        vis = ease(clamp(climb - i + 1.2)) if i <= 4 else ease(seg(p, .8, 1))
        c.rect(x, y, sw, h, mix(BG, mix(PANEL, col, .22), vis), mix(BG, col, vis), lw=2)
        c.text(x + sw / 2, y + 22, t, 26, fade(col, vis), True, anchor='mm')
        c.text(x + sw / 2, y + 52, ttl, 15, fade(FG, vis), anchor='mm')
        c.text(x + sw / 2, y + 74, sub, 12, fade(MUT, vis), anchor='mm')
        c.text(x + sw / 2, y + h - 18, lab.upper(), 14, fade(col, vis), True, anchor='mm')
    # marker
    mi = min(climb, 4.0); mx = x0 + mi * (sw + 6) + sw / 2; my = yb - (110 + mi * sh * .9) - 34
    pulse = 1 + .12 * math.sin(p * 40) if climb >= 4 else 1
    c.circle(mx, my, 15 * pulse, PUR, FG, 2); c.text(mx, my - 32, 'we are here', 17, PUR, True, anchor='mm')
    # pending tags
    tags = [('free-running generation vs HF greedy', STATUS['free_running']), ('HostGuard audit of the real driver', STATUS['hostguard']), ('speedup work', STATUS['speedup'])]
    for i, (t, s) in enumerate(tags):
        col = GRN if s == 'done' else MUT
        c.text(50, 150 + i * 26 if title else 20 + i * 26, f'{"[x]" if s == "done" else "[ ]"} {t}: {s}', 17, col, mono=True)
    if title: foot(c, 'Tier definitions and evidence: README "Status by tier". Pending items are not claimed as done.')

# result / caveats ----------------------------------------------------------------------------------------
def sc_result(c, p):
    head(c, 'What is verified today', 'real Qwen3-0.6B, real weights, full 151,936 vocab head, in-model argmax, stock Qwen3_5ForCausalLM')
    cards = [(f'{HEAD["outer_argmax"]} == {HEAD["hf_argmax"]}', 'outer argmax equals\nHugging Face', GRN), (sci(HEAD['maxrel']), 'max relative logit error\n(position 0)', BLUE), (f'{HEAD["sec"]:.1f} s', 'per token, shared A100\nfp32, untuned', YEL)]
    for i, (big, small, col) in enumerate(cards):
        a = ease(seg(p, i * .18, i * .18 + .25)); x = 60 + i * 392
        c.rect(x, 130, 372, 220, fade(PANEL, a), fade(col, a), lw=3)
        c.text(x + 186, 215, big, 54, fade(col, a), True, anchor='mm')
        for j, l in enumerate(small.split('\n')): c.text(x + 186, 290 + j * 26, l, 20, fade(FG, a), anchor='mm')
    a = ease(seg(p, .6, .85))
    c.text(60, 385, f'6-position teacher-forced run: argmax == HF at all 6 positions ({", ".join(str(s["outer_argmax"]) for s in STEPS)})', 19, fade(FG, a), True)
    c.text(60, 415, f'errors {sci(MAXREL_MIN)} to {sci(MAXREL_MAX)}, {SEC_MIN:.0f} to {SEC_MAX:.0f} s per position; {LAYERS} outer layers, D = {DDIM:,}, {FRAMES:,} frames', 18, fade(MUT, a))
    b = ease(seg(p, .75, 1))
    c.rect(60, 460, 1160, 215, fade(PANEL, b), fade(RED, b))
    c.text(80, 476, 'Honest caveats', 22, fade(RED, b), True)
    for i, t in enumerate(['teacher-forced positions of ONE prompt, not free-running generation: ' + STATUS['free_running'],
                           'HostGuard audit of this driver: ' + STATUS['hostguard'],
                           'host does: embedding lookup, flags, one-hot addresses, position ids, weight payloads,',
                           '    KV scratch truncation, lm_head eval through 36 columns',
                           'speed: untuned fp32, ~4.5 minutes per token (speedup work: ' + STATUS['speedup'] + ')']):
        c.text(80, 512 + i * 30, t, 18, fade(FG, b))
    foot(c, 'Source: experiments/vslice/logs/res_vs2_real151936_L28(_tok1).json and real_L28_gen.log (see docs/launch/PROVENANCE.md)')

# cost chart ---------------------------------------------------------------------------------------------
def sc_cost(c, p=1.0):
    head(c, 'Cost: absurd but measured', 'seconds per token on a shared A100, fp32, torch GDN; each bar is one measured position')
    x0, yb, hmax = 90, 470, 270
    for i, s in enumerate(STEPS):
        a = ease(seg(p, i * .1, i * .1 + .4)); h = hmax * s['sec'] / 320 * a; x = x0 + i * 100
        c.rect(x, yb - h, 76, h, BLUE, None, r=4); c.text(x + 38, yb - h - 16, f'{s["sec"]:.0f} s', 16, FG, True, anchor='mm')
        c.text(x + 38, yb + 18, f'pos {i}', 15, MUT, anchor='mm')
        c.rect(x, yb - hmax * s['run_s'] / 320 * a, 76, 4, YEL, None, r=0)
    c.line([(x0 - 10, yb), (x0 + 6 * 100, yb)], LINE, 2)
    c.text(x0, 520, 'blue: total per position (teacher-forced run); yellow tick: run time without layer building', 15, MUT)
    c.text(x0, 548, f'headline single-token run: {HEAD["sec"]:.1f} s ({HEAD["build_s"]:.1f} s build + {HEAD["run_s"]:.0f} s run)', 17, FG)
    # log comparison
    c.text(760, 150, 'vs the circuit route (log scale)', 20, FG, True)
    rows = [('this work, 1 token', HEAD['sec'], GRN, 'measured'), ('circuit route, matvecs only', CIRCUIT_S, RED, 'ESTIMATE (extrapolated)')]
    for i, (t, v, col, tag) in enumerate(rows):
        w = 440 * (math.log10(v) + 0) / 8.0 * ease(seg(p, .3, .9)); y = 200 + i * 110
        c.text(760, y, t, 17, FG); c.rect(760, y + 28, max(8, w), 34, col, None, r=4)
        c.text(760 + max(8, w) + 10, y + 45, f'{v:.3g} s  ({tag})', 16, MUT, anchor='lm')
    c.text(760, 440, 'circuit number: README cost model, 494M MACs at 14.7 MAC/s,', 15, MUT); c.text(760, 462, 'matvecs only, before any non-matvec op. Not a run.', 15, MUT)
    foot(c, 'Source: res_vs2_real151936_L28(_tok1).json; circuit baseline experiments/bench/RESULTS.md (estimate)')

# hero ---------------------------------------------------------------------------------------------------
def sc_hero(c, p):
    c.text(640, 110, 'Qwen inside Qwen', 88, FG, True, anchor='mm')
    c.text(640, 190, 'a real Qwen3-0.6B token, executed by the weights of a stock Qwen3.5 model', 26, MUT, anchor='mm')
    n = 70; sweep = seg(p, .05, .72); y0, h = 250, 150
    cols = [YEL, BLUE, PUR, GRN, BLUE, YEL, BLUE, YEL, BLUE]
    for i in range(n):
        x = 60 + i * (1160 / n); on = clamp(1 - abs(sweep * n - i) / 6) if 0 < sweep < 1 else 0
        lit = 1 if (sweep * n > i and sweep < 1) or sweep >= 1 else 0
        col = cols[i % len(cols)]
        c.rect(x, y0, 1160 / n - 3, h, mix(PANEL, col, .12 + .18 * lit + .65 * on), mix(LINE, col, .3 + .7 * on), r=3, lw=1)
    c.text(60, y0 + h + 22, f'{LAYERS} outer layers, one inner token (28 layers + LM head), shown as 70 slivers', 17, MUT)
    res = ease(seg(p, .75, .88))
    if sweep < 1:
        c.text(640, 480, 'frames flowing through the outer stack ...', 24, MUT, anchor='mm')
    else:
        c.text(640, 480, f'outer argmax {HEAD["outer_argmax"]}  ==  Hugging Face {HEAD["hf_argmax"]}', 40, fade(GRN, res), True, anchor='mm')
    c.text(213, 600, sci(HEAD['maxrel']), 40, BLUE, True, anchor='mm'); c.text(213, 640, 'max rel. logit error', 18, MUT, anchor='mm')
    c.text(640, 600, '6 / 6', 40, GRN, True, anchor='mm'); c.text(640, 640, 'positions match HF (teacher-forced)', 18, MUT, anchor='mm')
    c.text(1067, 600, f'{HEAD["sec"]:.1f} s', 40, YEL, True, anchor='mm'); c.text(1067, 640, 'per token, shared A100, fp32', 18, MUT, anchor='mm')
    foot(c, 'animation = illustration; numbers from experiments/vslice/logs; free-running generation: ' + STATUS['free_running'])

# demo ---------------------------------------------------------------------------------------------------
PROMPT = ['The', ' capital', ' of', ' France', ' is']
def sc_demo_intro(c, p):
    head(c, 'Demo: the real run', 'prompt "The capital of France is"; positions 0-5 run one by one, teacher-forced')
    a = ease(seg(p, 0, .3))
    for i, (w, t) in enumerate(zip(PROMPT, TOK)):
        x = 60 + i * 225; q = ease(seg(p, i * .08, i * .08 + .3))
        c.rect(x, 170, 205, 100, fade(PANEL, q), fade(YEL, q), lw=2); c.text(x + 102, 205, w.strip(), 28, fade(FG, q), True, anchor='mm'); c.text(x + 102, 245, f'id {t}', 18, fade(MUT, q), mono=True, anchor='mm')
    b = ease(seg(p, .5, .9))
    for i, t in enumerate([f'inner: real Qwen3-0.6B (28 layers, d=1024, vocab 151,936)', f'outer: stock Qwen3_5ForCausalLM, fp32, {LAYERS} layers, D = {DDIM:,}', 'hardware: one shared A100; torch reference GDN; layer-major executor',
                           f'per position: {FRAMES:,} frames. Each position takes ~4.5 minutes, so the demo is a time-lapse of the recorded run.']):
        c.text(60, 340 + i * 44, t, 22, fade(FG if i < 3 else YEL, b))
    foot(c, 'Replay of recorded results: experiments/vslice/logs/real_L28_gen.log and res_vs2_real151936_L28.json')

def demo_table(c, upto, cur=None, frac=0.0):
    cols = [('pos', 60), ('input id', 150), ('outer argmax', 330), ('HF argmax', 520), ('match', 690), ('max rel err', 810), ('time', 1040)]
    y0 = 150
    c.rect(48, y0 - 12, 1184, 46, PANEL, LINE)
    for t, x in cols: c.text(x, y0 + 11, t, 19, MUT, True, anchor='lm')
    for i, s in enumerate(STEPS):
        y = y0 + 54 + i * 62
        done = i < upto; run = (cur == i)
        c.rect(48, y - 8, 1184, 54, mix(BG, PANEL, .8), LINE if not run else YEL, lw=2 if run else 1, r=6)
        c.text(60, y + 19, f'{i}', 24, FG, True, mono=True, anchor='lm'); c.text(150, y + 19, f'{TOK[i]}', 24, FG, mono=True, anchor='lm')
        if done:
            c.text(330, y + 19, f'{s["outer_argmax"]}', 24, GRN, True, mono=True, anchor='lm'); c.text(520, y + 19, f'{s["hf_argmax"]}', 24, FG, mono=True, anchor='lm')
            c.text(690, y + 19, 'yes', 24, GRN, True, anchor='lm'); c.text(810, y + 19, sci(s['maxrel']), 24, FG, mono=True, anchor='lm'); c.text(1040, y + 19, f'{s["sec"]:.1f} s', 24, FG, mono=True, anchor='lm')
        elif run:
            c.rect(330, y + 6, 640, 26, PANEL, LINE, r=13, lw=1); c.rect(330, y + 6, max(26, 640 * frac), 26, BLUE, None, r=13)
            c.text(1040, y + 19, f'{s["sec"] * frac:.0f} s', 24, YEL, mono=True, anchor='lm')
def sc_demo_run(c, p, i):
    head(c, f'Demo: position {i} of 5', 'outer model computes the token; Hugging Face argmax is the reference (never used by the driver)')
    run = seg(p, 0, .8)
    demo_table(c, i + (1 if p >= .8 else 0), None if p >= .8 else i, run)
    c.text(60, 640, f'recorded run, time-lapse (x{STEPS[i]["sec"] / 4:.0f}); the real position took {STEPS[i]["sec"]:.1f} s ({STEPS[i]["build_s"]:.1f} s build, {STEPS[i]["run_s"]:.0f} s run)', 17, MUT)
    foot(c, 'Replay of recorded results (experiments/vslice/logs/real_L28_gen.log). Not a live run.')
def sc_demo_final(c, p):
    head(c, 'Demo result', 'argmax == Hugging Face at all 6 positions')
    demo_table(c, 6)
    a = ease(seg(p, .1, .5))
    c.text(60, 600, f'errors {sci(MAXREL_MIN)} to {sci(MAXREL_MAX)}   |   {SEC_MIN:.0f} to {SEC_MAX:.0f} s per position   |   teacher-forced, not free-running', 20, fade(FG, a), True)
    c.text(60, 635, f'free-running generation vs HF greedy: {STATUS["free_running"]}    |    HostGuard audit of this driver: {STATUS["hostguard"]}', 20, fade(YEL, a))
    foot(c, 'Replay of recorded results (experiments/vslice/logs/real_L28_gen.log). Not a live run.')

# ----------------------------------------------------------------------------------------------------------
EXPLAINER = [(sc_title, 4), (sc_idea, 9), (sc_memory, 12), (sc_pipeline, 11), (sc_tournament, 12), (lambda c, p: sc_tiers(c, p), 9), (sc_result, 9)]
DEMO = [(sc_demo_intro, 6)] + [((lambda i: lambda c, p: sc_demo_run(c, p, i))(i), 5) for i in range(6)] + [(sc_demo_final, 8)]
FPS = 15

def frames(scenes, fps=FPS):
    for fn, dur in scenes:
        n = int(dur * fps)
        for k in range(n):
            c = Canvas(); fn(c, k / (n - 1)); yield c.out((1280, 720))

def square(im, headline='Qwen inside Qwen', sub='a real Qwen3-0.6B token executed by a stock Qwen3.5 model'):
    c = Canvas(1080, 1080, 1.0); c.text(540, 70, headline, 54, FG, True, anchor='mm'); c.text(540, 125, sub, 22, MUT, anchor='mm')
    im2 = im.resize((1080, 607), Image.LANCZOS); c.im.paste(im2, (0, 236))
    c.text(540, 920, 'illustrations + replay of recorded logs | numbers from repo logs', 22, MUT, anchor='mm')
    c.text(540, 965, 'one token: 268.6 s, shared A100, fp32, argmax == HF', 24, FG, anchor='mm')
    c.text(540, 1010, 'github.com/Bobliuuu/qweninqwen', 24, BLUE, anchor='mm')
    return c.im

def ffmpeg(path, w, h):
    return subprocess.Popen(['nice', 'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', '-preset', 'medium', '-movflags', '+faststart', path], stdin=subprocess.PIPE)
def make_video(name, scenes, sub):
    outd = os.path.join(HERE, 'video'); os.makedirs(outd, exist_ok=True)
    a = ffmpeg(os.path.join(outd, f'{name}_16x9.mp4'), 1280, 720); b = ffmpeg(os.path.join(outd, f'{name}_square.mp4'), 1080, 1080)
    for im in frames(scenes):
        a.stdin.write(im.tobytes()); b.stdin.write(square(im, sub=sub).tobytes())
    for q in (a, b): q.stdin.close(); q.wait()

def make_gif(path, scene_fn, dur, fps=10, size=(800, 450), colors=64, hold=1.2, **kw):
    n = int(dur * fps); fr = []
    for k in range(n):
        c = Canvas(k=1.25); scene_fn(c, k / (n - 1)); fr.append(c.out(size).quantize(colors, dither=Image.Dither.NONE))
    durs = [int(1000 / fps)] * n; durs[-1] = int(hold * 1000)
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=durs, loop=0, optimize=True, disposal=1)

def png(name, fn, p=1.0, size=(1600, 900), k=2.0):
    c = Canvas(k=k); fn(c, p); c.out(size).save(os.path.join(ASSETS, name), optimize=True)

# ----------------------------------------------------------------------------- flow diagrams (mermaid source + PNG from the same data)
FLOWS = {
 'flow_architecture': dict(dir='LR', title='Who does what', nodes=[
    ('H', 'Host streamer|weights, ids, addresses,|position ids', 0, 0, MUT), ('G', 'DeltaNet layers|paged weight memory|(matvec banks, registers)', 1, 0, BLUE),
    ('O', 'Compiled norm / MLP layers|RMSNorm, SwiGLU,|residual, gadgets', 2, 0, YEL), ('A', 'Full-attention layers|RoPE, softmax, KV|(outer KV = inner KV)', 3, 0, GRN),
    ('L', 'In-model argmax|+ outer lm_head|36 rails -> token id', 4, 0, PUR), ('X', 'Host: outer argmax|+ append', 4, 1, MUT)],
    edges=[('H', 'G', ''), ('G', 'O', ''), ('O', 'A', ''), ('A', 'L', ''), ('L', 'X', ''), ('X', 'H', 'next token')]),
 'flow_tiers': dict(dir='LR', title='Tier dependencies (status from STATUS)', nodes=[], edges=[]),
 'flow_token': dict(dir='LR', title='One token, end to end', nodes=[
    ('E', 'Embed (host row)', 0, 0, MUT), ('L', 'Inner layers x28|norm, QKV bank, RoPE+attn,|SwiGLU, residuals', 1, 0, YEL), ('N', 'Final RMSNorm', 2, 0, YEL),
    ('B', 'LM head|594 blocks of 256', 3, 0, BLUE), ('M', 'Block max/argmax|(in model)', 3, 1, GRN), ('C', 'Hard-max attention|over 594 winners', 2, 1, GRN),
    ('R', '36 rails|outer lm_head', 1, 1, PUR), ('T', 'Host argmax|+ append token', 0, 1, MUT)],
    edges=[('E', 'L', ''), ('L', 'N', ''), ('N', 'B', ''), ('B', 'M', ''), ('M', 'C', ''), ('C', 'R', ''), ('R', 'T', '')]),
}
def tier_flow():
    r = {t[0]: t for t in tier_rows()}
    col = lambda k: GRN if r[k][3] == 'done' else YEL
    st = lambda s: GRN if s == 'done' else MUT
    nodes = [('T0', 'T0 gadget|' + r['T0'][4], 0, 1, col('T0')), ('T1', 'T1 projection|' + r['T1'][4], 1, 1, col('T1')), ('T2', 'T2 op chain|' + r['T2'][4], 2, 1, col('T2')),
             ('T3t', 'T3-toy|' + r['T3-toy'][4], 3, 1, col('T3-toy')), ('T3r', 'T3 real|' + r['T3 real'][4], 4, 1, col('T3 real')), ('T4', 'T4 cost model|' + r['T4'][4], 5, 1, col('T4')),
             ('FR', 'free-running vs HF|' + STATUS['free_running'], 3, 0, st(STATUS['free_running'])), ('HG', 'HostGuard audit|' + STATUS['hostguard'], 4, 0, st(STATUS['hostguard'])),
             ('SP', 'speedup|' + STATUS['speedup'], 5, 0, st(STATUS['speedup']))]
    edges = [('T0', 'T1', ''), ('T1', 'T2', ''), ('T2', 'T3t', ''), ('T3t', 'T3r', ''), ('T3r', 'T4', ''), ('FR', 'T3r', 'completes'), ('HG', 'T3r', 'completes'), ('SP', 'T4', 'improves')]
    return nodes, edges
def mermaid(name):
    f = FLOWS[name]
    if name == 'flow_tiers': f['nodes'], f['edges'] = tier_flow()
    L = [f'flowchart {f["dir"]}']
    for i, l, *_ in f['nodes']: L.append(f'  {i}["{l.replace("|", "<br/>")}"]')
    for a, b, t in f['edges']: L.append(f'  {a} -->{"|" + t + "|" if t else ""} {b}')
    return '\n'.join(L)
def flow_png(name):
    f = FLOWS[name]
    if name == 'flow_tiers': f['nodes'], f['edges'] = tier_flow()
    cols = max(n[2] for n in f['nodes']) + 1; rows = max(n[3] for n in f['nodes']) + 1
    bw, bh, gx, gy = 200, 96, 44, 80
    W = 60 + cols * (bw + gx); H = 130 + rows * (bh + gy) + 60
    c = Canvas(W, H, 2.0); c.text(30, 24, f['title'], 28, FG, True)
    pos = {}
    for i, l, cx, cy, col in f['nodes']:
        x = 30 + cx * (bw + gx); y = 90 + cy * (bh + gy); pos[i] = (x, y)
        c.rect(x, y, bw, bh, mix(PANEL, col, .2), col, lw=2)
        parts = l.split('|')
        for j, t in enumerate(parts): c.text(x + bw / 2, y + bh / 2 + (j - (len(parts) - 1) / 2) * 20, t, 15 if j else 17, FG if j == 0 else MUT, j == 0, anchor='mm')
    for a, b, t in f['edges']:
        (xa, ya), (xb, yb_) = pos[a], pos[b]
        if ya == yb_ and xb > xa: pts = [(xa + bw, ya + bh / 2), (xb - 3, yb_ + bh / 2)]
        elif ya == yb_ and xb < xa: pts = [(xa, ya + bh / 2), (xb + bw + 3, yb_ + bh / 2)]
        elif yb_ > ya and xa == xb: pts = [(xa + bw / 2, ya + bh), (xb + bw / 2, yb_ - 3)]
        elif yb_ < ya and xa == xb: pts = [(xa + bw / 2, ya), (xb + bw / 2, yb_ + bh + 3)]
        else: pts = [(xa + bw / 2, ya + bh if yb_ > ya else ya), (xb + bw / 2, yb_ - 3 if yb_ > ya else yb_ + bh + 3)]
        c.arrow(pts, MUT, 2.5, 10)
        if t: c.text((pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2 - 12, t, 14, MUT, anchor='mm')
    c.text(30, H - 24, 'Rendered from the same node/edge list as the mermaid block in the README (docs/launch/make_media.py).', 13, MUT)
    c.out((int(W * 1.6), int(H * 1.6))).save(os.path.join(ASSETS, name + '.png'), optimize=True)
    open(os.path.join(ASSETS, name + '.mmd'), 'w').write(mermaid(name) + '\n')

def results_table(c, p=1.0):
    head(c, 'Six positions, argmax == Hugging Face at every one', 'real Qwen3-0.6B, 28 layers, full 151,936 head, in-model argmax; teacher-forced prompt + continuation')
    demo_table(c, 6)
    c.text(60, 600, f'max rel. logit error {sci(MAXREL_MIN)} to {sci(MAXREL_MAX)}; {SEC_MIN:.0f} to {SEC_MAX:.0f} s per position on a shared A100 (fp32); {LAYERS} outer layers, D = {DDIM:,}', 19, FG)
    c.text(60, 635, 'teacher-forced: positions 1-5 feed the known continuation tokens through the KV cache. Not free-running generation.', 19, YEL)
    foot(c, 'Source: experiments/vslice/logs/real_L28_gen.log, res_vs2_real151936_L28.json')

def main(what):
    if 'png' in what:
        for n in FLOWS: flow_png(n)
        png('tiers_ladder.png', sc_tiers); png('tournament.png', sc_tournament); png('results_table.png', results_table); png('cost_chart.png', sc_cost)
        png('hero_still.png', sc_hero, 1.0); png('state_memory_toy.png', sc_memory); png('pipeline_flow.png', sc_pipeline)
    if 'gif' in what:
        make_gif(os.path.join(ASSETS, 'hero.gif'), sc_hero, 7, fps=12)
        make_gif(os.path.join(ASSETS, 'tiers_climb.gif'), sc_tiers, 8, fps=10)
        make_gif(os.path.join(ASSETS, 'tournament.gif'), sc_tournament, 10, fps=10)
        make_gif(os.path.join(ASSETS, 'pipeline.gif'), sc_pipeline, 9, fps=10)
        make_gif(os.path.join(ASSETS, 'state_memory.gif'), sc_memory, 9, fps=10)
        make_gif(os.path.join(ASSETS, 'idea.gif'), sc_idea, 7, fps=10)
    if 'video' in what:
        make_video('explainer', EXPLAINER, 'explainer: how a Qwen runs inside a Qwen')
        make_video('demo', DEMO, 'demo: the real 6-position run vs Hugging Face')

if __name__ == '__main__':
    main(sys.argv[1:] or ['png', 'gif', 'video'])
