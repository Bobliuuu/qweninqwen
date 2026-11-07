#!/usr/bin/env python3
"""Build the cinematic demo (1080p, 30 fps, white, one accent colour) for qweninqwen.

  docs/demo/demo.mp4         with a generated ambient/pulse audio track (numpy, local)
  docs/demo/demo_silent.mp4  same picture, no audio
  docs/demo/demo.gif         teaser (<= 8 MB, 900 px, loops)
  docs/demo/stills/*.png     one still per scene

Scenes re-use the README animations in docs/assets/viz (make_anim.py: hero, one_token, precision, seam_grid), as they are, rendered here at the
full 1920x1080 they are drawn at, and the static PNG figures animated with Ken-Burns moves and reveals.
This film only shows the final design and result; the text audit (--audit-text) keeps it free of any earlier answer or failing variant.
HARD RULE: every number drawn is read from a data file in this repo (see PROVENANCE.md); the asserts below refuse to build otherwise.
Schematic parts (layout/glow of the architecture figure, count-up easing) are marked in PROVENANCE.md.

usage: python3 docs/demo/make_demo.py [--stills-only]       (needs Pillow, numpy, ffmpeg; uses 8 processes)
"""
import json, math, os, shutil, subprocess, sys, types, wave
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ASSETS = os.path.join(REPO, 'docs', 'assets'); VIZ = os.path.join(ASSETS, 'viz')
FR = os.path.join(REPO, 'experiments', 'circuit', 'fullrun')
sys.path.insert(0, VIZ)
import anim_lib as L          # white toolkit, W,H = 1920,1080
from anim_lib import (INK, G1, G2, G3, G4, ACC, ACCD, ACCL, ACCM, WHITE, smooth, ease_out, ease_io, P, mix, clamp, lerp, commas)

# every string drawn through anim_lib.text is logged when AUDIT is set (see --audit-text)
AUDIT = []
_text0 = L.text
def _logged_text(d, x, y, s, *a, **k):
    if os.environ.get('DEMO_AUDIT'): AUDIT.append(str(s))
    return _text0(d, x, y, s, *a, **k)
L.text = _logged_text

import make_anim as M         # hero, one_token, seam_grid, precision (real-data scenes, used as they are)
W, H, FPS, XF = 1920, 1080, 30, 0.6
REPO_URL = 'github.com/Bobliuuu/qweninqwen'
FOOT_SCOPE = 'One token at position 0   ·   parallel jobs seeded from the reference, verified at 95 exact seams'

# ------------------------------------------------------------------ data + checks (all real)
def jl(p): return json.load(open(p))
verdict = jl(os.path.join(HERE, 'data', 'FINAL_VERDICT.json'))
seams = jl(os.path.join(FR, 'logs', 'cfram_q_vast', 'seams_0-95_20261107_004845.json'))
rows = {r['chunk']: r for r in seams['rows']}
dq1 = jl(os.path.join(FR, 'diag_quant', 'diag_quant.json'))
QR = os.path.join(FR, 'quantfix', 'results')
qf = {n: jl(os.path.join(QR, n + '.json')) for n in ('b8_apl2',)}
last = rows[95]; ro = last['readout']
n_seam = sum(1 for r in rows.values() if r.get('seam', {}).get('exact'))
sum_mism = sum(r['free_mismatches'] for r in rows.values()); sum_forced = sum(r['forced_tokens'] for r in rows.values()); sum_inj = sum(r['injections'] for r in rows.values())
TOTAL_TOKENS = M.T
assert len(rows) == 96 and seams['ok'] is True and 'PASS' in seams['verdict'] and verdict['passed'] is True
assert (n_seam, sum_mism, sum_forced, sum_inj) == (95, 0, 0, 0)
assert dq1['hf_argmax'] == 15846 and qf['b8_apl2']['argmax_fw'] == 15846
assert ro['circuit_word'] == 15846 and ro['circuit_halt_step'] == 57705 and ro['tail_equals_reference'] is True
assert TOTAL_TOKENS == 199287158 and 'Question' in open(os.path.join(FR, 'logs', 'final_token', 'REPORT.md')).read()

# ------------------------------------------------------------------ helpers
_emo = {}
def emoji(ch, sz):
    k = (ch, sz)
    if k not in _emo:
        f = ImageFont.truetype('/System/Library/Fonts/Apple Color Emoji.ttc', 160)
        im = Image.new('RGBA', (220, 220), (0, 0, 0, 0)); ImageDraw.Draw(im).text((0, 0), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox()); r = sz / im.height; _emo[k] = im.resize((max(1, int(im.width * r)), sz), Image.LANCZOS)
    return _emo[k]
def paste_emoji(img, ch, x, y, sz, alpha=1.0):
    e = emoji(ch, sz)
    if alpha < 1: e = e.copy(); e.putalpha(e.getchannel('A').point(lambda v: int(v * alpha)))
    img.paste(e, (int(x), int(y)), e)
_shadow = {}
def shadow_card(img, box, r=24, fill=WHITE, outline=G3, alpha=1.0):
    x0, y0, x1, y1 = [int(v) for v in box]; k = (x1 - x0, y1 - y0, r)
    if k not in _shadow:
        pad = 40; s = Image.new('RGBA', (k[0] + 2 * pad, k[1] + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(s).rounded_rectangle((pad, pad + 8, pad + k[0], pad + 8 + k[1]), r, fill=(17, 24, 39, 26)); _shadow[k] = s.filter(ImageFilter.GaussianBlur(16))
    sh = _shadow[k].copy()
    if alpha < 1: sh.putalpha(sh.getchannel('A').point(lambda v: int(v * alpha)))
    img.paste(sh, (x0 - 40, y0 - 40), sh)
    ImageDraw.Draw(img).rounded_rectangle((x0, y0, x1, y1), r, fill=mix(fill, alpha) if alpha < 1 else fill, outline=mix(outline, alpha) if alpha < 1 else outline, width=2)
def lower_third(img, s, a=1.0, y=952):
    if a <= 0: return
    f = L.font(34, 'Medium'); w = int(f.getlength(s)) + 90; x0 = (W - w) // 2
    shadow_card(img, (x0, y, x0 + w, y + 78), 39, WHITE, G3, a)
    d = ImageDraw.Draw(img); d.rounded_rectangle((x0 + 26, y + 22, x0 + 32, y + 56), 3, fill=mix(ACC, a) if a < 1 else ACC)
    L.text(d, x0 + 54, y + 39, s, 34, INK, 'Medium', 'lm', alpha=a)
def rect_glow(img, box, a, pulse=0.0):
    if a <= 0: return img
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov); x0, y0, x1, y1 = box; g = int(6 + 4 * pulse)
    for k in range(g, 0, -2): d.rounded_rectangle((x0 - k, y0 - k, x1 + k, y1 + k), 22 + k, outline=(79, 70, 229, int(a * 38)), width=3)
    d.rounded_rectangle(box, 22, fill=(79, 70, 229, int(a * 30)), outline=(79, 70, 229, int(a * 255)), width=5)
    return Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB')
def fade_white(img, a): return Image.blend(Image.new('RGB', img.size, WHITE), img, clamp(a))

# ------------------------------------------------------------------ scene: title
def title_scene(t):
    img, d = L.new(); a = smooth(P(t, 0.1, 1.0))
    f = L.font(132, 'Bold'); label = 'Qwen inside Qwen'; e = emoji('🧠', 120); tw = f.getlength(label)
    left = (W - (tw + e.width + 36)) / 2; dy = (1 - ease_out(P(t, 0.0, 1.0))) * 26
    paste_emoji(img, '🧠', left, 336 + dy, 120, a); d = ImageDraw.Draw(img)
    L.text(d, left + e.width + 36, 300 + dy, label, 132, INK, 'Bold', alpha=a)
    wl = 150 * ease_io(P(t, 0.7, 1.7)); d.line((W / 2 - wl, 520, W / 2 + wl, 520), fill=ACC, width=5)
    L.text(d, W / 2, 590, 'A real Qwen3-0.6B token, computed inside another Qwen.', 46, G1, 'Regular', 'ma', alpha=smooth(P(t, 1.2, 2.0)))
    L.text(d, W / 2, 700, REPO_URL, 40, ACC, 'Medium', 'ma', mono=True, alpha=smooth(P(t, 1.8, 2.6)))
    return img

# ------------------------------------------------------------------ scene: architecture (Ken Burns + host shrink + glows)
ARCH = Image.open(os.path.join(ASSETS, 'circuit_architecture.png')).convert('RGB')      # 2560x1544
HOST = (96, 301, 516, 900)                                                              # host card (px of the PNG)
HOSTPAD = (80, 288, 528, 918)
GLOW = {'table': (1476, 456, 1869, 1116), 'ram': (732, 648, 1032, 727), 'regs': (732, 544, 1032, 623), 'vdot': (1060, 648, 1388, 727),
        'alu': (1060, 544, 1388, 623), 'decode': (732, 753, 1388, 832), 'argmax': (732, 856, 1388, 936)}
_arch_cache = {}
def arch_base():
    if 'b' not in _arch_cache:
        x0, y0, x1, y1 = HOSTPAD; b = ARCH.copy(); crop = ARCH.crop(HOST)
        ImageDraw.Draw(b).rectangle((x0, y0, x1, y1), fill=WHITE); _arch_cache['b'] = b; _arch_cache['h'] = crop
    return _arch_cache['b'], _arch_cache['h']
def cam(t):   # (cx, cy, width) in PNG pixels
    K = [(0.0, 1280, 772, 2560), (1.2, 1280, 772, 2480), (2.6, 330, 640, 1150), (5.6, 330, 640, 1150), (7.0, 1180, 660, 1900), (11.0, 1180, 660, 1900), (12.5, 1280, 772, 2560)]
    for (t0, x0, y0, w0), (t1, x1, y1, w1) in zip(K, K[1:]):
        if t0 <= t <= t1:
            p = ease_io((t - t0) / (t1 - t0)); return lerp(x0, x1, p), lerp(y0, y1, p), math.exp(lerp(math.log(w0), math.log(w1), p))
    return K[-1][1:]
def arch_scene(t):
    base, hc = arch_base(); img = base.copy()
    s = lerp(1.0, 0.56, ease_io(P(t, 3.2, 4.8)))
    if s < 1: hc2 = hc.resize((int(hc.width * s), int(hc.height * s)), Image.LANCZOS)
    else: hc2 = hc
    hx = HOST[0] + (hc.width - hc2.width) / 2; hy = HOST[1] + 10
    img.paste(hc2, (int(hx), int(hy)))
    cx, cy, w = cam(t); h = w * 9 / 16; x0 = clamp(cx - w / 2, 0, 2560 - w); y0 = clamp(cy - h / 2, 0, 1544 - h)
    view = img.crop((int(x0), int(y0), int(x0 + w), int(y0 + h))).resize((W, H), Image.LANCZOS); k = W / w
    def tr(b): return ((b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k)
    pulse = 0.5 + 0.5 * math.sin(t * 5)
    hb = (hx - 4, hy - 4, hx + hc2.width + 4, hy + hc2.height + 4)
    view = rect_glow(view, tr(hb), smooth(P(t, 1.4, 2.2)) * (1 - 0.35 * smooth(P(t, 5.0, 6.0))), pulse)
    for key, (a0, a1, group) in {'table': (7.0, 8.4, ['table']), 'ram': (8.2, 9.6, ['ram', 'regs', 'vdot', 'alu', 'decode']), 'argmax': (9.6, 10.8, ['argmax'])}.items():
        a = smooth(P(t, a0, a0 + 0.5)) * (1 - 0.55 * smooth(P(t, a1 + 0.6, a1 + 1.4)) if key != 'argmax' else 1)
        for g in group: view = rect_glow(view, tr(GLOW[g]), a, pulse)
    view = fade_white(view, smooth(P(t, 0.0, 0.8)))
    caps = [(2.8, 5.6, 'The host: boot the firmware · the GO token · the embed-and-append loop · ecall I/O'), (7.0, 8.7, "Weights come from rows of the model's own embedding table"),
            (8.7, 10.0, 'Registers, RAM, VDOT and the address decode: inside the circuit'), (10.0, 11.4, "Argmax and HALT: the circuit's own"), (11.4, 12.5, 'Everything else runs in the circuit')]
    for a0, a1, s_ in caps:
        al = smooth(P(t, a0, a0 + 0.4)) * (1 - smooth(P(t, a1 - 0.3, a1)))
        if a1 == 12.5: al = smooth(P(t, a0, a0 + 0.4))
        lower_third(view, s_, al)
    return view

# ------------------------------------------------------------------ scene: kinetic numbers + heatstrip reveal
HEAT = Image.open(os.path.join(ASSETS, 'circuit_heatstrip.png')).convert('RGB')      # 2400x675
def counter(img, d, cx, y, val_s, label, a, col=INK, size=96):
    L.text(d, cx, y, val_s, size, col, 'Bold', 'mm', alpha=a); L.text(d, cx, y + 88, label, 26, G1, 'Regular', 'mm', alpha=a)
def numbers_scene(t):
    img, d = L.new(); L.text(d, 120, 96, 'The whole run, in numbers', 56, INK, 'Bold', alpha=smooth(P(t, 0, 0.6)))
    p = ease_out(P(t, 0.3, 3.2)); a = smooth(P(t, 0.2, 0.8))
    counter(img, d, 560, 250, commas(TOTAL_TOKENS * p), 'trace tokens for one token', a, ACC, 104)
    p2 = ease_out(P(t, 0.6, 2.2)); counter(img, d, 1400, 250, f'{int(round(96 * p2))} / 96', 'chunk jobs clean', smooth(P(t, 0.6, 1.2)), INK, 104)
    p3 = ease_out(P(t, 0.9, 2.6)); counter(img, d, 560, 410, f'{int(round(95 * p3))} / 95', 'seams exact', smooth(P(t, 0.9, 1.5)), ACC, 104)
    z = smooth(P(t, 2.4, 3.2)); counter(img, d, 1400, 410, '0 / 0 / 0', 'mismatches / forced tokens / host injections', z, INK, 104)
    hw = 1300; hh = int(HEAT.height * hw / HEAT.width); hv = HEAT.resize((hw, hh), Image.LANCZOS)
    reveal = ease_io(P(t, 3.0, 5.6)); rw = int(hw * reveal); ha = smooth(P(t, 3.0, 3.6))
    if rw > 0:
        crop = hv.crop((0, 0, rw, hh)); img.paste(crop, ((W - hw) // 2, 540))
        d = ImageDraw.Draw(img); d.rounded_rectangle(((W - hw) // 2 + rw - 2, 620, (W - hw) // 2 + rw + 2, 540 + hh - 110), 2, fill=ACCM) if 0.02 < reveal < 0.99 else None
    lower_third(img, FOOT_SCOPE, smooth(P(t, 5.4, 6.1)) * (1 - smooth(P(t, 8.4, 9.0))) + (0 if t < 8.4 else 0))
    return img

# ------------------------------------------------------------------ scene: fleet timeline (Ken Burns + wipe)
FLEET = Image.open(os.path.join(ASSETS, 'circuit_fleet.png')).convert('RGB')         # 2400x882
def fleet_scene(t):
    img, d = L.new(); L.text(d, 120, 96, 'How it was run: rented GPUs, restarts and all', 56, INK, 'Bold', alpha=smooth(P(t, 0, 0.6)))
    z = lerp(1.0, 1.07, ease_io(P(t, 0, 6.0))); fw = int(1680 * z); fh = int(FLEET.height * fw / FLEET.width); fv = FLEET.resize((fw, fh), Image.LANCZOS)
    reveal = ease_io(P(t, 0.5, 3.6)); rw = int(fw * reveal)
    x = (W - fw) // 2; y = 230
    if rw > 0: img.paste(fv.crop((0, 0, rw, fh)), (x, y))
    lower_third(img, 'The first fleet manager had bugs: two restarts, three boxes vanished. The seam checks decide, not the schedule.', smooth(P(t, 3.6, 4.2)))
    return img

# ------------------------------------------------------------------ scene: end card
def end_scene(t):
    img, d = L.new(); a = smooth(P(t, 0.1, 0.9)); dy = (1 - ease_out(P(t, 0, 1.0))) * 24
    f = L.font(112, 'Bold'); label = 'Qwen inside Qwen'; e = emoji('✅', 100); tw = f.getlength(label)
    left = (W - (tw + e.width + 32)) / 2; paste_emoji(img, '✅', left, 270 + dy, 100, a); d = ImageDraw.Draw(img)
    L.text(d, left + e.width + 32, 240 + dy, label, 112, INK, 'Bold', alpha=a)
    wl = 130 * ease_io(P(t, 0.6, 1.5)); d.line((W / 2 - wl, 450, W / 2 + wl, 450), fill=ACC, width=5)
    L.text(d, W / 2, 530, f"96 / 96 chunks clean   ·   {n_seam} / 95 seams exact   ·   final word {ro['circuit_word']}", 44, INK, 'Medium', 'ma', alpha=smooth(P(t, 1.0, 1.8)))
    L.text(d, W / 2, 620, FOOT_SCOPE, 30, G1, 'Regular', 'ma', alpha=smooth(P(t, 1.5, 2.3)))
    L.text(d, W / 2, 740, REPO_URL, 44, ACC, 'Medium', 'ma', mono=True, alpha=smooth(P(t, 2.0, 2.8)))
    L.text(d, W / 2, 810, 'every number in this video: docs/demo/PROVENANCE.md', 26, G2, 'Regular', 'ma', mono=True, alpha=smooth(P(t, 2.4, 3.2)))
    return img

# ------------------------------------------------------------------ timeline
SEG = [('hero', M.hero, 10.0, 'dissolve'), ('title', title_scene, 4.5, 'slide'), ('architecture', arch_scene, 12.5, 'dissolve'),
       ('one_token', M.one_token, 24.0, 'dissolve'), ('precision', M.precision, 10.5, 'dissolve'),
       ('seam_grid', M.seam_grid, 11.0, 'dissolve'), ('numbers', numbers_scene, 9.0, 'dissolve'), ('fleet', fleet_scene, 6.5, 'dissolve'), ('end', end_scene, 6.0, 'dissolve')]
STARTS = []; _s = 0.0
for _i, (_n, _f, _d, _k) in enumerate(SEG):
    STARTS.append(_s); _s += _d - (XF if _i < len(SEG) - 1 else 0)
TOTAL = _s

def frame_at(T):
    act = [(i, T - STARTS[i]) for i in range(len(SEG)) if STARTS[i] <= T < STARTS[i] + SEG[i][2] + 1e-9]
    if len(act) == 1:
        i, lt = act[0]; return SEG[i][1](min(lt, SEG[i][2] - 1e-3))
    (i, ta), (j, tb) = act[0], act[1]; p = smooth((T - STARTS[j]) / XF); A = SEG[i][1](min(ta, SEG[i][2] - 1e-3)); B = SEG[j][1](max(0.0, tb))
    if SEG[j][3] == 'slide':
        q = ease_io((T - STARTS[j]) / XF); off = int(W * (1 - q)); out = Image.new('RGB', (W, H), WHITE)
        out.paste(A, (-int(W * 0.18 * q), 0)); sh = Image.new('RGBA', (60, H), (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        for xx in range(60): sd.line((xx, 0, xx, H), fill=(17, 24, 39, int(30 * (xx / 60) ** 2)))
        out.paste(sh, (off - 60, 0), sh); out.paste(B, (off, 0)); return out
    return Image.blend(A, B, p)
def worker(T): return frame_at(T).tobytes()
import re
# flagged: an earlier answer shown AS an answer, failing-variant answers shown as answers, and wording about earlier results
# (the digit pair inside an unrelated number or a layer / chunk index is fine)
BAD_RE = [re.compile(r"(answer|token|word|firmware|gives|says)\W{0,4}13(?![\d,.])", re.I), re.compile(r"(?<![\d,.])13(?![\d,.])\s*['\"]\.['\"]"),
          re.compile(r"(?<![\d,.])(838|3405|408|454|661)(?![\d,.])"), re.compile(r"old firmware|before the fix|int4 everywhere|first answer|wrong|why the|why 13|was 13", re.I)]
def bad(sx): return any(r.search(sx) for r in BAD_RE)
def audit_worker(T):
    AUDIT.clear(); frame_at(T)
    return sorted({sx for sx in AUDIT if bad(sx)})

# ------------------------------------------------------------------ audio (numpy, local, calm pad + soft pulse + chimes)
def make_audio(path, total):
    sr = 44100; n = int(total * sr); t = np.arange(n) / sr; rng = np.random.default_rng(7)
    chords = [(73.42, 146.83, 220.0, 329.63, 369.99), (61.74, 123.47, 185.0, 246.94, 293.66), (98.0, 196.0, 246.94, 293.66, 369.99), (55.0, 110.0, 220.0, 277.18, 329.63)]
    L_ch = np.zeros(n); R_ch = np.zeros(n); seg = total / 8.0
    for ci in range(8):
        a0, a1 = ci * seg - 2.0, (ci + 1) * seg + 2.0; ch = chords[ci % 4]
        env = np.clip(np.minimum((t - a0) / 3.0, (a1 - t) / 3.0), 0, 1) ** 1.5
        for k, fq in enumerate(ch):
            lfo = 0.75 + 0.25 * np.sin(2 * np.pi * (0.07 + 0.013 * k) * t + k)
            for det, pan in ((-0.8, 0.8), (0.8, 1.0)):
                w = np.sin(2 * np.pi * (fq + det) * t + k) * env * lfo * (0.05 if fq > 100 else 0.07)
                L_ch += w * (2 - pan) / 2; R_ch += w * pan
    beat = 60.0 / 72.0; pulse = np.zeros(n)
    for bi in range(int(total / beat)):
        tb = bi * beat
        if tb < 3 or tb > total - 5: continue
        lvl = 0.10 + 0.08 * (1 if 12 < tb < total - 15 else 0); i0 = int(tb * sr); m = min(int(0.5 * sr), n - i0); tt = np.arange(m) / sr
        pulse[i0:i0 + m] += np.sin(2 * np.pi * (52 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9) * lvl
        if bi % 2 == 1:
            m2 = min(int(0.25 * sr), n - i0); t2 = np.arange(m2) / sr; pulse[i0:i0 + m2] += rng.standard_normal(m2) * np.exp(-t2 * 60) * 0.012
    chime = np.zeros(n)
    for T0 in STARTS[1:]:
        i0 = int((T0 + 0.2) * sr); m = min(int(1.6 * sr), n - i0); tt = np.arange(m) / sr
        for fq, g in ((880.0, 0.05), (1318.5, 0.03), (1760.0, 0.015)): chime[i0:i0 + m] += np.sin(2 * np.pi * fq * tt) * np.exp(-tt * 3.2) * g
    left = L_ch + pulse + chime; right = R_ch + pulse + chime * 0.9
    fade = np.clip(np.minimum(t / 1.5, (total - t) / 3.0), 0, 1); left *= fade; right *= fade
    pk = max(np.abs(left).max(), np.abs(right).max()); g = 0.42 / pk
    st = np.stack([left * g, right * g], 1); pcm = (st * 32767).astype('<i2')
    with wave.open(path, 'wb') as wf: wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())

# ------------------------------------------------------------------ main
if __name__ == '__main__':
    stills = os.path.join(HERE, 'stills'); shutil.rmtree(stills, ignore_errors=True); os.makedirs(stills)
    for i, (name, fn, dur, kind) in enumerate(SEG):
        frame_at(STARTS[i] + dur * 0.85 if name not in ('title',) else STARTS[i] + 3.2).save(os.path.join(stills, f'{i + 1:02d}_{name}.png'))
    print('timeline: total %.1f s' % TOTAL, [(n, round(s, 1)) for (n, *_), s in zip(SEG, STARTS)])
    if '--audit-text' in sys.argv:
        os.environ['DEMO_AUDIT'] = '1'; ts = [i * 0.2 for i in range(int(TOTAL / 0.2))]
        with Pool(8) as pool: res = pool.map(audit_worker, ts, chunksize=8)
        hits = sorted({sx for r in res for sx in r}); print('audited', len(ts), 'frames; strings drawn through the text helper that contain a banned token:', hits or 'NONE'); sys.exit()
    if '--stills-only' in sys.argv: sys.exit()
    silent = os.path.join(HERE, 'demo_silent.mp4'); N = int(TOTAL * FPS)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '19',
                           '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', silent], stdin=subprocess.PIPE)
    with Pool(8) as pool:
        for k, b in enumerate(pool.imap(worker, [i / FPS for i in range(N)], chunksize=3)):
            ff.stdin.write(b)
            if k % 150 == 0: print('frame', k, '/', N, flush=True)
    ff.stdin.close(); ff.wait()
    wav = os.path.join(HERE, '_audio.wav'); make_audio(wav, N / FPS)
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', wav, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', os.path.join(HERE, 'demo.mp4')])
    os.remove(wav)
    # teaser gif: hero (0-6 s) + title + architecture glow + seam grid + end
    ix = {n: STARTS[i] for i, (n, *_r) in enumerate(SEG)}
    cuts = [(0.0, 6.0), (ix['architecture'] + 6.8, ix['architecture'] + 11.0), (ix['precision'] + 3.0, ix['precision'] + 9.0), (ix['seam_grid'] + 1.0, ix['seam_grid'] + 9.0), (ix['end'], ix['end'] + 4.0)]
    parts = ''.join(f'[0:v]trim={a:.2f}:{b:.2f},setpts=PTS-STARTPTS,fps=15,scale=900:-1:flags=lanczos[v{i}];' for i, (a, b) in enumerate(cuts))
    cc = ''.join(f'[v{i}]' for i in range(len(cuts))) + f'concat=n={len(cuts)}:v=1:a=0[c];[c]split[s0][s1];[s0]palettegen=max_colors=80:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[g]'
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-filter_complex', parts + cc, '-map', '[g]', '-loop', '0', os.path.join(HERE, 'demo.gif')])
    for fn in ('demo.mp4', 'demo_silent.mp4', 'demo.gif'): print(fn, round(os.path.getsize(os.path.join(HERE, fn)) / 1e6, 2), 'MB')
