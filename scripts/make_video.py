"""Render the demo video/GIF from assets/trace.json (real tokens + timings + per-layer subspace energy
recorded on an A100 by scripts/trace_demo.py). The per-token layer sweep is a visual of the forward pass;
bar lengths are the measured residual-stream norms of each subspace after each layer (last generated token).
  python scripts/make_video.py assets/trace.json assets/demo.mp4
"""
import sys, json, math, subprocess, os, tempfile
from PIL import Image, ImageDraw, ImageFont

W, Hh, FPS = 1280, 720, 24
BG, FG, DIM = (13, 17, 23), (230, 237, 243), (125, 133, 144)
BLUE, ORANGE, GREEN, PURPLE = (88, 166, 255), (255, 166, 87), (63, 185, 80), (188, 140, 255)
def font(sz, mono=True):
    for p in (['/System/Library/Fonts/Menlo.ttc'] if mono else ['/System/Library/Fonts/Helvetica.ttc']) + ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
F, FS, FB, FT = font(20), font(15), font(28, False), font(46, False)
d = json.load(open(sys.argv[1])); NL = d['n_layers']; types = d['types']
runs = d['runs']

def wrap(text, n):
    out = []
    for para in text.split('\n'):
        while len(para) > n: out.append(para[:n]); para = para[n:]
        out.append(para)
    return out

def ebar(e):  # log scale 0..~1000 -> 0..1
    return 0 if e <= 0 else max(0.02, min(1, math.log10(1 + e) / 3.1))

def scene_frames(run, secs_min=5.0):
    toks = run['tokens']; tt = [t['t'] for t in toks]; dur = max(secs_min, tt[-1] + 1.2)
    n = int(dur * FPS); frames = []
    for f in range(n):
        t = f / FPS
        img = Image.new('RGB', (W, Hh), BG); g = ImageDraw.Draw(img)
        inner = run['mode'] == 'inner'
        g.text((40, 28), 'qweninqwen', font=FB, fill=FG)
        g.text((40, 70), 'Qwen3-0.6B running inside Qwen3.5-0.8B  -  one stock Qwen3.5 checkpoint', font=FS, fill=DIM)
        # terminal
        g.rounded_rectangle((40, 110, 760, 640), 10, fill=(22, 27, 34), outline=(48, 54, 61))
        col = ORANGE if inner else BLUE
        g.text((60, 124), 'active model: ' + run['model'], font=F, fill=col)
        g.text((60, 152), 'token ids: ' + ('248320 + id  (inner vocab rows)' if inner else '0 .. 248319  (outer vocab)'), font=FS, fill=DIM)
        k = sum(1 for x in tt if x <= t - 0.2)
        gen = ''.join(x['text'] for x in toks[:k])
        lines = wrap(run['prompt'] + '\x00' + gen, 56)
        y = 190
        for ln in lines[:17]:
            if '\x00' in ln:
                a, b = ln.split('\x00', 1); g.text((60, y), a, font=F, fill=FG); x0 = 60 + g.textlength(a, font=F); g.text((x0, y), b, font=F, fill=col)
            else:
                g.text((60, y), ln, font=F, fill=col if any(('\x00' in l) for l in lines[:lines.index(ln)]) else FG)
            y += 26
        g.text((60, 610), f'{k} tokens  |  real timings from A100 trace', font=FS, fill=DIM)
        # layer panel
        px, py, rh = 800, 130, 9.6
        g.text((px, 106), 'residual stream after each layer', font=FS, fill=FG)
        g.text((px + 40, py + NL * rh + 6), 'outer dims', font=FS, fill=BLUE); g.text((px + 250, py + NL * rh + 6), 'inner dims', font=FS, fill=ORANGE)
        if k < len(toks) and k >= 0:
            prev = tt[k - 1] if k > 0 else 0; cur = max(0.05, tt[k] - prev)
            sweep = int(min(1, max(0, (t - 0.2 - prev) / cur)) * NL)
        else: sweep = NL
        for i in range(NL):
            yy = py + i * rh
            g.rectangle((px, yy, px + 6, yy + rh - 2), fill=GREEN if types[i] == 'full_attention' else PURPLE)
            eo, ei = run['layer_energy'][i]
            lit = i <= sweep
            g.rectangle((px + 40, yy, px + 220, yy + rh - 2), fill=(30, 36, 44))
            g.rectangle((px + 250, yy, px + 430, yy + rh - 2), fill=(30, 36, 44))
            if lit:
                g.rectangle((px + 40, yy, px + 40 + 180 * ebar(eo), yy + rh - 2), fill=BLUE)
                g.rectangle((px + 250, yy, px + 250 + 180 * ebar(ei), yy + rh - 2), fill=ORANGE)
            if i == sweep: g.rectangle((px - 4, yy - 1, px + 440, yy + rh), outline=FG)
        g.text((px, 660), 'green = full attention, purple = gated delta net', font=FS, fill=DIM)
        g.text((40, 665), 'the other half stays exactly zero: block-diagonal weights', font=FS, fill=DIM)
        frames.append(img)
    return frames

def card(lines, secs, big=True):
    out = []
    for f in range(int(secs * FPS)):
        img = Image.new('RGB', (W, Hh), BG); g = ImageDraw.Draw(img); y = 200
        for i, (txt, c) in enumerate(lines):
            fnt = FT if i == 0 else FB if i == 1 else F
            w = g.textlength(txt, font=fnt); g.text(((W - w) / 2, y), txt, font=fnt, fill=c); y += 80 if i < 2 else 44
        out.append(img)
    return out

frames = card([('qweninqwen', FG), ('the smallest Qwen running inside a Qwen', BLUE),
               ('Qwen3-0.6B (28 layers) merged into the real Qwen3.5-0.8B', DIM), ('one checkpoint, stock transformers code, weights only', DIM)], 3)
for r in runs: frames += scene_frames(r)
frames += card([('exact, not emulated', FG), ('fp32: logits match the standalone models to ~1e-4', GREEN),
                ('Qwen3-0.6B inside Qwen3.5-0.8B  -  github.com/Bobliuuu/qweninqwen', DIM)], 4)
out = sys.argv[2]
with tempfile.TemporaryDirectory() as td:
    for i, im in enumerate(frames): im.save(f'{td}/f{i:05d}.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{td}/f%05d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out], check=True)
    gif = out.replace('.mp4', '.gif')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{td}/f%05d.png', '-vf', 'fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse', gif], check=True)
print('wrote', out, len(frames) / FPS, 's')
