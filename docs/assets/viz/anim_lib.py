"""Tiny white-background animation toolkit (Pillow + ffmpeg). Frames are drawn at 1920x1080, downscaled to 1280x720 (lanczos) and encoded to MP4 and GIF."""
import os, shutil, subprocess, math
from PIL import Image, ImageDraw, ImageFont
W, H = 1920, 1080; OW, OH = 1280, 720
WHITE = (255, 255, 255); INK = (17, 24, 39); G1 = (107, 114, 128); G2 = (156, 163, 175); G3 = (229, 231, 235); G4 = (243, 244, 246)
ACC = (79, 70, 229); ACCD = (67, 56, 202); ACCL = (238, 242, 255); ACCM = (199, 210, 254); ACC2 = (129, 140, 248)
SF = '/System/Library/Fonts/SFNS.ttf'; SFM = '/System/Library/Fonts/SFNSMono.ttf'
_fc = {}
def font(sz, wt='Regular', mono=False):
    k = (sz, wt, mono)
    if k not in _fc:
        f = ImageFont.truetype(SFM if mono else SF, sz)
        try: f.set_variation_by_name(wt)
        except Exception: pass
        _fc[k] = f
    return _fc[k]
def mix(c, a):
    a = max(0.0, min(1.0, a)); return tuple(int(255 + (c[i] - 255) * a) for i in range(3))
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def P(t, a, b): return clamp((t - a) / (b - a))
def smooth(x): x = clamp(x); return x * x * (3 - 2 * x)
def ease_out(x): x = clamp(x); return 1 - (1 - x) ** 3
def ease_io(x): x = clamp(x); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def lerp(a, b, t): return a + (b - a) * t
def new(): img = Image.new('RGB', (W, H), WHITE); return img, ImageDraw.Draw(img)
def text(d, x, y, s, size=28, fill=INK, wt='Regular', anchor='la', mono=False, alpha=1.0):
    d.text((x, y), s, font=font(size, wt, mono), fill=mix(fill, alpha) if alpha < 1 else fill, anchor=anchor)
def tw(s, size, wt='Regular', mono=False): return font(size, wt, mono).getlength(s)
def rr(d, x0, y0, x1, y1, r=16, fill=None, outline=None, w=2, alpha=1.0):
    if x1 <= x0 or y1 <= y0: return
    f = mix(fill, alpha) if fill is not None and alpha < 1 else fill
    o = mix(outline, alpha) if outline is not None and alpha < 1 else outline
    d.rounded_rectangle((x0, y0, x1, y1), radius=min(r, (x1 - x0) / 2, (y1 - y0) / 2), fill=f, outline=o, width=w)
def line(d, x0, y0, x1, y1, fill=G3, w=2, alpha=1.0): d.line((x0, y0, x1, y1), fill=mix(fill, alpha) if alpha < 1 else fill, width=w)
def arrow(d, x0, y0, x1, y1, fill=ACC, w=3, alpha=1.0):
    c = mix(fill, alpha) if alpha < 1 else fill; d.line((x0, y0, x1, y1), fill=c, width=w)
    a = math.atan2(y1 - y0, x1 - x0); s = 12
    d.polygon([(x1, y1), (x1 - s * math.cos(a - .45), y1 - s * math.sin(a - .45)), (x1 - s * math.cos(a + .45), y1 - s * math.sin(a + .45))], fill=c)
def commas(n): return f'{int(n):,}'

def render(name, outdir, scene, seconds, fps=30, gif_fps=20, gif_w=960, keep=False):
    tmp = os.path.join(outdir, '_frames_' + name); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    n = int(seconds * fps)
    for i in range(n):
        img = scene(i / fps)
        img.resize((OW, OH), Image.LANCZOS).save(f'{tmp}/f{i:05d}.png', optimize=False)
    mp4 = os.path.join(outdir, name + '.mp4'); gif = os.path.join(outdir, name + '.gif')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/f%05d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mp4], check=True)
    vf = f'fps={gif_fps},scale={gif_w}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/f%05d.png', '-vf', vf, '-loop', '0', gif], check=True)
    if not keep: shutil.rmtree(tmp, ignore_errors=True)
    print(name, f'{os.path.getsize(mp4) / 1e6:.2f} MB mp4, {os.path.getsize(gif) / 1e6:.2f} MB gif, {n} frames')
    return mp4, gif
def still(name, outdir, scene, t):
    img = scene(t); p = os.path.join(outdir, '_still_' + name + '.png'); img.resize((1280, 720), Image.LANCZOS).save(p); return p
