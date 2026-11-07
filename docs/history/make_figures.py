"""Generates the hand-laid-out static figures in docs/assets/ as PNG (needs rsvg-convert on PATH; no SVG files are kept).
Run: python docs/history/make_figures.py. Animated/launch media: docs/history/launch/make_media.py (history: earlier compiled route)."""
import os, math
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--panel:#f6f8fa;
--state:#0969da;--statef:#ddf4ff;--op:#9a6700;--opf:#fff8c5;--att:#1a7f37;--attf:#dafbe1;--host:#57606a;--hostf:#eaeef2;--bad:#cf222e;--badf:#ffebe9;--acc:#8250df;--accf:#fbefff}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#3d444d;--panel:#151b23;
--state:#4493f8;--statef:#0c2d6b;--op:#d4a72c;--opf:#3b2e00;--att:#3fb950;--attf:#0f3a1b;--host:#9198a1;--hostf:#262c36;--bad:#ff7b72;--badf:#4a1a1a;--acc:#bc8cff;--accf:#2d1b4e}}
text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;fill:var(--fg);font-size:13px}
.t{font-size:20px;font-weight:600}.s{font-size:12px;fill:var(--mut)}.b{font-weight:600}.m{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.wt{fill:#fff}.c{text-anchor:middle}.r{text-anchor:end}
.bg{fill:var(--bg)}.pn{fill:var(--panel);stroke:var(--line)}
.st{fill:var(--statef);stroke:var(--state);stroke-width:1.5}.op{fill:var(--opf);stroke:var(--op);stroke-width:1.5}
.at{fill:var(--attf);stroke:var(--att);stroke-width:1.5}.ho{fill:var(--hostf);stroke:var(--host);stroke-width:1.5}
.ac{fill:var(--accf);stroke:var(--acc);stroke-width:1.5}.bd{fill:var(--badf);stroke:var(--bad);stroke-width:1.5}
.ln{stroke:var(--mut);stroke-width:1.6;fill:none}.dash{stroke-dasharray:6 5}
.flow{stroke-dasharray:7 6;animation:fl 1.2s linear infinite}@keyframes fl{to{stroke-dashoffset:-13}}
"""
def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">'
            f'<title id="t">{title}</title><desc id="d">{desc}</desc><style>{CSS}</style>'
            f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--mut)"/></marker></defs>'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>{body}</svg>')
def box(x, y, w, h, cls, rx=8): return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'
def tx(x, y, s, cls='', anchor=None):
    c = (cls + (' c' if anchor == 'c' else ' r' if anchor == 'r' else '')).strip()
    s = s.replace('&', '&amp;').replace('<', '&lt;')
    return f'<text x="{x}" y="{y}" class="{c}">{s}</text>'
def arrow(d, flow=False, extra=''): return f'<path d="{d}" class="ln{" flow" if flow else ""}" marker-end="url(#ar)" {extra}/>'
def save(name, s):
    """Rasterise to PNG with rsvg-convert (no SVG files are kept). CSS variables are resolved to the light theme first
    because rsvg does not implement var() or prefers-color-scheme."""
    import subprocess, re
    root = re.search(r':root\{([^}]*)\}', CSS).group(1)
    vals = dict(re.findall(r'--(\w+):(#[0-9a-fA-F]+)', root))
    s = re.sub(r'@media[^{]*\{:root\{[^}]*\}\}', '', s)
    s = re.sub(r'var\(--(\w+)\)', lambda m: vals[m.group(1)], s)
    s = re.sub(r'animation:[^;}]*;?', '', s)
    png = os.path.join(OUT, name.replace('.svg', '.png'))
    subprocess.run(['rsvg-convert', '-z', '2', '-b', 'white', '-o', png], input=s.encode(), check=True)

# ------------------------------------------------------------------ (a) architecture
def architecture():
    b = tx(24, 36, 'Architecture: an inner Qwen executed by an outer Qwen', 't')
    b += tx(24, 56, 'The outer model is stock Qwen3.5 code with compiled weights. The host never computes an inner activation.', 's')
    # host
    b += box(24, 90, 190, 300, 'ho')
    b += tx(119, 116, 'Host streamer', 'b', 'c')
    for i, l in enumerate(['inner weight rows', 'token ids / embeddings', 'one-hot addresses, flags', 'position ids (clock)', 'BALLAST constant', '', 'outer argmax + append']):
        if l: b += tx(119, 146 + i * 26, l, 's' if i != 6 else 'b', 'c')
    b += tx(119, 346, 'no inner matmul, norm,', 's', 'c') + tx(119, 362, 'softmax, RoPE or sampling', 's', 'c')
    # outer
    b += box(268, 78, 400, 324, 'pn') + tx(468, 102, 'Outer Qwen3.5 (stock Qwen3_5ForCausalLM, fp32)', 'b', 'c')
    layers = [('st', 'DeltaNet layers = paged weight memory', 'state S holds a tile of W; write (k, v), read q -> S q'),
              ('op', 'Compiled op layers', 'RMSNorm, SwiGLU, residual, gadgets (weights only)'),
              ('at', 'Full-attention layers', 'RoPE + softmax over the outer KV = inner KV cache'),
              ('ho', 'Outer lm_head', 'decodes the in-model argmax bit code')]
    for i, (c, t1, t2) in enumerate(layers):
        y = 118 + i * 70
        b += box(288, y, 360, 58, c) + tx(468, y + 24, t1, 'b', 'c') + tx(468, y + 44, t2, 's', 'c')
    b += arrow('M214 240 L268 240', True) + tx(241, 228, 'frames', 's', 'c')
    # return
    b += arrow('M648 372 L690 372 L690 440 L119 440 L119 392', True)
    b += tx(430, 458, 'outer token (argmax) is appended and becomes the next inner step', 's', 'c')
    # inner block
    b += box(712, 78, 264, 300, 'pn') + tx(844, 102, 'Inner Qwen decoder block', 'b', 'c') + tx(844, 120, '(the program being executed)', 's', 'c')
    ops = [('op', 'RMSNorm'), ('st', 'q, k, v projection'), ('at', 'RoPE + attention (KV)'), ('st', 'o projection, + residual'), ('op', 'RMSNorm'), ('st', 'gate, up projection'), ('op', 'SwiGLU'), ('st', 'down projection, + residual')]
    for i, (c, t) in enumerate(ops):
        y = 132 + i * 29
        b += box(732, y, 224, 23, c, 5) + tx(844, y + 16, t, 'c', 'c')
    b += tx(844, 396, 'blue: runs in DeltaNet state', 's', 'c') + tx(844, 412, 'amber: compiled op layer', 's', 'c') + tx(844, 428, 'green: outer attention layer and KV cache', 's', 'c')
    b += '<path d="M668 240 L712 240" class="ln dash" marker-end="url(#ar)"/>'
    return svg(1000, 476, b, 'Architecture', 'Host streams weights and tokens into the outer Qwen3.5 stack, which executes the inner Qwen decoder block; the outer argmax is appended as the next token.')

# ------------------------------------------------------------------ (b) DeltaNet state as weight memory (animated)
def state_memory():
    R, C = 4, 6; cx, cy, cw, ch = 330, 150, 54, 40
    T = 14.0
    def anim(t0, t1, vals='0;1', fill_to=None):
        # opacity is 0 before t0, 1 from t0 to the end of the loop
        kt = f'0;{t0/T:.4f};{min(t0/T+0.01,1):.4f};1'
        return f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
    b = tx(24, 34, 'DeltaNet state as weight memory', 't')
    b += tx(24, 54, 'Delta rule: S <- S + k (beta (v - S^T k))^T, read: out = S^T q. With a one-hot k and beta = 1, a write overwrites row r with v.', 's')
    rows = [[0.9, 0.2, 0.5, 0.7, 0.3, 0.6], [0.4, 0.8, 0.1, 0.5, 0.9, 0.2], [0.7, 0.3, 0.6, 0.2, 0.5, 0.8], [0.2, 0.6, 0.9, 0.4, 0.1, 0.7]]
    b += tx(cx + C * cw / 2, cy - 24, 'state S (key rows x value columns)', 'b', 'c')
    # grid frame
    b += box(cx - 4, cy - 4, C * cw + 8, R * ch + 8, 'pn', 4)
    for r in range(R):
        t0 = 1.0 + r * 2.0
        for c in range(C):
            w = rows[r][c]
            b += f'<rect x="{cx+c*cw+2}" y="{cy+r*ch+2}" width="{cw-4}" height="{ch-4}" rx="3" class="st" style="fill-opacity:{0.15+0.7*w:.2f}">{anim(t0, t0)}</rect>'
        # incoming weight row v and key k one-hot
        b += f'<g>{anim(t0-0.8, t0)}' + f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{(t0-0.9)/T:.4f};{(t0-0.8)/T:.4f};{(t0+0.5)/T:.4f};{(t0+0.8)/T:.4f};1" dur="{T}s" repeatCount="indefinite"/>' \
             + f'<path d="M{cx+C*cw+70} {cy+r*ch+20} L{cx+C*cw+10} {cy+r*ch+20}" class="ln" marker-end="url(#ar)"/>' \
             + tx(cx + C * cw + 76, cy + r * ch + 24, f'write: k = e{r}, v = W[{r}, :]', 'm') + '</g>'
    # read
    t0 = 9.5
    b += f'<g><animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{t0/T:.4f};{(t0+.3)/T:.4f};1" dur="{T}s" repeatCount="indefinite"/>'
    q = [0.5, -1.0, 0.25, 2.0]
    for r in range(R):
        b += tx(cx - 70, cy + r * ch + 25, f'q{r} = {q[r]:+.2f}', 'm', 'r') + f'<path d="M{cx-66} {cy+r*ch+20} L{cx-8} {cy+r*ch+20}" class="ln" marker-end="url(#ar)"/>'
    out = [sum(q[r] * rows[r][c] for r in range(R)) for c in range(C)]
    b += box(cx - 4, cy + R * ch + 24, C * cw + 8, 34, 'ac', 4)
    for c in range(C):
        b += tx(cx + c * cw + cw / 2, cy + R * ch + 46, f'{out[c]:+.2f}', 'm', 'c')
    b += tx(cx - 14, cy + R * ch + 46, 'out = S^T q', 'b', 'r') + '</g>'
    # side notes
    b += tx(24, 410, 'WRITE frames (host transport): one-hot address, weight payload, flags.', 's')
    b += tx(24, 430, 'READ frame (in-model): the activation q is produced by outer layers; the contraction S^T q happens inside the delta-rule layer.', 's')
    b += tx(24, 450, 'Per-head q, k are L2-normalised and pass a conv SiLU, v goes through a gated RMSNorm: the engine compensates with ballast channels and a', 's')
    b += tx(24, 468, 'pre-inverted SiLU payload (weights only). See experiments/state/README.md. Values above are illustrative, not data.', 's')
    return svg(1000, 492, b, 'DeltaNet state as weight memory', 'Animated: weight rows are written into a DeltaNet state matrix one key row at a time, then a query vector reads out S transposed times q.')

# ------------------------------------------------------------------ (c) unrolled pipeline in depth
def layer_timeline():
    b = tx(24, 34, 'The unrolled pipeline: one inner token in one outer forward call', 't')
    b += tx(24, 54, 'State and KV only carry data deeper (to layers at or below the layer that holds them), so each inner op gets its own outer layers, in dependency order.', 's')
    chips = [('op', 'RMSNorm', 'GDN gated-norm head'), ('st', 'W_qkv bank', 'GDN state read'), ('st', 'register', 'GDN state'), ('op', 'q/k norm', 'norm + gain product'), ('at', 'RoPE + attention', 'outer KV = inner KV'), ('st', 'register', 'GDN state'),
             ('st', 'W_o bank', 'read + residual'), ('op', 'RMSNorm', 'GDN gated-norm head'), ('st', 'W_gate, W_up bank', 'GDN state read'), ('op', 'SwiGLU', 'MLP gadget'), ('st', 'register', 'GDN state'), ('st', 'W_down bank', 'read + residual')]
    b += tx(24, 90, 'depth (outer layers)  ->', 'b') + tx(976, 90, 'real model: 1006 outer layers, D = 40775', 's', 'r')
    cw, gap, h = 140, 22, 64
    for row in range(2):
        y = 106 + row * 100
        for i in range(6):
            c, t1, t2 = chips[row * 6 + i]; x = 24 + i * (cw + gap)
            b += box(x, y, cw, h, c, 6) + tx(x + cw / 2, y + 28, t1, 'b', 'c') + tx(x + cw / 2, y + 48, t2, 's', 'c')
            if i < 5: b += arrow(f'M{x+cw+2} {y+h/2} L{x+cw+gap-2} {y+h/2}')
        if row == 0: b += arrow(f'M{24+5*(cw+gap)+cw/2} {y+h+2} L{24+5*(cw+gap)+cw/2} {y+h+14} L{24+cw/2} {y+h+14} L{24+cw/2} {y+98}')
    b += tx(24, 316, 'repeat for each inner layer, then:', 's')
    tail = [('op', 'final RMSNorm'), ('st', 'LM-head bank'), ('at', 'argmax attention'), ('ho', 'outer lm_head')]
    for i, (c, t) in enumerate(tail):
        x = 24 + i * 176
        b += box(x, 326, 150, 40, c, 6) + tx(x + 75, 351, t, 'b', 'c')
        if i < 3: b += arrow(f'M{x+152} 346 L{x+174} 346')
    b += arrow('M706 346 L760 346') + box(762, 326, 214, 40, 'ho', 6) + tx(869, 351, 'host: argmax + append', 'b', 'c')
    b += tx(24, 400, 'Weights are paged in the same call: WRITE frames precede the READ frame of each bank, one chunked prefill per inner token.', 's')
    b += tx(24, 418, 'The only deep-to-shallow channel is the emitted token: outer argmax, host append, next embedding row. Real Qwen3-0.6B: 293,068 outer frames per token. Toy: 1530. Schematic order.', 's')
    return svg(1000, 440, b, 'Unrolled pipeline', 'Each inner op occupies its own outer layers in dependency order; registers are DeltaNet layers placed between producer and consumer; one forward call per inner token.')

# ------------------------------------------------------------------ log bar helper
def logbars(title, sub, rows, xmin, xmax, unit_ticks, footer, w=1000):
    L, Rr = 330, w - 70; top = 92; bh = 30; gap = 16
    h = top + len(rows) * (bh + gap) + 92
    def X(v): return L + (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (Rr - L)
    b = tx(24, 34, title, 't') + tx(24, 54, sub, 's')
    for v, lab in unit_ticks:
        b += f'<line x1="{X(v)}" x2="{X(v)}" y1="{top-14}" y2="{h-80}" stroke="var(--line)"/>' + tx(X(v), h - 64, lab, 's', 'c')
    for i, (name, v, cls, note, hatched) in enumerate(rows):
        y = top + i * (bh + gap)
        b += tx(L - 12, y + 19, name, 'b', 'r')
        fill = {'st': 'var(--state)', 'bd': 'var(--bad)', 'at': 'var(--att)', 'ho': 'var(--host)', 'ac': 'var(--acc)', 'op': 'var(--op)'}[cls]
        b += f'<rect x="{L}" y="{y}" width="{X(v)-L:.1f}" height="{bh}" rx="4" fill="{fill}" fill-opacity="{0.35 if hatched else 0.9}" stroke="{fill}" stroke-width="1.5"{" stroke-dasharray=\"5 4\"" if hatched else ""}/>'
        lab = (f'{v:,.0f} s' if v >= 10 else f'{v:.3g} s') + '  ' + note
        if X(v) + 8 + 7.2 * len(lab) < w - 10: b += tx(X(v) + 8, y + 19, lab, 'm')
        else: b += tx(L + 10, y + 19, lab, 'm wt')
    b += tx(L, h - 40, 'seconds per inner token (log scale)  -  solid = measured, translucent dashed = estimate / extrapolation', 's')
    for i, f in enumerate(footer): b += tx(24, h - 22 + i * 16, f, 's')
    return svg(w, h + 16 * (len(footer) - 1), b, title, sub), h

def cost_model():
    rows = [('full token, real Qwen3-0.6B (measured)', 268.6, 'st', 'shared A100, 293,068 frames, 1006 layers', False),
            ('paging bound (design estimate)', 110, 'ho', '2048 floats/step; untested', True),
            ('weight streaming only, nL=4 bank (measured)', 41.95, 'st', 'A10G, 519,430 frames', False),
            ('weight streaming only, nL=2 (extrapolated)', 19, 'st', 'plan() estimate', True),
            ('weight streaming only, nL=1 (extrapolated)', 7, 'st', 'plan() estimate', True)]
    s, _ = logbars('Cost model: Qwen3-0.6B inside the outer model, seconds per token',
        'Top row: one full token incl. all norms, attention, LM head and argmax (single position). Other rows: weight streaming alone through a shallow bank (experiments/run/).',
        rows, 1, 1000, [(1, '1 s'), (3, '3 s'), (10, '10 s'), (30, '30 s'), (100, '100 s'), (300, '300 s'), (1000, '1000 s')],
        ['Full token: 13.9 s layer building + ~242 s running, torch-reference GDN, fp32, untuned, shared GPU (experiments/vslice/FULL_TOKEN_RESULT.md).',
         'Streaming-only rows use harness-fed random activations and a shallow stack: they do not add up to, or compare directly with, the full-token row.'])
    save('cost_model.svg', s)

def comparison():
    rows = [('circuit route, real 896-row (measured)', 494e6 / 14.71, 'bd', '14.7 MAC/s, 389 days', False),
            ('circuit route, best synthetic (measured)', 494e6 / 488.6, 'bd', '488 MAC/s, 11.7 days', False),
            ('state route, full token (measured)', 268.6, 'st', 'real Qwen3-0.6B, 1 position, A100', False),
            ('state route, matvecs only (measured)', 41.95, 'st', 'A10G, 596M weights', False),
            ('MLP-multiplier, MACs only (estimate)', 0.64, 'ac', '0.77 GMAC/s A100; host copies x (flagged)', True)]
    s, _ = logbars('Circuit route vs state-as-memory route (seconds per inner token)',
        'Same inner model size (~0.5-0.6 B weights). Circuit: bit-level arithmetic in GDN state (reference/). State route: weights written into the state, contraction done by the delta-rule read.',
        rows, 0.3, 1e8, [(1, '1 s'), (60, '1 min'), (3600, '1 h'), (86400, '1 day'), (86400 * 30, '30 d'), (86400 * 365, '1 yr')],
        ['Circuit rows extrapolate measured MAC/s (reference/nested-qwen-lab, experiments/bench/circuit_baseline) to 494M MACs; they exclude all non-matvec ops.',
         'State rows: full token incl. all ops (experiments/vslice) and weight streaming alone. MLP-multiplier row: estimate (experiments/alt/FINDINGS.md).'])
    save('comparison.svg', s)

def lm_head():
    b = tx(24, 34, 'Full-vocabulary LM head and argmax, in-model', 't')
    b += tx(24, 54, 'Logits for 151,936 tokens are produced in 594 blocks of 256; the argmax is a two-level hard max done by outer layers.', 's')
    xs = [24, 262, 500, 738]; w = 214; y = 96; h = 120
    stages = [('st', 'Blocks of logits', ['594 blocks x 256 logits', 'state-bank matvec on the', '151,936 x 1024 head weights']),
              ('op', 'Per-block tournament', ['max and local argmax', 'inside each block', 'output: best logit + 8 bits']),
              ('at', 'Hard-max attention', ['key = block best logit', 'value = 36 dual-rail bits', '(8 local + 10 block index)']),
              ('ho', 'Outer lm_head', ['36 non-zero columns', 'decode rails -> token id', 'host: argmax + append'])]
    for i, (c, t, ls) in enumerate(stages):
        b += box(xs[i], y, w, h, c) + tx(xs[i] + w / 2, y + 28, t, 'b', 'c')
        for k, l in enumerate(ls): b += tx(xs[i] + w / 2, y + 56 + k * 20, l, 's', 'c')
        if i < 3: b += arrow(f'M{xs[i]+w+2} {y+h/2} L{xs[i+1]-2} {y+h/2}', True)
    b += tx(24, 262, 'softmax over block winners is a hard max: the winning block gets weight 1 - e^-40 when the top-2 gap exceeds 1e-3, so the value read out is its code.', 's')
    b += tx(24, 282, 'Full-token run (real Qwen3-0.6B): outer argmax 15846 == Hugging Face; block-best logit error 5.8e-5 of max|logit|; winning rails clean (score 18.0007 of ideal 18).', 's')
    b += tx(24, 302, 'Exact ties blend the code and give an invalid token id; the op test notes this is practically never hit (experiments/ops/README.md).', 's')
    return svg(1000, 326, b, 'LM head in-model', 'Blocks of 256 logits, per-block max and argmax, hard-max attention over block winners, 36 dual-rail bits decoded by the outer lm_head.')

os.makedirs(OUT, exist_ok=True)
save('architecture.svg', architecture()); save('state_memory.svg', state_memory()); save('unrolled_pipeline.svg', layer_timeline())
cost_model(); comparison(); save('lm_head.svg', lm_head())
print('ok')
