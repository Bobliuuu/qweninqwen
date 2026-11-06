"""Generate assets/subspace_energy.svg (real data from assets/trace.json) and assets/architecture.svg."""
import json, math
d = json.load(open('assets/trace.json')); types = d['types']
def chart(run, x0, title):
    s = f'<text x="{x0}" y="30" fill="#e6edf3" font-size="15" font-family="monospace">{title}</text>'
    for i, (eo, ei) in enumerate(run['layer_energy']):
        y = 50 + i * 9
        c = '#3fb950' if types[i] == 'full_attention' else '#bc8cff'
        s += f'<rect x="{x0}" y="{y}" width="5" height="7" fill="{c}"/>'
        for off, e, colr in ((20, eo, '#58a6ff'), (190, ei, '#ffa657')):
            w = 0 if e <= 0 else max(2, 150 * min(1, math.log10(1 + e) / 3.1))
            s += f'<rect x="{x0+off}" y="{y}" width="150" height="7" fill="#1e242c"/><rect x="{x0+off}" y="{y}" width="{w:.1f}" height="7" fill="{colr}"/>'
    return s
svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 540" width="800"><rect width="800" height="540" fill="#0d1117"/>'
svg += chart(d['runs'][0], 20, 'inner token: Qwen3-0.6B active') + chart(d['runs'][1], 420, 'outer token: Qwen3.5-0.8B active')
svg += '<text x="40" y="528" fill="#58a6ff" font-size="12" font-family="monospace">blue = outer dims</text><text x="200" y="528" fill="#ffa657" font-size="12" font-family="monospace">orange = inner dims (log norm after each layer)</text></svg>'
open('assets/subspace_energy.svg', 'w').write(svg)
a = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 420" width="900" font-family="monospace">
<rect width="900" height="420" fill="#0d1117"/>
<style>.t{fill:#e6edf3;font-size:14px}.d{fill:#7d8590;font-size:12px}
@keyframes flow{0%{transform:translateY(0);opacity:0}8%{opacity:1}92%{opacity:1}100%{transform:translateY(235px);opacity:0}}
.po{animation:flow 3s linear infinite}.pi{animation:flow 3s linear infinite;animation-delay:1.5s}
@keyframes glowO{0%,100%{opacity:.35}25%{opacity:1}}@keyframes glowI{0%,100%{opacity:.35}75%{opacity:1}}
.go{animation:glowO 6s ease-in-out infinite}.gi{animation:glowI 6s ease-in-out infinite}</style>
<text x="450" y="28" text-anchor="middle" class="t" style="font-size:18px">one checkpoint, two models, one residual stream</text>
<text x="60" y="62" class="t">hidden state (2080 dims)</text>
<rect x="60" y="75" width="200" height="40" rx="4" fill="#1e242c" stroke="#30363d"/>
<rect x="60" y="75" width="98" height="40" rx="4" fill="#58a6ff" class="go"/><text x="109" y="100" text-anchor="middle" fill="#0d1117" font-size="12">outer 1024</text>
<rect x="162" y="75" width="98" height="40" rx="4" fill="#ffa657" class="gi"/><text x="211" y="100" text-anchor="middle" fill="#0d1117" font-size="12">inner 1024</text>
<text x="60" y="135" class="d">+ 1 flag dim (2^-10, inner tokens only)</text>
<rect x="320" y="70" width="520" height="250" rx="8" fill="#161b22" stroke="#30363d"/>
<text x="340" y="92" class="t">layers 0-23: real Qwen3.5-0.8B (18 GDN + 6 attn)</text>
<text x="340" y="112" class="d">inner subspace = exact zeros here</text>
<rect x="340" y="125" width="480" height="70" rx="4" fill="#58a6ff" opacity=".22"/>
<text x="340" y="210" class="t">layers 24-51: Qwen3-0.6B, 28 appended full-attention layers</text>
<text x="340" y="230" class="d">outer subspace = exact zeros here</text>
<rect x="340" y="243" width="480" height="62" rx="4" fill="#ffa657" opacity=".22"/>
<circle class="po" cx="360" cy="75" r="7" fill="#58a6ff"/><circle class="pi" cx="800" cy="75" r="7" fill="#ffa657"/>
<text x="60" y="190" class="t">tricks (weights only):</text>
<text x="60" y="212" class="d">- block-diagonal q/k/v/o + MLP</text><text x="60" y="230" class="d">- shared RoPE grid 10^(-k/32)</text>
<text x="60" y="248" class="d">- norm width + eps rescale</text><text x="60" y="266" class="d">- attn gate saturated by flag dim</text>
<text x="60" y="284" class="d">- lm_head: outer rows read -flag</text>
<text x="450" y="350" text-anchor="middle" class="t" style="fill:#58a6ff">tokens 0..248319 &#8594; Qwen3.5-0.8B</text>
<text x="450" y="374" text-anchor="middle" class="t" style="fill:#ffa657">tokens 248320+ &#8594; Qwen3-0.6B</text>
<text x="450" y="400" text-anchor="middle" class="d">stock Qwen3_5ForCausalLM code, no patches</text></svg>'''
open('assets/architecture.svg', 'w').write(a)
