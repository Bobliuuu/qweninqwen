"""Fake-quant simulation (GPU, HF torch) of weight/activation precision knobs; top-1 agreement vs fp32 HF on algo's 12 prompts + held-out passage,
position 0 included.  Finds the cheapest scheme reaching >=99%.  usage: quant_sim.py  (run under flock on the GPU box)"""
import sys, json, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, '.'); 
import importlib.util
spec = importlib.util.spec_from_file_location('fp', 'fidelity_prompts_data.py'); fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
name = 'Qwen/Qwen3-0.6B'; tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
seqs = [tok(p).input_ids for p in fp.PROMPTS] + [tok(fp.HELD).input_ids[:128]]
with torch.no_grad(): refs = [m(torch.tensor([s])).logits[0].argmax(-1) for s in seqs]
orig = {n: p.detach().clone() for n, p in m.named_parameters()}
def fq_w(W, bits, group):
    if bits >= 16: return W
    qmax = 2 ** (bits - 1) - 1
    if group:
        R, C = W.shape; Wg = W.reshape(R, C // group, group); s = Wg.abs().amax(-1, keepdim=True) / qmax
        return (torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s).reshape(R, C)
    s = W.abs().amax(1, keepdim=True) / qmax; return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s
def fq_a(x, bits):
    if bits >= 16 and bits != 16: return x
    if bits >= 32: return x
    qmax = 2 ** (bits - 1) - 1; s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
    return torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
class QL(nn.Module):
    def __init__(self, lin, abits): super().__init__(); self.lin = lin; self.abits = abits
    def forward(self, x): return nn.functional.linear(fq_a(x, self.abits), self.lin.weight)
def run(label, wb, ab, group=None, keep=(), head_wb=None, head_ab=None, lastmlp_fp=False, sinkfp=False):
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(orig[n])
        for n, mod in m.named_modules():
            if isinstance(mod, QL): pass
        hs = []
        for n, mod in list(m.named_modules()):
            if isinstance(mod, nn.Linear) and n != 'lm_head' and 'layers.' in n:
                lay = int(n.split('layers.')[1].split('.')[0])
                fpk = (lastmlp_fp and lay == 27 and 'mlp' in n) or any(k in n for k in keep)
                w = mod.weight; w.copy_(w if fpk else fq_w(orig[n + '.weight'], wb, group)); b = 32 if fpk else ab
                hs.append(mod.register_forward_pre_hook(lambda mo, a, b=b: (fq_a(a[0], b),)))
        hw = head_wb or wb; hb = head_ab or ab
        m.lm_head.weight.copy_(fq_w(orig['lm_head.weight'] if 'lm_head.weight' in orig else orig['model.embed_tokens.weight'], hw, None))
        hs.append(m.lm_head.register_forward_pre_hook(lambda mo, a: (fq_a(a[0], hb),)))
        ok = tot = 0; ok0 = 0; okh = toth = 0
        for i, s in enumerate(seqs):
            am = m(torch.tensor([s])).logits[0].argmax(-1); eq = (am == refs[i])
            if i < 12: ok += int(eq.sum()); tot += len(s)
            else: okh, toth = int(eq.sum()), len(s)
            ok0 += int(eq[0])
        for h in hs: h.remove()
    r = dict(label=label, wb=wb, ab=ab, group=group, diverse=f'{ok}/{tot}', diverse_pct=round(100 * ok / tot, 1), held=f'{okh}/{toth}', held_pct=round(100 * okh / toth, 1), pos0_ok=f'{ok0}/13')
    print(json.dumps(r), flush=True); return r
if __name__ == "__main__":
    grid = [('W16 A16(fp)', 16, 32), ('W8 row, A8', 8, 8), ('W8 row, A16', 8, 16), ('W8 row, A12', 8, 12), ('W16 A8 (act only)', 16, 8), ('W12 row, A16', 12, 16), ('W12 row, A8', 12, 8),
            ('W8 g64, A16', 8, 16, 64), ('W8 g128, A16', 8, 16, 128), ('W6 row, A16', 6, 16), ('W4 g64, A16', 4, 16, 64), ('W4 row, A16', 4, 16), ('W4 g32, A16', 4, 16, 32), ('W5 row A16', 5, 16), ('W6 g64, A16', 6, 16, 64)]
    out = [run(*(g if len(g) > 3 else (*g, None))[:1], *[None] * 0, wb=g[1], ab=g[2], group=(g[3] if len(g) > 3 else None)) for g in grid]
    out.append(run('W8 row A16, last-layer MLP fp', 8, 16, lastmlp_fp=True)); out.append(run('W8 row A16, lm_head W16', 8, 16, head_wb=16))
    json.dump(out, open('res_quant_sim.json', 'w'), indent=1)
