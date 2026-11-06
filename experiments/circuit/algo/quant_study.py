"""How few bits does real Qwen3-0.6B need for argmax == HF?  Integer-exact fake quantisation of every linear (incl. lm_head):
weights symmetric RTN per group, activations symmetric per token.  Also measures weight/activation zero fractions and activation bit density
(inputs to the skip-zero ideas).  The dequantised fp32 matmul equals the integer matmul times the scales (up to fp32 rounding)."""
import torch, json, time, math
import torch.nn as nn

PROMPTS = ["The capital of France is", "In 1969, humans first landed on the Moon, and", "def fibonacci(n):\n    if n < 2:\n        return n",
           "Water boils at 100 degrees Celsius at sea level because", "The quick brown fox jumps over the lazy dog. The dog", "Photosynthesis is the process by which plants",
           "Once upon a time, in a small village by the sea, there lived", "The theory of general relativity describes gravity as", "Q: What is 12 times 12?\nA:",
           "import numpy as np\nx = np.arange(10)\nprint(x.sum())", "Le chat est sur la table et le chien", "The largest planet in the solar system is"]


class QLinear(nn.Module):
    def __init__(self, lin, wb, ab, gs, stats, name):
        super().__init__()
        W = lin.weight.data.float(); out, inn = W.shape; g = inn if gs <= 0 else gs
        Wg = W.reshape(out, inn // g, g); qmax = 2 ** (wb - 1) - 1
        sc = (Wg.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
        q = torch.round(Wg / sc).clamp(-qmax - 1, qmax)
        self.register_buffer('w', (q * sc).reshape(out, inn)); self.ab, self.name, self.stats = ab, name, stats
        stats.setdefault('wzero', []).append(((q == 0).float().mean().item(), W.numel(), name))

    def forward(self, x):
        if self.ab:
            qm = 2 ** (self.ab - 1) - 1; s = (x.abs().amax(-1, keepdim=True) / qm).clamp(min=1e-12)
            q = torch.round(x / s).clamp(-qm - 1, qm); x = q * s
            self.stats.setdefault('azero', {}).setdefault(self.name, []).append((q == 0).float().mean().item())
            # bit density of |q| (sign-magnitude): average fraction of 1 bits over the ab-1 magnitude bits
            for bs in (8, 32):
                if q.shape[-1] % bs == 0: self.stats.setdefault('blk%d' % bs, {}).setdefault(self.name, []).append((q.reshape(-1, q.shape[-1] // bs, bs) == 0).all(-1).float().mean().item())
            m = q.abs().long(); bits = sum(((m >> b) & 1).float().mean().item() for b in range(self.ab - 1)) / max(self.ab - 1, 1)
            self.stats.setdefault('abit', {}).setdefault(self.name, []).append(bits)
        return nn.functional.linear(x, self.w)


def quantize(model, wb, ab, gs, head_wb=None, stats=None):
    stats = {} if stats is None else stats
    for n, m in list(model.named_modules()):
        for cn, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                full = (n + '.' + cn).strip('.'); kind = cn
                setattr(m, cn, QLinear(c, head_wb or wb if cn == 'lm_head' else wb, ab, gs, stats, kind))
    return stats


@torch.no_grad()
def evaluate(model, ref_logits, ids_list):
    agree = tot = top5 = 0; errs = []
    for ids, ref in zip(ids_list, ref_logits):
        lg = model(ids.cuda()[None]).logits[0].float()
        agree += (lg.argmax(-1) == ref.argmax(-1)).sum().item(); tot += ids.numel()
        top5 += (ref.argmax(-1)[:, None] == lg.topk(5, -1).indices).any(-1).sum().item()
        errs.append(((lg - ref).abs().amax(-1) / ref.abs().amax(-1)).cpu())
    e = torch.cat(errs); return dict(argmax_agree=agree / tot, top5_contains_ref=top5 / tot, rel_err_mean=e.mean().item(), rel_err_max=e.max().item(), positions=tot)


def run(grid=None, gs_list=(128,)):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    ids_list = [torch.tensor(tok(p)['input_ids']) for p in PROMPTS]
    out = []
    grid = grid or [(16, 0, 0, None), (8, 8, 128, None), (6, 8, 128, None), (4, 8, 128, None), (4, 8, 0, None), (4, 6, 128, None), (4, 4, 128, None), (3, 8, 128, None), (3, 6, 128, None), (3, 4, 128, None),
                    (2, 8, 128, None), (4, 8, 128, 8), (3, 8, 128, 8), (3, 6, 128, 8), (2, 8, 128, 8), (4, 5, 128, None), (5, 6, 128, None), (3, 5, 128, 6)]
    base = None
    for wb, ab, gs, hwb in grid:
        m = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32).cuda().eval()
        if base is None:
            with torch.no_grad(): base = [m(i.cuda()[None]).logits[0].float() for i in ids_list]
        if wb == 16: continue
        t = time.time(); st = quantize(m, wb, ab, gs, hwb); r = evaluate(m, base, ids_list)
        wz = sum(a * n for a, n, _ in st['wzero']) / sum(n for _, n, _ in st['wzero'])
        az = {k: sum(v) / len(v) for k, v in st.get('azero', {}).items()}; ab_ = {k: sum(v) / len(v) for k, v in st.get('abit', {}).items()}
        bz = {b: {k: sum(v) / len(v) for k, v in st.get(b, {}).items()} for b in ('blk8', 'blk32')}
        rec = dict(block_zero=bz, wb=wb, ab=ab, group=gs, head_wb=hwb, weight_zero_frac=wz, act_zero_frac=az, act_bit_density=ab_, **r)
        print('QUANT ' + json.dumps({k: v for k, v in rec.items() if k not in ('act_zero_frac', 'act_bit_density', 'block_zero')}), flush=True)
        out.append(rec); del m; torch.cuda.empty_cache()
    return dict(quant=out, prompts=len(PROMPTS))
