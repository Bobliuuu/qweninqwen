"""Second grid: split weight/activation precision by matrix class.  usage: quant_sim2.py"""
import json, torch, torch.nn as nn, importlib.util
import quant_sim as Q
m, orig, seqs, refs, fq_w, fq_a = Q.m, Q.orig, Q.seqs, Q.refs, Q.fq_w, Q.fq_a
def run(label, wb_attn, wb_mlp, ab, ab_down=None, ab_o=None, head_wb=16, head_ab=None, group=None, wb_down=None):
    ab_down = ab_down or ab; ab_o = ab_o or ab; head_ab = head_ab or ab; wb_down = wb_down or wb_mlp
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(orig[n])
        hs = []
        for n, mod in list(m.named_modules()):
            if isinstance(mod, nn.Linear) and n != 'lm_head' and 'layers.' in n:
                wb = wb_attn if 'self_attn' in n else (wb_down if 'down_proj' in n else wb_mlp)
                a = ab_down if 'down_proj' in n else (ab_o if 'o_proj' in n else ab)
                mod.weight.copy_(fq_w(orig[n + '.weight'], wb, group))
                hs.append(mod.register_forward_pre_hook(lambda mo, x, b=a: (fq_a(x[0], b),)))
        m.lm_head.weight.copy_(fq_w(orig['model.embed_tokens.weight'], head_wb, None))
        hs.append(m.lm_head.register_forward_pre_hook(lambda mo, x: (fq_a(x[0], head_ab),)))
        ok = tot = ok0 = okh = toth = 0
        for i, s in enumerate(seqs):
            eq = m(torch.tensor([s])).logits[0].argmax(-1) == refs[i]
            if i < 12: ok += int(eq.sum()); tot += len(s)
            else: okh, toth = int(eq.sum()), len(s)
            ok0 += int(eq[0])
        for h in hs: h.remove()
    r = dict(label=label, diverse_pct=round(100 * ok / tot, 1), held_pct=round(100 * okh / toth, 1), pos0_ok=f'{ok0}/13'); print(json.dumps(r), flush=True); return r
if __name__ == '__main__':
    out = [run('W12 A16 (ref point)', 12, 12, 16), run('W12 A12', 12, 12, 12), run('W12 A10', 12, 12, 10),
           run('W12 A8, down_proj A16', 12, 12, 8, ab_down=16), run('W12 A8, down+o A16', 12, 12, 8, ab_down=16, ab_o=16), run('W12 A10 down A16', 12, 12, 10, ab_down=16),
           run('W10 A16', 10, 10, 16), run('W11 A16', 11, 11, 16), run('W9 A16', 9, 9, 16),
           run('W8 attn, W12 mlp, A16', 8, 12, 16), run('W12 attn, W8 mlp, A16', 12, 8, 16), run('W12 gate/up,attn; W8 down; A16', 12, 12, 16, wb_down=8),
           run('W12 A16 head W8', 12, 12, 16, head_wb=8), run('W12 A16 head W12', 12, 12, 16, head_wb=12),
           run('W8 g32 A16', 8, 8, 16, group=32), run('W6 g32 A16', 6, 6, 16, group=32), run('W4 g32 A16', 4, 4, 16, group=32)]
    json.dump(out, open('res_quant_sim2.json', 'w'), indent=1)
