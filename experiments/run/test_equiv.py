"""layer-major (sparse write-frame) executor vs full stock forward on the vslice toy: same frames, compare control-frame logits."""
import os
import sys, os, time, torch
sys.path.insert(0, os.path.expanduser('~/qweninqwen-work/trueqiq')); sys.path.insert(0, os.path.expanduser('~/qweninqwen-work/trueqiq_run'))
from vslice import *
from toy_test import make_toy, weights
from lm_runner import LayerMajor
c = Cfg(); toy = make_toy(c); Wd, gains = weights(toy, c)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m, S = build_outer(c); m = m.eval().to(dev)
emb = lambda t: toy.model.embed_tokens.weight[t].double()
pr = Prog(S, c, m); init_steps(pr); token_program(pr, 5, 0, Wd, gains, emb)
fr = torch.stack(pr.frames); print('frames', fr.shape, flush=True)
t0 = time.time(); cache = DynamicCache(config=m.config)
lmx = LayerMajor(S, c, S.plan, dev=dev); ctrl = lmx.frame_banks(fr.float()) < 0
st = {}
hooks = [ly.register_forward_hook((lambda i: lambda mod, inp, out: st.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0, ctrl.to(dev)].cpu()))(i)) for i, ly in enumerate(m.model.layers)]
ref, _ = run_frames(m, fr, pr.pos, cache, 0, dev=dev, seg=100000)
for h in hooks: h.remove()
print('stock', time.time() - t0, flush=True)
t0 = time.time(); lm = LayerMajor(S, c, S.plan, dev=dev); tr = {}; H = lm.run(fr, pr.pos, trace=tr)
for i in range(len(S.plan)):
    if i in st:
        d = (tr[i] - st[i]).abs().max().item(); print('layer', i, S.plan[i], 'maxdiff vs stock', d, 'scale', st[i].abs().max().item())
        if d > 1e-2 * max(1, st[i].abs().max().item()): break
x = H[-1:].to(dev); nm = m.model.norm; logits = m.lm_head(nm(x))[0]
print('layer-major', time.time() - t0, lm.stats)
r = ref[-1]; logits = logits.cpu(); r = r.cpu(); print('max abs diff', (logits - r).abs().max().item(), 'max|ref|', r.abs().max().item(), 'argmax', int(logits.argmax()), int(r.argmax()))
hf = toy(torch.tensor([[5]])).logits[0, -1]; print('vs HF toy maxrel', ((logits - hf).abs().max() / hf.abs().max()).item())
