"""Record real data for the README visuals: streamed tokens + per-layer subspace energy.
  python scripts/trace_demo.py CKPT OUT.json
"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from qweninqwen.merge import NO, INNER_REPO, OUTER_DIR, HO, HI

m = Qwen3_5ForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).cuda().eval()
ti, to = AutoTokenizer.from_pretrained(INNER_REPO), AutoTokenizer.from_pretrained(OUTER_DIR)
layers = m.model.layers; energy = {}
def hook(i):
    def f(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out)[0, -1].float()
        energy.setdefault(i, []).append([h[:HO].norm().item(), h[HO:HO + HI].norm().item()])
    return f
for i, l in enumerate(layers): l.register_forward_hook(hook(i))
res = {'n_layers': len(layers), 'types': m.config.layer_types, 'runs': []}
jobs = [('inner', 'Qwen3-0.6B (inner)', "The capital of France is", ti, NO),
        ('outer', 'Qwen3.5-0.8B (outer)', "The capital of France is", to, 0),
        ('inner', 'Qwen3-0.6B (inner)', "def fibonacci(n):\n    if n < 2:\n        return n\n", ti, NO)]
for mode, name, p, tok, off in jobs:
    energy.clear()
    x = tok(p, return_tensors='pt').input_ids.cuda() + off
    t0 = time.time(); ids = x; toks = []
    with torch.no_grad():
        for step in range(40):
            out = m(ids if step == 0 else ids[:, -1:], past_key_values=None if step == 0 else past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True); ids = torch.cat([ids, nxt], 1)
            toks.append(dict(t=round(time.time() - t0, 4), text=tok.decode(nxt[0] - off)))
    last = {i: e[-1] for i, e in energy.items()}  # energies at the last generated token
    res['runs'].append(dict(mode=mode, model=name, prompt=p, tokens=toks, layer_energy=[last[i] for i in range(len(layers))]))
json.dump(res, open(sys.argv[2], 'w'))
print('ok')
