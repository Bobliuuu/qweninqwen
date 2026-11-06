import sys, torch, json
from common import *
import ops
from ops import *
B = ops.BALLAST; res = {}
for name in ['q25', 'q3']:
    inner, tok = load_inner(name); cfg = inner.config; d = cfg.hidden_size; eps = cfg.rms_norm_eps
    t = ids(tok); T = t.shape[1]
    with torch.no_grad(): hs = inner(t, output_hidden_states=True).hidden_states
    lo = Layout(); X = lo.alloc('X', d); OUT = lo.alloc('OUT', d)
    sc = {n: lo.alloc(n, 1).start for n in ('MS', 'R0', 'QQ', 'TT', 'ZZ')}
    class P: pass
    scal = {n: slice(v, v + 1) for n, v in sc.items()}; D = lo.n
    m = make_outer2(D, ['full_attention'] * 6, 4096, attn=dict(heads=1, kv=1, hd=8), rot=6)
    unit = RMSNormUnit([m.model.layers[i] for i in range(6)], D, X, OUT, scal, d, eps, B)
    print(name, 'neurons per layer', unit.neurons, flush=True)
    def run(x):
        n = x.shape[0]; fr = torch.zeros(n, 1, D); fr[:, 0, 0] = B; fr[:, 0, X] = x
        pos = torch.zeros(n, 1, dtype=torch.long); pe = m.model.rotary_emb(fr, pos)
        with torch.no_grad():
            for l in m.model.layers:
                o = l(fr, position_embeddings=pe, attention_mask=None); fr = o[0] if isinstance(o, tuple) else o
        return fr
    r = res[name] = {}
    for L in [0, 5, 12, cfg.num_hidden_layers - 1, cfg.num_hidden_layers]:
        x = hs[L][0].float(); xd = x.double()
        ref = (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)).float()
        fr = run(x)
        r[f'L{L}'] = err(fr[:, 0, OUT], ref) | dict(maxabs_x=x.abs().max().item(), scalars_left=fr[:, 0, [s for s in sc.values()]].abs().max().item(), other_slot_x_err=(fr[:, 0, X] - x).abs().max().item())
        print(name, L, r[f'L{L}'], flush=True)
    # dynamic range
    for sc_ in [1e-4, 1e-2, 1.0, 1e2, 1e4]:
        x = hs[10][0].float() * sc_; xd = x.double()
        ref = (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)).float()
        fr = run(x); r[f'scale{sc_:g}'] = err(fr[:, 0, OUT], ref)['max_rel_to_maxabs']
    print(name, {k: v for k, v in r.items() if k.startswith('scale')})
json.dump(res, open('res_rmsnorm_unit.json', 'w'), indent=1)
