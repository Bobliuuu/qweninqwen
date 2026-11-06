"""Real-dim (Qwen3-0.6B: 16q/8kv x128, theta 1e6, per-head q/k-norm with streamed gain) test of vslice's N2 (GDN gated-norm heads + gain product) + attention layer."""
import sys, torch, json, math
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'qweninqwen'))
from common import *
import vslice
from vslice import *
from machine import Layout
name = sys.argv[1] if len(sys.argv) > 1 else 'q3'
inner, tok = load_inner(name); cfg = inner.config
nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // nq
assert name == 'q3'
class C(Cfg): pass
c = C(); c.d = cfg.hidden_size; c.nq = nq; c.nkv = nkv; c.hd = hd; c.inter = cfg.intermediate_size; c.L = 1; c.V = 8
c.theta = 1e6; c.H = nq + nkv; c.dk = 128; c.dv = 288; c.I_MLP = 4096; c.O = c.dv - 2
t = ids(tok); T = t.shape[1]
lay = Layout(); a = lay.alloc; S = Slots(); S.lay = lay
S.QKVR = a('QKVR', (nq + 2 * nkv) * hd); S.QN = a('QN', nq * hd); S.KN = a('KN', nkv * hd); S.GQ = a('GQ', nq * hd); S.GK = a('GK', nkv * hd)
S.QG = a('QG', nq * hd); S.KG = a('KG', nkv * hd); S.AO = [a('AO0', nq * hd)]; S.MASK = [a('MASK0', 1)]
D = lay.n
m = make_model(D, ['linear_attention', 'full_attention'], c)
heads = [dict(src=slice(S.QKVR.start + h * hd, S.QKVR.start + (h + 1) * hd), dst=slice(S.QN.start + h * hd, S.QN.start + (h + 1) * hd)) for h in range(nq)]
for g in range(nkv):
    o = S.QKVR.start + nq * hd + g * hd
    heads.append(dict(src=slice(o, o + hd), dst=slice(S.KN.start + g * hd, S.KN.start + (g + 1) * hd)))
build_norm(m.model.layers[0], D, c, S, heads)
mb = MLPBuilder(m.model.layers[0], D, ballast=B); mb.product(S.QN, S.GQ, S.QG); mb.product(S.KN, S.GK, S.KG)
build_attention(m.model.layers[1], D, c, S, 0)
res = {}
for L in [0, 13, 27]:
    a_ = inner.model.layers[L].self_attn; cap = {}
    hk = [a_.q_proj.register_forward_hook(lambda mm, i, o: cap.__setitem__('q', o.reshape(T, -1).clone())),
          a_.k_proj.register_forward_hook(lambda mm, i, o: cap.__setitem__('k', o.reshape(T, -1).clone())),
          a_.v_proj.register_forward_hook(lambda mm, i, o: cap.__setitem__('v', o.reshape(T, -1).clone())),
          a_.o_proj.register_forward_pre_hook(lambda mm, i: cap.__setitem__('o', i[0].reshape(T, -1).clone()))]
    with torch.no_grad(): inner(t)
    for h in hk: h.remove()
    fr = torch.zeros(1, T, D); fr[0, :, 0] = B
    fr[0, :, S.QKVR.start:S.QKVR.start + (nq + 2 * nkv) * hd] = torch.cat([cap['q'], cap['k'], cap['v']], -1)
    fr[0, :, S.GQ] = a_.q_norm.weight.detach().repeat(nq)[None]; fr[0, :, S.GK] = a_.k_norm.weight.detach().repeat(nkv)[None]
    # full-attn mask dim: attention frames MASK=0 (default 0 here)
    from transformers import DynamicCache
    pos = torch.arange(T)[None]
    # run layers manually, causal mask for full-attn layer
    from transformers.masking_utils import create_causal_mask
    with torch.no_grad():
        o = m.model.layers[0](fr, position_ids=pos) if False else None
        cache = DynamicCache(config=m.config)
        out = m.model(inputs_embeds=fr, position_ids=pos, past_key_values=cache, use_cache=True, output_hidden_states=True)
        # hidden_states[2] is post final norm; final norm weight zero => (1+0)*rms-normalized: recompute with hooks instead
    hs = out.hidden_states
    # capture pre-final-norm: hook on layers[1]
    store = {}
    h = m.model.layers[1].register_forward_hook(lambda mm, i, o: store.__setitem__('o', o[0] if isinstance(o, tuple) else o))
    with torch.no_grad(): m.model(inputs_embeds=fr, position_ids=pos, past_key_values=DynamicCache(config=m.config), use_cache=True)
    h.remove()
    e = err(store['o'][0, :, S.AO[0]], cap['o'])
    res[f'L{L}'] = e; print(name, L, e, flush=True)
json.dump(res, open('res_vslice_real.json', 'w'), indent=1)
