import sys, torch, json
from common import *
import ops
from ops import *
B = ops.BALLAST; dtype = torch.float32; res = {}
def run_layer(m, frame, T):
    pos = torch.zeros(T, 1, dtype=torch.long)
    with torch.no_grad(): o = m.model.layers[0](frame, position_embeddings=m.model.rotary_emb(frame, pos), attention_mask=None)
    return o[0] if isinstance(o, tuple) else o
# ---- Qwen3 per-head q_norm: norm (d=128, eps 1e-6) + streamed gain product, real activations, all 16 heads x T positions
inner, tok = load_inner('q3'); cfg = inner.config; hd = cfg.head_dim; nq = cfg.num_attention_heads
t = ids(tok); T = t.shape[1]
for L in [0, 13, 27]:
    a = inner.model.layers[L].self_attn; cap = {}
    h1 = a.q_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('q', o.reshape(T * nq, hd).clone()))
    h2 = a.q_norm.register_forward_hook(lambda m, i, o: cap.__setitem__('qn', o.reshape(T * nq, hd).clone()))
    with torch.no_grad(): inner(t)
    h1.remove(); h2.remove(); N = T * nq
    q = cap['q']; g = a.q_norm.weight.detach()
    lo = Layout(); XS = lo.alloc('XS', hd); Y = lo.alloc('Y', hd); D = lo.n
    m = make_outer2(D, ['full_attention'], 8, attn=dict(heads=1, kv=1, hd=hd), dtype=dtype, rot=hd - 2)
    lay0 = m.model.layers[0]; c = rmsnorm_front(lay0, XS, D, hd, which='in'); sa = lay0.self_attn
    with torch.no_grad():
        sa.v_proj.weight[torch.arange(hd), XS.start + torch.arange(hd)] = 1.0
        sa.o_proj.weight[Y.start + torch.arange(hd), torch.arange(hd)] = 2.0 * c
    s = rms_scale(hd, cfg.rms_norm_eps, B)
    frame = torch.zeros(N, 1, D); frame[:, 0, 0] = B; frame[:, 0, XS] = q * s
    y = run_layer(m, frame, N)[:, 0, Y]
    # next round: transparent layer multiplies by streamed gain g (product gadget); stream = [ballast, n, g, out]
    lo = Layout(); Nn = lo.alloc('N', hd); Gs = lo.alloc('G', hd); P = lo.alloc('P', hd); D = lo.n
    m2 = make_outer2(D, ['full_attention'], 2 * hd, attn=dict(heads=1, kv=1, hd=8), dtype=dtype, rot=6)
    mb = MLPBuilder(m2.model.layers[0], D, B); transparent(m2.model.layers[0], D, B, ('in',)); mb.product(Nn, Gs, P)
    fr = torch.zeros(N, 1, D); fr[:, 0, 0] = B; fr[:, 0, Nn] = y; fr[:, 0, Gs] = g[None]
    yp = run_layer(m2, fr, N)[:, 0, P]
    res[f'q3_qnorm_L{L}'] = dict(norm_only=err(y, (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps))), norm_x_gain_product=err(yp, cap['qn']))
    print('q3 qnorm', L, res[f'q3_qnorm_L{L}'], flush=True)
# ---- residual add / embedding row / copy+scale at full width (Qwen2.5 d=896): residual by slot-sharing (free), copy gadget exact
inner, tok = load_inner('q25'); cfg = inner.config; d = cfg.hidden_size
t = ids(tok); T = t.shape[1]
with torch.no_grad(): hs = inner(t, output_hidden_states=True).hidden_states
x = hs[10][0]; delta = hs[11][0] - hs[10][0]                  # real residual + real sublayer delta
lo = Layout(); A = lo.alloc('A', d); Bs = lo.alloc('B', d); Out = lo.alloc('OUT', d); D = lo.n
m = make_outer2(D, ['full_attention'], 4 * d, attn=dict(heads=1, kv=1, hd=8), dtype=dtype, rot=6)
mb = MLPBuilder(m.model.layers[0], D, B); transparent(m.model.layers[0], D, B, ('in',))
mb.copy(Bs, A, 1.0)                 # residual add: A(+=)B through the MLP identity gadget  (or free via slot-sharing)
mb.copy(A, Out, 1.0)                # (reads stream A before the add: layers are parallel) -> Out = x
fr = torch.zeros(T, 1, D); fr[:, 0, 0] = B; fr[:, 0, A] = x; fr[:, 0, Bs] = delta
o = run_layer(m, fr, T)
res['residual_add_gadget'] = err(o[:, 0, A], hs[11][0]); res['identity_copy'] = err(o[:, 0, Out], x)
print(res['residual_add_gadget'], res['identity_copy'])
# embedding lookup: token -> embedding row streamed into slot (host transport of inner weight row; verified trivially equal)
E = inner.model.embed_tokens.weight; res['embedding_row'] = err(E[t[0]], hs[0][0])
json.dump(res, open('res_misc.json', 'w'), indent=1)
