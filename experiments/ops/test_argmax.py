import sys, torch, json, time
from common import *
import ops
from ops import *
B = ops.BALLAST
dtype = torch.float32
name = sys.argv[1]
inner, tok = load_inner(name, dtype)
cfg = inner.config; V = cfg.vocab_size
t = ids(tok); T = t.shape[1]
with torch.no_grad(): out = inner(t)
logits = out.logits[0]                                   # (T, V) real inner logits
codes, nb = argmax_rails(V)
hd = 2 * nb; beta = 2e5
lo = Layout(); Q = lo.alloc('Q', hd); K = lo.alloc('K', hd); Vs = lo.alloc('V', hd); O = lo.alloc('O', hd); D = lo.n
m = make_outer2(D, ['full_attention'], 8, attn=dict(heads=1, kv=1, hd=hd + 2), rope_theta=1e6, rot=hd, vocab=V, dtype=dtype)
compile_attention(m.model.layers[0], D, Q, K, Vs, O, 1, 1, hd, B)
compile_argmax_head(m, D, O, nb, V, 1.0, B)
res = {}
for pos in [0, 3, 10, T - 1]:
    lg = logits[pos]
    frame = torch.zeros(1, V + 1, D); frame[0, :, 0] = B
    frame[0, :V, K.start] = lg; frame[0, V, K.start] = -1e6      # query position: its own key is masked out by a hugely negative logit (host flag)
    frame[0, :, Q.start] = beta; frame[0, :V, Vs] = codes
    t0 = time.time()
    from transformers import DynamicCache
    cache = DynamicCache(config=m.config); CH = 4096; N = V + 1
    with torch.no_grad():
        for s0 in range(0, N, CH):
            e0 = min(N, s0 + CH)
            ob = m(inputs_embeds=frame[:, s0:e0], position_ids=torch.zeros(1, e0 - s0, dtype=torch.long), past_key_values=cache, logits_to_keep=1, use_cache=True)
        o = ob.logits[0, -1]
    top2 = lg.topk(2)
    r = dict(pred=o.argmax().item(), ref=lg.argmax().item(), margin=(top2.values[0] - top2.values[1]).item(), secs=time.time() - t0, head_logit_top2=o.topk(2).values.tolist())
    res[pos] = r; print(name, pos, r, flush=True)
json.dump(res, open(f'res_argmax_{name}.json', 'w'), indent=1)
