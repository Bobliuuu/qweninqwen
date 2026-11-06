import sys, torch, json
from common import *
from ops import *
import ops
ops_B = ops.BALLAST
dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[sys.argv[1]]
B = ops_B
res = {}
for name in ['q25', 'q3']:
    inner, tok = load_inner(name, dtype)
    cfg = inner.config; d = cfg.hidden_size; nq = cfg.num_attention_heads; nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', None) or d // nq; theta = cfg.rope_parameters['rope_theta'] if hasattr(cfg, 'rope_parameters') and cfg.rope_parameters else cfg.rope_theta
    t = ids(tok); T = t.shape[1]
    cap = {}
    def mk(L):
        a = inner.model.layers[L].self_attn
        hooks = []
        qsrc, ksrc = (a.q_norm, a.k_norm) if name == 'q3' else (a.q_proj, a.k_proj)
        hooks.append(qsrc.register_forward_hook(lambda m, i, o: cap.__setitem__('q', o.reshape(T, -1))))
        hooks.append(ksrc.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o.reshape(T, -1))))
        hooks.append(a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o.reshape(T, -1))))
        hooks.append(a.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__('o', i[0].reshape(T, -1))))
        return hooks
    r = res[name] = {}
    for L in [0, 7, 15, cfg.num_hidden_layers - 1]:
        hk = mk(L)
        with torch.no_grad(): inner(t)
        for h in hk: h.remove()
        lo = Layout(); Q = lo.alloc('Q', nq * hd); K = lo.alloc('K', nkv * hd); V = lo.alloc('V', nkv * hd); O = lo.alloc('O', nq * hd); D = lo.n
        g = attn_config(nq, nkv, hd, theta)
        m = make_outer2(D, ['full_attention'], 8, dtype=dtype, **g)
        compile_attention(m.model.layers[0], D, Q, K, V, O, nq, nkv, hd, B)
        frame = torch.zeros(1, T, D, dtype=dtype); frame[0, :, 0] = B
        frame[0, :, Q] = cap['q']; frame[0, :, K] = cap['k']; frame[0, :, V] = cap['v']
        pos = torch.arange(T)[None]
        with torch.no_grad(): o = m.model.layers[0](frame, position_embeddings=m.model.rotary_emb(frame, pos), attention_mask=None) if False else None
        with torch.no_grad(): o = m.model(inputs_embeds=frame, position_ids=pos, output_hidden_states=True).hidden_states
        # hidden_states[-1] is post final norm for the last layer; use 2-layer approach instead -> handled below
        r[f'attn_L{L}'] = None
        # run layer directly with causal mask through the model internals
        lay = m.model.layers[0]
        from transformers.masking_utils import create_causal_mask
        mask = create_causal_mask(config=m.config, inputs_embeds=frame, attention_mask=None, past_key_values=None, position_ids=pos)
        with torch.no_grad(): ol = lay(frame, position_embeddings=m.model.rotary_emb(frame, pos), attention_mask=mask, position_ids=pos)
        if isinstance(ol, tuple): ol = ol[0]
        e = err(ol[0, :, O].float(), cap['o'].float())
        r[f'attn_L{L}'] = e
        print(name, L, T, e, flush=True)
        del m
json.dump(res, open(f'res_attn_{sys.argv[1]}.json', 'w'), indent=1)
