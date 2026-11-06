import os
import sys, torch, json
from common import *
import ops
from ops import *
dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[sys.argv[1]]
B = float(sys.argv[2]) if len(sys.argv) > 2 else 1e8
res = {}
for name in ['q25', 'q3']:
    inner, tok = load_inner(name, dtype)
    cfg = inner.config; d = cfg.hidden_size; I = cfg.intermediate_size; eps = cfg.rms_norm_eps
    t = ids(tok); T = t.shape[1]
    with torch.no_grad(): out = inner(t, output_hidden_states=True)
    hs = out.hidden_states
    r = res[name] = {}
    for L in ([0, 5, 12, cfg.num_hidden_layers - 1]):
        lay = inner.model.layers[L]
        x = hs[L][0]                                    # (T,d) residual into layer L
        # ---------- (1) RMSNorm (no gain; gain folded into consumer)
        with torch.no_grad():
            xf = x.float(); ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)   # inner RMSNorm w/o gain (fp32 ref)
            ref_gain = lay.input_layernorm(x)           # HF inner incl gain (in dtype)
        lo = Layout(); XS = lo.alloc('XS', d); Y = lo.alloc('Y', d); D = lo.n
        m = make_outer2(D, ['full_attention'] * 1, 2 * d + 8, attn=dict(heads=1, kv=1, hd=8), dtype=dtype, rot=6)
        lay0 = m.model.layers[0]
        c = rmsnorm_front(lay0, XS, D, d, which='in')     # consumer = full-attn v_proj (linear reader of the normalised slot), 1 token/sequence
        sa = lay0.self_attn; hdo = 8
        with torch.no_grad():
            i = torch.arange(min(d, hdo))
        # v_proj rows (kv=1, hd=8): head dim j reads XS[j]; use d//8 repeated passes? -> use a wide head instead
        m = make_outer2(D, ['full_attention'], 8, attn=dict(heads=1, kv=1, hd=d), dtype=dtype, rot=d - 2)
        lay0 = m.model.layers[0]; c = rmsnorm_front(lay0, XS, D, d, which='in'); sa = lay0.self_attn
        with torch.no_grad():
            sa.v_proj.weight[torch.arange(d), XS.start + torch.arange(d)] = 1.0
            sa.o_proj.weight[Y.start + torch.arange(d), torch.arange(d)] = 2.0 * c      # gate=sigmoid(0)=1/2
        s = rms_scale(d, eps, B)
        frame = torch.zeros(T, 1, D, dtype=dtype); frame[:, 0, 0] = B; frame[:, 0, XS] = (x.float() * s).to(dtype)
        with torch.no_grad(): o = m.model.layers[0](frame, position_embeddings=m.model.rotary_emb(frame, torch.zeros(T, 1, dtype=torch.long)), attention_mask=None)
        if isinstance(o, tuple): o = o[0]
        y = o[:, 0, Y]
        r[f'rmsnorm_L{L}'] = err(y.float(), ref) | dict(ref_in=err(ref.to(dtype).float(), ref))
        # gain folded check: (y*g) vs HF
        g = lay.input_layernorm.weight.float()
        r[f'rmsnorm_gain_folded_L{L}'] = err((y.float() * g).to(dtype).float(), ref_gain.float())
        # ---------- (2) SwiGLU combine on real gate/up
        hn = lay.post_attention_layernorm(hs[L + 1][0] if False else x)   # any real normalized activation
        with torch.no_grad():
            gt = lay.mlp.gate_proj(hn); up = lay.mlp.up_proj(hn); refm = lay.mlp.act_fn(gt.float()) * up.float()
        lo = Layout(); G = lo.alloc('G', I); U = lo.alloc('U', I); M = lo.alloc('M', I); D = lo.n
        m = make_outer2(D, ['full_attention'], I, attn=dict(heads=1, kv=1, hd=8), dtype=dtype, rot=6)
        lay0 = m.model.layers[0]; transparent(lay0, D, B, ('in',))
        mb = MLPBuilder(lay0, D, B); mb.swiglu(G, U, M)
        frame = torch.zeros(1, T, D, dtype=dtype); frame[0, :, 0] = B; frame[0, :, G] = gt; frame[0, :, U] = up
        pe = m.model.rotary_emb(frame, torch.arange(T)[None])
        with torch.no_grad(): o = m.model.layers[0](frame, position_embeddings=pe, attention_mask=None)
        if isinstance(o, tuple): o = o[0]
        r[f'swiglu_L{L}'] = err(o[0, :, M].float(), refm) | dict(ref_in=err(refm.to(dtype).float(), refm), n_neurons=mb.n)
        print(name, L, json.dumps({k: v['max_rel_to_maxabs'] for k, v in r.items() if k.endswith(f'L{L}')}), flush=True)
        del m
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'res_norm_swiglu_{sys.argv[1]}_{B:g}.json'), 'w'), indent=1)
