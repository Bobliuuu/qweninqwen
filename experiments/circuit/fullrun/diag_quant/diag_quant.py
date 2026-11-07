"""Why does the firmware give 13 ('.') at position 0 for prompt [785] while HF fp32 gives 15846? Independent float re-implementation of the
firmware's numeric contract (compiler/gen_fx.py + compiler/fw/qwen3_fx.c), position 0 only (T=1 => softmax=1, attention output = v, q/k/RoPE are irrelevant).
Contract: weights per-row symmetric RTN (int4: qmax 7, clip[-8,7]; int8 = two int4 planes: qmax 119, clip[-120,119]); lm_head = tied embedding, quantised;
input embedding NOT quantised (Q16); dynamic per-tensor int8 activations (x*127/amax) before q/k/v, o, gate/up, down, lm_head; v int8 per kv-head (kv_store).
usage: python diag_quant.py OUT.json"""
import sys, json, torch, numpy as np
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)
name = 'Qwen/Qwen3-0.6B'; TOK = 785
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
cfg = m.config; D, NL, NH, NKV, HD = cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
W = {k: v.clone() for k, v in m.state_dict().items()}
EPS = cfg.rms_norm_eps
def rms(x, g): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * g
def wq(Wt, bits, group=None):
    if bits is None: return Wt
    qmax = 7 if bits == 4 else 119; lo, hi = -qmax - 1, qmax
    if group:
        R, C = Wt.shape; Wg = Wt.reshape(R, C // group, group)
        s = Wg.abs().amax(-1, keepdim=True) / qmax; s = torch.where(s == 0, torch.full_like(s, 1e-6), s)
        return (torch.clamp(torch.round(Wg / s), lo, hi) * s).reshape(R, C)
    s = Wt.abs().amax(1, keepdim=True) / qmax; s = torch.where(s == 0, torch.full_like(s, 1e-6), s)
    return torch.clamp(torch.round(Wt / s), lo, hi) * s
def aq(v, on, keep_outliers=0):
    if not on: return v
    v = v.clone()
    if keep_outliers:   # LLM.int8-style: top-k |v| channels stay fp, the rest int8 with their own amax
        idx = v.abs().topk(keep_outliers).indices; o = v[idx].clone(); v[idx] = 0
        a = v.abs().max().clamp_min(1e-30); q = torch.round(v * 127 / a) * a / 127; q[idx] = o; return q
    a = v.abs().max().clamp_min(1e-30); return torch.round(v * 127 / a) * a / 127
def run(wbits=None, lmbits='same', act=(), group=None, q16=False, keep=0, layers=None, vq=True, emb_q=False):
    """act: set of sites among qkv,o,gu,down,lm.  layers: if given, only these layers get quantised weights/acts (others fp)."""
    lmbits = wbits if lmbits == 'same' else lmbits
    emb = W['model.embed_tokens.weight']
    x = emb[TOK].clone()
    if emb_q: x = wq(emb, wbits)[TOK]
    if q16: x = torch.round(x * 65536) / 65536
    xs = []; stats = []
    for l in range(NL):
        p = f'model.layers.{l}.'; on = layers is None or l in layers; wb = wbits if on else None
        sel = lambda s: s if on else ()
        h = rms(x, W[p + 'input_layernorm.weight'])
        hq = aq(h, 'qkv' in act and on, keep)
        v = wq(W[p + 'self_attn.v_proj.weight'], wb, group) @ hq            # (NKV*HD)
        if vq and on and wbits is not None:                                   # kv_store: int8 per kv head
            v = torch.stack([aq(v[g * HD:(g + 1) * HD], True) for g in range(NKV)]).reshape(-1)
        ao = v.reshape(NKV, HD).repeat_interleave(NH // NKV, 0).reshape(-1)   # T=1: softmax==1
        o = wq(W[p + 'self_attn.o_proj.weight'], wb, group) @ aq(ao, 'o' in act and on, keep)
        x = x + o
        h = rms(x, W[p + 'post_attention_layernorm.weight']); hq = aq(h, 'gu' in act and on, keep)
        g_ = wq(W[p + 'mlp.gate_proj.weight'], wb, group) @ hq; u_ = wq(W[p + 'mlp.up_proj.weight'], wb, group) @ hq
        hh = torch.nn.functional.silu(g_) * u_
        d = wq(W[p + 'mlp.down_proj.weight'], wb, group) @ aq(hh, 'down' in act and on, keep)
        x = x + d
        stats.append(dict(l=l, xmax=float(x.abs().max()), dmax=float(d.abs().max()), hhmax=float(hh.abs().max()), vmax=float(v.abs().max())))
        xs.append(x.clone())
    hN = rms(x, W['model.norm.weight'])
    lm = wq(W['lm_head.weight'] if 'lm_head.weight' in W else emb, lmbits, group)
    lg = lm @ aq(hN, 'lm' in act, keep)
    return lg, xs, stats
ALL = {'qkv', 'o', 'gu', 'down', 'lm'}
ref_lg, ref_xs, ref_stats = run()
hf = m(torch.tensor([[TOK]])).logits[0, 0]
assert hf.argmax() == ref_lg.argmax() == 15846 and (hf - ref_lg).abs().max() < 1e-2, ('independent fp32 must equal HF', hf.argmax(), ref_lg.argmax(), (hf - ref_lg).abs().max())
def summ(lg, xs):
    top = lg.topk(5); srt = lg.sort(descending=True).indices
    cs = [float(torch.nn.functional.cosine_similarity(a, b, dim=0)) for a, b in zip(xs, ref_xs)]
    re = [float((a - b).norm() / b.norm()) for a, b in zip(xs, ref_xs)]
    return dict(argmax=int(lg.argmax()), top5=[int(i) for i in top.indices], gap=float(top.values[0] - top.values[1]),
                rank13=int((srt == 13).nonzero()), rank15846=int((srt == 15846).nonzero()), cos_final=cs[-1], relerr_layers=[round(r, 4) for r in re], cos_layers=[round(c, 4) for c in cs],
                first_layer_relerr_gt_0p1=next((i for i, r in enumerate(re) if r > 0.1), None))
V = {}
def add(n, **kw): lg, xs, st = run(**kw); V[n] = summ(lg, xs); print(n, V[n]['argmax'], V[n]['top5'], 'rank13', V[n]['rank13'], 'rank15846', V[n]['rank15846'], 'cosN', round(V[n]['cos_final'], 4), 'firstbad', V[n]['first_layer_relerr_gt_0p1'], flush=True)
add('fp32_independent')
add('W4_wonly', wbits=4); add('W8_wonly', wbits=8)
add('fpW_actint8_all', act=ALL)
add('FW_W4_actint8_all (firmware contract, float sim)', wbits=4, act=ALL)
add('FW_W8_actint8_all (firmware contract, float sim)', wbits=8, act=ALL)
add('FW_W4_q16_emb', wbits=4, act=ALL, q16=True)
for s in ['qkv', 'o', 'gu', 'down', 'lm']: add(f'W4_act_only_{s}', wbits=4, act={s})
for s in ['qkv', 'o', 'gu', 'down', 'lm']: add(f'W4_act_all_except_{s}', wbits=4, act=ALL - {s})
add('W4_lm_fp', wbits=4, lmbits=None, act=ALL); add('W4_lm_int8', wbits=4, lmbits=8, act=ALL)
add('W4_g128_actint8', wbits=4, act=ALL, group=128); add('W8_g128_actint8', wbits=8, act=ALL, group=128)
add('W4_actint8_keep8outliers', wbits=4, act=ALL, keep=8); add('W8_actint8_keep8outliers', wbits=8, act=ALL, keep=8)
add('fpW_act_keep8', act=ALL, keep=8)
add('W4_emb_quantised_too (run_full_fx fake-quant style)', wbits=4, emb_q=True)
for k in (2, 4, 8, 14, 21, 28): add(f'W4_act_first{k}layers_only', wbits=4, act=ALL, layers=set(range(k)))
add('W4_act_last14layers_only', wbits=4, act=ALL, layers=set(range(14, 28)))
# massive-activation profile of the fp32 model at position 0
prof = [dict(l=s['l'], xmax=round(s['xmax'], 1), dmax=round(s['dmax'], 1), hhmax=round(s['hhmax'], 1), vmax=round(s['vmax'], 2)) for s in ref_stats]
q16_over = [s['l'] for s in ref_stats if s['xmax'] >= 32768]; q16_dover = [s['l'] for s in ref_stats if s['dmax'] >= 32768]
out = dict(model=name, tok=TOK, hf_argmax=int(hf.argmax()), firmware_reported_argmax=dict(w4=13, w8=14582), variants=V, fp32_profile=prof, resid_exceeds_Q16_layers=q16_over, down_exceeds_Q16_layers=q16_dover)
json.dump(out, open(sys.argv[1], 'w'), indent=1); print('profile', prof[:6], '...', 'Q16 overflow layers', q16_over, q16_dover)
