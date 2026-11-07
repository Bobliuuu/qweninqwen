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
        hq = aq(h, 'qkv' in act and on and l not in AOFF, keep)
        v = wq(W[p + 'self_attn.v_proj.weight'], OVR.get((l, 'v_proj'), wb) if (l, 'v_proj') in OVR else wb, group) @ hq            # (NKV*HD)
        if vq and on and wbits is not None:                                   # kv_store: int8 per kv head
            v = torch.stack([aq(v[g * HD:(g + 1) * HD], True) for g in range(NKV)]).reshape(-1)
        ao = v.reshape(NKV, HD).repeat_interleave(NH // NKV, 0).reshape(-1)   # T=1: softmax==1
        o = wq(W[p + 'self_attn.o_proj.weight'], OVR.get((l, 'o_proj'), wb) if (l, 'o_proj') in OVR else wb, group) @ aq(ao, 'o' in act and on and l not in AOFF, keep)
        x = x + o
        h = rms(x, W[p + 'post_attention_layernorm.weight']); hq = aq(h, 'gu' in act and on and l not in AOFF, keep)
        g_ = wq(W[p + 'mlp.gate_proj.weight'], OVR.get((l, 'gate_proj'), wb) if (l, 'gate_proj') in OVR else wb, group) @ hq; u_ = wq(W[p + 'mlp.up_proj.weight'], OVR.get((l, 'up_proj'), wb) if (l, 'up_proj') in OVR else wb, group) @ hq
        hh = torch.nn.functional.silu(g_) * u_
        d = wq(W[p + 'mlp.down_proj.weight'], OVR.get((l, 'down_proj'), wb) if (l, 'down_proj') in OVR else wb, group) @ aq(hh, 'down' in act and on and l not in AOFF, keep)
        x = x + d
        stats.append(dict(l=l, xmax=float(x.abs().max()), dmax=float(d.abs().max()), hhmax=float(hh.abs().max()), vmax=float(v.abs().max())))
        xs.append(x.clone())
    hN = rms(x, W['model.norm.weight'])
    lm = wq(W['lm_head.weight'] if 'lm_head.weight' in W else emb, lmbits, group)
    lg = lm @ aq(hN, 'lm' in act, keep)
    return lg, xs, stats
OVR = {}; AOFF = set()
ALL = {'qkv', 'o', 'gu', 'down', 'lm'}
ref_lg, ref_xs, ref_stats = run()
def summ(lg, xs):
    top = lg.topk(5); srt = lg.sort(descending=True).indices
    cs = float(torch.nn.functional.cosine_similarity(xs[-1], ref_xs[-1], dim=0))
    return dict(argmax=int(lg.argmax()), top5=[int(i) for i in top.indices], rank13=int((srt == 13).nonzero()), rank15846=int((srt == 15846).nonzero()), cos_final=cs)
V = {}
def add(n, **kw):
    lg, xs, st = run(**kw); V[n] = summ(lg, xs); v = V[n]; print(f"{n:58s} am={v['argmax']:6d} top5={v['top5']} r15846={v['rank15846']:6d} cosN={v['cos_final']:.3f}", flush=True)

def cfg(ovr=None, aoff=()):
    OVR.clear(); OVR.update(ovr or {}); AOFF.clear(); AOFF.update(aoff)
MATS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
def allm(ls, b): return {(l, mname): b for l in ls for mname in MATS}
S = [2, 3, 26, 27]
runs = [
 ('W4+act; down_proj fp @ layers 2,27', dict(ovr={(2, 'down_proj'): None, (27, 'down_proj'): None}), {}),
 ('W4+act; ALL mats fp @ layer 2 only', dict(ovr=allm([2], None)), {}),
 ('W4+act; ALL mats fp @ layer 27 only', dict(ovr=allm([27], None)), {}),
 ('W4+act; ALL mats fp @ 2,26,27 (act still int8 there)', dict(ovr=allm([2, 26, 27], None)), {}),
 ('W4+act; ALL mats fp @ 2,3,26,27 (act still int8 there)', dict(ovr=allm(S, None)), {}),
 ('W4+act; W8 @ 2,3,26,27 (act int8 there)', dict(ovr=allm(S, 8)), {}),
 ('W4+act; W8 @ 2,3,26,27, act OFF there', dict(ovr=allm(S, 8), aoff=S), {}),
 ('W4+act; fp weights @ 2,3,26,27, act OFF there (= EXCEPT)', dict(ovr=allm(S, None), aoff=S), {}),
 ('W4+act; W8 @ 2,26,27, act OFF there', dict(ovr=allm([2, 26, 27], 8), aoff=[2, 26, 27]), {}),
 ('W4+act; down_proj W8 @ 2,27 and act OFF on all of layers 2,27', dict(ovr={(2, 'down_proj'): 8, (27, 'down_proj'): 8}, aoff=[2, 27]), {}),
 ('W4+act; fp down_proj @ 2,27 and act OFF on layers 2,27', dict(ovr={(2, 'down_proj'): None, (27, 'down_proj'): None}, aoff=[2, 27]), {}),
 ('W8+act everywhere except fp @ 27 (weights+act off)', dict(ovr=allm([27], None), aoff=[27]), dict(wbits=8)),
 ('W8+act everywhere except fp @ 2,27 (weights+act off)', dict(ovr=allm([2, 27], None), aoff=[2, 27]), dict(wbits=8)),
]
for n, c, kw in runs:
    cfg(**c); add(n, **{**dict(wbits=4, act=ALL), **kw})
json.dump(V, open(sys.argv[1], 'w'), indent=1)
