"""Tiny Qwen3-architecture LLM -> int4 WROM + C tables for the Qwenputer RV32 CPU.

python gen_q3.py [--nl 1] [--vocab 128] [--seed 0] [--ntok 16]
Writes weights_q3.h, wrom_q3.c, ref_q3.json (float reference tokens, HF check).
Architecture (real Qwen3): pre-norm RMSNorm, GQA (8 q heads / 4 kv heads, hd 8), per-head q_norm/k_norm,
half-split (rotate_half) RoPE, SwiGLU, untied lm_head, final RMSNorm. RMSNorm gains of input/post/final
norms are folded into the matrices that read them; q/k-norm gains are applied by the firmware.
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--nl", type=int, default=1)
ap.add_argument("--vocab", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ntok", type=int, default=16)
ap.add_argument("--maxpos", type=int, default=64)
ap.add_argument("--hf", action="store_true", help="also run Hugging Face Qwen3ForCausalLM (needs torch+transformers)")
a = ap.parse_args()
dim, nh, nkv, hs, hid, nl, vocab = 64, 8, 4, 8, 128, a.nl, a.vocab
kvd, THETA, EPS = nkv * hs, 1e6, 1e-6
rng = np.random.default_rng(a.seed)
N = lambda *s, sd=1.0: rng.normal(0, sd, s)

emb = N(vocab, dim)
ga, gf, gfin = 1 + .2 * N(nl, dim), 1 + .2 * N(nl, dim), 1 + .2 * N(dim)
qn, kn = 1 + .3 * N(nl, hs), 1 + .3 * N(nl, hs)
wq, wk, wv = N(nl, dim, dim, sd=1.5 / 8), N(nl, kvd, dim, sd=1.5 / 8), N(nl, kvd, dim, sd=1 / 8)
wo, w1, w3 = N(nl, dim, dim, sd=.5 / 8), N(nl, hid, dim, sd=1 / 8), N(nl, hid, dim, sd=1 / 8)
w2 = N(nl, dim, hid, sd=.5 / np.sqrt(hid))
cls = N(vocab, dim, sd=3 / 8)


def mse_codes(w):
    best = None
    for c in np.linspace(0.5, 1.0, 26):
        s = np.abs(w).max(-1, keepdims=True) * c / 7
        s[s == 0] = 1e-12
        q = np.clip(np.round(w / s), -8, 7)
        e = ((q * s - w) ** 2).sum(-1, keepdims=True)
        best = (q, s, e) if best is None else (np.where(e < best[2], q, best[0]),
                                               np.where(e < best[2], s, best[1]), np.minimum(e, best[2]))
    return best[0].astype(np.int8), best[1][..., 0]

# fold input/post/final norm gains, quantize every matmul to int4 per-row
src = {"cls": cls * gfin, "wq": wq * ga[:, None], "wk": wk * ga[:, None], "wv": wv * ga[:, None], "wo": wo,
       "w1": w1 * gf[:, None], "w3": w3 * gf[:, None], "w2": w2}
Q, SC, DQ = {}, {}, {}
for k, w in src.items():
    if w.ndim == 3:
        r = [mse_codes(w[l]) for l in range(nl)]
        Q[k], SC[k] = np.stack([x for x, _ in r]), np.stack([y for _, y in r])
    else:
        Q[k], SC[k] = mse_codes(w)
    DQ[k] = Q[k] * SC[k][..., None]          # dequantized (what the CPU's weights represent)


def rmsn(x, g=None):
    y = x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS)
    return y if g is None else y * g


def rope(v, pos):  # v (heads, hs), rotate_half
    h = hs // 2
    fr = 1.0 / THETA ** (np.arange(h) * 2 / hs)
    c, s = np.cos(pos * fr), np.sin(pos * fr)
    return np.concatenate([v[:, :h] * c - v[:, h:] * s, v[:, h:] * c + v[:, :h] * s], 1)


def fwd(tok, pos, kc, vc, W=DQ, stats=None):
    x = emb[tok].copy()
    for l in range(nl):
        h = rmsn(x)                                    # gains folded into W
        q, k, v = (W["wq"][l] @ h).reshape(nh, hs), (W["wk"][l] @ h).reshape(nkv, hs), W["wv"][l] @ h
        q, k = rope(rmsn(q, qn[l]), pos), rope(rmsn(k, kn[l]), pos)
        kc[l].append(k); vc[l].append(v.reshape(nkv, hs))
        if stats is not None: stats[0][l] = max(stats[0][l], np.abs(k).max()); stats[1][l] = max(stats[1][l], np.abs(v).max())
        out = np.zeros((nh, hs))
        for hh in range(nh):
            g = hh // (nh // nkv)
            sc = np.array([q[hh] @ kk[g] for kk in kc[l]]) / np.sqrt(hs)
            p = np.exp(sc - sc.max()); p /= p.sum()
            out[hh] = sum(pi * vv[g] for pi, vv in zip(p, vc[l]))
        x = x + W["wo"][l] @ out.reshape(-1)
        h = rmsn(x)
        a1, b1 = W["w1"][l] @ h, W["w3"][l] @ h
        x = x + W["w2"][l] @ (a1 / (1 + np.exp(-a1)) * b1)
    return W["cls"] @ rmsn(x)


def gen(n, W=DQ, stats=None, sample=None):
    kc, vc, tok, out, lg = [[] for _ in range(nl)], [[] for _ in range(nl)], 1, [], []
    for pos in range(n):
        l_ = fwd(tok, pos, kc, vc, W, stats)
        tok = int(np.argmax(l_)) if sample is None else int(sample.choice(vocab, p=np.exp(l_ - l_.max()) / np.exp(l_ - l_.max()).sum()))
        out.append(tok); lg.append(l_)
    return out, np.array(lg)

toks, lg = gen(a.ntok)
tf, _ = gen(a.ntok, src)  # unquantized-weight float tokens (fold only)
srt = np.sort(lg, 1)
print("float(dequant) tokens", toks)
print("float(fp32 weights) tokens", tf, "agree", sum(x == y for x, y in zip(toks, tf)))
print("top1-top2 margin per step", np.round(srt[:, -1] - srt[:, -2], 2).tolist())
ref = {"tokens_dequant": toks, "tokens_fp": tf, "cfg": dict(dim=dim, nh=nh, nkv=nkv, hs=hs, hid=hid, nl=nl, vocab=vocab, seed=a.seed)}

if a.hf:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=dim, intermediate_size=hid, num_hidden_layers=nl,
                      num_attention_heads=nh, num_key_value_heads=nkv, head_dim=hs, rms_norm_eps=EPS,
                      rope_theta=THETA, tie_word_embeddings=False, max_position_embeddings=256, attention_bias=False)
    m = Qwen3ForCausalLM(cfg).double().eval()
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)
    with torch.no_grad():
        m.model.embed_tokens.weight.copy_(T(emb)); m.lm_head.weight.copy_(T(DQ["cls"]))
        m.model.norm.weight.fill_(1.0)
        for l, ly in enumerate(m.model.layers):
            ly.input_layernorm.weight.fill_(1.0); ly.post_attention_layernorm.weight.fill_(1.0)
            ly.self_attn.q_norm.weight.copy_(T(qn[l])); ly.self_attn.k_norm.weight.copy_(T(kn[l]))
            for nm, key in (("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv"), ("o_proj", "wo")):
                getattr(ly.self_attn, nm).weight.copy_(T(DQ[key][l]))
            ly.mlp.gate_proj.weight.copy_(T(DQ["w1"][l])); ly.mlp.up_proj.weight.copy_(T(DQ["w3"][l]))
            ly.mlp.down_proj.weight.copy_(T(DQ["w2"][l]))
        ids = torch.tensor([[1]])
        g = m.generate(ids, max_new_tokens=a.ntok - 1, do_sample=False)[0].tolist()
        full = m(torch.tensor([[1] + toks[:-1]])).logits[0].numpy()
    print("HF generate tokens (start tok 1, incl. start)", g)
    print("HF == numpy ref:", g[1:] == toks[:len(g) - 1], "max logit err (teacher forced)", float(np.abs(full - lg).max()))
    ref.update(hf_tokens=g[1:], hf_logit_maxerr=float(np.abs(full - lg).max()), hf_first=g[1] if len(g) > 1 else None,
               hf_matches_numpy=bool(g[1:] == toks[:len(g) - 1]))
# calibration of KV scales on sampled sequences
st = [np.zeros(nl), np.zeros(nl)]
for sd in range(4):
    gen(64, DQ, st, np.random.default_rng(100 + sd))
cells = bytearray()
def add_matrix(q):
    rows, n = q.shape
    npad = -(-n // 8) * 8
    first = len(cells) // 8
    for r in range(rows):
        row = np.zeros(npad, np.int64); row[:n] = q[r]
        cells.extend((row + 8).astype(np.uint8).tobytes())
    return first
wb = {"cls": add_matrix(Q["cls"])}
for l in range(nl):
    for k in ["wq", "wk", "wv", "wo", "w1", "w3"]:
        wb[k, l] = add_matrix(Q[k][l])
    for ci in range(2):
        wb[f"w2c{ci}", l] = add_matrix(Q["w2"][l][:, ci * 64:(ci + 1) * 64])
assert len(cells) // 8 <= 32640
def carr(name, ctype, v):
    v = np.asarray(v).ravel()
    return f"static const {ctype} {name}[{len(v)}] = {{{','.join(str(int(x)) for x in v)}}};\n"
h = [f"#define DIM {dim}\n#define HID {hid}\n#define NL {nl}\n#define NH {nh}\n#define NKV {nkv}\n#define HS {hs}\n"
     f"#define KVD {kvd}\n#define VOCAB {vocab}\n#define MAXPOS {a.maxpos}\n#define WB_CLS {wb['cls']}\n"]
for k in ["wq", "wk", "wv", "wo", "w1", "w3", "w2c0", "w2c1"]:
    h.append(carr(f"WB_{k}", "int32_t", [wb[k, l] for l in range(nl)]))
for k in ["cls", "wq", "wk", "wv", "wo", "w1", "w3", "w2"]:
    s = np.round(np.asarray(SC[k], np.float64) * 2**15).astype(np.int64)
    assert s.max() < 2**15, (k, s.max())
    h.append(carr(f"S_{k}", "int32_t", s))
h.append(carr("SILU", "int32_t", [round((x := (i - 256) / 32) / (1 + np.exp(-x)) * 4096) for i in range(512)]))
s8 = np.abs(emb).max(1) / 127.0
q8 = np.clip(np.round(emb / s8[:, None]), -127, 127).astype(np.int8)
h.append(carr("Q_emb", "int8_t", q8)); h.append(carr("S_emb", "int32_t", np.round(s8 * 2**32)))
cs = []
for pos in range(a.maxpos):
    for i in range(hs // 2):
        fr = 1.0 / THETA ** (2 * i / hs)
        cs += [round(np.cos(pos * fr) * 16384), round(np.sin(pos * fr) * 16384)]
h.append(carr("ROPE", "int16_t", cs))
# q/k-norm gains: sqrt(hs) * gain, Q12
h.append(carr("QNG", "int32_t", np.round(qn * np.sqrt(hs) * 4096))); h.append(carr("KNG", "int32_t", np.round(kn * np.sqrt(hs) * 4096)))
h.append(carr("KMAX", "int32_t", np.round(st[0] * 1.25 * 65536))); h.append(carr("VMAX", "int32_t", np.round(st[1] * 1.25 * 65536)))
h.append(carr("P8", "uint8_t", [round(127 * np.exp(-i / 32)) for i in range(256)]))
open(f"{HERE}/weights_vd2.h", "w").write("#include <stdint.h>\n" + "".join(h))
open(f"{HERE}/wrom_data2.c", "w").write("#include <stdint.h>\n__attribute__((section(\".wrom\"), used)) const uint8_t WROM_DATA["
                                        f"{len(cells)}] = {{{','.join(map(str, cells))}}};\n")
json.dump(ref, open(f"{HERE}/ref_q3.json", "w"))
print(f"{len(cells) // 8} WROM words of 32640 ({len(cells)} int4 weights)")
