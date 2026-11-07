"""Qwen3 HF state dict -> integer firmware inputs for fw/qwen3_fx.c: cfg.h, blob.bin (int32 tables), wrom.bin (int4 cells).
wbits=4: one int4 plane (per-row scale); wbits=8: two int4 planes (hi*16+lo), 2 VDOT passes."""
import numpy as np, os, math

MATS = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

def scale_word(ws):
    """per-row scale ws (<1) -> int32: 26-bit mantissa (f in [0.5,1), Q31 with low 5 bits dropped) | e (ws = f * 2^-e, e in 0..31)"""
    f, p = np.frexp(ws.astype(np.float64)); e = -p
    assert e.min() >= 0 and e.max() <= 31, (e.min(), e.max())
    m = np.minimum(np.rint(f * 2.0 ** 31), 2 ** 31 - 32).astype(np.int64) & ~31
    return (m | e).astype(np.int32)

def quant(W, wbits):
    P = wbits // 4; qmax = 7 * (16 ** P - 1) // 15          # 7, 119, 1911: signed base-16 digits in [-8,7]
    s = np.abs(W).max(1) / qmax; s[s == 0] = 1e-6
    q = np.clip(np.rint(W / s[:, None]), -qmax - 1, qmax).astype(np.int64)
    planes = []
    for i in range(P):
        d = ((q + 8) & 15) - 8; planes.append(d); q = (q - d) // 16
    assert (q == 0).all()
    return planes[::-1], s       # most significant digit plane first

def chunks(q):
    R, C = q.shape; assert C % 64 == 0
    return (q.reshape(R, C // 64, 64).transpose(1, 0, 2) + 8).astype(np.uint8)

class Blob:
    def __init__(self): self.b = bytearray()
    def add(self, a, dt=np.int32):
        a = np.ascontiguousarray(a, dtype=dt); off = len(self.b); self.b += a.tobytes(); return off

def gen(sd, cfg, prompt, outdir, wbits=4, ext=False, maxpos=None, dump=False, boost=None, apl=1, aplmap=None):
    """wbits: default weight bits (4/8/12); boost: {(layer, matname-substring): bits} overrides, e.g. {(27,'mlp'):12}"""
    boost = boost or {}; aplmap = aplmap or {}; aplv = [aplmap.get(l, 1) for l in range(cfg['num_hidden_layers'] + 1)]; apl = max(aplv)
    D, NL, NH, NKV, HD, HID, V = (cfg[k] for k in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size'))
    maxpos = maxpos or len(prompt); os.makedirs(outdir, exist_ok=True)
    g = lambda n: np.asarray(sd[n], dtype=np.float64)
    B = Blob(); wrom = []; wp = 0
    emb = g('model.embed_tokens.weight'); head = g('lm_head.weight') if 'lm_head.weight' in sd else emb
    EMB = B.add(np.rint(emb * 65536))
    pos = np.arange(maxpos)[:, None]; inv = cfg['rope_theta'] ** (-2.0 * np.arange(HD // 2) / HD); ang = pos * inv[None, :]
    c = np.rint(np.cos(ang) * 32767).astype(np.int64); s_ = np.rint(np.sin(ang) * 32767).astype(np.int64)
    ROPE = B.add(((s_ & 0xFFFF) << 16) | (c & 0xFFFF), np.uint32)
    gain = lambda n: B.add(np.rint(g(n) * 4096))
    FN = gain('model.norm.weight'); norms, soff, wbw, plns = [], [], [], []
    def addmat(W, wb_=None):
        nonlocal wp
        planes, s = quant(W, wb_ or wbits); so = B.add(scale_word(s)); base = wp
        for pl in planes: cc = chunks(pl); wrom.append(cc.ravel()); wp += cc.size // 8
        return so, base
    for l in range(NL):
        p = f'model.layers.{l}.'
        norms.append([gain(p + 'input_layernorm.weight'), gain(p + 'self_attn.q_norm.weight'), gain(p + 'self_attn.k_norm.weight'), gain(p + 'post_attention_layernorm.weight')])
        so, wb, pl = [], [], []
        for m in MATS:
            bits = next((v for (ll, sub), v in boost.items() if ll == l and sub in m), wbits)
            a, b = addmat(g(p + m + '.weight'), bits); so.append(a); wb.append(b); pl.append(bits // 4)
        soff.append(so); wbw.append(wb); plns.append(pl)
    LMS, LMW = addmat(head)
    arr = lambda a: '{' + ','.join(('{' + ','.join(str(int(v)) for v in r) + '}') for r in a) + '}'
    cq = round(math.log2(math.e) * 65536 / (127 ** 2 * 1024 ** 2 * math.sqrt(HD)) * 2 ** 50)
    eps = cfg['rms_norm_eps']
    h = f"""#pragma once
{'#define EXT 1' if ext else ''}
{'#define DUMP 1' if dump else ''}
#define APL {apl}
#define LM_PL {wbits // 4}
#define DIM {D}
#define NL {NL}
#define NH {NH}
#define NKV {NKV}
#define HD {HD}
#define HID {HID}
#define VOCAB {V}
#define MAXPOS {maxpos}
#define NTOK {len(prompt)}
#define DLOG {int(math.log2(D))}
#define HLOG {int(math.log2(HD))}
#define EPSQ16 {int(eps * 2 ** 32 + 0.5)}LL
#define EPSQ12 {int(eps * 2 ** 24 + 0.5)}LL
#define CQ {cq}LL
#define PROMPT {{{','.join(map(str, prompt))}}}
#define EMB_OFF {EMB}u
#define ROPE_OFF {ROPE}u
#define FINAL_NORM_OFF {FN}u
#define LM_SOFF {LMS}u
#define LM_WB {LMW}
static const uint32_t NORM_OFF[NL][4] = {arr(norms)};
static const uint32_t SOFF[NL][7] = {arr(soff)};
static const int WBW[NL][7] = {arr(wbw)};
static const int PLN[NL][7] = {arr(plns)};
static const int APLN[NL + 1] = {{{",".join(map(str, aplv))}}};
"""
    assert D & (D - 1) == 0 and HD & (HD - 1) == 0
    open(f'{outdir}/cfg.h', 'w').write(h); open(f'{outdir}/blob.bin', 'wb').write(bytes(B.b))
    open(f'{outdir}/wrom.bin', 'wb').write(np.concatenate(wrom).tobytes())
    return dict(blob_bytes=len(B.b), wrom_words=wp, wrom_weights=wp * 8)
