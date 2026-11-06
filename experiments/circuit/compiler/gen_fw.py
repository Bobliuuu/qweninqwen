"""Qwen3 (HF state dict) -> firmware inputs: cfg.h + blob.bin (float tables, row scales, embedding) + wrom.bin (int4 cells).
Usage as library: gen(sd, cfg, prompt, outdir, mode='f32'|'q4v')."""
import numpy as np, os, math

MATS = [('self_attn.q_proj', 0), ('self_attn.k_proj', 1), ('self_attn.v_proj', 2), ('self_attn.o_proj', 3),
        ('mlp.gate_proj', 4), ('mlp.up_proj', 5), ('mlp.down_proj', 6)]

class Blob:
    def __init__(self): self.b = bytearray()
    def add(self, a):
        a = np.ascontiguousarray(a, dtype=np.float32); off = len(self.b); self.b += a.tobytes(); return off

def quant4_rows(W):
    """symmetric per-row int4 in [-8,7] -> (cells u=w+8 uint8, scale float32[R])"""
    s = np.abs(W).max(1) / 7.0; s[s == 0] = 1.0
    q = np.clip(np.rint(W / s[:, None]), -8, 7).astype(np.int8)
    return q, s.astype(np.float32)

def wrom_chunks(q):
    """q [R][C] int4 -> chunk-major cells [C/64][R][64] (+8)"""
    R, C = q.shape; assert C % 64 == 0, 'inner dim must be a multiple of 64 for the VDOT kernels'
    return (q.reshape(R, C // 64, 64).transpose(1, 0, 2) + 8).astype(np.uint8)

def gen(sd, cfg, prompt, outdir, mode='f32', maxpos=None, dump=True):
    D, NL, NH, NKV, HD, HID, V = (cfg[k] for k in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'intermediate_size', 'vocab_size'))
    maxpos = maxpos or len(prompt); os.makedirs(outdir, exist_ok=True)
    g = lambda n: np.asarray(sd[n], dtype=np.float32)
    B = Blob(); wrom = []; wp = 0
    emb = g('model.embed_tokens.weight'); head = g('lm_head.weight') if 'lm_head.weight' in sd else emb
    EMB = B.add(emb)
    pos = np.arange(maxpos)[:, None]; inv = cfg['rope_theta'] ** (-2.0 * np.arange(HD // 2) / HD)
    ang = pos * inv[None, :]; ROPE = B.add(np.cos(ang)); B.add(np.sin(ang))
    FN = B.add(g('model.norm.weight'))
    norms, soff, wbw = [], [], []
    for l in range(NL):
        p = f'model.layers.{l}.'
        norms.append([B.add(g(p + 'input_layernorm.weight')), B.add(g(p + 'self_attn.q_norm.weight')), B.add(g(p + 'self_attn.k_norm.weight')), B.add(g(p + 'post_attention_layernorm.weight'))])
        so, wb = [], []
        for name, _ in MATS:
            W = g(p + name + '.weight')
            if mode == 'f32': so.append(B.add(W)); wb.append(0)
            else:
                q, s = quant4_rows(W); so.append(B.add(s)); wb.append(wp); c = wrom_chunks(q); wrom.append(c.ravel()); wp += c.size // 8
        soff.append(so); wbw.append(wb)
    lm_soff = lm_wb = 0
    if mode == 'q4v':
        q, s = quant4_rows(head); lm_soff = B.add(s); lm_wb = wp; c = wrom_chunks(q); wrom.append(c.ravel()); wp += c.size // 8
    arr = lambda a: '{' + ','.join(('{' + ','.join(str(int(v)) for v in r) + '}') if hasattr(r, '__len__') else str(int(r)) for r in a) + '}'
    h = f"""#pragma once
#define {'MODE_F32' if mode == 'f32' else 'MODE_Q4V'} 1
{'#define DUMP 1' if dump else ''}
#define DIM {D}
#define NL {NL}
#define NH {NH}
#define NKV {NKV}
#define HD {HD}
#define HID {HID}
#define VOCAB {V}
#define MAXPOS {maxpos}
#define NTOK {len(prompt)}
#define EPS {cfg['rms_norm_eps']!r}f
#define ATT_SCALE {1.0 / math.sqrt(HD)!r}f
#define PROMPT {{{','.join(map(str, prompt))}}}
#define EMB_OFF {EMB}u
#define ROPE_OFF {ROPE}u
#define FINAL_NORM_OFF {FN}u
#define LM_SOFF {lm_soff}u
#define LM_WB {lm_wb}
static const uint32_t NORM_OFF[NL][4] = {arr(norms)};
static const uint32_t SOFF[NL][7] = {arr(soff)};
static const int WBW[NL][7] = {arr(wbw)};
"""
    open(f'{outdir}/cfg.h', 'w').write(h); open(f'{outdir}/blob.bin', 'wb').write(bytes(B.b))
    if wrom: open(f'{outdir}/wrom.bin', 'wb').write(np.concatenate(wrom).tobytes())
    return dict(blob_bytes=len(B.b), wrom_words=wp, wrom_weights=wp * 8)
