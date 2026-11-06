"""trueqiq_run/stream.py (agent D): full-inner-checkpoint -> outer-input weight stream, vectorised on GPU.

Matches agent B's StateMV frame protocol exactly (engine.py write_frames/tile_weights) but builds a whole matrix at once.
TRANSPORT ONLY: tile reformatting (+c/-c pairs), silu pre-inversion of weights (fp64), per-row shift, norm-scale
pre-compensation of weight-only frames.  Nothing here touches an activation.

Compact stored form per matrix: payload [nOB, 126, nL*H*127] fp32 (already silu-inverted and norm-compensated); a full
outer frame is assembled on the fly by `assemble()` (BALLAST, WF flags, SEL one-hot address + this payload slice).
The stream is token-independent, so it is built once and kept resident (GPU) / pinned / memmapped.
"""
import os
import math, sys, os, glob, json, time
import torch
if os.environ.get('NOFLA'): sys.modules['fla'] = None
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'state')); 
from engine import StateMV, silu_inv, SILU_MIN

def silu_inv_fast(t, iters=60):
    return silu_inv(t, iters)

def build_matrix_payload(spec, W, blocks_per_chunk=4, device='cuda', out_dtype=torch.float32, nL_act=None):
    """W [Oall,Kall] (any dtype). returns payload [nOB, 2P, nL*H*(O+1)] on `device` (out_dtype), identical to
    spec.write_frames(*spec.tile_weights(W,o0)) PAY region."""
    nL, H, P, O = (nL_act or spec.nL), spec.H, spec.P, spec.O
    Oall, Kall = W.shape; T = nL * H; assert Kall <= T * P, (Kall, T * P)
    nOB = -(-Oall // O)
    out = torch.empty(nOB, 2 * P, nL * H * (O + 1), dtype=out_dtype, device=device)
    for b0 in range(0, nOB, blocks_per_chunk):
        b1 = min(nOB, b0 + blocks_per_chunk); nb = b1 - b0
        Wb = torch.zeros(nb * O, T * P, dtype=torch.float64, device=device)
        rows = W[b0 * O:min(Oall, b1 * O)].to(device=device, dtype=torch.float64)
        Wb[:rows.shape[0], :Kall] = rows
        Wt = Wb.reshape(nb, O, T, P)                               # [b, o, t, j]
        pos = Wt.permute(0, 3, 2, 1)                               # [b, j, t, o]
        c = torch.cat([pos, -pos], 1)                              # [b, 2P, t, o]
        s = (-0.27 - c.min(dim=3).values).clamp(min=0)             # [b, 2P, t]
        v = silu_inv(c + s[..., None]); ref = silu_inv(s)
        blk = torch.cat([v, ref[..., None]], 3).reshape(nb, 2 * P, T * (O + 1))
        # norm compensation (frame = BALLAST(zeroed) + SEL one-hot (1) + nL WF flags): o2 = 1 + nL
        o2 = float(1 + nL); p2 = (blk ** 2).sum(2)                 # [b, 2P]
        g = torch.ones_like(p2)
        for _ in range(30):
            g = spec.B / torch.sqrt(spec.B ** 2 + o2 + p2 / g ** 2)
        out[b0:b1] = (blk / g[..., None]).to(out_dtype)
    return out

class FrameAssembler:
    """assembles [n, D] fp32 outer inputs from compact payload; pure transport (flags/addresses/payload)."""
    def __init__(self, spec, device='cuda'):
        self.s, self.dev = spec, device
        self.pay0 = spec.PAY[0].start; self.pay1 = spec.PAY[-1].stop
        assert all(spec.PAY[i].stop == spec.PAY[i + 1].start for i in range(spec.nL - 1))
    def write(self, payload_rows, rows_idx, nL_act=None):
        """payload_rows [n, nL*H*127]; rows_idx LongTensor [n] key rows -> frames [n, D]"""
        s = self.s; n = payload_rows.shape[0]
        f = torch.zeros(n, s.layout.n, dtype=torch.float32, device=self.dev)
        f[:, 0] = s.B
        f[torch.arange(n, device=self.dev), s.SEL.start + rows_idx.to(self.dev)] = 1.0
        na = nL_act or s.nL
        f[:, s.WF.start:s.WF.start + na] = 1.0
        f[:, self.pay0:self.pay0 + payload_rows.shape[1]] = payload_rows
        return f
    def read(self, X):
        s = self.s; m = X.shape[0]
        f = torch.zeros(m, s.layout.n, dtype=torch.float32, device=self.dev)
        f[:, 0] = s.B; f[:, s.RF.start] = 1; f[:, s.X] = X.to(self.dev, torch.float32)
        return f


# ---------------- inner checkpoint -> program (list of matrices in execution order) -----------------
def inner_cfg(name):
    r = {'q3': 'Qwen3-0.6B', 'q25': 'Qwen2.5-0.5B'}[name]
    p = glob.glob(os.path.expanduser(f'~/.cache/huggingface/hub/models--Qwen--{r}/snapshots/*/'))[0]
    return json.load(open(p + 'config.json')), p + 'model.safetensors'

def inner_program(cfg):
    """one inner token: ordered list of (name, O, K). norms/biases/qk-norm vectors listed separately."""
    d, I, L, V = cfg['hidden_size'], cfg['intermediate_size'], cfg['num_hidden_layers'], cfg['vocab_size']
    nh, nkv = cfg['num_attention_heads'], cfg['num_key_value_heads']; hd = cfg.get('head_dim') or d // nh
    prog = []
    for l in range(L):
        p = f'model.layers.{l}.'
        prog += [(p + 'self_attn.q_proj.weight', nh * hd, d), (p + 'self_attn.k_proj.weight', nkv * hd, d),
                 (p + 'self_attn.v_proj.weight', nkv * hd, d), (p + 'self_attn.o_proj.weight', d, nh * hd),
                 (p + 'mlp.gate_proj.weight', I, d), (p + 'mlp.up_proj.weight', I, d), (p + 'mlp.down_proj.weight', d, I)]
    prog.append(('lm_head.weight', V, d))
    vec = []
    for l in range(L):
        vec += [d, d] + ([hd, hd] if 'qwen3' == cfg['model_type'] else []) + ([nh * hd + 2 * nkv * hd] if cfg.get('model_type') == 'qwen2' else [])
    vec.append(d)
    return prog, vec

def plan(cfg, nL=1, H=16, P=63, O=126, write_pair=True, read_m=1):
    """counts for one inner token with a bank of T=nL*H tiles; each output block = 2P write frames + read frame(s)."""
    prog, vec = inner_program(cfg); T = nL * H
    frames = 0; blocks = 0; wts = 0; padded = 0; need_tiles = 0
    for n, o, k in prog:
        kt = -(-k // P); need_tiles = max(need_tiles, kt)
        nk = -(-kt // T)                       # tile-sets along K (needs multi-pass accumulation if >1)
        ob = -(-o // O); blocks += ob * nk; frames += ob * nk * (2 * P + read_m); wts += o * k; padded += ob * O * kt * P
    D = 1 + nL + 1 + 1 + 128 + nL * H * (O + 1) + T * P + O
    return dict(nL=nL, H=H, tiles=T, tiles_needed_one_pass=need_tiles, blocks=blocks, frames=frames, weights=wts, padded=padded,
                util=wts / (blocks * T * P * O), D=D, vec_params=sum(vec),
                dense_frame_bytes=frames * D * 4, payload_bytes_compact=blocks * 2 * P * T * (O + 1) * 4, weight_bytes_fp32=wts * 4)

if __name__ == '__main__':
    for nm in ['q3', 'q25']:
        cfg, _ = inner_cfg(nm)
        for nL in (1, 2, 3, 4, 5):
            r = plan(cfg, nL)
            print(nm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
