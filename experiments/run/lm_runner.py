"""trueqiq_run/lm_runner.py (agent D): LAYER-MAJOR executor for the unrolled outer pipeline (vslice design).

Why: the unrolled outer model has ~15 outer layers per inner layer (hundreds of layers), uniform D, so (a) the parameters of the
whole stack do not fit in memory, (b) every WRITE frame would cost a pass through every layer.  Layer-major evaluation runs layer 0
over ALL frames of the token, then layer 1 over all frames, ... (mathematically the same as one forward over the frame sequence:
layer l's input at frame t only depends on layers < l at frame t, and its state only on earlier frames of layer l).  Each outer
layer is a STOCK Qwen3_5DecoderLayer built on demand (compiled constants from the program) and freed afterwards; only its state
(GDN recurrent state / attention KV) is kept in a DynamicCache.
Frames of kind WRITE (host weight transport into bank b: flags WF[b]) are fed only to bank layer b (idle everywhere else:
flags off => exact zero contribution / masked KV; verified against the full stock forward by `equiv_test`).  All other
(control/read) frames pass through EVERY layer.
"""
import os
import math, sys, os, time
import torch
sys.path.insert(0, os.path.expanduser('~/qweninqwen-work/trueqiq')); sys.path.insert(0, os.path.expanduser('~/qweninqwen-work/trueqiq_state'))
sys.path.insert(0, os.path.expanduser('~/qweninqwen-work/trueqiq_ops'))
import vslice as V
from transformers import Qwen3_5TextConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm
from transformers.masking_utils import create_causal_mask


def make_cfg(D, layer_types, c, eps=1e-30):
    rot = c.hd; hdo = c.hd + 3
    rp = dict(rope_type='default', rope_theta=c.theta, partial_rotary_factor=(rot + 0.5) / hdo,
              mrope_section=[rot // 2 // 3 + (1 if rot // 2 % 3 > i else 0) for i in range(3)])
    return Qwen3_5TextConfig(hidden_size=D, intermediate_size=c.I_MLP, num_hidden_layers=len(layer_types), layer_types=layer_types,
                             vocab_size=8, rms_norm_eps=eps, tie_word_embeddings=False, linear_num_key_heads=c.H, linear_num_value_heads=c.H,
                             linear_key_head_dim=c.dk, linear_value_head_dim=c.dv, linear_conv_kernel_dim=4, num_attention_heads=c.nq,
                             num_key_value_heads=c.nkv, head_dim=hdo, max_position_embeddings=4096, rope_parameters=rp, mtp_num_hidden_layers=0)


def compile_layer(layer, k, l, S, c, D):
    """same per-kind compilation as vslice.build_outer (kept in sync by reading vslice)."""
    d = c.d; B = V.B
    with torch.no_grad():
        for p in layer.parameters(): p.zero_()
    if k == 'R1':
        L = c.L
        src = [(S.EMBX, 1.0, 'E')] if l == 0 else [(S.Y, 1.0, 'Y'), (S.RESH, 1.0, 'E')]
        heads = [dict(lane=0, dst=S.NIN[('n1', l)], src=src), dict(lane=0, dst=S.RESX, src=src)] if l < L else \
                [dict(lane=0, dst=S.NIN[('nf', 0)], src=src)]
        V.build_reg(layer, D, c, S, f'R1_{l}', heads)
    elif k == 'R2': V.build_reg(layer, D, c, S, f'R2_{l}', [dict(lane=i, dst=S.QKVR, src=[(S.Y, 1.0, 'Y')]) for i in range(2)])
    elif k == 'R3': V.build_reg(layer, D, c, S, f'R3_{l}', [dict(lane=0, dst=S.X, src=[(S.AO, 1.0, 'E')])])
    elif k == 'R4':
        src = [(S.Y, 1.0, 'Y'), (S.RESX, 1.0, 'E')]
        V.build_reg(layer, D, c, S, f'R4_{l}', [dict(lane=0, dst=S.NIN[('n3', l)], src=src), dict(lane=0, dst=S.RESH, src=src)])
    elif k == 'R5': V.build_reg(layer, D, c, S, f'R5_{l}', [dict(lane=i, dst=S.GU, src=[(S.Y, 1.0, 'Y')]) for i in range(4)])
    elif k == 'R6': V.build_reg(layer, D, c, S, f'R6_{l}', [dict(lane=i, dst=S.X, src=[(S.ACT, 1.0, 'E')]) for i in range(2)])
    elif k == 'Rlg': V.build_reg(layer, D, c, S, 'Rlg', [dict(lane=i, dst=S.LOG, src=[(S.Y, 1.0, 'Y')]) for i in range(4)])
    elif k in ('N1', 'N3', 'NF'):
        key = {'N1': ('n1', l), 'N3': ('n3', l), 'NF': ('nf', 0)}[k]
        V.build_norm(layer, D, c, S, [dict(src=S.NIN[key], dst=slice(S.X.start, S.X.start + d))])
    elif k == 'N2':
        hd = c.hd; heads = []
        for h in range(c.nq):
            heads.append(dict(src=slice(S.QKVR.start + h * hd, S.QKVR.start + (h + 1) * hd), dst=slice(S.QN.start + h * hd, S.QN.start + (h + 1) * hd)))
        for g in range(c.nkv):
            o = S.QKVR.start + c.nq * hd + g * hd
            heads.append(dict(src=slice(o, o + hd), dst=slice(S.KN.start + g * hd, S.KN.start + (g + 1) * hd)))
        V.build_norm(layer, D, c, S, heads)
        mb = V.MLPBuilder(layer, D, ballast=B); mb.product(S.QN, S.GQ, S.QG); mb.product(S.KN, S.GK, S.KG)
    elif k == 'W': V.build_bank(layer, D, c, S, l)
    elif k == 'A': V.build_attention(layer, D, c, S, l)
    elif k == 'S':
        mb = V.MLPBuilder(layer, D, ballast=B)
        mb.swiglu(slice(S.GU.start, S.GU.start + c.inter), slice(S.GU.start + c.inter, S.GU.start + 2 * c.inter), S.ACT)
    else: raise ValueError(k)


class LayerMajor:
    def __init__(self, S, c, plan, dev='cpu', eps=1e-30):
        self.S, self.c, self.plan, self.dev = S, c, plan, dev
        self.D = S.D
        types = ['full_attention' if k == 'A' else 'linear_attention' for k, _ in plan]
        self.cfg = make_cfg(self.D, types, c, eps)
        self.cfg._attn_implementation = 'sdpa'
        self.cache = DynamicCache(config=self.cfg)
        self.rot = Qwen3_5TextRotaryEmbedding(config=self.cfg).to(dev)
        self.bank_li = {l: li for li, (k, l) in enumerate(plan) if k == 'W'}
        self.stats = dict(layer_calls=0, rows=0)
        self.off = 0                                   # absolute (outer) position offset for attention caches

    def frame_banks(self, frames):
        """data-independent classification from the host flag fields: write frame -> bank id, else -1 (control)."""
        S = self.S; wf = frames[:, S.WF.start:S.WF.stop]; rf = frames[:, S.RF.start:S.RF.stop]
        isw = (wf.abs().sum(1) > 0) & (rf.abs().sum(1) == 0)
        return torch.where(isw, wf.argmax(1), torch.full((frames.shape[0],), -1))

    @torch.no_grad()
    def run(self, frames, pos, seg=4096, trace=None):
        """frames [n,D] float (control + write frames in order); returns final-layer outputs of control frames (index map)."""
        dev = self.dev; n = frames.shape[0]
        frames = frames.float(); banks = self.frame_banks(frames)
        H = frames.clone()
        posT = torch.tensor(pos)
        for li, (k, l) in enumerate(self.plan):
            act = (banks < 0) | (banks == (l if k == 'W' else -2))
            idx = act.nonzero()[:, 0]
            layer = Qwen3_5DecoderLayer(self.cfg, li).to(dev).eval(); compile_layer(layer, k, l, self.S, self.c, self.D)
            for a in range(0, idx.numel(), seg):
                ii = idx[a:a + seg]; x = H[ii].to(dev)[None]
                cp = torch.arange(self.off_l(li), self.off_l(li) + ii.numel(), device=dev)
                pid = posT[ii].to(dev)[None]
                pe = self.rot(x, pid[None].expand(3, 1, -1))
                mask = None
                if k == 'A':
                    past = self.off_l(li); nn_ = ii.numel()
                    mask = (torch.arange(past + nn_, device=dev)[None, :] <= (torch.arange(nn_, device=dev)[:, None] + past))[None, None]
                y = layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pid, past_key_values=self.cache, cache_position=cp, use_cache=True)
                self.adv(li, ii.numel())
                H[ii] = y[0].to(H.dtype).cpu()
                self.stats['rows'] += ii.numel()
            self.stats['layer_calls'] += 1
            if trace is not None: trace[li] = H[banks < 0].clone()
            del layer
        return H

    # per-layer offset bookkeeping (attention caches only need monotone cache_position)
    _offs = None
    def off_l(self, li):
        if self._offs is None: self._offs = {}
        return self._offs.get(li, 0)
    def adv(self, li, n): self._offs[li] = self.off_l(li) + n
