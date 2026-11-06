"""Vertical slice: TOY inner Qwen3-architecture model executed by a stock Qwen3_5ForCausalLM outer model (debug vehicle).

Design (see INTERFACE.md 5b): unrolled pipeline in depth; one forward call per inner token; weights paged through GDN-state
'W banks' by host WRITE frames (transport only); registers = GDN layers between producer and consumer; norms = GDN gated-RMSNorm
heads; SwiGLU / products = outer MLP gadgets; softmax attention + RoPE + KV = outer full-attention layers; LM head = outer lm_head,
argmax by host on outer logits.
"""
import math, sys, os, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ops'))
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, DynamicCache
import os, sys  # PATHS_SET
_h=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[os.path.join(_h,'..','..','qweninqwen'), os.path.join(_h,'..','state'), os.path.join(_h,'..','ops')]
from machine import Layout, BALLAST
from engine import silu_inv, ZSTAR, SILU_MIN          # agent B (host weight-payload pre-inversion only)
from ops import MLPBuilder                              # agent C exact gadgets

B = BALLAST
Rq, Bg = 100.0, 1e8
KB = 60.0
EPS_IN = 1e-6


def sl(s): return s.start, s.stop - s.start


class Cfg:
    d = 128; nq = 4; nkv = 2; hd = 32; inter = 256; L = 2; V = 512; theta = 1e4; eps = 1e-6
    # outer machine geometry
    H = 7; dk = 128; dv = 288            # uniform GDN geometry
    O = dv - 2                           # outputs per W-bank block
    P = (dk - 2) // 2                    # contraction elements per head tile (63)
    HW = 5                               # heads used by W banks (K <= HW*P)
    LW = 143                             # register lane width: 2*LW+1 <= dv, O == 2*LW
    I_MLP = 1024


class Slots:
    pass


def build_layout(c):
    lay = Layout(); a = lay.alloc; S = Slots(); S.lay = lay
    S.EMBX = a('EMBX', c.d)
    S.PAY = a('PAY', c.HW * (c.O + 1)); S.SEL = a('SEL', c.dk); S.INIT = a('INIT', 1); S.RINIT = a('RINIT', 1)
    S.X = a('X', c.HW * c.P); S.Y = a('Y', c.O)
    nbank = 4 * c.L + 1
    S.WF = a('WF', nbank); S.RF = a('RF', nbank)
    S.NIN = {}
    for l in range(c.L):
        S.NIN[('n1', l)] = a(f'NIN_n1_{l}', c.d); S.NIN[('n3', l)] = a(f'NIN_n3_{l}', c.d)
    S.NIN[('nf', 0)] = a('NIN_nf', c.d)
    S.RESX = a('RESX', c.d); S.RESH = a('RESH', c.d)
    S.QKVR = a('QKVR', (c.nq + 2 * c.nkv) * c.hd)
    S.QN = a('QN', c.nq * c.hd); S.KN = a('KN', c.nkv * c.hd); S.GQ = a('GQ', c.nq * c.hd); S.GK = a('GK', c.nkv * c.hd)
    S.QG = a('QG', c.nq * c.hd); S.KG = a('KG', c.nkv * c.hd); S.AO = [a(f'AO{l}', c.nq * c.hd) for l in range(c.L)]
    S.MASK = [a(f'MASK{l}', 1) for l in range(c.L)]
    S.GU = a('GU', 2 * c.inter); S.ACT = a('ACT', c.inter); S.LOG = a('LOG', c.V)
    S.REG = {}   # name -> dict(RA=slice, WA=slice)
    return S


def reg_slots(S, name, nheads):
    a = S.lay.alloc
    S.REG[name] = dict(RA=a(f'{name}_RA', nheads), WA=a(f'{name}_WA', nheads), n=nheads)


# ----------------------------------------------------------------------------- model + layer builders
def make_model(D, layer_types, c, eps=1e-30, dtype=torch.float32):
    rot = c.hd; hdo = c.hd + 3
    rp = dict(rope_type='default', rope_theta=c.theta, partial_rotary_factor=(rot + 0.5) / hdo,
              mrope_section=[rot // 2 // 3 + (1 if rot // 2 % 3 > i else 0) for i in range(3)])
    cfg = Qwen3_5TextConfig(hidden_size=D, intermediate_size=c.I_MLP, num_hidden_layers=len(layer_types),
                            layer_types=layer_types, vocab_size=c.V, rms_norm_eps=eps, tie_word_embeddings=False,
                            linear_num_key_heads=c.H, linear_num_value_heads=c.H, linear_key_head_dim=c.dk,
                            linear_value_head_dim=c.dv, linear_conv_kernel_dim=4, num_attention_heads=c.nq,
                            num_key_value_heads=c.nkv, head_dim=hdo, max_position_embeddings=4096,
                            rope_parameters=rp, mtp_num_hidden_layers=0)
    m = Qwen3_5ForCausalLM(cfg).to(dtype)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
    return m


def transparent(layer, D, which=('in', 'post')):
    with torch.no_grad():
        v = B / math.sqrt(D) - 1.0
        if 'in' in which: layer.input_layernorm.weight.fill_(v)
        if 'post' in which: layer.post_attention_layernorm.weight.fill_(v)


def gdn_base(layer, D, c):
    """beta~1, decay~1, conv=current tap, gate const; transparent norms"""
    g = layer.linear_attn
    transparent(layer, D)
    with torch.no_grad():
        g.conv1d.weight.zero_(); g.conv1d.weight[:, 0, -1] = 1.0
        g.A_log.zero_(); g.dt_bias.zero_()
        g.in_proj_qkv.weight.zero_(); g.out_proj.weight.zero_()
        g.in_proj_b.weight.zero_(); g.in_proj_b.weight[:, 0] = 30.0 / B
        g.in_proj_a.weight.zero_(); g.in_proj_a.weight[:, 0] = -30.0 / B
        g.in_proj_z.weight.zero_(); g.in_proj_z.weight[:, 0] = 30.0 / B
        g.norm.weight.fill_(1.0 / 30.0)
    return g


def build_reg(layer, D, c, S, name, heads):
    """GDN register layer.  heads: list of dict(lane, dst=slice|None, src=list[(slot, coef, mode)]).
    state per head: row0 = register lane vector (pos cols [0,LW), neg cols [LW,2LW)), row1 = ballast row (col dv-1 = Bg)."""
    g = gdn_base(layer, D, c); H = len(heads); kd = c.H * c.dk; LW = c.LW; dv = c.dv
    assert H <= c.H and 2 * LW + 1 <= dv
    RA, WA = S.REG[name]['RA'], S.REG[name]['WA']
    Gr = Rq * Bg / (30.0 * math.sqrt(dv))
    with torch.no_grad():
        Wq, Wo = g.in_proj_qkv.weight, g.out_proj.weight
        for h, hd in enumerate(heads):
            # query: address row0 (30*RA) + ballast coordinate row1 (Rq)
            Wq[h * c.dk + 0, RA.start + h] = 30.0
            Wq[h * c.dk + 1, 0] = Rq / B
            # key: row0 <- WA[h], row1 <- RINIT
            Wq[kd + h * c.dk + 0, WA.start + h] = 30.0
            Wq[kd + h * c.dk + 1, S.RINIT.start] = 30.0
            # value
            for i in range(LW):
                for (slot, coef, mode) in hd['src']:
                    s0, n = sl(slot)
                    idx = (hd['lane'] % 2) * LW + i if mode == 'Y' else hd['lane'] * LW + i
                    if idx < n:
                        Wq[2 * kd + h * dv + i, s0 + idx] += coef
                        Wq[2 * kd + h * dv + LW + i, s0 + idx] -= coef
            Wq[2 * kd + h * dv + dv - 1, S.RINIT.start] = Bg
            # read out
            if hd.get('dst') is not None:
                d0, n = sl(hd['dst'])
                for i in range(LW):
                    e = hd['lane'] * LW + i
                    if e < n:
                        Wo[d0 + e, h * dv + i] += Gr; Wo[d0 + e, h * dv + LW + i] -= Gr


def build_norm(layer, D, c, S, heads):
    """GDN gated-RMSNorm heads.  heads: list of dict(src=slice (d floats), dst=slice (d floats)); out = x/sqrt(mean x^2+EPS_IN)."""
    g = gdn_base(layer, D, c); kd = c.H * c.dk; dv = c.dv
    eps_v = 1e-4        # scale in silu small-signal regime (|x*eps_v| <= ~1e-3 for |x|<=10; relative err (eps_v x)^2/4)
    with torch.no_grad():
        Wq, Wo = g.in_proj_qkv.weight, g.out_proj.weight
        for h, hd in enumerate(heads):
            s0, d = sl(hd['src']); d0, dd = sl(hd['dst']); assert d == dd and 2 * d + 2 <= dv
            Wq[h * c.dk + 0, 0] = 30.0 / B           # q = e0
            Wq[kd + h * c.dk + 0, 0] = 30.0 / B      # k = e0
            for i in range(d):
                Wq[2 * kd + h * dv + i, s0 + i] = eps_v
                Wq[2 * kd + h * dv + d + i, s0 + i] = -eps_v
            E = eps_v * math.sqrt(d * EPS_IN)
            Wq[2 * kd + h * dv + 2 * d, 0] = E / B
            Wq[2 * kd + h * dv + 2 * d + 1, 0] = -E / B
            # (on+ - on-) = eps_v x / rms(v);  rms(v)^2 = eps_v^2 (sum x^2 + d EPS)/(2 dv)  -> = xhat*sqrt(2dv/d)
            k = math.sqrt(d / (2.0 * dv))
            for i in range(d):
                Wo[d0 + i, h * dv + i] = k; Wo[d0 + i, h * dv + d + i] = -k


def build_bank(layer, D, c, S, l):
    """B-style paged float matvec bank (GDN state) for bank index l (flags WF[l], RF[l]); shared PAY/SEL/X/Y slots."""
    g = gdn_base(layer, D, c); kd = c.H * c.dk; dv = c.dv; O = c.O; P = c.P
    G = Bg * Rq / (math.sqrt(dv) * SILU_MIN)
    with torch.no_grad():
        Wq, Wo = g.in_proj_qkv.weight, g.out_proj.weight
        for h in range(c.HW):
            for j in range(P):
                xi = h * P + j
                Wq[h * c.dk + j, S.X.start + xi] = 1.0
                Wq[h * c.dk + P + j, S.X.start + xi] = -1.0
            Wq[h * c.dk + c.dk - 1, 0] = Rq / B
            for r in range(c.dk):
                Wq[kd + h * c.dk + r, 0] = -KB / B
                Wq[kd + h * c.dk + r, S.SEL.start + r] = 2 * KB
            for cc in range(O + 1):
                Wq[2 * kd + h * dv + cc, S.PAY.start + h * (O + 1) + cc] = 1.0
            Wq[2 * kd + h * dv + dv - 1, S.INIT.start] = Bg
            for cc in range(O):
                Wo[S.Y.start + cc, h * dv + cc] = G
                Wo[S.Y.start + cc, h * dv + O] = -G
        g.in_proj_z.weight.zero_(); g.in_proj_z.weight[:, 0] = -40.0 / B; g.in_proj_z.weight[:, S.RF.start + l] = 40.0 + ZSTAR
        g.in_proj_b.weight.zero_(); g.in_proj_b.weight[:, 0] = -30.0 / B; g.in_proj_b.weight[:, S.WF.start + l] = 60.0
        g.norm.weight.fill_(1.0)


def build_attention(layer, D, c, S, l):
    """Outer full-attention layer = inner softmax attention w/ RoPE/GQA/KV (C's construction + a mask dim for non-attention steps).
    head_dim_o = hd+3: dims [0,hd) rotated, R=hd: const in q, R+1: const in k, R+2: const in q / -MASK in k."""
    hd = c.hd; hdo = hd + 3; nq, nkv = c.nq, c.nkv; sa = layer.self_attn; R = hd; cc = 2.0 ** 23
    transparent(layer, D, ('in',))
    # q_hat = Gq*q/(sqrt2*cc/sqrt(hdo)) (two const dims), k_hat = Gk*k/(cc/sqrt(hdo)); want q.k/sqrt(hd)
    G = math.sqrt(math.sqrt(2.0) * cc * cc / (math.sqrt(hdo) * math.sqrt(hd)))   # outer attn applies hdo^-1/2 itself
    with torch.no_grad():
        Wq = sa.q_proj.weight.view(nq, 2, hdo, D); Wk = sa.k_proj.weight.view(nkv, hdo, D); Wv = sa.v_proj.weight.view(nkv, hdo, D)
        j = torch.arange(hd)
        for h in range(nq):
            Wq[h, 0, j, S.QG.start + h * hd + j] = 1.0
            Wq[h, 0, R, 0] = cc / B
            Wq[h, 0, R + 2, 0] = cc / B
            Wq[h, 1, j, 0] = 30.0 / B
        for g in range(nkv):
            Wk[g, j, S.KG.start + g * hd + j] = 1.0; Wk[g, R + 1, 0] = cc / B
            Wk[g, R + 2, S.MASK[l].start] = -cc * 4
            Wv[g, j, S.QKVR.start + (nq + nkv) * hd + g * hd + j] = 1.0
        Wo = sa.o_proj.weight.view(D, nq, hdo)
        for h in range(nq): Wo[S.AO[l].start + h * hd + j, h, j] = 1.0
        sa.q_norm.weight.fill_(G - 1.0); sa.k_norm.weight.fill_(G - 1.0)


# ----------------------------------------------------------------------------- assembly
def build_outer(c, dtype=torch.float32):
    S = build_layout(c)
    L = c.L
    # register layers' slots (RA/WA per head)
    for l in range(L + 1): reg_slots(S, f'R1_{l}', 2 if l < L else 1)
    for l in range(L):
        reg_slots(S, f'R2_{l}', 2); reg_slots(S, f'R3_{l}', 1); reg_slots(S, f'R4_{l}', 2)
        reg_slots(S, f'R5_{l}', 4); reg_slots(S, f'R6_{l}', 2)
    reg_slots(S, 'Rlg', 4)
    D = S.lay.n
    plan = []   # list of (kind, args)
    for l in range(L):
        plan += [('R1', l), ('N1', l), ('W', 4 * l + 0), ('R2', l), ('N2', l), ('A', l), ('R3', l), ('W', 4 * l + 1),
                 ('R4', l), ('N3', l), ('W', 4 * l + 2), ('R5', l), ('S', l), ('R6', l), ('W', 4 * l + 3)]
    plan += [('R1', L), ('NF', 0), ('W', 4 * L), ('Rlg', 0)]
    types = ['full_attention' if k == 'A' else 'linear_attention' for k, _ in plan]
    m = make_model(D, types, c, dtype=dtype)
    d = c.d; LWd = c.LW
    for li, (k, l) in enumerate(plan):
        layer = m.model.layers[li]
        if k == 'R1':
            if l == 0:
                src = [(S.EMBX, 1.0, 'E')]
            else:
                src = [(S.Y, 1.0, 'Y'), (S.RESH, 1.0, 'E')]
            heads = [dict(lane=0, dst=S.NIN[('n1', l)], src=src), dict(lane=0, dst=S.RESX, src=src)] if l < L else \
                    [dict(lane=0, dst=S.NIN[('nf', 0)], src=src)]
            build_reg(layer, D, c, S, f'R1_{l}', heads)
        elif k == 'R2':
            heads = [dict(lane=i, dst=S.QKVR, src=[(S.Y, 1.0, 'Y')]) for i in range(2)]
            build_reg(layer, D, c, S, f'R2_{l}', heads)
        elif k == 'R3':
            build_reg(layer, D, c, S, f'R3_{l}', [dict(lane=0, dst=S.X, src=[(S.AO[l], 1.0, 'E')])])
        elif k == 'R4':
            src = [(S.Y, 1.0, 'Y'), (S.RESX, 1.0, 'E')]
            build_reg(layer, D, c, S, f'R4_{l}', [dict(lane=0, dst=S.NIN[('n3', l)], src=src), dict(lane=0, dst=S.RESH, src=src)])
        elif k == 'R5':
            build_reg(layer, D, c, S, f'R5_{l}', [dict(lane=i, dst=S.GU, src=[(S.Y, 1.0, 'Y')]) for i in range(4)])
        elif k == 'R6':
            build_reg(layer, D, c, S, f'R6_{l}', [dict(lane=i, dst=S.X, src=[(S.ACT, 1.0, 'E')]) for i in range(2)])
        elif k == 'Rlg':
            build_reg(layer, D, c, S, 'Rlg', [dict(lane=i, dst=S.LOG, src=[(S.Y, 1.0, 'Y')]) for i in range(4)])
        elif k == 'N1':
            gdn_dst = slice(S.X.start, S.X.start + d)
            build_norm(layer, D, c, S, [dict(src=S.NIN[('n1', l)], dst=gdn_dst)])
        elif k == 'N3':
            build_norm(layer, D, c, S, [dict(src=S.NIN[('n3', l)], dst=slice(S.X.start, S.X.start + d))])
        elif k == 'NF':
            build_norm(layer, D, c, S, [dict(src=S.NIN[('nf', 0)], dst=slice(S.X.start, S.X.start + d))])
        elif k == 'N2':
            hd = c.hd; heads = []
            for h in range(c.nq):
                heads.append(dict(src=slice(S.QKVR.start + h * hd, S.QKVR.start + (h + 1) * hd), dst=slice(S.QN.start + h * hd, S.QN.start + (h + 1) * hd)))
            for g in range(c.nkv):
                o = S.QKVR.start + c.nq * hd + g * hd
                heads.append(dict(src=slice(o, o + hd), dst=slice(S.KN.start + g * hd, S.KN.start + (g + 1) * hd)))
            build_norm(layer, D, c, S, heads)
            mb = MLPBuilder(layer, D, ballast=B)
            mb.product(S.QN, S.GQ, S.QG); mb.product(S.KN, S.GK, S.KG)
        elif k == 'W':
            build_bank(layer, D, c, S, l)
        elif k == 'A':
            build_attention(layer, D, c, S, l)
        elif k == 'S':
            mb = MLPBuilder(layer, D, ballast=B)
            mb.swiglu(slice(S.GU.start, S.GU.start + c.inter), slice(S.GU.start + c.inter, S.GU.start + 2 * c.inter), S.ACT)
    with torch.no_grad():
        m.model.norm.weight.fill_(B / math.sqrt(D) - 1.0)
        m.lm_head.weight[:, S.LOG.start:S.LOG.stop] = torch.eye(c.V)
    S.plan = plan; S.D = D
    return m, S


# ----------------------------------------------------------------------------- host frame program (flags, addresses, weight payloads, gains)
class Prog:
    def __init__(self, S, c, model):
        self.S, self.c, self.m = S, c, model; self.D = S.D; self.frames = []; self.pos = []; self.cur_pos = 0; self.marks = {}
    def new(self, pos=None):
        f = torch.zeros(self.D, dtype=torch.float64); f[0] = B
        for l in range(self.c.L): f[self.S.MASK[l].start] = 1.0     # default: keys of this step masked in every attention layer
        return f
    def push(self, f, pos=None):
        self.frames.append(f); self.pos.append(self.cur_pos if pos is None else pos)
    def mark(self, n): self.marks[n] = len(self.frames)
    def ra(self, f, name, heads):
        for h in heads: f[self.S.REG[name]['RA'].start + h] = 1.0
    def wa(self, f, name, heads):
        for h in heads: f[self.S.REG[name]['WA'].start + h] = 1.0


def tile_payload(W, o0, c):
    """W [out,in] -> payload rows [2P, HW*(O+1)] (host weight transport: silu pre-inversion, B's engine)."""
    P, HW, O = c.P, c.HW, c.O
    Wb = W[o0:o0 + O].to(torch.float64)
    if Wb.shape[0] < O: Wb = torch.cat([Wb, torch.zeros(O - Wb.shape[0], Wb.shape[1], dtype=torch.float64)])
    Wp = torch.zeros(O, HW * P, dtype=torch.float64); Wp[:, :Wb.shape[1]] = Wb
    Wt = Wp.reshape(O, HW, P).permute(2, 1, 0)            # [j, h, c]
    rows = torch.cat([Wt, -Wt], 0)                         # [2P, HW, O]
    s = (-0.27 - rows.min(dim=2).values).clamp(min=0)
    v = silu_inv(rows + s[:, :, None]); ref = silu_inv(s)
    return torch.cat([v, ref[:, :, None]], 2).reshape(2 * P, HW * (O + 1))


def write_steps(pr, bank, W, o0):
    S, c = pr.S, pr.c
    pay = tile_payload(W, o0, c)
    for r in range(2 * c.P):
        f = pr.new(); f[S.SEL.start + r] = 1.0; f[S.WF.start + bank] = 1.0
        f[S.PAY.start:S.PAY.stop] = pay[r]; pr.push(f)


def init_steps(pr):
    S, c = pr.S, pr.c
    for bank in range(4 * c.L + 1):
        f = pr.new(); f[S.SEL.start + c.dk - 1] = 1.0; f[S.WF.start + bank] = 1.0; f[S.INIT.start] = 1.0; pr.push(f)
    f = pr.new(); f[S.RINIT.start] = 1.0; pr.push(f)


def token_program(pr, tok, pos, Wd, gains, emb):
    """Frames for ONE inner token (pos = inner position).  Wd[bank] = list of (matrix W_folded, ...) ; gains = per-layer (gq,gk)."""
    S, c = pr.S, pr.c; L = c.L
    pr.cur_pos = pos
    pr.mark('emb'); f = pr.new(); f[S.EMBX.start:S.EMBX.stop] = emb(tok); pr.wa(f, 'R1_0', [0, 1]); pr.push(f)
    for l in range(L):
        b = 4 * l
        write_steps(pr, b, Wd[b], 0)
        pr.mark(f'qkv{l}'); f = pr.new(); f[S.RF.start + b] = 1; pr.ra(f, f'R1_{l}', [0]); pr.wa(f, f'R2_{l}', [0, 1]); pr.push(f)
        pr.mark(f'attn{l}'); f = pr.new(); pr.ra(f, f'R2_{l}', [0, 1]); f[S.MASK[l].start] = 0.0
        gq, gk = gains[l]; f[S.GQ.start:S.GQ.stop] = gq.double(); f[S.GK.start:S.GK.stop] = gk.double()
        pr.wa(f, f'R3_{l}', [0]); pr.push(f)
        write_steps(pr, b + 1, Wd[b + 1], 0)
        pr.mark(f'o{l}'); f = pr.new(); f[S.RF.start + b + 1] = 1; pr.ra(f, f'R3_{l}', [0]); pr.ra(f, f'R1_{l}', [1]); pr.wa(f, f'R4_{l}', [0, 1]); pr.push(f)
        for blk in range(2):
            write_steps(pr, b + 2, Wd[b + 2], blk * c.O)
            pr.mark(f'gu{l}_{blk}'); f = pr.new(); f[S.RF.start + b + 2] = 1; pr.ra(f, f'R4_{l}', [0]); pr.wa(f, f'R5_{l}', [2 * blk, 2 * blk + 1]); pr.push(f)
        pr.mark(f'sw{l}'); f = pr.new(); pr.ra(f, f'R5_{l}', [0, 1, 2, 3]); pr.wa(f, f'R6_{l}', [0, 1]); pr.push(f)
        write_steps(pr, b + 3, Wd[b + 3], 0)
        pr.mark(f'down{l}'); f = pr.new(); f[S.RF.start + b + 3] = 1; pr.ra(f, f'R6_{l}', [0, 1]); pr.ra(f, f'R4_{l}', [1])
        pr.wa(f, f'R1_{l + 1}', [0, 1] if l + 1 < L else [0]); pr.push(f)
    for blk in range(2):
        write_steps(pr, 4 * L, Wd[4 * L], blk * c.O)
        pr.mark(f'lm{blk}'); f = pr.new(); f[S.RF.start + 4 * L] = 1; pr.ra(f, f'R1_{L}', [0]); pr.wa(f, 'Rlg', [2 * blk, 2 * blk + 1]); pr.push(f)
    f = pr.new(); pr.ra(f, 'Rlg', [0, 1, 2, 3]); pr.push(f)       # EMIT step: outer lm_head reads LOG
    return len(pr.frames) - 1


@torch.no_grad()
def run_frames(model, frames, pos, cache, off, seg=384, dev='cpu', capture=None):
    out = None
    for a in range(0, frames.shape[0], seg):
        fb = frames[a:a + seg].to(dev).float()
        cp = torch.arange(off + a, off + a + fb.shape[0], device=dev)
        pid = torch.tensor(pos[a:a + seg], device=dev)[None] if pos is not None else cp[None]
        o = model(inputs_embeds=fb[None], past_key_values=cache, use_cache=True, cache_position=cp, position_ids=pid)
        out = o.logits[0]
    return out, off + frames.shape[0]
