"""vs2: scalable vertical slice (generalises vslice.py).  Real-dim capable: geometry H=16, dk=130 (P=64), dv=258 (LW=128, O=256),
multi-layer K-split banks, REG groups spanning layers, C's RMSNormUnit stages for residual norms, GDN heads for per-head q/k norm,
outer full attention for softmax/RoPE/KV, per-layer slots to avoid inactive-layer leakage, layer-major executor (layers materialised on demand).
Host: flags, one-hot addresses, silu-pre-inverted weight payloads (gains folded), embedding row, position ids, argmax/append."""
import math, sys, os, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ('.', '../state', '../ops', '../../qweninqwen'): sys.path.insert(0, os.path.join(HERE, p))
from transformers import Qwen3_5TextConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm
import os, sys  # PATHS_SET
_h=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[os.path.join(_h,'..','..','qweninqwen'), os.path.join(_h,'..','state'), os.path.join(_h,'..','ops')]
from machine import Layout, BALLAST
from engine import silu_inv, ZSTAR, SILU_MIN
from ops import MLPBuilder

B = BALLAST; Rq, Bg, KB = 100.0, 1e8, 60.0


def sl(s): return s.start, s.stop - s.start


class Cfg:
    def __init__(self, d, nq, nkv, hd, inter, L, V, theta=1e4, eps=1e-6, H=16, LW=128, P=64, I_MLP=4096, qk_norm=True):
        self.d, self.nq, self.nkv, self.hd, self.inter, self.L, self.V = d, nq, nkv, hd, inter, L, V
        self.theta, self.eps, self.H, self.LW, self.P, self.I_MLP = theta, eps, H, LW, P, I_MLP
        self.dk = 2 * P + 2; self.dv = 2 * LW + 2; self.O = 2 * LW
        self.qkv_n = (nq + 2 * nkv) * hd
        self.tourn = V > 8192          # in-model block argmax (tournament + hard-max attention) instead of Rlg->LOG
        self.nblk = -(-V // self.O)


def nlanes(n, c): return -(-n // c.LW)


class Slots: pass


def op_dims(c, op):
    """(n_out, K) of bank op index"""
    L = c.L
    if op == 4 * L: return c.V, c.d
    k = op % 4
    return [(c.qkv_n, c.d), (c.d, c.nq * c.hd), (2 * c.inter, c.d), (c.d, c.inter)][k]


def build_layout(c):
    lay = Layout(); a = lay.alloc; S = Slots(); S.lay = lay; L = c.L
    S.EMBX = a('EMBX', c.d)
    S.PAY = a('PAY', c.H * (c.O + 1)); S.SEL = a('SEL', c.dk); S.INIT = a('INIT', 1); S.RINIT = a('RINIT', 1)
    nops = 4 * L + 1
    S.nbl = {op: -(-op_dims(c, op)[1] // (c.H * c.P)) for op in range(nops)}
    Kmax = max(S.nbl[op] * c.H * c.P for op in range(nops))
    S.X = a('X', Kmax); S.Y = a('Y', c.O)
    S.wf_idx = {}; n = 0
    for op in range(nops):
        for j in range(S.nbl[op]): S.wf_idx[(op, j)] = n; n += 1
    S.WF = a('WF', n); S.RF = a('RF', nops)
    S.NIN = {}
    S.NINS = a('NIN', c.d)                                   # one shared norm-input slot (norm instances are flag-gated)
    S.NIN[('nf', 0)] = S.NINS
    for l in range(L): S.NIN[('n1', l)] = S.NINS; S.NIN[('n3', l)] = S.NINS
    S.ngkeys = [('n1', l) for l in range(L)] + [('n3', l) for l in range(L)] + [('nf', 0)]
    S.ACTF = a('ACTF', 3 * L + len(S.ngkeys))                # per-layer activity flags (host control): A_l, NH_l, S_l, NG instances
    S.fl = {}
    for l in range(L): S.fl[('A', l)] = l; S.fl[('NH', l)] = L + l; S.fl[('S', l)] = 2 * L + l
    for i, key in enumerate(S.ngkeys): S.fl[('NG',) + key] = 3 * L + i
    S.scal = {}
    for key in S.ngkeys: S.scal[key] = {nme: a(f's_{key[0]}{key[1]}_' + nme, 1) for nme in ('MS', 'R0', 'QQ', 'TT', 'ZZ', 'ZF')}
    S.RESX = a('RESX', c.d); S.RESH = a('RESH', c.d)
    S.QKVR = a('QKVR', c.qkv_n); S.QN = a('QN', c.nq * c.hd); S.KN = a('KN', c.nkv * c.hd); S.AO = a('AO', c.nq * c.hd)
    S.MASK = [a(f'MASK{l}', 1) for l in range(L)]
    S.GU = a('GU', 2 * c.inter)
    S.ACT = a('ACT', c.inter)
    if not c.tourn: S.LOG = a('LOG', c.V)
    else:
        S.PAD = a('PAD', c.O); S.BLKCODE = a('BLKCODE', 20); S.MASKM = a('MASKM', 1); S.AMO = a('AMO', 36)
        S.VL = {l: a(f'VL{l}', c.O >> l) for l in range(1, 9)}
        S.BL = {l: a(f'BL{l}', l * (c.O >> l)) for l in range(1, 9)}
        S.SL = {l: a(f'SL{l}', c.O >> (l + 1)) for l in range(8)}
    # registers: name -> heads (global); RA/WA slots
    S.regs = {}
    def reg(name, n, ports, src):
        nl = nlanes(n, c); heads = []
        for p, dst in enumerate(ports):
            for ln in range(nl): heads.append(dict(port=p, lane=ln, dst=dst, src=src))
        nh = len(heads)
        S.regs[name] = dict(heads=heads, nl=nl, RA=a(name + '_RA', nh), WA=a(name + '_WA', nh), nlayers=-(-nh // c.H))
    for l in range(L + 1):
        src = [(S.EMBX, 1.0, 'E')] if l == 0 else [(S.Y, 1.0, 'Y'), (S.RESH, 1.0, 'E')]
        reg(f'R1_{l}', c.d, [S.NIN[('n1', l)], S.RESX] if l < L else [S.NIN[('nf', 0)]], src)
    for l in range(L):
        reg(f'R2_{l}', c.qkv_n, [S.QKVR], [(S.Y, 1.0, 'Y')])
        reg(f'R3_{l}', c.nq * c.hd, [S.X], [(S.AO, 1.0, 'E')])
        reg(f'R4_{l}', c.d, [S.NIN[('n3', l)], S.RESH], [(S.Y, 1.0, 'Y'), (S.RESX, 1.0, 'E')])
        reg(f'R5_{l}', 2 * c.inter, [S.GU], [(S.Y, 1.0, 'Y')])
        reg(f'R6_{l}', c.inter, [S.X], [(S.ACT, 1.0, 'E')])
    if not c.tourn: reg('Rlg', c.V, [S.LOG], [(S.Y, 1.0, 'Y')])
    S.D = lay.n
    return S


def make_plan(S, c):
    L = c.L; plan = []
    def R(name):
        for j in range(S.regs[name]['nlayers']): plan.append(('R', name, j))
    def W(op):
        for j in range(S.nbl[op]): plan.append(('W', op, j))
    for l in range(L):
        R(f'R1_{l}')
        for s in range(7): plan.append(('NG', ('n1', l), s))
        W(4 * l); R(f'R2_{l}'); [plan.append(('NH', l, jl)) for jl in range(-(-(c.nq + c.nkv) // c.H))]; plan.append(('A', l, 0)); R(f'R3_{l}'); W(4 * l + 1)
        R(f'R4_{l}')
        for s in range(7): plan.append(('NG', ('n3', l), s))
        W(4 * l + 2); R(f'R5_{l}'); plan.append(('S', l, 0)); R(f'R6_{l}'); W(4 * l + 3)
    R(f'R1_{L}')
    for s in range(7): plan.append(('NG', ('nf', 0), s))
    W(4 * L)
    if not c.tourn: R('Rlg')
    else:
        for l in range(8): plan.append(('TS', l, 0)); plan.append(('TM', l, 0))
        plan.append(('AM', 0, 0))
    return plan


# ----------------------------------------------------------------------------- per-layer compile (vectorised)
def transparent(layer, D, which=('in', 'post')):
    with torch.no_grad():
        v = B / math.sqrt(D) - 1.0
        if 'in' in which: layer.input_layernorm.weight.fill_(v)
        if 'post' in which: layer.post_attention_layernorm.weight.fill_(v)


def gdn_base(layer, D):
    g = layer.linear_attn; transparent(layer, D)
    with torch.no_grad():
        g.conv1d.weight[:, 0, -1] = 1.0
        g.in_proj_b.weight[:, 0] = 30.0 / B
        g.in_proj_a.weight[:, 0] = -30.0 / B
        g.in_proj_z.weight[:, 0] = 30.0 / B
        g.norm.weight.fill_(1.0 / 30.0)
    return g


def build_reg(layer, D, c, S, name, j):
    g = gdn_base(layer, D); heads = S.regs[name]['heads'][j * c.H:(j + 1) * c.H]; H0 = j * c.H
    kd = c.H * c.dk; LW, dv = c.LW, c.dv; RA, WA = S.regs[name]['RA'], S.regs[name]['WA']
    Gr = Rq * Bg / (30.0 * math.sqrt(dv)); i = torch.arange(LW, device=g.in_proj_qkv.weight.device)
    with torch.no_grad():
        Wq, Wo = g.in_proj_qkv.weight, g.out_proj.weight
        for lh, hd in enumerate(heads):
            gh = H0 + lh
            Wq[lh * c.dk + 0, RA.start + gh] = 30.0; Wq[lh * c.dk + 1, 0] = Rq / B
            Wq[kd + lh * c.dk + 0, WA.start + gh] = 30.0; Wq[kd + lh * c.dk + 1, S.RINIT.start] = 30.0
            for (slot, coef, mode) in hd['src']:
                s0, n = sl(slot)
                idx = (hd['lane'] % 2) * LW + i if mode == 'Y' else hd['lane'] * LW + i
                ok = idx < n
                Wq[2 * kd + lh * dv + i[ok], s0 + idx[ok]] += coef
                Wq[2 * kd + lh * dv + LW + i[ok], s0 + idx[ok]] -= coef
            Wq[2 * kd + lh * dv + dv - 1, S.RINIT.start] = Bg
            d0, n = sl(hd['dst']); e = hd['lane'] * LW + i; ok = e < n
            Wo[d0 + e[ok], lh * dv + i[ok]] += Gr; Wo[d0 + e[ok], lh * dv + LW + i[ok]] -= Gr


def build_bank(layer, D, c, S, op, j):
    g = gdn_base(layer, D); kd = c.H * c.dk; dv = c.dv; O = c.O; P = c.P; H = c.H
    G = Bg * Rq / (math.sqrt(dv) * SILU_MIN); dev = g.in_proj_qkv.weight.device
    with torch.no_grad():
        Wq, Wo = g.in_proj_qkv.weight, g.out_proj.weight
        jj = torch.arange(P, device=dev); rr = torch.arange(c.dk, device=dev); cc = torch.arange(O + 1, device=dev); co = torch.arange(O, device=dev)
        for h in range(H):
            xi = j * H * P + h * P + jj
            Wq[h * c.dk + jj, S.X.start + xi] = 1.0
            Wq[h * c.dk + P + jj, S.X.start + xi] = -1.0
            Wq[h * c.dk + c.dk - 1, 0] = Rq / B
            Wq[kd + h * c.dk + rr, 0] = -KB / B
            Wq[kd + h * c.dk + rr, S.SEL.start + rr] = 2 * KB
            Wq[2 * kd + h * dv + cc, S.PAY.start + h * (O + 1) + cc] = 1.0
            Wq[2 * kd + h * dv + dv - 1, S.INIT.start] = Bg
            Wo[S.Y.start + co, h * dv + co] = G
            Wo[S.Y.start + co, h * dv + O] = -G
        g.in_proj_z.weight.zero_(); g.in_proj_z.weight[:, 0] = -40.0 / B; g.in_proj_z.weight[:, S.RF.start + op] = 40.0 + ZSTAR
        g.in_proj_b.weight.zero_(); g.in_proj_b.weight[:, 0] = -30.0 / B; g.in_proj_b.weight[:, S.WF.start + S.wf_idx[(op, j)]] = 60.0
        g.norm.weight.fill_(1.0)


def build_qknorm(layer, D, c, S, l, jl=0):
    """GDN gated-RMSNorm heads over each q/k head (size hd; needs dv >= 2*hd+2): QKVR_l -> QN_l / KN_l."""
    g = gdn_base(layer, D); kd = c.H * c.dk; dv = c.dv; hd = c.hd; eps_v = 1e-7
    assert 2 * hd + 2 <= dv
    i = torch.arange(hd, device=g.in_proj_qkv.weight.device)
    heads = [(S.QKVR.start + h * hd, S.QN.start + h * hd) for h in range(c.nq)] + \
            [(S.QKVR.start + c.nq * hd + k * hd, S.KN.start + k * hd) for k in range(c.nkv)]
    heads = heads[jl * c.H:(jl + 1) * c.H]
    with torch.no_grad():
        g.in_proj_z.weight.zero_(); g.in_proj_z.weight[:, 0] = -40.0 / B; g.in_proj_z.weight[:, S.ACTF.start + S.fl[('NH', l)]] = 70.0   # gate: silu(30) when active, ~0 when inactive
        Wq, Wo = g.in_proj_qkv.weight, g.out_proj.weight
        for h, (s0, d0) in enumerate(heads):
            Wq[h * c.dk + 0, 0] = 30.0 / B; Wq[kd + h * c.dk + 0, 0] = 30.0 / B
            Wq[2 * kd + h * dv + i, s0 + i] = eps_v; Wq[2 * kd + h * dv + hd + i, s0 + i] = -eps_v
            E = eps_v * math.sqrt(hd * c.eps)
            Wq[2 * kd + h * dv + 2 * hd, 0] = E / B; Wq[2 * kd + h * dv + 2 * hd + 1, 0] = -E / B
            k = math.sqrt(hd / (2.0 * dv))
            Wo[d0 + i, h * dv + i] = k; Wo[d0 + i, h * dv + hd + i] = -k


def build_attention(layer, D, c, S, l, gq, gk):
    """as vslice.build_attention (mask dim, outer scale hdo^-1/2 compensated) + inner q/k-norm gains folded into q/k projections."""
    hd = c.hd; hdo = hd + 3; nq, nkv = c.nq, c.nkv; sa = layer.self_attn; R = hd; cc = 2.0 ** 23
    transparent(layer, D, ('in',)); dev = sa.q_proj.weight.device
    G = math.sqrt(math.sqrt(2.0) * cc * cc / (math.sqrt(hdo) * math.sqrt(hd)))
    with torch.no_grad():
        Wq = sa.q_proj.weight.view(nq, 2, hdo, D); Wk = sa.k_proj.weight.view(nkv, hdo, D); Wv = sa.v_proj.weight.view(nkv, hdo, D)
        j = torch.arange(hd, device=dev); gq = gq.to(dev); gk = gk.to(dev)
        for h in range(nq):
            Wq[h, 0, j, S.QN.start + h * hd + j] = gq
            Wq[h, 0, R, 0] = cc / B; Wq[h, 0, R + 2, 0] = cc / B; Wq[h, 1, j, 0] = -30.0 / B; Wq[h, 1, j, S.ACTF.start + S.fl[('A', l)]] = 60.0
        for g in range(nkv):
            Wk[g, j, S.KN.start + g * hd + j] = gk; Wk[g, R + 1, 0] = cc / B
            Wk[g, R + 2, S.MASK[l].start] = -cc * 4
            Wv[g, j, S.QKVR.start + (nq + nkv) * hd + g * hd + j] = 1.0
        Wo = sa.o_proj.weight.view(D, nq, hdo)
        for h in range(nq): Wo[S.AO.start + h * hd + j, h, j] = 1.0
        sa.q_norm.weight.fill_(G - 1.0); sa.k_norm.weight.fill_(G - 1.0)


def build_swiglu(layer, D, c, S, l):
    """O += silu(G - 1e4(1-f)) * U : exactly zero when this layer's flag f=0 (shared GU slot)."""
    mb = MLPBuilder(layer, D, ballast=B); n = c.inter; r = mb._alloc(n); i = torch.arange(n, device=mb.g.device); nn_ = r.start + i
    with torch.no_grad():
        mb.g[nn_, S.GU.start + i] = 1.0; mb.g[nn_, 0] = -1e4 / B; mb.g[nn_, S.ACTF.start + S.fl[('S', l)]] = 1e4
        mb.u[nn_, S.GU.start + n + i] = 1.0; mb.d[S.ACT.start + i, nn_] = 1.0


def ng_stage(layer, D, c, S, key, s, m_max=1e8, rho=1.012, kappa=1e-3):
    """stage s of C's RMSNormUnit (ops.RMSNormUnit), one outer layer at a time. X_IN=NIN[key], OUT=X[0:d]."""
    d = c.d; eps_in = c.eps; xi0 = S.NIN[key].start; o0 = S.X.start
    MS, R0, QQ, TT, ZZ, ZF = [S.scal[key][n].start for n in ('MS', 'R0', 'QQ', 'TT', 'ZZ', 'ZF')]; FL = S.ACTF.start + S.fl[('NG',) + key]
    lam = 1e12; b = MLPBuilder(layer, D, B); dev = b.g.device
    with torch.no_grad():
        i = torch.arange(d, device=dev)
        if s == 0:
            r = b._alloc(2 * d)
            for sg_, off in ((1.0, 0), (-1.0, d)):
                n = r.start + off + i; b.g[n, xi0 + i] = sg_; b.u[n, xi0 + i] = sg_; b.d[MS, n] = kappa / d
        elif s == 1:
            nk = int(math.log(m_max / eps_in) / math.log(rho)) + 2
            t = eps_in * rho ** torch.arange(nk, dtype=torch.float64); f = t ** -0.5
            sgm = (f[1:] - f[:-1]) / (t[1:] - t[:-1]); a_k = torch.cat([sgm[1:] - sgm[:-1], -sgm[-1:]]); tk = t[1:]
            r = b._alloc(len(tk) + 1); idx = (r.start + torch.arange(len(tk))).to(dev)
            b.g[idx, MS] = -lam / kappa; b.g[idx, 0] = ((lam * (tk - eps_in)) / B).float().to(dev)
            b.u[idx, 0] = 1.0 / B; b.d[R0, idx] = (a_k / lam).float().to(dev)
            n0 = r.start + len(tk); b.g[n0, 0] = 1.0 / B; b.u[n0, 0] = float(f[-1]) / (B * 0.7310585786300049); b.d[R0, n0] = 1.0
        elif s == 2:
            r = b._alloc(2)
            for j, sg_ in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, R0] = sg_; b.u[n, R0] = sg_; b.d[QQ, n] = kappa
        elif s == 3:
            r = b._alloc(2)
            for j, sg_ in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, QQ] = sg_ / kappa; b.u[n, MS] += sg_ / kappa; b.u[n, 0] += sg_ * eps_in / B; b.d[TT, n] = 1.0
        elif s == 4:
            r = b._alloc(2)
            for j, sg_ in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, R0] = sg_; b.u[n, TT] = -0.5 * sg_; b.u[n, 0] += 1.5 * sg_ / B; b.d[ZZ, n] = 1.0
        elif s == 5:                      # ZF = ZZ * f   (instance gate)
            r = b._alloc(2)
            for j, sg_ in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, ZZ] = sg_; b.u[n, FL] = sg_; b.d[ZF, n] = 1.0
        else:
            r = b._alloc(2 * d)
            for sg_, off in ((1.0, 0), (-1.0, d)):
                n = r.start + off + i; b.g[n, xi0 + i] = sg_; b.u[n, ZF] = sg_; b.d[o0 + i, n] = 1.0
            for slot in (MS, R0, QQ, TT, ZZ, ZF):
                r = b._alloc(2)
                for j, sg_ in enumerate((1.0, -1.0)):
                    n = r.start + j; b.g[n, slot] = sg_; b.u[n, 0] = sg_ / B; b.d[slot, n] = -1.0



class NW:
    """custom neuron writer on a layer's MLP (transparent norm). terms = [(slot_index, coef)], constants via slot 0 coef c/B."""
    def __init__(self, layer, D):
        transparent(layer, D, ('post',))
        self.g, self.u, self.d = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight
        self.n = 0; self.cap = self.g.shape[0]
    def _a(self):
        assert self.n < self.cap, 'MLP too small'; self.n += 1; return self.n - 1
    def _row(self, W, n, terms, mul=1.0):
        for (i, cf) in terms: W[n, i] += mul * cf
    def const(self, v): return [(0, v / B)]
    def product(self, a_terms, b_terms, out_terms, scale=1.0):
        for sg in (1.0, -1.0):
            n = self._a(); self._row(self.g, n, a_terms, sg); self._row(self.u, n, b_terms, sg)
            for (i, cf) in out_terms: self.d[i, n] += scale * cf
    def copy(self, a_terms, out_terms, scale=1.0):
        for sg in (1.0, -1.0):
            n = self._a(); self._row(self.g, n, a_terms, sg); self.u[n, 0] += sg / B
            for (i, cf) in out_terms: self.d[i, n] += scale * cf
    def hinge(self, z_terms, out_terms, lam=1e3):
        """out += relu(z) via silu(lam z)/lam"""
        n = self._a(); self._row(self.g, n, z_terms, lam); self.u[n, 0] += 1.0 / B
        for (i, cf) in out_terms: self.d[i, n] += cf / lam


def _vt(c, S, l, k):
    if l == 0: return [(S.Y.start + k, 1.0), (S.PAD.start + k, -1e4)]
    return [(S.VL[l].start + k, 1.0)]
def _bt(c, S, l, j, k): return [(S.BL[l].start + j * (c.O >> l) + k, 1.0)]
TW = 1e-3


def build_ts(layer, D, c, S, l):
    nw = NW(layer, D); n_next = c.O >> (l + 1)
    for k in range(n_next):
        a = _vt(c, S, l, 2 * k); b = _vt(c, S, l, 2 * k + 1)
        zt = [(i, cf / TW) for (i, cf) in a] + [(i, -cf / TW) for (i, cf) in b]
        out = [(S.SL[l].start + k, 1.0)]
        nw.hinge(zt + nw.const(0.5), out); nw.hinge(zt + nw.const(-0.5), [(i, -cf) for (i, cf) in out])


def build_tm(layer, D, c, S, l):
    nw = NW(layer, D); n_next = c.O >> (l + 1)
    for k in range(n_next):
        s_ = [(S.SL[l].start + k, 1.0)]
        a = _vt(c, S, l, 2 * k); b = _vt(c, S, l, 2 * k + 1)
        diff = a + [(i, -cf) for (i, cf) in b]; out = [(S.VL[l + 1].start + k, 1.0)]
        nw.hinge(diff, out, lam=1e5); nw.copy(b, out)       # max(a,b) = b + relu(a-b): no large-number cancellation
        for j in range(l):
            aj = _bt(c, S, l, j, 2 * k); bj = _bt(c, S, l, j, 2 * k + 1)
            oj = [(S.BL[l + 1].start + j * n_next + k, 1.0)]
            nw.product(s_, aj + [(i, -cf) for (i, cf) in bj], oj); nw.copy(bj, oj)
        # new bit l = 1 - s  (1 if right child wins)
        ob = [(S.BL[l + 1].start + l * n_next + k, 1.0)]
        nw.copy(nw.const(1.0), ob); nw.copy(s_, ob, scale=-1.0)


def build_am(layer, D, c, S):
    """hard-max attention over LM-block steps: key = block best logit, value = 36 rails (8 local bits + 10 block bits, dual rail)."""
    hd = c.hd; hdo = hd + 3; nq, nkv = c.nq, c.nkv; sa = layer.self_attn; R = hd; cc = 2.0 ** 23; beta = 1e5
    transparent(layer, D, ('in',)); dev = sa.q_proj.weight.device
    G = math.sqrt(math.sqrt(2.0) * cc * cc / (math.sqrt(hdo) * math.sqrt(hd)))
    with torch.no_grad():
        Wq = sa.q_proj.weight.view(nq, 2, hdo, D); Wk = sa.k_proj.weight.view(nkv, hdo, D); Wv = sa.v_proj.weight.view(nkv, hdo, D)
        Wq[0, 0, 0, 0] = beta * math.sqrt(hd) / B
        Wq[0, 0, R, 0] = cc / B; Wq[0, 0, R + 2, 0] = cc / B; Wq[0, 1, :, 0] = 30.0 / B
        for h in range(1, nq): Wq[h, 1, :, 0] = 30.0 / B
        Wk[0, 0, S.VL[8].start] = 1.0; Wk[0, R + 1, 0] = cc / B; Wk[0, R + 2, S.MASKM.start] = -cc * 4
        for j in range(8):
            Wv[0, 2 * j, S.BL[8].start + j * 1] = 1.0                      # rail bit
            Wv[0, 2 * j + 1, S.BL[8].start + j] = -1.0; Wv[0, 2 * j + 1, 0] = 1.0 / B   # rail 1-bit
        for j in range(10):
            Wv[0, 16 + 2 * j, S.BLKCODE.start + 2 * j] = 1.0; Wv[0, 16 + 2 * j + 1, S.BLKCODE.start + 2 * j + 1] = 1.0
        Wo = sa.o_proj.weight.view(D, nq, hdo)
        for j in range(36): Wo[S.AMO.start + j, 0, j] = 1.0
        sa.q_norm.weight.fill_(G - 1.0); sa.k_norm.weight.fill_(G - 1.0)


def compile_entry(layer, entry, D, c, S, P):
    """P: per-inner-layer params dict: P['gq'][l], P['gk'][l]"""
    k = entry[0]
    with torch.no_grad():
        for p in layer.parameters(): p.zero_()
    if k == 'R': build_reg(layer, D, c, S, entry[1], entry[2])
    elif k == 'NG': ng_stage(layer, D, c, S, entry[1], entry[2])
    elif k == 'NH': build_qknorm(layer, D, c, S, entry[1], entry[2])
    elif k == 'W': build_bank(layer, D, c, S, entry[1], entry[2])
    elif k == 'A': build_attention(layer, D, c, S, entry[1], P['gq'][entry[1]], P['gk'][entry[1]])
    elif k == 'S': build_swiglu(layer, D, c, S, entry[1])
    elif k == 'TS': build_ts(layer, D, c, S, entry[1])
    elif k == 'TM': build_tm(layer, D, c, S, entry[1])
    elif k == 'AM': build_am(layer, D, c, S)
    else: raise ValueError(entry)


# ----------------------------------------------------------------------------- host frame program
@torch.no_grad()
def tile_payload(W, o0, j, c, dev):
    """W [out,in] -> payload rows [2P, H*(O+1)] for bank-layer j (contraction columns j*H*P..)."""
    P, H, O = c.P, c.H, c.O
    Wb = W[o0:o0 + O, j * H * P:(j + 1) * H * P].to(dev).to(torch.float64)
    Wp = torch.zeros(O, H * P, dtype=torch.float64, device=dev); Wp[:Wb.shape[0], :Wb.shape[1]] = Wb
    Wt = Wp.reshape(O, H, P).permute(2, 1, 0)
    rows = torch.cat([Wt, -Wt], 0)
    s = (-0.27 - rows.min(dim=2).values).clamp(min=0)
    v = silu_inv(rows + s[:, :, None]); ref = silu_inv(s)
    return torch.cat([v, ref[:, :, None]], 2).reshape(2 * P, H * (O + 1)).float().cpu()


class Prog:
    """Compact frame program.  Control frames are dense rows (flags/addresses); WRITE frames are compact jobs (payload + row/flag
    info) expanded on the GPU only inside the bank layer that consumes them (all other layers see them as exactly idle)."""
    def __init__(self, S, c, plan, dev):
        self.S, self.c, self.D, self.dev = S, c, S.D, dev
        self.ctrl = []; self.cpos = []; self.jobs = []; self.timeline = []; self.marks = {}; self.cur_pos = 0; self.total = 0
        self.bank_li = {(p[1], p[2]): li for li, p in enumerate(plan) if p[0] == 'W'}
    def new(self):
        f = torch.zeros(self.D); f[0] = B
        for l in range(self.c.L): f[self.S.MASK[l].start] = 1.0
        if self.c.tourn: f[self.S.MASKM.start] = 1.0
        return f
    def push(self, f, kind=-1):
        if kind < 0:
            self.timeline.append(('c', len(self.ctrl))); self.ctrl.append(f); self.cpos.append(self.cur_pos)
        else:
            self.jobs.append(dict(li=kind, dense=f[None])); self.timeline.append(('w', len(self.jobs) - 1))
        self.total += 1
    def push_write(self, li, pay, wf_slot):
        self.jobs.append(dict(li=li, pay=pay, wf=wf_slot)); self.timeline.append(('w', len(self.jobs) - 1)); self.total += pay.shape[0]
    def mark(self, n): self.marks[n] = len(self.ctrl)
    def heads(self, name, port=None, lanes=None):
        r = self.S.regs[name]; out = []
        for gh, h in enumerate(r['heads']):
            if (port is None or h['port'] == port) and (lanes is None or h['lane'] in lanes): out.append(gh)
        return out
    def fl(self, f, key): f[self.S.ACTF.start + self.S.fl[key]] = 1.0
    def ra(self, f, name, **kw):
        for gh in self.heads(name, **kw): f[self.S.regs[name]['RA'].start + gh] = 1.0
    def wa(self, f, name, **kw):
        for gh in self.heads(name, **kw): f[self.S.regs[name]['WA'].start + gh] = 1.0
    def ctrl_frames(self): return torch.stack(self.ctrl)
    def dense_job(self, job, dev):
        if 'dense' in job: return job['dense'].to(dev)
        S, c = self.S, self.c
        pay = job['pay'].to(dev) if 'pay' in job else tile_payload(job['W'], job['o0'], job['j'], c, dev).to(dev); m = pay.shape[0]
        F = torch.zeros(m, self.D, device=dev); F[:, 0] = B
        ar = torch.arange(m, device=dev); F[ar, S.SEL.start + ar] = 1.0
        F[:, S.WF.start + job['wf']] = 1.0; F[:, S.PAY.start:S.PAY.stop] = pay
        return F


def init_steps(pr):
    S, c = pr.S, pr.c
    for (op, j), li in pr.bank_li.items():
        f = pr.new(); f[S.SEL.start + c.dk - 1] = 1.0; f[S.WF.start + S.wf_idx[(op, j)]] = 1.0; f[S.INIT.start] = 1.0; pr.push(f, li)
    f = pr.new(); f[S.RINIT.start] = 1.0; pr.push(f)


def write_block(pr, op, W, blk):
    S, c = pr.S, pr.c
    for j in range(S.nbl[op]):
        pr.jobs.append(dict(li=pr.bank_li[(op, j)], W=W, o0=blk * c.O, j=j, m=2 * c.P, wf=S.wf_idx[(op, j)]))
        pr.timeline.append(('w', len(pr.jobs) - 1)); pr.total += 2 * c.P


def token_program(pr, tok, pos, Wd, emb):
    S, c = pr.S, pr.c; L = c.L; pr.cur_pos = pos
    pr.mark('emb'); f = pr.new(); f[S.EMBX.start:S.EMBX.stop] = emb(tok)
    pr.wa(f, 'R1_0'); pr.push(f)
    def nblocks(op): return -(-op_dims(c, op)[0] // c.O)
    for l in range(L):
        op = 4 * l
        for b in range(nblocks(op)):
            write_block(pr, op, Wd[op], b)
            pr.mark(f'qkv{l}_{b}'); f = pr.new(); f[S.RF.start + op] = 1; pr.ra(f, f'R1_{l}', port=0); pr.fl(f, ('NG', 'n1', l))
            pr.wa(f, f'R2_{l}', lanes=(2 * b, 2 * b + 1)); pr.push(f)
        pr.mark(f'attn{l}'); f = pr.new(); pr.ra(f, f'R2_{l}'); f[S.MASK[l].start] = 0.0; pr.fl(f, ('A', l)); pr.fl(f, ('NH', l)); pr.wa(f, f'R3_{l}'); pr.push(f)
        op = 4 * l + 1
        for b in range(nblocks(op)):
            write_block(pr, op, Wd[op], b)
            pr.mark(f'o{l}_{b}'); f = pr.new(); f[S.RF.start + op] = 1; pr.ra(f, f'R3_{l}'); pr.ra(f, f'R1_{l}', port=1)
            pr.wa(f, f'R4_{l}', lanes=(2 * b, 2 * b + 1)); pr.push(f)
        op = 4 * l + 2
        for b in range(nblocks(op)):
            write_block(pr, op, Wd[op], b)
            pr.mark(f'gu{l}_{b}'); f = pr.new(); f[S.RF.start + op] = 1; pr.ra(f, f'R4_{l}', port=0); pr.fl(f, ('NG', 'n3', l))
            pr.wa(f, f'R5_{l}', lanes=(2 * b, 2 * b + 1)); pr.push(f)
        pr.mark(f'sw{l}'); f = pr.new(); pr.ra(f, f'R5_{l}'); pr.fl(f, ('S', l)); pr.wa(f, f'R6_{l}'); pr.push(f)
        op = 4 * l + 3
        for b in range(nblocks(op)):
            write_block(pr, op, Wd[op], b)
            pr.mark(f'down{l}_{b}'); f = pr.new(); f[S.RF.start + op] = 1; pr.ra(f, f'R6_{l}'); pr.ra(f, f'R4_{l}', port=1)
            pr.wa(f, f'R1_{l + 1}', lanes=(2 * b, 2 * b + 1)); pr.push(f)
    op = 4 * L
    for b in range(nblocks(op)):
        write_block(pr, op, Wd[op], b)
        pr.mark(f'lm{b}'); f = pr.new(); f[S.RF.start + op] = 1; pr.ra(f, f'R1_{L}', port=0); pr.fl(f, ('NG', 'nf', 0))
        if not c.tourn: pr.wa(f, 'Rlg', lanes=(2 * b, 2 * b + 1))
        else:
            nv = min(c.O, c.V - b * c.O); f[S.PAD.start + nv:S.PAD.start + c.O] = 1.0
            for j in range(10):
                bit = (b >> j) & 1; f[S.BLKCODE.start + 2 * j] = bit; f[S.BLKCODE.start + 2 * j + 1] = 1 - bit
            f[S.MASKM.start] = 0.0
        pr.push(f)
    pr.mark('emit'); f = pr.new()
    if not c.tourn: pr.ra(f, 'Rlg')
    pr.push(f)
    return len(pr.ctrl) - 1


# ----------------------------------------------------------------------------- layer-major executor
class Runner:
    def __init__(self, S, c, plan, P, dev='cuda', eps=1e-30):
        self.S, self.c, self.plan, self.P, self.dev, self.D = S, c, plan, P, dev, S.D
        hdo = c.hd + 3; rot = c.hd
        rp = dict(rope_type='default', rope_theta=c.theta, partial_rotary_factor=(rot + 0.5) / hdo,
                  mrope_section=[rot // 2 // 3 + (1 if rot // 2 % 3 > i else 0) for i in range(3)])
        types = ['full_attention' if e[0] in ('A', 'AM') else 'linear_attention' for e in plan]
        self.cfg = Qwen3_5TextConfig(hidden_size=self.D, intermediate_size=c.I_MLP, num_hidden_layers=len(plan), layer_types=types,
                                     vocab_size=8, rms_norm_eps=eps, tie_word_embeddings=False, linear_num_key_heads=c.H,
                                     linear_num_value_heads=c.H, linear_key_head_dim=c.dk, linear_value_head_dim=c.dv,
                                     linear_conv_kernel_dim=4, num_attention_heads=c.nq, num_key_value_heads=c.nkv, head_dim=hdo,
                                     max_position_embeddings=4096, rope_parameters=rp, mtp_num_hidden_layers=0)
        self.cfg._attn_implementation = 'sdpa'
        import copy
        def mk(**kw):
            cf = copy.deepcopy(self.cfg)
            for k_, v_ in kw.items(): setattr(cf, k_, v_)
            cf._attn_implementation = 'sdpa'; return cf
        self.cfgv = {'mlp': mk(linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=2, linear_value_head_dim=2),
                     'gdn': mk(intermediate_size=16), 'attn': mk(intermediate_size=16)}
        self.kind_cfg = {'NG': 'mlp', 'S': 'mlp', 'TS': 'mlp', 'TM': 'mlp', 'R': 'gdn', 'W': 'gdn', 'NH': 'gdn', 'A': 'attn', 'AM': 'attn'}
        self.cache = DynamicCache(config=self.cfg)
        self.rot = Qwen3_5TextRotaryEmbedding(config=self.cfg).to(dev)
        self.offs = {}; self.stats = dict(rows=0, build_s=0.0, run_s=0.0)

    @torch.no_grad()
    def run(self, pr, seg=2048, trace_rows=None, trace_layers=None):
        dev = self.dev; Hs = pr.ctrl_frames().to(dev); posT = torch.tensor(pr.cpos); trace = {}
        for li, e in enumerate(self.plan):          # AM key/value cache is per-token scratch: truncated at token boundary (data-independent)
            if e[0] == 'AM' and self.offs.get(li, 0) > 0:
                self.cache.layers[li].reset(); self.offs[li] = 0
        for li, entry in enumerate(self.plan):
            t0 = time.time()
            with torch.device(dev): layer = Qwen3_5DecoderLayer(self.cfgv[self.kind_cfg[entry[0]]], li).eval()
            compile_entry(layer, entry, self.D, self.c, self.S, self.P)
            if dev == 'cuda': torch.cuda.synchronize()
            self.stats['build_s'] += time.time() - t0; t0 = time.time()
            off = self.offs.get(li, 0)
            # items for this layer in timeline order: all control rows + write jobs addressed to this layer
            items = [it for it in pr.timeline if it[0] == 'c' or pr.jobs[it[1]]['li'] == li]
            buf = []; nbuf = 0
            def flush():
                nonlocal off, buf, nbuf
                if not buf: return
                rows = []; cidx = []; pids = []; pos_ = 0
                i = 0
                while i < len(buf):
                    if buf[i][0] == 'c':
                        j = i
                        while j < len(buf) and buf[j][0] == 'c': j += 1
                        ids = torch.tensor([b_[1] for b_ in buf[i:j]], device=dev)
                        rows.append(Hs[ids]); cidx.append((pos_, ids)); pids += posT[ids.cpu()].tolist(); pos_ += len(ids); i = j
                    else:
                        F = pr.dense_job(pr.jobs[buf[i][1]], dev); rows.append(F); pids += [pr.cur_pos] * F.shape[0]; pos_ += F.shape[0]; i += 1
                x = torch.cat(rows)[None]; n = x.shape[1]
                cp = torch.arange(off, off + n, device=dev); pid = torch.tensor(pids, device=dev)[None]
                pe = self.rot(x, pid[None].expand(3, 1, -1)); mask = None
                if entry[0] in ('A', 'AM'):
                    mask = (torch.arange(off + n, device=dev)[None, :] <= (torch.arange(n, device=dev)[:, None] + off))[None, None]
                y = layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pid, past_key_values=self.cache, cache_position=cp, use_cache=True)
                if isinstance(y, tuple): y = y[0]
                for (p0, ids) in cidx: Hs[ids] = y[0, p0:p0 + len(ids)].float()
                off += n; self.stats['rows'] += n; buf = []; nbuf = 0
            for it in items:
                m = 1 if it[0] == 'c' else pr.jobs[it[1]].get('m', 1)
                if nbuf + m > seg: flush()
                buf.append(it); nbuf += m
            flush()
            self.offs[li] = off
            if dev == 'cuda': torch.cuda.synchronize()
            self.stats['run_s'] += time.time() - t0
            if trace_rows is not None and (trace_layers is None or li in trace_layers):
                for nme, r in trace_rows.items(): trace[(nme, li)] = Hs[r.to(dev) if torch.is_tensor(r) else r].cpu().clone()
            del layer
            if dev == 'cuda': torch.cuda.empty_cache()
            if os.environ.get('VS2_PROGRESS') and li % 50 == 0: print(f'  [progress] layer {li}/{len(self.plan)} {entry[0]} build_s={self.stats["build_s"]:.1f} run_s={self.stats["run_s"]:.1f} rows={self.stats["rows"]}', flush=True)
        return Hs.cpu(), trace
