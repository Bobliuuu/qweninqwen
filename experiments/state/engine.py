"""trueqiq_state engine: GDN recurrent state as float weight memory (agent B).

Design summary (see README.md):
  state S[layer,head] : dk x dv fp32.  row r (key index) <- one write frame (k = one-hot(r)); read q gives out = S^T q.
  * activation x enters q through the conv SiLU as the exact pair silu(x)-silu(-x)=x : tile rows (j, P+j) hold (+c,-c)
    P=(dk-2)//2 contraction elements per head.  q[dk-1] = Rq constant (ballast coordinate).
  * weights enter v (host transport).  v goes through SiLU: host pre-inverts silu on the payload (weights only), with a per-row
    shift s>=0.3 (silu^-1 needs t>-0.278); the shift is cancelled by a reference column (dv-2) holding s.
  * l2norm of q and the per-head gated RMSNorm both only multiply by an x-dependent scalar; a huge constant column (dv-1,
    value Bg at row dk-1, written once by an INIT frame) dominates the RMSNorm rms so the scalar cancels exactly.
  * RMSNorm gate silu(z) is set at the flat minimum of silu (z*=-1.2785) so the transparent-outer-norm scale error enters only
    in 2nd order.  Write frames: gate z=-40 -> 0.   beta = sigmoid(+-30), decay exp(g)=1 -> exact overwrite.
  * out_proj sums (col - ref) over heads (and layers via residual add): K-tile accumulation happens IN the model.
"""
import os
import math, sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'qweninqwen'))
from machine import Layout, make_outer, BALLAST as BALLAST0

ZSTAR = -1.2784645427610738          # argmin silu
def silu64(u): return u / (1 + torch.exp(-u))
SILU_MIN = float(silu64(torch.tensor(ZSTAR, dtype=torch.float64)))

def silu_inv(t, bis=14, newton=5):
    """inverse of silu on [-0.27846, inf) (right branch), fp64: coarse bisection then Newton polish (to ~1e-16)."""
    lo = torch.full_like(t, ZSTAR); hi = t.clamp(min=0) + 2.0
    for _ in range(bis):
        mid = (lo + hi) / 2
        big = silu64(mid) > t
        hi = torch.where(big, mid, hi); lo = torch.where(big, lo, mid)
    u = (lo + hi) / 2
    for _ in range(newton):
        sg = torch.sigmoid(u)
        u = u - (u * sg - t) / (sg * (1 + u * (1 - sg)))
    return u


class StateMV:
    """Spec of a bank of nL GDN layers (H heads each) acting as a paged float matvec unit."""
    def __init__(self, layout, nL=1, H=16, dk=128, dv=128, K=None, B=BALLAST0, Rq=100.0, Bg=1e8, name='B', G=1):
        assert nL % G == 0
        self.G_, self.KL = G, nL // G          # G output groups x KL layers (K-tile layers) per group
        self.nL, self.H, self.dk, self.dv, self.B, self.Rq, self.Bg = nL, H, dk, dv, B, Rq, Bg
        self.P = (dk - 2) // 2                  # contraction elems per head tile
        self.O = dv - 2                         # outputs per head (cols; col dv-2 = ref, dv-1 = ballast)
        self.K = K if K is not None else self.KL * H * self.P
        assert self.K <= self.KL * H * self.P, f'K={self.K} exceeds capacity {self.KL*H*self.P} of {self.KL} K-layers x {H} heads x {self.P}'
        self.layout = layout
        a = layout.alloc
        self.WF = a(name + '_WF', nL); self.RF = a(name + '_RF', 1); self.INIT = a(name + '_INIT', 1)
        self.SEL = a(name + '_SEL', dk)
        self.PAYn = H * (self.O + 1)                   # payload floats per (layer): O outputs + ref per head
        self.PAY = [a(f'{name}_PAY{l}', self.PAYn) for l in range(nL)]
        self.X = a(name + '_X', self.K); self.Y = a(name + '_Y', self.O * G)
        self.G = Bg * Rq / (math.sqrt(dv) * SILU_MIN)
        self.dev = 'cpu'; self.ft = torch.float64      # ft: dtype of host frame prep (float32 is ~10x faster on GPU, same accuracy to ~1e-7)

    # ---- compile into a model whose layers l0..l0+nL-1 are 'linear_attention' ----
    def compile(self, model, l0=0):
        D, H, dk, dv, P, O, B = self.layout.n, self.H, self.dk, self.dv, self.P, self.O, self.B
        assert model.config.hidden_size == D
        kd = H * dk; vd = H * dv; KB = 60.0
        for l in range(self.nL):
            L = model.model.layers[l0 + l]; g = L.linear_attn
            with torch.no_grad():
                L.input_layernorm.weight.fill_(B / math.sqrt(D) - 1.0)
                g.conv1d.weight.zero_(); g.conv1d.weight[:, 0, -1] = 1.0
                g.A_log.zero_(); g.dt_bias.zero_(); g.norm.weight.fill_(1.0)
                Wqkv = g.in_proj_qkv.weight; Wqkv.zero_()
                for h in range(H):
                    gl, kl = divmod(l, self.KL); t = kl * H + h
                    for j in range(P):
                        xi = t * P + j
                        if xi < self.K:
                            Wqkv[h * dk + j, self.X.start + xi] = 1.0
                            Wqkv[h * dk + P + j, self.X.start + xi] = -1.0
                    Wqkv[h * dk + dk - 1, 0] = self.Rq / B
                    for r in range(dk):
                        Wqkv[kd + h * dk + r, 0] = -KB / B
                        Wqkv[kd + h * dk + r, self.SEL.start + r] = 2 * KB
                    for c in range(O + 1):
                        Wqkv[2 * kd + h * dv + c, self.PAY[l].start + h * (O + 1) + c] = 1.0
                    Wqkv[2 * kd + h * dv + dv - 1, self.INIT.start] = self.Bg
                g.in_proj_z.weight.zero_()
                g.in_proj_z.weight[:, 0] = -40.0 / B; g.in_proj_z.weight[:, self.RF.start] = 40.0 + ZSTAR
                g.in_proj_b.weight.zero_(); g.in_proj_b.weight[:, 0] = -30.0 / B; g.in_proj_b.weight[:, self.WF.start + l] = 60.0
                g.in_proj_a.weight.zero_(); g.in_proj_a.weight[:, 0] = -30.0 / B
                Wo = g.out_proj.weight; Wo.zero_(); gl_ = l // self.KL
                for h in range(H):
                    for c in range(O):
                        Wo[self.Y.start + gl_ * O + c, h * dv + c] = self.G
                        Wo[self.Y.start + gl_ * O + c, h * dv + O] = -self.G
        return model

    # ---- host frame builders: TRANSPORT ONLY (flags, addresses, weight payloads) ----
    def new_frames(self, n):
        return torch.zeros(n, self.layout.n, dtype=self.ft, device=self.dev)

    def _norm_comp(self, u, pay_idx):
        """pre-compensate the transparent outer RMSNorm for a frame made only of host-known values (weights/flags):
        outer norm gives h = u*B/sqrt(sum u^2); we want h_payload exactly = intended payload."""
        p = u[:, pay_idx].clone(); other = u.clone(); other[:, pay_idx] = 0; other[:, 0] = 0
        o2 = (other ** 2).sum(1).double(); p2 = (p ** 2).sum(1).double(); g = torch.ones(u.shape[0], dtype=torch.float64, device=u.device)
        for _ in range(30):
            g = self.B / torch.sqrt(self.B ** 2 + o2 + p2 / g ** 2)
        u[:, pay_idx] = (p / g[:, None]).to(u.dtype)
        return u

    def write_frames(self, rows, layers_payload, row_sign=None):
        """rows: LongTensor [n] key rows to write; layers_payload: list (len nL) of tensors [n, H, O] target weights c
        (float64/32; None for a layer not written).  Returns frames [n, D] (float64)."""
        n = len(rows); u = self.new_frames(n); u[:, 0] = self.B
        u[torch.arange(n, device=self.dev), self.SEL.start + rows.to(self.dev)] = 1.0
        pay_idx = []
        for l, c in enumerate(layers_payload):
            if c is None: continue
            u[:, self.WF.start + l] = 1.0
            c = c.to(self.ft)
            s = (-0.27 - c.min(dim=2).values).clamp(min=0)    # [n,H] shift per head/row: 0 unless some w < -0.27 (silu^-1 domain)
            v = silu_inv(c + s[:, :, None])
            ref = silu_inv(s)
            blk = torch.cat([v, ref[:, :, None]], dim=2).reshape(n, -1)
            u[:, self.PAY[l]] = blk
            pay_idx += list(range(self.PAY[l].start, self.PAY[l].stop))
        return self._norm_comp(u, pay_idx) if pay_idx else u

    def init_frames(self):
        out = self.new_frames(self.nL); out[:, 0] = self.B
        for l in range(self.nL):
            out[l, self.WF.start + l] = 1; out[l, self.INIT.start] = 1; out[l, self.SEL.start + self.dk - 1] = 1
        return self._norm_comp(out, [self.INIT.start])

    def read_frames(self, X):
        """X [m, K] activation vectors.  (TEST HARNESS: in the full model X slot is filled by outer layers.)"""
        m = X.shape[0]; u = self.new_frames(m); u[:, 0] = self.B; u[:, self.RF.start] = 1
        u[:, self.X] = X.to(self.ft)
        return u

    def tile_weights(self, W, o0):
        """Slice real weight matrix W[Oall,Kall] -> write rows and payload per layer for the output super-block o0 : o0+G*O.
        Layer l = (group g, K-layer kl) holds W[o0+g*O : +O, tiles kl*H..(kl+1)*H) .  returns (rows [2P], [payload [2P,H,O] per layer])
        with row r<P: +W[:,j] (read via silu(+x_j)), row P+j: -W[:,j] (read via silu(-x_j))."""
        P, O, H, KL = self.P, self.O, self.H, self.KL
        Kall = W.shape[1]; Kp = KL * H * P
        Wb = W[o0:o0 + self.G_ * O].to(self.dev).to(self.ft)
        Wp = torch.zeros(self.G_ * O, Kp, dtype=self.ft, device=self.dev); Wp[:Wb.shape[0], :min(Kall, Kp)] = Wb[:, :min(Kall, Kp)]
        Wt = Wp.reshape(self.G_, O, KL, H, P)
        pl = []
        for l in range(self.nL):
            g, kl = divmod(l, KL)
            pos = Wt[g, :, kl].permute(2, 1, 0)            # [j, h, O]
            pl.append(torch.cat([pos, -pos], 0))
        return torch.arange(2 * P, device=self.dev), pl


def build_state_matvec(layout, nL=1, H=16, K=896, B=BALLAST0, name='B', dk=128, dv=128, G=1):
    """Step 1 of the INTERFACE contract: allocate fields (call BEFORE make_outer so layout.n is final).
    spec = build_state_matvec(layout, nL, H, K);  model = make_outer(layout, [...'linear_attention'...], gdn=dict(hk=H,hv=H,dk=dk,dv=dv));
    spec.compile(model, l0)   # writes weights into layers l0..l0+nL-1.  Needs frame BALLAST value spec.B (>=1e6 recommended)."""
    return StateMV(layout, nL=nL, H=H, dk=dk, dv=dv, K=K, B=B, name=name, G=G)
