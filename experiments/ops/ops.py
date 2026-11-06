"""trueqiq_ops/ops.py -- FLOAT inner-Qwen non-matvec ops compiled into stock Qwen3_5 outer layers (agent C).

Conventions (see trueqiq/INTERFACE.md): residual stream ('frame') of D floats, slot 0 = BALLAST constant.
Outer norms are made TRANSPARENT ((1+w)=BALLAST/sqrt(D)) so every sublayer sees ~ the raw stream
(rel. error |stream_without_ballast|^2/(2 BALLAST^2)  -> use BALLAST>=1e8 in fp32 for <1e-9; 1e4 gives ~1e-3 on big acts).
All writers use only linear maps + SwiGLU gadgets:
   identity  z = silu(z)-silu(-z)        product  x*y = silu(x)*y + silu(-x)*(-y)     (both EXACT, 2 neurons / element)
"""
import math, sys, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'qweninqwen'))
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from machine import Layout, GATE_ON

BALLAST = 2.0 ** 27     # power of two: exact in bf16 (and c/ballast weights exact)


def sl(s):  # slice -> (start, size)
    return s.start, s.stop - s.start


def make_outer2(D, layer_types, intermediate, attn=dict(heads=1, kv=1, hd=64), rope_theta=1e6, rot=None,
                vocab=8, eps=1e-12, dtype=torch.float32, rope_override=None):
    """machine.make_outer + explicit rope config (theta, rotated dims). rot = #rotated head dims (default hd_o-2)."""
    hd = attn['hd']; rot = rot or hd - 2
    rp = rope_override or dict(rope_type='default', rope_theta=rope_theta, partial_rotary_factor=(rot + 0.5) / hd,
                               mrope_section=[rot // 2 // 3 + (1 if rot // 2 % 3 > i else 0) for i in range(3)])
    cfg = Qwen3_5TextConfig(hidden_size=D, intermediate_size=intermediate, num_hidden_layers=len(layer_types),
                            layer_types=layer_types, vocab_size=vocab, rms_norm_eps=eps, tie_word_embeddings=False,
                            linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
                            linear_conv_kernel_dim=4, num_attention_heads=attn['heads'], num_key_value_heads=attn['kv'],
                            head_dim=hd, max_position_embeddings=4096, rope_parameters=rp, mtp_num_hidden_layers=0)
    m = Qwen3_5ForCausalLM(cfg).to(dtype)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
    return m


def transparent(layer, D, ballast=BALLAST, which=('in', 'post')):
    with torch.no_grad():
        v = ballast / math.sqrt(D) - 1.0
        if 'in' in which: layer.input_layernorm.weight.fill_(v)
        if 'post' in which: layer.post_attention_layernorm.weight.fill_(v)


class MLPBuilder:
    """Fills neurons of layer.mlp.  Sublayer input = post_attention_layernorm(stream); transparent unless `norm_slots` given."""
    def __init__(self, layer, D, ballast=BALLAST, transparent_norm=True):
        self.l, self.D, self.B = layer, D, ballast; self.n = 0
        self.g, self.u, self.d = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight
        self.cap = self.g.shape[0]
        if transparent_norm: transparent(layer, D, ballast, ('post',))

    def _alloc(self, k):
        assert self.n + k <= self.cap, f'MLP too small: need {self.n + k} have {self.cap}'
        r = slice(self.n, self.n + k); self.n += k; return r

    def swiglu(self, G, U, O, scale=1.0, gscale=1.0):
        """O += scale * silu(gscale*G) * U   (native outer SwiGLU), elementwise over slots of equal size."""
        (g0, k), (u0, _), (o0, _) = sl(G), sl(U), sl(O); r = self._alloc(k); i = torch.arange(k)
        with torch.no_grad():
            self.g[r.start + i, g0 + i] = gscale; self.u[r.start + i, u0 + i] = 1.0; self.d[o0 + i, r.start + i] = scale

    def copy(self, A, O, scale=1.0):
        """O += scale*A, exact identity gadget (2 neurons/elt). up = ballast read (=1)."""
        (a0, k), (o0, _) = sl(A), sl(O); r = self._alloc(2 * k); i = torch.arange(k)
        with torch.no_grad():
            for s, off in ((1.0, 0), (-1.0, k)):
                n = r.start + off + i
                self.g[n, a0 + i] = s; self.u[n, 0] = s / self.B; self.d[o0 + i, n] = scale

    def product(self, X, Y, O, scale=1.0):
        """O += scale * X*Y elementwise, exact (polarisation of silu: x = silu(x)-silu(-x))."""
        (x0, k), (y0, _), (o0, _) = sl(X), sl(Y), sl(O); r = self._alloc(2 * k); i = torch.arange(k)
        with torch.no_grad():
            for s, off in ((1.0, 0), (-1.0, k)):
                n = r.start + off + i
                self.g[n, x0 + i] = s; self.u[n, y0 + i] = s; self.d[o0 + i, n] = scale

    def square(self, X, O, scale=1.0): self.product(X, X, O, scale)


# ------------------------------------------------------------------ (1) RMSNorm
def rms_scale(d, eps_in, ballast=BALLAST):
    """s such that XS = s*x makes  sum(stream^2) = ballast^2 + s^2|x|^2 = s^2 (|x|^2 + d*eps_in)."""
    return ballast / math.sqrt(d * eps_in)


def rmsnorm_front(layer, XS, D, d, which='post'):
    """Make `layer`'s sublayer input = normalised x.  Stream must hold XS = rms_scale*x (producer writes it: any
    linear writer / identity gadget with scale rms_scale).  Layer norm weight (1+w)=1 on XS slots, 0 elsewhere.
    Sublayer input slice at XS then equals  sqrt(D/d) * x/sqrt(mean(x^2)+eps_in)   (inner RMSNorm without gain).
    Returns the factor c=sqrt(d/D) consumers must multiply by.  NOTE: afterwards XS (~1e12) poisons transparency of
    LATER layers in the same step -> norm layer must be the last compute layer of a round (or XS erased).  Gain
    g is folded into the consuming matvec weights W' = W diag(g) (weight reformatting)."""
    x0, k = sl(XS); assert k == d
    nm = layer.post_attention_layernorm if which == 'post' else layer.input_layernorm
    with torch.no_grad():
        nm.weight.fill_(-1.0); nm.weight[x0:x0 + k] = 0.0
    return math.sqrt(d / D)


# ------------------------------------------------------------------ (3)(4) RoPE + softmax attention + KV cache
def attn_config(nq, nkv, hd_in, theta, pad=2):
    """outer attention geometry for an inner head layout: outer heads == inner heads (GQA native), head_dim_o = hd_in+pad;
    the first hd_in dims are ROTATED (partial_rotary_factor) with inner theta -> rotate_half pairs (i,i+hd/2) identical."""
    return dict(attn=dict(heads=nq, kv=nkv, hd=hd_in + pad), rope_theta=theta, rot=hd_in)


def compile_attention(layer, D, Q, K, V, O, nq, nkv, hd, ballast=BALLAST, c=2.0 ** 23, which='in', kscale=1.0):
    """Layer's full-attention sublayer executes inner softmax attention.
    Stream slots: Q (nq*hd, head-major, already normed/biased), K (nkv*hd), V (nkv*hd): written by producers (matvec results);
    O (nq*hd) receives per-head attention output (before inner o_proj).  Outer KV cache == inner KV cache (post-norm, post-RoPE K).
    Neutralisations: outer q/k-norm -> constant (c dims, saturation: rel err |q|^2/2c^2) with gain G chosen so that
    score = q.k/sqrt(hd_in) exactly; output gate: sigmoid(GATE_ON)=1; causal mask native; RoPE via config (theta, partial rotary)."""
    hdo = hd + 2; sa = layer.self_attn; R = hd
    if which == 'in': transparent(layer, D, ballast, ('in',))
    q0, _ = sl(Q); k0, _ = sl(K); v0, _ = sl(V); o0, _ = sl(O)
    G = Gq = Gk = c / (hdo * hd) ** 0.25
    dt = sa.q_norm.weight.dtype
    if dt != torch.float32:      # bf16: store w=G-1 on the bf16 grid; choose the PAIR (Gq,Gk) whose product is closest to target
        tgt = c * c / math.sqrt(hdo * hd)
        def rnd(x): return (torch.tensor(float(x)) - 1).to(dt).float().item() + 1
        best = (9, 0, 0)
        for a in range(-300, 301):
            gq = rnd(G * (1 + a * 2.0 ** -9 / 2)); gk = rnd(tgt / gq)
            e = abs(gq * gk / tgt - 1)
            if e < best[0]: best = (e, gq, gk)
        _, Gq, Gk = best
    with torch.no_grad():
        Wq = sa.q_proj.weight.view(nq, 2, hdo, D); Wk = sa.k_proj.weight.view(nkv, hdo, D); Wv = sa.v_proj.weight.view(nkv, hdo, D)
        j = torch.arange(hd)
        for h in range(nq):
            Wq[h, 0, j, q0 + h * hd + j] = 1.0
            Wq[h, 0, R, 0] = c / ballast
            Wq[h, 1, j, 0] = GATE_ON / ballast
        for g in range(nkv):
            Wk[g, j, k0 + g * hd + j] = kscale; Wk[g, R + 1, 0] = c / ballast
            Wv[g, j, v0 + g * hd + j] = 1.0
        Wo = sa.o_proj.weight.view(D, nq, hdo)
        for h in range(nq): Wo[o0 + h * hd + j, h, j] = 1.0
        sa.q_norm.weight.fill_(Gq - 1.0); sa.k_norm.weight.fill_(Gk - 1.0)


# ------------------------------------------------------------------ (6) LM-head argmax in-model (hard-max attention) + bit decode head
def argmax_rails(vocab, nbits=None):
    """dual-rail address code of every vocab row: (vocab, 2*nbits) float; rails [b, 1-b]."""
    nb = nbits or max(1, (vocab - 1).bit_length())
    i = torch.arange(vocab)[:, None]; b = ((i >> torch.arange(nb)[None]) & 1).float()
    return torch.stack([b, 1 - b], -1).reshape(vocab, 2 * nb), nb


def compile_argmax_head(model, D, O, nb, vocab, kappa=1.0, ballast=BALLAST):
    """outer lm_head rows v: logit_v = kappa * sum_rails (code_v . O)  (= kappa*nb at the winning token, <= kappa*(nb-1) elsewhere).
    Outer greedy argmax over the outer vocab == inner argmax token id (the PERMITTED outer argmax/append)."""
    codes, _ = argmax_rails(vocab, nb); o0, k = sl(O); assert k == 2 * nb
    with torch.no_grad():
        model.lm_head.weight[:, o0:o0 + k] = kappa * codes.to(model.lm_head.weight.dtype)
        model.model.norm.weight.fill_(ballast / math.sqrt(D) - 1.0)


# ------------------------------------------------------------------ (1b) NARROW real-width RMSNorm: 6 transparent MLP layers, any d, any eps, no poison
class RMSNormUnit:
    """Exact RMSNorm of a d-wide stream slot X_IN (d = 896/1024/...) using ONLY transparent MLP gadgets (no GDN width needed, no XS poison):
         m = sum(x^2)/d + eps ;  r = m^-1/2 by  piecewise-linear init (silu-hinges) + ONE Newton step ;  out_i = x_i * r.
    layers[0..5] = six outer layers (any type; only their MLPs are used).  Scalar scratch slots are erased by layer 5 (reusable).
      L0  MS  = kappa*sum x_i^2/d                     (square gadget, 2d neurons)
      L1  R0  = PWL(m)  (relative error ~ (ln rho)^2*3/32)   (n_knots neurons)
      L2  QQ  = kappa*R0^2
      L3  TT  = QQ*m/kappa             (~ 1+O(e))
      L4  ZZ  = R0*(1.5-0.5*TT)        (Newton: error 1.5 e^2 ~ 1e-10)
      L5  OUT_i = x_i*ZZ  (+ erase MS,R0,QQ,TT,ZZ)
    The consumer folds the inner gain: W' = W diag(g)."""
    def __init__(self, layers, D, X_IN, OUT, scal, d, eps_in, ballast=BALLAST, m_max=1e8, rho=1.012, kappa=1e-3, accumulate=True):
        assert len(layers) == 6
        self.kappa, self.d, self.eps = kappa, d, eps_in
        B = ballast; xi0, k = sl(X_IN); assert k == d; o0, _ = sl(OUT)
        MS, R0, QQ, TT, ZZ = [scal[n].start for n in ('MS', 'R0', 'QQ', 'TT', 'ZZ')]
        lam = 1e12
        mb = [MLPBuilder(l, D, B) for l in layers]
        def mlin_up(mbd, row, c=1.0):          # up_proj row <- c*(m) , m = MS/kappa + eps
            mbd.u[row, MS] += c / kappa; mbd.u[row, 0] += c * eps_in / B
        with torch.no_grad():
            # L0: square gadget with scalar sink
            b = mb[0]; r = b._alloc(2 * d); i = torch.arange(d)
            for s, off in ((1.0, 0), (-1.0, d)):
                n = r.start + off + i
                b.g[n, xi0 + i] = s; b.u[n, xi0 + i] = s; b.d[MS, n] = kappa / d
            # L1: PWL of m^-1/2 on geometric knots t_0=eps .. m_max
            nk = int(math.log(m_max / eps_in) / math.log(rho)) + 2
            t = eps_in * rho ** torch.arange(nk, dtype=torch.float64); f = t ** -0.5
            sl_ = (f[1:] - f[:-1]) / (t[1:] - t[:-1]); sl_ = torch.cat([sl_, sl_[-1:]])
            c = sl_.clone(); c[1:] -= sl_[:-1]
            b = mb[1]; r = b._alloc(nk + 1)
            idx = r.start + torch.arange(nk)
            b.g[idx, MS] = lam / kappa; b.g[idx, 0] = ((lam * (eps_in - t)) / B).float()
            b.u[idx, 0] = 1.0 / B; b.d[R0, idx] = (c / lam).float()
            # constant neuron: R0 += f(t0): gate=ballast(=1 after weight), up=const
            n0 = r.start + nk; b.g[n0, 0] = 1.0 / B; b.u[n0, 0] = float(f[0]) / (B * 0.7310585786300049); b.d[R0, n0] = 1.0
            # L2: QQ = kappa*R0^2
            b = mb[2]; r = b._alloc(2)
            for j, s in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, R0] = s; b.u[n, R0] = s; b.d[QQ, n] = kappa
            # L3: TT = (QQ/kappa) * m = R0^2 m ~ 1   (up = m linear combo)
            b = mb[3]; r = b._alloc(2)
            for j, s in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, QQ] = s / kappa; mlin_up(b, n, s); b.d[TT, n] = 1.0
            # L4: ZZ = R0*(1.5 - 0.5 TT)
            b = mb[4]; r = b._alloc(2)
            for j, s in enumerate((1.0, -1.0)):
                n = r.start + j; b.g[n, R0] = s; b.u[n, TT] = -0.5 * s; b.u[n, 0] += 1.5 * s / B; b.d[ZZ, n] = 1.0
            # L5: OUT_i = x_i*ZZ ; erase scalars (identity gadget)
            b = mb[5]; r = b._alloc(2 * d); i = torch.arange(d)
            for s, off in ((1.0, 0), (-1.0, d)):
                n = r.start + off + i
                b.g[n, xi0 + i] = s; b.u[n, ZZ] = s; b.d[o0 + i, n] = 1.0
            for slot in (MS, R0, QQ, TT, ZZ):
                r = b._alloc(2)
                for j, s in enumerate((1.0, -1.0)):
                    n = r.start + j; b.g[n, slot] = s; b.u[n, 0] = s / B; b.d[slot, n] = -1.0
        self.neurons = [m_.n for m_ in mb]
