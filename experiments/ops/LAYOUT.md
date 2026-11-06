# LAYOUT.md: real-dim pieces from agent C (Qwen3-0.6B d=1024 / Qwen2.5-0.5B d=896)

Geometry assumed (agreed with a87): uniform GDN H=16, dk=130, dv=258; MLP intermediate 8192 for every layer; BALLAST=2^27, eps_outer tiny (1e-30 as vslice);
all outer norms transparent; attention head_dim_o = hd+3 = 131 (vslice.build_attention), nq=16, nkv=8.

## 1. Residual RMSNorm (n1, n3, nf): `ops.RMSNormUnit` (6 outer layers, MLP-only; use GDN-type layers)
API (stable): `RMSNormUnit(layers[6], D, X_IN, OUT, scal={'MS','R0','QQ','TT','ZZ'} (1-slot slices), d, eps_in, ballast)`; input slot X_IN (d floats), output slot OUT (d floats, ADDED; = x/sqrt(mean x^2+eps), no gain -> fold g into the next bank's W' = W diag(g)).
| layer | MLP neurons | does |
|---|---|---|
| 0 | 2d | MS = kappa*sum x^2/d (square gadget, scalar sink) |
| 1 | 2704 (eps=1e-6, m_max=1e8, rho=1.012) | R0 = PWL(m^-1/2), downward hinges |
| 2,3,4 | 2 each | QQ=kappa R0^2 ; TT=R0^2 m ; ZZ=R0(1.5-0.5TT) (Newton) |
| 5 | 2d + 10 | OUT_i = x_i*ZZ ; exact erase of the 5 scalars |
Rules: (a) **X_IN must be a per-site slot that is EXACTLY zero in all frames where the site is inactive** (a87's NIN_n1_l etc. are fine; do not erase+reuse X_IN: a 1e-7 residue would be normalised to O(1) garbage); (b) OUT may be a shared slot (inactive sites add exactly 0); (c) the 5 scalar slots are shared by ALL sites (erased each time; magnitudes <= 1e4 << B); (d) the 6 layers of a site are consecutive and in the same frame as the register read that fills X_IN; (e) fp32.
D footprint: 5 floats. Depth: 6 layers/site -> Qwen3: 57 sites*6 = 342 layers, Qwen2.5: 49 sites*6 = 294.
Measured (res_rmsnorm_unit.json, real activations all layers): Qwen2.5 5e-7..3e-6, Qwen3 5e-7..5e-6 (max err/max ref); valid for rms(x) in [1e-3,1e4].

## 2. Per-head q/k-norm + streamed gain (Qwen3 only) = vslice.build_norm heads + MLP product (a87 keeps these)
H=16 heads/layer: 16 q-heads in one GDN layer, 8 k-heads in a second (hd=128 fits dv=258: 2*128+2). q: product(QN,GQ->QG) 4096 neurons in the q-layer MLP; k: product(KN,GK->KG) 2048 neurons.
Slots per layer-in-flight (sharable across inner layers if the producing register reads are flag-gated): QKVR 4096 (q2048,k1024,v1024), QN 2048, KN 1024, GQ 2048, GK 1024 (gain payload, host transport), QG 2048, KG 1024. => ~13.3k floats.
Layers: 2 GDN + 1 attention per inner layer. Qwen2.5 has no q/k-norm and uses QG=q(+bias payload), KG=k directly.

## 3. Attention (vslice.build_attention geometry at real dims)
Q slot = QG (16*128), K slot = KG (8*128), V = QKVR v part (8*128), MASK_l 1, output AO_l 16*128 (per layer or shared+gated). One full-attention layer; KV cache = inner KV. position_ids = inner position, theta=1e6 via config (partial_rotary_factor=(hd+.5)/hd_o).

## 4. Per inner-layer outer-layer count (op part; banks/regs are B's)
RMSNorm n1: 6; q/k-norm: 2 (Qwen3 only); attention: 1; RMSNorm n3: 6; SwiGLU: 1 (3072 or 4864 neurons <= 8192); residuals: free (register write adds).
=> Qwen3: 16 op layers/inner layer (28 layers -> 448 + final norm 6); Qwen2.5: 14 (24 -> 336 + 6). Rounds are set by the matvec chain (D/a87), not by these ops.

## 5. Final norm + LM head
nf = RMSNormUnit; logits via B's blocks into a SEPARATE argmax stack (own small frame: K slot (logit), V rails 36, Q const; `ops.compile_attention` + `compile_argmax_head`, verified exact on 151936 vocab for Qwen2.5 and Qwen3 at 4/4 positions): logits arrive in chunks as KV entries of one attention layer (one position per logit), the main frame carries no LOG slot.

## 6. Verified at real dims on A10G with a87's geometry (test_vslice_real5.py / real5b.py, res on box ~/work/ops)
Qwen3-0.6B layers 0/13/27, real q/k/v: vslice.build_norm (16 q-heads layer + 8 k-heads layer, dk=130 dv=258 H=16) + gain product + vslice.build_attention (16q/8kv x 128, hd_o=131, theta 1e6, GQA, causal, KV): attention output vs HF max err/max|ref| = 2.8e-6, 8.5e-7, 2.6e-6 (q/k-norm+gain alone 3e-7) with eps_v=1e-7.
With a87's current eps_v=1e-4: 2.9e-6, 1.1e-6, 1.8e-5 (layer 27, large q) -> **recommend eps_v=1e-7 in vslice.build_norm** (single-line change, still >> eps_outer 1e-30; code in vslice_c.py).
**CRITICAL (initial 1e-3 error root cause):** fla `chunk_gated_delta_rule` is inexact (TF32 inside): ~1e-3 error. ALL GDN layers must run with `trueqiq_state/gdn_impl.use_best()` (fla fused_recurrent) before building/ running; with it the error drops to 1e-6.
