# trueqiq_ops: float inner-Qwen non-matvec ops in stock Qwen3_5 outer layers (agent C)

Code: `ops.py` (API), tests `test_norm_swiglu.py test_attn.py test_argmax.py test_misc.py`, results `res_*.json`.
All tests run at FULL inner dims (Qwen2.5-0.5B d=896/I=4864/14q-2kv-hd64/vocab 151936; Qwen3-0.6B d=1024/I=3072/16q-8kv-hd128/vocab 151936),
on real activations of layers 0..last, vs HF inner. Outer layers are stock `Qwen3_5ForCausalLM` modules, all weights compiled (`machine.make_outer`
style; `make_outer2` adds rope config: theta + partial_rotary_factor, which `make_outer` lacks -> integration please adopt `rope_parameters`).

## Conventions
* stream (frame) of D floats, slot 0 = BALLAST. **Use BALLAST = 2**27 (power of 2) not 1e4**: transparent-norm error is |data|^2/(2 BALLAST^2);
  at 1e4 real SwiGLU data gives 3e-5..5e-4 rel error (measured, res_norm_swiglu_fp32_10000.json), at 2^27 it is below fp32 eps. Pad D to a power of 4 for bf16.
* `transparent(layer, D, ballast)`: (1+w)=ballast/sqrt(D) -> sublayer input == raw stream.
* gadgets (exact): identity z=silu(z)-silu(-z); product x*y=silu(x)y+silu(-x)(-y); square; native swiglu. `MLPBuilder(layer,D,ballast)` with `.swiglu .copy .product .square` (neurons allocated sequentially; `intermediate` must be >= `.n`).

## API
| call | does | outer layer |
|---|---|---|
| `rms_scale(d, eps_in, B)` -> s | producer must write `XS = s*x` (any linear writer / out_proj scale, free) | - |
| `rmsnorm_front(layer, XS, D, d, which='in'/'post')` -> c | sublayer input slice at XS becomes `sqrt(D/d) * x/sqrt(mean x^2+eps_in)`; consumer reads it with weights*`c` (any linear reader: GDN in_proj, attn v_proj, MLP gate/up) | norm of 1 layer |
| `MLPBuilder.swiglu(G,U,O)` | O += silu(G)*U (inner SwiGLU act*up on streamed gate/up slots) | MLP |
| `MLPBuilder.copy/product/square` | exact identity / elementwise product (streamed gain g, bias-free scalings, residual add into other slot) | MLP |
| `attn_config(nq,nkv,hd,theta)` + `make_outer2(...)` + `compile_attention(layer,D,Q,K,V,O,nq,nkv,hd,B)` | inner softmax attention: RoPE(theta), GQA, causal, 1/sqrt(hd), KV cache | full_attention |
| `argmax_rails`, `compile_argmax_head` | in-model hard-max argmax via attention + bit-code V, outer lm_head decodes bits | full_attention + lm_head |

### RMSNorm details (exact for any d, any eps, any outer width)
stream sumsq = B^2 + s^2|x|^2 = s^2(|x|^2 + d*eps_in) with s=B/sqrt(d eps_in): the BALLAST slot itself supplies the eps term, no padding or config eps change. Outer norm weight (1+w)=1 on XS, 0 elsewhere.
**Limits/hazards**: (i) XS~1e12 poisons the transparency of LATER layers in the same step -> the norm layer must be the last compute layer of a round and its consumer must be in the same layer (e.g. B's GDN in_proj / write), or XS erased (not solved exactly; see below). (ii) the constant ballast slot read inside that layer is scaled by data-dependent r, so "up=1" gadgets cannot live in the norm layer (use linear consumers). (iii) the gain g is NOT applied by the norm: either fold `W' = W diag(g)` into the streamed matvec weights (weight reformatting; exact, 0 ops) or run a product round (verified: Qwen3 q_norm = norm + streamed-gain product err 3e-7, `res_misc.json`).
Per-head qk-norm (Qwen3): norm each head as its own outer POSITION (N=T*heads positions in one pass, d=hd=128), then product round with gamma.
Qwen2.5 qkv bias: free: host puts b in the destination slot (payload) and the matvec result is added by the residual. Residual add: free (producer writes into the residual slot), or exact copy gadget (verified 5e-11).
Embedding lookup: host transports row E[tok] into the x slot (token->row, no compute).

### Attention details
outer heads == inner heads (nq,nkv native GQA), head_dim_o=hd+2: first hd dims ROTATED (partial_rotary_factor=(hd+.5)/hd_o, theta=inner theta, rotate_half pairing identical => RoPE exact, no permutation; different theta -> longrope table as recursiveqwen/merge2.py), dim R holds const c in q, dim R+1 const c in k (saturates outer q/k-norm: q'=G q sqrt(hd_o)/c, rel err |q|^2/2c^2~1e-12 with c=2^23), G=c/(hd_o hd)^(1/4) restores the 1/sqrt(hd_in) scale (bf16: pair (Gq,Gk) searched on the bf16 grid), gate rows read ballast*30 -> sigmoid==1. The outer KV cache holds the inner (post-norm, post-RoPE) K and V. Q/K/V slots must hold the FINAL inner q (biased/normed), k, v. position_ids = inner token index.

### LM-head argmax (full 151936 vocab, in-model)
logit_j sits in the K slot of outer position j (151936 positions; produced by B's matvec tiles or per-position product-gadget dot), V = dual-rail 18-bit address code of j (host-supplied address, like B's row addresses), query position (last, own key set to -1e6 by host flag) has Q=beta=2e5. Softmax is a hard max (winner p=1-e^-40 for top-2 gap>1e-3; fp32 score resolution -> logit resolution ~3e-6; exact ties blend -> invalid code, practically never). O = bit code of argmax; outer `lm_head` rows (code_v . O, compiled once, 151936 x 36 nonzeros) -> outer greedy argmax over outer vocab == inner token id (the permitted outer argmax/append). Alternative not built: pairwise max tournament (18 layers, |z| via silu(lam z)+silu(-lam z), index select by product gadgets) - more depth, same exactness class; attention version is 1 layer.

## Verified results (max err / max|ref|, fp32 outer, B=2^27)
| op | Qwen2.5-0.5B | Qwen3-0.6B | notes |
|---|---|---|---|
| RMSNorm (d=896/1024), layers 0,5,12,last | 1.0e-7..1.6e-7 | 1.2e-7..1.7e-7 | with folded gain 1.0e-7..2.8e-7; eps handled exactly |
| SwiGLU act*up (4864 / 3072 neurons) | 1.8e-8..1.9e-7 | 6e-8..1.2e-7 | |
| attention incl RoPE/GQA/causal, T=25, layers 0,7,15,last | 6.6e-7..1.3e-5 | 1.2e-6..3.0e-6 | L0 of Qwen2.5 1.3e-5: huge q bias outliers |
| qk-norm per head + streamed gain | n/a | 2.9e-7..3.1e-7 | 16 heads x 25 pos |
| residual add gadget / copy | 5.5e-11 / 1.2e-9 | | |
| argmax over 151936 | 4/4 positions exact (Qwen3 see below) | | 22-39 s per argmax on Mac CPU (1 layer, 152k positions, chunked KV) |
bf16 outer vs HF bf16 inner: RMSNorm 0.3-0.5% , SwiGLU 0.1-1.4%, attention 0.5-3.6% (Qwen2.5 layer0 13%) = bf16 rounding noise level of the inner itself (both sides round); fp32 is the recommended mode.

## Per inner token op/step count (op part only; matvecs are B's)
Per inner layer: norm1 (1 layer), attention (1 layer), norm2 (1), SwiGLU (1), residual free; Qwen3 adds per-head q,k norm (1 norm layer over (nq+nkv)*T? positions=24) + 2 gain-product rounds -> +4.
Qwen2.5-0.5B (24 L): 4*24 + final norm 1 = 97 op-layers; Qwen3-0.6B (28 L): 8*28 + 1 = 225 op-layers; + argmax 1 layer over 151936 positions.
Per layer op work (MACs, fp32): SwiGLU 3*I (14.6k/9.2k nonzeros, 4864/3072 neurons), attention 2*T*nq*hd_o (T = context), norms O(d). Gadget neuron counts: product/copy 2/element.
Depth rule (INTERFACE sec.5): each op above is separated from the next by a matvec -> number of outer rounds per inner layer ~ (norm->qkv matvec, attn->o matvec, norm->gate/up matvec, swiglu->down matvec) = 4 rounds (+2 for Qwen3 qk-norm) because every op consumes a matvec result of an earlier round (norm layer last in round).

## Not done / honest gaps
* XS poison erase not solved: norm must end a round. Streamed gain not applicable inside the norm layer (fold into W).
* GDN-gated-RMSNorm alternative route analysed (conv SiLU kills sign; needs small-signal or offset) - not built.
* bf16 is noise-level not exact; no longrope (different theta) test; attention tested at T=25 not long contexts.
