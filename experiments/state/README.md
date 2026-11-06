# trueqiq_state (agent B): outer GDN recurrent state as float weight memory

Real Qwen2.5-0.5B weights are paged tile by tile into the recurrent state S of stock `Qwen3_5GatedDeltaNet` layers and read back as
y = W x by the layer's own delta-rule read, in fp32, no bit circuits. Everything below was measured (A10G, fp32, no TF32; Mac CPU for tests).

## Results
| item | result |
|---|---|
| single real tile 126x63 (1 head) | rel L2 4.7e-7 (state vs fp64 matmul) |
| full q_proj/k_proj/gate_proj/up_proj (896/128/4864 x 896), real weights | rel L2 2.2e-7 .. 8.6e-7 |
| full down_proj 896x4864 (5 K-layers x 16 heads, K-accumulation inside the model) | rel L2 2.9e-7 |
| FULL LM head 151936x896 (tied embedding, 136M weights) | rel L2 1.6e-7, argmax identical, 1.0 s outer compute (153k frames) |
| in-model chained W2 (W1 x0) via RegFile hand-off (host frame has X=0) | rel L2 6e-7 .. 9e-7 |
| in-model K accumulation of 2 passes (multi-row register read) | rel L2 7e-7 .. 8e-7 |
| activation scale 1..300 (|x|<=9000), outlier weights (-4, +3), bf16 weights | no loss at B=2^27 (results_accuracy_cuda.json) |
| bf16-rounded x | engine exact on the rounded x (1e-6); error vs fp32 x is the inherent 2.3e-3 of bf16 rounding |
| ballast B=1e4 | FAILS for big x: 1e-5 at |x|~900, 5% at |x|~9000 (transparent-norm scale error |x|^2/2B^2). Use B=2^27 (default) |
| fla `chunk_gated_delta_rule` | NOT exact (TF32 inside): 6e-4 .. 2e-3. fla `fused_recurrent` (gdn_impl 'fla_rec') exact AND fastest. HF torch chunk exact |

Throughput (A10G, results_steps_cuda.json): nL=1 (D=3186): 153k frames/s streamed = ~155M unique weights/s; single cached frame call ~630/s (launch bound).
nL=5 (D=15286, needed for K=4864): 9.6k frames/s = 48M weights/s. Output groups G>1 do NOT help (cost/frame grows ~T*D): G=1 best.
Unique weights per frame = 1008 per layer (16 heads x 63 contraction x ... pair rows duplicate each weight as +c/-c: the 2x tax of exact SiLU linearisation).
State: 1.05 MB per layer (16 heads x 128x128 fp32). Whole Qwen2.5-0.5B (494M weights): ~490k frames per token for bank nL=1, compute ~4-6 s on A10G if all
matvecs were independent; the real cost is set by the DEPTH RULE below.

## DEPTH RULE (critical for the integration)
A layer sees only its own state + residual of shallower layers (same or earlier position). Data therefore flows only to deeper-or-equal layers.
Re-using the same bank layers in the next round for the next dependent op is impossible (its X would have to come from shallower state).
A chain of N dependent inner ops needs an outer stack [bank_1][REG_1][bank_2][REG_2]..., >= 2 outer layers per dependent matvec (+ the nonlinear op layers).
Banks are reused only across INDEPENDENT work (blocks of one projection, tokens). The only deep->shallow channel is the outer token (host argmax + embed_tokens),
~17 bits/step: that is the fallback if depth must stay small (not built).

## In-model hand-off + K accumulation (RegFile, tested in test_chain.py)
```python
lay = Layout(); b1 = StateMV(lay, nL=1, H=16, K=896, name='B1'); b2 = StateMV(lay, nL=1, H=16, K=896, name='B2'); reg = RegFile(lay, b1, b2, nmax=4)
m = make_outer(lay, ['linear_attention']*3, ...); b1.compile(m,0); reg.compile(m,1); b2.compile(m,2)
frame = reg.add_write(b1.read_frames(x0), b=block, row=0)      # Y block -> register row (in-model, same frame as the read)
frame = reg.add_read(b2.read_frames(zeros), rows=[0])           # next round: b2.X = register content (sum of rows) filled in-model
```
REG_j stores Y as silu(+y)/silu(-y) pair columns + a ballast column; the transparent-norm scale cancels; unflagged heads use beta=sigmoid(-60).

## Exact in-model vs host split
In-model: all products/sums (state read S^T q, K-tile sums by out_proj + residual across heads/layers/rows), l2norm/RMSNorm/SiLU gadgets (compiled to be exact scalings),
delta-rule overwrite, register hand-off. Host: BALLAST constant, flags (WF/RF/INIT/HP/RR), one-hot addresses (SEL/SELR/SELQ), weight payload = silu-pre-inverted weights with
per-row shift for weights < -0.27 and the host's own frame-norm compensation (weights only; no activation touches the host). In tests/demo the program input x0 is host-fed (it is the embedding).
Honest limits: 2x capacity tax (pair rows), D grows with banks (cost/frame ~ T*D), idle outer compute is dense (compiled weights are ~99.9% zeros), depth rule above.

## Files
engine.py (StateMV, RegFile, silu_inv, build_state_matvec) | tiled.py (project: full W through the stream) | gdn_impl.py (kernel selection, use_best) | test_engine.py, test_accuracy.py,
test_chain.py, test_tiled.py | gpu_bench.py, bench_steps.py | demo.py | results_*.json | run_block.py (load_w)

## Frame protocol (all host-written fields are transport only)
* slot 0 BALLAST = spec.B (**B >= 1e6 required**: with 1e4 the transparent outer norm adds x-dependent scale error |x|^2/(2B^2) ~1e-3 for |x|~1e3;
  measured below).  RF=1 read frame; WF[l]=1 write for layer l; INIT=1 + SEL[dk-1]=1 init frame; SEL one-hot = key row; PAY[l] = payload.
* WRITE frame: SEL[r]=1, WF[l]=1, PAY[l][h*127+c] = silu^-1(w_c + s) (c<126), PAY[..+126]=silu^-1(s)  (s = per-row/head shift, weights only).
  The host also pre-compensates the transparent-norm scale of the *write* frame (fixed point on weights only).  Writes of all nL layers x 16 heads
  happen in the SAME frame (each (layer,head) has its own payload channels) => nL*H*126 weights per frame.
* READ frame: x in slot X (K floats).  Layer l head h contracts x tile t=l*H+h (63 elems).  out_proj sums heads (and the residual sums layers):
  K-tile accumulation is inside the model.   Y[c] = sum_t W_t x_t.
* Non-written/overwritten semantics: each write frame overwrites row r completely (beta=1,g=0): no clear needed between tiles; reads are non-destructive (beta=0).
