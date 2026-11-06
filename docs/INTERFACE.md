# qweninqwen: outer machine interface (state engine = experiments/state, ops = experiments/ops)

Goal: a STOCK `transformers` 5.x `Qwen3_5ForCausalLM` (weights compiled, fp32) whose forward executes an
inner Qwen. Shared code: `trueqiq/machine.py` (Layout, make_outer, gdn_const_init). Import it, don't fork it.

## 1. The outer model
* `make_outer(layout, layer_types, intermediate, gdn=..., attn=..., vocab=...)` returns a stock
  Qwen3_5ForCausalLM with ALL parameters zeroed (each layer is the identity on the residual stream) in fp32.
  Compilers then write their weights. `layer_types` may mix 'linear_attention' (GatedDeltaNet) and
  'full_attention' (C uses full-attention layers for inner softmax attention + KV; B uses GDN state).
* One shared `Layout` (frame/residual layout): slot 0 `BALLAST` = 2^27 constant (standardised; 1e4 gives 1e-3..5e-4 transparent-norm error) in every frame. Each of B/C
  allocs its own named slots from the same Layout (so names must not collide: prefix `B_` / `C_`).
  For integration the final model's D is `layout.n`.  Dims are free (e.g. D = a few thousand).
* The outer RMSNorm in every layer (input_layernorm/post_attention_layernorm) is made *transparent*: set
  weight so (1+w) = BALLAST/sqrt(D) for all slots (rms(stream) ~ BALLAST/sqrt(D) as long as |other slots|^2 << BALLAST^2,
  rel. error |x|^2/(2 BALLAST^2)).  Ops that need a real normalisation (inner RMSNorm) use the GDN gated RMSNorm
  head or a layer whose norm weight is not transparent and whose stream has been cleaned.  eps=1e-12.
* dtype fp32 reference; bf16 is an optional later experiment, not assumed.

## 2. What is an "outer machine step"
One step = one sequence position of `inputs_embeds` (a frame of D floats) pushed through all outer layers
(equivalent whether you call prefill over many positions or cached single-token decode - same math).
* Residual stream is PER STEP (per position). Within a step, data flows through the depth of the layers.
* Data crosses steps ONLY through outer state: GDN recurrent state (S: heads x 128 x 128 fp32 per layer) and
  outer KV cache of full-attention layers. So:
  - weight banks  = GDN state of "W layers"  (paged, overwritten per tile-set)       [B]
  - register file / inner KV cache = GDN state of "REG layers" or outer full-attn KV [B / C]
* A GDN layer = associative memory with the delta rule  S <- S*exp(g) + k (beta*(v - S^T k))^T ; out = S^T q.
  `gdn_const_init` makes beta~1,g~0 => exact overwrite at row k.  WRITE: k = one-hot row address (x flag), v = payload.
  READ: q = vector, out = S^T q (native float matvec).  Hazards to design for (each owner handles):
  (a) q,k are L2-normalised (eps 1e-6): magnitude of q lost -> append ballast/completion coordinate so ||q|| is a
      known constant, or read with unit-norm queries;  (b) q,k,v pass the conv SiLU: use z>=20 offset (silu(z)=z to 1e-8)
      or small-signal scaling; (c) head output goes through per-head gated RMSNorm  w*o/rms(o)*silu(z): put a known-magnitude
      ballast channel in the value rows so rms(o) is a constant; (d) 16 heads x (128 key x 128 value) per layer
      (config is free: gdn=dict(hk,hv,dk,dv)).
* Frame layout (host-writable part).  The HOST MAY: set BALLAST (constant), control flags (one-hot step type:
  WRITE / READ / TOKEN / ...), one-hot ADDRESS fields, weight-row payloads (transport of inner weights, any
  reformatting: transpose/pad/tile/quantise-dequantise), token ids (via `embed_tokens`), and `position_ids`
  (integer clock = inner token index, needed for RoPE continuity because weight-stream steps occupy outer positions).
  The HOST MAY NOT: compute any inner activation, matmul, norm, softmax, RoPE rotation, nonlinearity, sampling.
  Activations are produced only by outer layers; the only host-visible results are the outer logits at designated
  steps; host does argmax + append of the emitted outer token (and may read intermediate slots for VALIDATION only,
  never fed back).
* Non-token steps (pure weight-write steps) also create outer KV entries in full-attention layers: key flag slot
  must give those keys a large negative score (BALLAST trick) so they get zero attention weight.

## 3. Program structure (paging)
Inner token = a program: for each inner op that needs weights: [WRITE steps: stream tile rows into W-state] then
[READ step: activation (already in a register/state or in the residual of this step) queries tile, result lands in
residual -> REG write]. Nonlinear parts (RMSNorm, SiLU*up, softmax, RoPE) run inside the layers' MLP/GDN/attn
using weights compiled once (not paged): SwiGLU gadgets: square: eps*y^2 = silu(eps*y)*y + silu(-eps*y)*(-y);
identity: z = silu(z)-silu(-z); product via polarisation.
Cost accounting per inner token: n_steps(token) x t_step(measured), n_steps ~ weights/(rows-per-step x banks).

## 4. Deliverable contract for B and C
* B: a function `build_state_matvec(layout, ...)` that adds to an outer model the layers+frame fields so that a
  host stream of WRITE frames loads a real W tile set into GDN state and a READ step returns W^T x in float.
  Report max rel err, tiles/step, MAC/s measured, bytes of state.
* C: a library `ops.py`: functions that compile one inner op into given layers of the same outer model on the same Layout
  (slots in/out named), each with a verification test vs HF inner (Qwen2.5-0.5B, Qwen3-0.6B real weights).
* Both: expose `compile(model, layout, ...)` + `run_test()` and write results JSON. Don't edit machine.py;
  send requests to integration.

## 5. Dependency rule (important for composition)
Layers are strictly feed-forward: layer l at step t sees only layer l-1 outputs of step t and state of layer l from steps < t
(within one forward call) or from earlier calls (cache).  Hence a value computed at depth d of step t can feed shallow layers
only in a LATER forward call (via the state/KV cache), never in the same call.  So an inner token is executed as a sequence of
outer forward calls ("rounds"): each round = [WRITE frames ..., READ frame(s)] processed as one chunked prefill, with the
state cache (DynamicCache) carried between rounds.  Number of rounds per inner token = depth of the inner dependency chain
that cannot be folded in one pass; the host only decides WHICH frames to send (static schedule), never values.
Within one round the whole write-then-read stream runs as a single parallel chunked GDN pass (fast).

## 6. Agent D (trueqiq_run/) and the NO-SCALING-DOWN rule
Directive: T3 = the FULL real inner model (all layers, full hidden/heads/vocab, real weights; Qwen2.5-0.5B preferred:
24 layers, d=896, 14 q-heads/2 kv-heads x64, inter 4864, vocab 151936 tied). Scaled-down instances are debugging only and must
never be presented as a result. Agent D owns the full-model weight-streaming runner (schedule of rounds, WRITE-frame generation
from safetensors, state paging, KV/registers, outer-step throughput engineering on the A100, and measuring whole layers).
D consumes B's state-matvec and C's ops through the `compile(model, layout, ...)` contracts above; D's host schedule
may only contain tokens/flags/addresses/weight rows/position_ids (section 2). If a full token is not physically feasible in time,
D runs as many COMPLETE inner layers as possible end-to-end (each checked vs HF) and extrapolates with measured per-round numbers.
Measured so far (integration, bench/step_bench.py): bit-circuit baseline 60-195 MAC/s on contended Mac CPU BF16 (ref: 458-488).

## 5b. CORRECTION to section 5 (found in vertical-slice design, integration owner)
Section 5 is WRONG where it says a value computed deep in step t can feed shallow layers in a later forward call via state.
Layer l's state/KV at step t' only contains functions of layers < l at step t.  So state can only carry data DOWNWARD in depth
(to consumers at layers >= the layer holding the state).  Consequences, now the design basis of trueqiq/vslice:
* The inner token is an UNROLLED PIPELINE in depth: each inner op gets its own outer layers, in dependency order.
* Registers = GDN layers placed BETWEEN producer and consumer: REG_r layer sits deeper than the op that writes it and
  shallower than the op that reads it (it writes at one step, reads at a later step; outputs of the read only feed deeper layers).
* One forward call per inner token (chunked prefill over ~#weights/payload frames).  Within the call, op r+1's READ step comes
  after op r's last output block in step order, and deeper in layer order.  Paged weights are written in the same call by WRITE frames.
* Outer depth ~ (#inner ops) ~ 15 outer layers per inner layer; every frame flows through all layers (cost multiplier);
  stock config is uniform per layer so D x depth parameters get large (real model: tens of B params -> layer-streamed execution or smaller D).
* Pure-GDN stacks crash in the stock mask builder; include a full_attention layer (C's attention layers do) or pass explicit masks.
* The only deep->shallow loop-back channel is the emitted token (outer argmax -> host append -> embedding).
