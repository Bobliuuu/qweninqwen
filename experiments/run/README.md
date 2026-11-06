# trueqiq_run (agent D): full-model runner + performance

Code: stream.py (GPU-vectorised weight->outer-frame stream builder + planner), run_proj.py (one matrix), run_token.py (a whole inner token's
matrices), bench_step.py (stand-in outer stack throughput). Run on A10G with `NOFLA=1` (forces HF torch GDN path; see fla note).

## Measured (all fp32, no TF32), Qwen3-0.6B real weights, B's engine.py frame protocol
* Whole LM head (151936x1024, 155.6M weights): A100 3.70 s (nL=2,D=5347); A10G 4.9-5.3 s (nL=2,D=6339), 12.5 s (nL=4,D=12421), 1.8 s (nL=1,H=17,D=3488).
  maxrel vs fp64 ref 3e-7 on all 151936 outputs, argmax identical.
* Whole layer 0 (q,k,v,o,gate,up,down; 15.7M weights): 1.06 s (A10G, nL=4 bank).
* FULL TOKEN of weight streaming (all 28 layers + LM head, 197 matrices, 595,984,384 real weights, 519,430 outer frames): 41.95 s on A10G, nL=4,H=16 bank,
  12.4k frames/s, worst rel err 5.6e-7, argmax ok on all 197. Activations were harness-fed (random), NOT a chained token yet.
* frames/s vs bank size (A10G, LM head): nL=1,H=17 (T=17 tiles,D=3488) 86k/s; nL=2 (T=32,D=6339) 28-31k/s; nL=3 19.7k/s; nL=4 (T=64,D=12421) 12.4k/s.
  Cost/frame ~ T*D ~ T^2 (fp32 matmul bound).  Extrapolation (plan(), NOT measured): T=17 bank with multi-pass K accumulation
  (615k frames) ~ 7 s/token; nL=2: ~19 s; nL=4: 42 s (measured).
* Stand-in outer stack (A100, D=1024,I=3584,24 layers mixed): single-step 24-28 steps/s (launch bound), prefill 4096: 15k steps/s fp32, 55k bf16, 36k TF32.
  GDN-only layer cost ~1.3 us per 1000 D per step (D=2048: 3.3 us/layer/step) = fp32 FLOP bound.  Round length >=512 frames needed to leave launch-bound regime.
* fla: TF32 (triton default) gives 5e-4 error (unusable for the float design); TRITON_F32_DEFAULT=ieee fixes error (3.6e-7) but ran 11 frames/s (A100).
  HF torch chunk fallback = 3e-7 and fastest correct path. bf16 would be 3.7x faster but breaks the 1e6 ballast design.
## Bandwidth: compact resident payload 9.15 GB/token (nL=4 active-layer packing, fits GPU); 25.8 GB of dense frames written per token = ~0.1 s at 600 GB/s;
  PCIe stream of payload 9.15 GB ~0.4 s.  Compute-bound, not bandwidth-bound.  Stream is token independent: build once (61 s) and keep resident.
## Design bound check: "2048 floats/step" is pessimistic; B's frame carries T*63 unique weights (T=64: 4032); measured 596M weights in 42 s.
## Host vs in-model split (strict)
Host: static schedule, flags, one-hot addresses, silu-pre-inverted/norm-compensated weight payload, token ids, position ids, ballast, outer argmax+append.
In-model: RMSNorm, q/k/v/o/gate/up/down/LM-head contractions (state reads), q/k norm, RoPE, softmax attention over inner KV, SwiGLU, residuals, accumulation.
Rounds: ~4 dependent matvecs/layer -> ~113 rounds/token (est.); per round >=1143 frames.
OPEN: per-layer norm gain vectors (65.5k params) need a vector write path; read output -> next X slot round hand-off; multi-pass K accumulation for T<49.
