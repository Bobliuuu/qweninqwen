# Circuit-route plan: the entire Qwen3-0.6B, gate by gate, inside a stock Qwen3.5

Status: first version. Every number is tagged **[M]** measured (file cited), **[E]** estimated (derivation shown) or **[?]** unknown. Projected and measured columns in the tier table are updated as the build agents' results land under `experiments/circuit/`.

The circuit route is the goal. The compiled route (real fp32 matvecs in outer layers, 268.6 s/token, see README) is a verified stepping stone and is not what this plan is about. The tiers below are successively faster builds of the circuit-level run: tier S0 is the naive baseline (about 1.3 years per token), and each later tier removes one named bottleneck.

## 1. Definition

**Entire Qwen inside Qwen (circuit route).** One token of the real, unscaled Qwen3-0.6B (28 layers, 151,936-token head, real weights) is produced by a stock-architecture `Qwen3_5ForCausalLM` whose weights are a compiled gate netlist and whose DeltaNet state and KV cache are its registers and memory. Every inner arithmetic op (multiply, add, accumulate, normalise, exponentiate, divide, compare, argmax) is a gate-level computation under the **G-contract**:

1. The outer model runs in **bf16** (so it cannot silently perform multi-bit fp arithmetic; this is how Qwenputer and qwentpu are checked).
2. Data between ops are bits or small integers (at most 8 significant bits per value, fan-in at most 256 so counts are bf16-exact). Bit-plane popcounts (a multi-hot GDN read, or a 0/1 linear layer) are allowed gates; this is what qwentpu's `signed_tile.py` already uses. A real-valued fp32 matvec on multi-bit data is not allowed (that is the compiled route).
3. Wide values live as limbs; carries, shifts, rounding and comparison are threshold gates (SiLU/ReLU neurons).

Allowed for the host (same rules as the README): embedding row / token id, control flags, one-hot addresses, position ids, the outer argmax and append, weight transport (unpack, pad, tile, quantise offline), validation reads. Not allowed: any inner activation arithmetic, or feeding a readback activation into the next input.

**Where do the inner weights live?** Options, honestly:

| Option | What it is | Cost | Honesty |
|---|---|---|---|
| W-stream | host streams weight bits as token ids / frame payloads into DeltaNet state each token (qwentpu/Qwenputer style, the 14.7 MAC/s baseline) | tokens/MAC about 3.6 [M] | allowed (transport); weights are not "in the outer's weights" |
| W-state | weights bulk-loaded into GDN state once (parallel_gdn design), reused across tokens | 4 bit-planes x fp32 per weight = 9.6 GB for 596 M weights x 4 bits [E] | allowed as a driver capability; state is not a parameter |
| W-param | inner weight bits are literally the 0/1 entries of outer linear layers; a read is a bit-plane count | 4.8 GB bf16 per plane set [E]; dense-stock zero-tax can multiply this 8-32x depending on layout [E] | strongest reading of "inside the weights"; then the matvec is a counting matmul, so the line to the compiled route is the G-contract above, which the user should confirm |

Recommendation: build against W-state first (fastest to measure), keep W-param as the end-state tier.

Quantisation caveat: qwentpu's measured tile is INT8 activation x INT4 weight with integer output [M]. Qwen3-0.6B at W4A8 will not reproduce HF argmax in general (massive-activation outliers). The "match HF" claim must be defined against a pinned fixed-point oracle (W8A12 or similar) and, separately, the argmax agreement with bf16 HF is reported. Bits per MAC scale the cost (section 3).

## 2. What exists and what is missing

| Op (Qwen3-0.6B) | Verified today | Measured throughput | Missing | Feasibility |
|---|---|---|---|---|
| MAC / matvec | signed INT8xINT4 tile (qwentpu `signed_tile.py`), 32x31 tile exact on real Qwen2.5 weights; 896-weight real row exact (result -883) | 14.7 MAC/s real row, 19.4 with direct readout (`real_weight_*.json`); 458-488 MAC/s 32x31 tile (`parallel_complete_results.json`); 2852 GDN reads/s = 2.83 M logical MAC/s primitive only, excludes load/accumulate (`parallel_gdn_results.json`) [M] | GPU batching, group scales, W8 or wider, persistent K-tile accumulate in the same machine, scheduling | feasible; throughput is the whole issue |
| accumulate over K tiles | 32-bit persistent accumulator in GDN state, bf16 stock layers, 10 layers / 108.5 M params (`persistent_results.json`, `qwen_layer_results.json`) [M] | 8.5 s for 4 updates (CPU) [M] | merge with tile, scale handling | feasible |
| RMSNorm (incl. q/k norm) | only as compiled-route fp32 op | n/a for circuit | gate version: square (multiplier), sum, reciprocal sqrt (LUT or Newton), scale (multiplier). 28x(2x1024 + 24x128) elements | feasible, deep (est. 40-80 gate layers); nonlinear agent is building |
| RoPE | compiled fp32 only | n/a | sin/cos table as constants + 2 fixed-point multiplies per pair | feasible, easy |
| softmax attention | compiled fp32 only | n/a | exp (LUT/poly), max, sum, reciprocal/divide, QK and AV MACs (114,688 x T MACs/token [E]), KV in outer cache. Largest unknown: exp+divide accuracy | feasible but biggest nonlinear risk |
| SiLU / SwiGLU | silu exact as thresholds in compiled route | n/a | fixed-point silu LUT (3072 x 28 elements) + 8x8 or 12x12 multiply | feasible |
| residual add | compiled only | n/a | adder (already exists) | easy |
| LM head + argmax | MACs as above; argmax tournament exists (compiled) | n/a | 151,936-way comparator tree on 32-bit logits (comparison gates) | feasible, depth about 17 compare levels |

Missing infrastructure that matters more than any single op: (a) a sequencer inside the machine (today the host schedules QUERY/COMMIT: this has to be an in-model program or a data-independent frame schedule), (b) the depth rule: without token feedback, dependent ops need strictly deeper outer layers; the only deep-to-shallow channel is the emitted token (about 17 bits/step) [per `docs/INTERFACE.md` and `experiments/state/README.md`], (c) bit-exact equivalence between the stock HF forward and any custom fused executor (Qwenputer's `execcheck` does this; Qwenputer: 250 tok/s HF path vs 3,200 tok/s Triton executor on L4 [M, external repo]).

## 3. Work for one token

All counts for Qwen3-0.6B (d=1024, 16 q / 8 kv heads x 128, inter 3072, 28 layers, vocab 151,936) [E, computed from config]:

- Per layer: q 2.10 M + k 1.05 M + v 1.05 M + o 2.10 M + MLP 9.44 M = 15.73 M MACs; x28 = **440.4 M**; LM head **155.6 M**; total **596.0 M MACs** (README's 3.4e7 s used Qwen2.5-0.5B's 494 M; with 596 M the same 14.7 MAC/s gives **4.05e7 s = 1.28 years** [E from M]).
- Attention: 114,688 x T MACs (T = context) [E]; 14.7 M at T=128, under 3 % of total.
- Non-matvec: about 0.5 M normalised/activation elements per token [E], under 1 % of gate evaluations but they set the **depth** (latency), not the volume.
- Partial products per MAC: W4A8 = 32, W8A8 = 64, W8A12 = 96 [E]. Roughly 5 threshold neurons per partial product (AND plus carry-save cell) gives 160 / 320 / 480 neuron evaluations per MAC, i.e. **1e11 to 3e11 neuron evaluations per token** for the matvecs [E].
- Measured protocol cost, token-serial: 3,252 outer tokens for a 896-MAC row = **3.63 tokens/MAC** (1,904 tile + 1,348 accumulator tokens, `real_weight_results.json`) [M]; 596 M MAC x 3.63 = 2.2e9 outer steps per token [E].

Throughput model: time = steps_sequential x time_per_step, where time_per_step on a GPU is max(weight-read time, 2 x params x batch_frames / sustained FLOPs). Anchors: outer fwd 2,219 steps/s fp32 and 4,145 steps/s bf16 for a 3.4 B random outer, chunked prefill, A100 shared, torch reference GDN (`experiments/bench/RESULTS.md`) [M]; Qwenputer 3,200 tok/s on L4, 250 tok/s HF [M, external]; qwentensor tiny chip 307 tok/s CUDA graph A100 (3.25 ms/token) [M, external]. Honest limit on batching: batch rows from separate sequences cannot exchange data inside the model, so a host that gathers activations between sessions is computing. Parallelism must be inside one sequence (many frames/positions, many heads, many lanes per token).

## 4. Tier table (speed tiers S0..S6)

Time per token = matvec-dominated; "x" is the cumulative factor claimed by the tier over the previous one. Projected values are estimates with the stated assumption; "measured" is filled from `experiments/circuit/*`.

| Tier | Build | Assumption behind the projection | Projected s/token | Measured s/token | Needs |
|---|---|---|---|---|---|
| S0 | naive: HF stock, CPU bf16, token-serial, 1 row at a time | 14.7 MAC/s [M], 596 M MAC | **4.05e7** (1.28 y) [E] | 4.05e7 extrapolated from [M] 14.7 MAC/s | done |
| S1 | tile parallelism on CPU (32x31 tile, 8 heads) | 458-488 MAC/s [M] (tile only: no K-accumulate, no scales) | 1.2e6 (14 d) [E] | pending | nothing new |
| S2 | GPU-batched engine (`experiments/circuit/engine`, one forward carries B tile circuits, CUDA graphs) | 60-200x over CPU at S1 per `bench_batched.py` batch sweep; Qwenputer saw 13x from executor alone | 6e3 - 2e4 (2-6 h matvec-only) [E, optimistic] | pending (engine agent, gpuB) | 1 GPU |
| S3 | algorithmic step reduction (`experiments/circuit/algo`: fuse READ+ACT 2x, lanes L=2 about 2x, bulk weight load removes about 2 of 3.6 tokens/MAC, wider GDN bit-plane reads) | tokens/MAC 3.6 -> 0.3-0.8 | 1e3 - 4e3 [E] | pending (algo agent, Modal) | 1 GPU |
| S4 | nonlinear gadgets (`experiments/circuit/nonlinear`: RMSNorm, RoPE, softmax, SiLU, argmax) | adds about 1.2-2x work and a very deep critical path; correctness tier, not a speedup | 1.5e3 - 8e3 for a COMPLETE token [E] | pending (nonlinear agent, gpuA) | 1 GPU |
| S5 | layer pipelining / depth unrolling with resident weights (W-param or W-state), 1 long sequence of frames, no per-MAC token feedback | steps per token about 1e5-1e6 (funnel bound: about 3.5e5 intermediate values x 16 bits / 17 bits per token) x 13-40 ms/step for a 10-20 GB outer, weight-read bound | 5e2 - 4e4 [E] | pending | 1 x 80 GB GPU |
| S6 | multi-GPU pipeline (layers split across GPUs; frames of independent positions in flight) | 4-8x over S5 | 1e2 - 1e4 [E] | pending | 4-8 GPUs |

What would make it hours instead of a year: S2+S3 alone project to roughly 1-10 h/token if the batched engine reaches the stated factors; S5 is the route to minutes. These ranges are not demonstrated. The most likely failure points: (1) GDN state is per-sequence, so cross-row gathering must be in-model; (2) dense stock layers waste 99.9 % of FLOPs on zeros (measured: sparse COO matmul 1.4 ms vs dense fp32 110 ms for the compiled route's D=40775, `RUNS.md`), so a GPU executor must exploit sparsity or bit-pack, which is a custom kernel and needs a lock-step equivalence check against stock HF; (3) quantisation fidelity; (4) exp/divide accuracy in gates.

Summary range per token, circuit route, whole token, assuming all tiers work: **best about 5e2 s (S5/S6), likely 1e3-1e4 s (a few hours), worst 4e7 s (S0, 1.3 years)** [E, nothing but S0 is measured].

## 5. Verification ladder (what correct means at each size)

| Tier | Claim | Cost estimate | Hardware |
|---|---|---|---|
| C0 | one signed MAC exact in stock bf16 outer (done in qwentpu [M]) | done | CPU |
| C1 | one matvec row K=1024 with real Qwen3-0.6B weights, persistent accumulate + scale, vs fixed-point oracle | S0 speed: about 70 s (14.7 MAC/s) [E]; minutes on GPU | CPU or any GPU |
| C2 | one full decoder layer (norm, QKV, RoPE, attention T=8, SwiGLU, residuals) at real dims | 15.7 M MAC: S0 12 days [E]; S2 minutes-hours | needs S2 engine + S4 gadgets |
| C3 | all layers of a tiny Qwen (e.g. d=128, 2 layers, vocab 512; the vslice toy config) free-running | about 1 M MAC/token, S0 18 h, S2 seconds [E] | 1 GPU |
| C4 | real Qwen3-0.6B, one token, argmax vs fixed-point oracle and vs bf16 HF | per tier table | 1-8 GPUs |

## 6. Hardware and next 24 hours

AWS G-family vCPU limit is 16 and both g5 boxes are in use, so no new G instance without a quota request. Available: Lambda A100 (`lambda-final`, shared, use `flock -w 7200 ~/.gpu.lock`; observed 25.9 GB used at 100 % utilisation when checked, so it is contended), Modal A100-80GB (`modal run --detach`; one run was queued behind a workspace GPU limit), gpuA/gpuB (in use by the build agents). Cost per tier: C1 under 1 GPU-hour; C3 a few GPU-hours; C4 at S2/S3 speeds tens of GPU-hours per token (estimate), at S5 a few GPU-hours.

Doable in 24 h: C1 on CPU or GPU with the existing qwentpu stack; the S2 batch sweep; the S3 token-count measurements (cheap, count-only on CPU).

## 7. Recommended first experiment

Measure what decides everything, the steps/s and MAC/s of the existing exact circuit as a function of batch on a GPU:

```
cd experiments/circuit/engine
python bench_batched.py 32 31 8 1,8,64,256,1024 --dtype bf16 --out res_bench_gpu.json
```

(ROWS=32, COLS=31, NVEC=8 reproduces the 32x31 tile of the 458-488 MAC/s CPU number.) Report exactness (must be 100 %), MAC/s per batch and the step time versus batch, and whether it is weight-read-bound or FLOP-bound. Decision rule: if MAC/s at B=1024 is at least 3e4, S2 is on track; if under 3e3, the dense-zero tax dominates and a sparse/packed executor must be built before anything else. The engine agent is running this on gpuB; results land in `experiments/circuit/engine`.
