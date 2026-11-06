# Tier leaderboard: Qwen3-0.6B as a program on the RV32IM CPU inside a real Qwen3.5

Maintained by the integration lead. Rule: a number in a **measured** column comes from a committed log/JSON of a run of that exact build; everything else is **projected** and says how it was derived. Nothing on the circuit route has been run end to end yet, so every full-model row below is PROJECTED. Definition of the goal and the arithmetic: [`CIRCUIT_ROUTE_PLAN.md`](CIRCUIT_ROUTE_PLAN.md).

Not this: the README's 268.6 s/token result is the earlier compiled route (custom giant outer, 1006 layers, D=40775). It is a stepping stone, not a circuit-route tier.

## Headline (2026-11-06)

Best build with MEASURED instruction counts on real Qwen3-0.6B weights: v2 integer-only int4 VDOT + custom nonlinear instructions: **1.34e8 instr/token at ctx 1, 1.55e8 at ctx 129** (compiler emulator, `experiments/circuit/compiler/RESULTS.md`). Seconds are computed, not measured: **about 8 to 10 hours per token at ctx 1** (model-based token count 1.154 tok/instr = 1.57e8 tokens; 7.9 h at 5,500 tok/s, 8.8 h at 4,963, 10.3 h at 4,225; the wider 7.7-12.4 h range below used assumed 1.02-1.40 tok/instr), 9 to 14 h at ctx 129, 17 to 27 h at ctx 1024 (extrapolated at 1.6e5 instr per context token). The literal T0 build is also measured as instructions: 1.10e11 per token, argmax exact vs HF (0.71 to 0.98 years computed). No Qwen3-0.6B token has been run on the circuit executor; fidelity is 92 % top-1 vs HF, not exact (R4).

Range arithmetic (explicit): trace tokens = instr x tok/instr with tok/instr in 1.02 (VDOT-dominated, kernels agent, token-accurate) to 1.40 (gcc-C mix, Qwenputer docs): ctx 1 gives 1.37e8 to 1.88e8 tokens. Rates are MEASURED but on other firmware: 4,963 tok/s (A10G, tuned, stories260k, `engine/results/autotune.json`) and 4,225 tok/s (A100, untuned, our tiny q3). Best = 1.37e8 / 4,963 = 2.76e4 s = 7.7 h; worst = 1.88e8 / 4,225 = 4.45e4 s = 12.4 h. ctx 129: 1.58e8 to 2.17e8 tokens, 8.8 to 14.3 h. ctx 1024: 2.98e8 instr, 3.04e8 to 4.17e8 tokens, 17 to 27 h. Whether the real 0.6B firmware reaches the stories260k-class rate on the 4B chip circuit is unmeasured (chip layers and per-step arithmetic are the same circuit, but the hidden-4096 core was timed only on those two firmwares).

## Leaderboard (last updated 2026-11-06, commit of this file)

s/token = trace tokens per token / trace-tok/s. Rate column for T1+ is Qwenputer's measured executor figure on a stories260K-class program, not yet measured on a Qwen3-0.6B-shaped program.

| Tier | Optimizations included | Instr/token | Trace tok/token | Trace tok/s | Instances | Hardware | s/token | Status | Evidence | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | literal: fp32 soft-float llama2.c-style firmware on RV32IM, real Qwen3-0.6B weights, ctx 1 | 1.10e11 [M: 1.097e11/1.100e11/1.100e11 at 3 positions, 184 instr/MAC, argmax == HF at all 3 incl. position 0] | 1.12e11 - 1.54e11 [computed, 1.02-1.4 tok/instr] | 4,995 [M on q3.elf, tuned executor, A10G, other firmware] | 1 | A10G | 2.25e7 - 3.08e7 s (0.71 - 0.98 yr) [computed from measured instructions] | INSTRUCTIONS MEASURED, seconds computed, not run | `experiments/circuit/compiler/res_full_f32.json`, `RESULTS.md` | 960918b |
| T1 | T0 + fused executor (Triton + CUDA graph) | 4.0e9 [P] | 5.6e9 [P] | 5,562 [M upstream] | 1 | L4 | 1.0e6 (11.6 d) | PROJECTED | plan sec. 5 | e713778 |
| T2 | T1 + SMAQA int8 build (ctx 1 throughout; see context table) (VDOT int4 is T3) | 5.4e8 matvec only [P from M instr/MAC 0.913] | 7.6e8 at 1.4 tok/instr [P] | 5,562 | 1 | L4 | 1.4e5 (39 h) | PROJECTED | plan sec. 3, 5 | e713778 |
| T3 | T2 + VDOT int4 (0.176 instr/MAC) + nonlinear ISA ext + algorithmic cuts | 1.05e8 matvec only [P from M 0.176] + nonlinear, not yet counted | 1.25e8-1.5e8 matvec only [P] (plan said 1.2e8) | 3,300-5,562 | 1 | L4/A10G | 2.3e4 - 4.5e4 (6-12 h), nonlinear extra |  PROJECTED | plan sec. 5 | e713778 |
| T4 | T3 + persistent kernel, faster GPU | same | 1.2e8 | 1.0e4-1.6e4 [E] | 1 | L40S/A100/H100 | 7.5e3 - 1.2e4 | PROJECTED | plan sec. 5 | e713778 |
| T5 | T3 + in-forward-pass lanes (SIMT) | same | ~1.3e8 + join | 5,562 per executor [M upstream, batch 1] | 2-4 lanes (4B only; 0.8B 1 lane) | 1 GPU, fused executor needed | 8.3-12.4 h/token at 16 lanes, 8.7-10.1 h at 4 lanes (multicore model, nonlinear unsplit); gain bounded ~1.4x by token bus | PROJECTED [needs R2, R4] | `experiments/circuit/multicore/res_model.json` | efc7dad |
| T5b | T3 + independent sessions splitting matvec rows (aggregate batch 21-38k tok/s) | same | 1.3e8 | 2.1e4-3.8e4 agg. (independent sessions [M upstream]) | 8-32 | 1 GPU | ~3.5e3-6e3 if cross-session transport allowed | PROJECTED [needs R2, R3, R4] | plan sec. 5; scaleout model `experiments/circuit/scaleout/model_out.json` | efc7dad |
| T6 | T5 on several GPUs, layer pipelining | same | 1.2e8 | >=1e5 agg. [speculative] | many | 4-8 GPU | ~1.2e3 | PROJECTED | plan sec. 5 | e713778 |

Instr/token = trace tok/token / 1.4 tok per instruction (Qwenputer range 1.0-1.55). [P] projected, [E] estimate, [M] measured by someone else upstream (Qwenputer repo), not by us.

#Matvec-only instruction totals use 596.0 M MACs/token x measured instr/MAC: plain 4.18e9, SMAQA 5.44e8, VDOT 1.05e8. Plan estimates (5.6e9, 5.2e8, 1.2e8 trace tokens) were roughly right for plain, optimistic by 1.5x for SMAQA, close for VDOT. Attention and nonlinear ops are not included yet.

Correction (multicore, 44ba97c, modeled not run): lane-level parallelism inside one forward pass is limited by the token bus (one stock-vocabulary token per joint step, 17.9 bits). Pure lane gain is <=1.7x, 4 lanes about 1.4x on the VDOT build, and 0.8B can host only 1 lane. The plan's "8-32 CPUs" T5 therefore only holds as independent sessions (T5b), which needs R3. T6 inherits R3 and is not reducible below T5b.

## Combination rules

- Effects multiply only when they act on different factors: instruction count (compiler, kernels, algo, nonlinear), trace tokens per instruction (ISA fusion), and tokens per second (engine, hardware, lanes). Within the same factor they overlap (for example nonlinear ISA and kernels both cut instructions) and must be re-measured jointly, not multiplied.
- Lanes (T5) multiply the rate only if the work is divisible without a serial dependency; layers are serial, so lane speedup applies within a layer's matvec rows, and the per-layer synchronisation cost is unmeasured.
- Stories260K scaling to Qwen3-0.6B assumes cost proportional to parameters; plan sec. 3 gives caveats (LM head 26 % of MACs, nonlinear ops not proportional). Treat as +-2x until the compiler agent's measured instr/MAC replaces it.

## Measured tiny-model runs (NOT Qwen3-0.6B; model size stated)

| Run | Inner model | Outer / executor | Trace tokens | Rate | Exactness | Evidence | Commit |
|---|---|---|---|---|---|---|---|
| tiny Qwen3-architecture firmware, 16 generated tokens | dim 64, 8 heads / 4 kv, head size 8, hidden 128, 1 layer, vocab 128, random weights | Qwenputer fused executor on A100-40GB (lambda-final), running ONLY the compiled math of the 11 chip layers (hidden 4096 layout of the Qwen3.5-4B Qwenputer merge), not the HF 4B checkpoint; HF-module check of the real merged checkpoint is `e2e/hfcheck.py` (separate, see its log) | 708,920 steps (reference 708,921 tokens, 529,035 instr: 1.34 tok/instr) = 44.3k trace tokens per generated token | 4,225 tok/s (0.237 ms/step) = 10.5 s per tiny token [M] | fused trace "exact" vs reference emulator; HF check of the same firmware at f32 rel err 6.3e-7 (compiler `res_tiny.json`) | `experiments/circuit/e2e/logs/fused_q3_16tok.log`, `.meta.json` | eb4f4f3, 135cfdc |

Base checkpoint note from the run metadata: Qwen3.5-0.8B is NOT supported by the Qwenputer core (it has 1 value head per key head, the core needs 2), so 4B is the base; the README's "0.8B/4B" outer should read 4B until the core is re-scheduled. Update 3dd3ddc: the stock HF-module check of the real merged checkpoint now passes for the first token of this tiny firmware (harness echo bug found and fixed); see `e2e` hf log. This is the first MEASURED circuit-route run. It validates the pipeline (firmware -> executor -> exact trace) at toy size only. Nothing here is a Qwen3-0.6B number.

## Qwen3-0.6B: MEASURED instruction counts, s/token COMPUTED from assumed rates

Instructions measured by the compiler's RV32IM emulator on real HF weights (positions 0..3; larger contexts extrapolate the measured per-context-token slope). Seconds are NOT measured: s = instr x tok/instr / rate, with 1.40 tok/instr and 5,500 tok/s (best 1.19 @5,500, worst 1.40 @3,200 give 1.3e5 / 2.6e5 s at ctx 1). Evidence: `experiments/circuit/compiler/RESULTS.md`, `res_full_q4v.json`.

| Build | ctx | instr/token [M] | s/token [computed] | days |
|---|---|---|---|---|
| q4v (VDOT int4, soft-float glue) | 1 | 5.87e8 | 1.49e5 | 1.73 |
| q4v | 128 | 3.47e9 | 8.84e5 | 10.2 |
| q4v | 1024 | 2.38e10 | 6.07e6 | 70.2 |

Context dimension: attention grows linearly, about 2.3e7 instructions per context token in the q4v build (about 200 instr per attention MAC, all soft-float). Every tier row above is at ctx 1; at ctx 128 the q4v build is 5.9x more expensive and at ctx 1024 40x. Tier rows must name their context length.

Reconciliation of q4v 1.7 days with the T3 projection of 6-12 h: the q4v build has VDOT (and SMAQA/custom class 3.1e8 of 2.6e9 instr over 4 positions) but still runs RMSNorm, softmax, exp, SwiGLU and RoPE as compiled soft-float: `__mulsf3` 1.25e9, `__addsf3`/`__subsf3` 2.5e8 each, `expf` 3.8e7 of 2.6e9 total. The nonlinear agent's fixed-point build with EXP2/RSQ/SIGM/AMX/ROT16 instructions costs 6.1e6 (T=1) to 9.4e6 (T=512) instructions for all nonlinear plus glue, against about 4.9e8 of non-matvec instructions in q4v at ctx 1. So the custom instructions and fixed-point kernels (9096c9d, ef9a958) are NOT yet in the q4v build; folding them in is the next integration step (compiler + nonlinear). The nonlinear agent's own matvec figure is 9.7e7 (VDOT, 8 rows sharing activations) vs 2.5e8 for single-row VDOT; q4v matvec uses the 0.176 instr/MAC kernel (1.05e8). Expected combined instruction count at ctx 1 is about 1.1e8-1.2e8, i.e. the T3 range, once both are merged; at long context the fixed-point softmax cost (14.4 instr per score element, 448 rows x T) must be added, not the plain soft-float one. PROJECTED until the merged build runs.

Fidelity flag: `res_full_q4v.json` records `argmax_fw` [13, 315, 279, 374] vs `argmax_hf` [15846, 315, 279, 374] on the 4-token prompt: the q4v firmware differs from HF at position 0 (agrees at 1-3). Argmax vs HF is not yet a pass; this belongs under R4 (numeric contract).

## Fidelity rows (open ruling R4: numeric contract)

| Build | Top-1 agreement with HF fp32 | Notes | Evidence |
|---|---|---|---|
| firmware int4 per-row, EXT, real weights | 120/130 positions = 92.3 % | float weight-only int4: 91.5 %, so the firmware adds little beyond int4 RTN | `compiler/RESULTS.md`, `res_full_fx_*.json` |
| float weight-only int8 per-row | 47/48 | position 0 WRONG | `res_fidelity_ablation.json` |
| firmware int8 weights | positions 1-3 match | position 0 WRONG (argmax 14582 vs HF 15846); 12-bit last-layer MLP does not fix it | same |

### Fidelity reconciliation (compiler 120/130 = 92.3 % vs algo 81 % / 61 %): not comparable as reported

| | Compiler (`compiler/fidelity_ablation.py`, `run_full_fx.py`) | Algo (`algo/quant_study.py`) |
|---|---|---|
| Positions | 130 tokens of one prompt: a three-sentence text ("The capital of France is Paris. ...") repeated 40 times; the fidelity ablation uses a 48-token prompt of repeated text | 140 positions over 12 short, diverse prompts (facts, code, French, arithmetic), teacher-forced; many are first-few-token positions with high entropy |
| Weights | int4 per-row RTN (VDOT scale per row), real firmware; float ablation: int8 per-row, int4 per-row, group 64/128 | RTN per group 128 (W8A8 g128, W4A8 g128; also per-row variant), fake-quantised in torch, every linear incl. tied lm_head |
| Activations | firmware: dynamic int8 with fixed-point nonlinears; float ablation: activations fp32 (weight-only) | symmetric per token (absmax) quantised at the input of every linear incl. down_proj, nonlinears fp32 |
| Result | firmware int4 120/130 (92.3 %), float weight-only int4 91.5 % on that prompt but 33/48 (69 %) on the 48-token prompt; int8 weight-only 47/48 | W8A8 81 %, W4A8 g128 61 %, W4A6 54 % |

Reading: the two numbers are not in conflict, they test different things. (1) The compiler's 130-token prompt is a heavily repeated text, so most positions are an easy copy of an earlier occurrence; its own 48-token prompt gives only 69 % for float int4 per-row versus 91.5 % on the 130 tokens, which shows the prompt dependence (about 20 points). (2) The algo grid adds per-token int8 activation quantisation at every linear including down_proj, which is where Qwen3's outlier channels hurt; the compiler's weight-only float rows never test it, and its firmware tested activation int8 only on 4 positions for int8 weights. (3) Algo positions are diverse and short, so more top-1 margins are small. Implication for R4: neither number is the contract. We need one pinned recipe measured on the same positions. Proposed comparable recipe: the algo's 12 prompts plus a held-out 128-token natural-text passage (not repeated), teacher-forced, run through (a) the compiler's firmware int4 EXT (the real contract candidate), (b) the algo torch fake-quant at the same settings. Until that runs, quote the firmware figure as "92 % on a repeated-text prompt; 69 % float on a 48-token prompt; diverse-prompt fake-quant W4A8 g128 61 %", and plan for QAT/GPTQ-class weights if the contract needs HF argmax equality. Asked both agents to run the cross-check.

Position 0 is the attention-sink token with top1-top2 logit gap 0.136; even int8 misses it. State plainly: no current firmware is argmax-exact against HF. Whether the target is "equals HF" or "equals the pinned integer emulator" is the open numeric-contract ruling (R4). The integer emulator itself is the correctness reference for trace equality; HF agreement is reported separately.

Reconciled per-op split per token (compiler, 8bcc898; v2 int4 EXT): VDOT kernels 1.05e8 (78 %), mv dequant epilogue 1.3e7, rms 6.0e6, quantize 3.0e6, attention 2.3e6, swiglu 2.0e6, rope 1.0e6, total 1.344e8. Trace tokens from the emulator's model-based counter (rv32.md record sizes, NOT a ref.Machine run): 1.154 tok/instr for int4 EXT (1.110 for int8 2-plane), so 1.57e8 tokens per inner token at ctx 1, replacing the assumed 1.02-1.40 range for this firmware. Hours at ctx 1: 7.9 h at 5,500 tok/s, 8.8 h at 4,963, 10.3 h at 4,225 (still computed, model-based token count). The 8-12 h headline range is therefore tightened to 8-10 h.

## Merged integer-only build (MEASURED instructions on real weights; seconds computed)

Compiler, aa7b367 / 3011004, `experiments/circuit/compiler/RESULTS.md` (emulated RV32IM firmware, real Qwen3-0.6B weights, nonlinear agent's EXP2/RSQ/SIGM/AMX/ROT16 merged as `-DEXT`, plain RV32IM fallback available). This is the first build that approaches the PROJECTED builds above, and it is measured as instructions.

| Build | ctx | instr/token [M] | s/token (computed: 1.4 tok/instr, 5,500 tok/s) | at our measured 4,225 tok/s |
|---|---|---|---|---|
| fp32 soft-float (T0 literal) | 1 | ~1.18e11 [extrapolated at 198 instr/MAC from tiny; full measurement still running] | 3.0e7 s (0.95 yr) | not run |
| int4 VDOT + soft-float glue (q4v) | 1 | 5.87e8 | 1.49e5 | |
| int4 VDOT, integer-only, EXT | 1 | 1.34e8 | 3.4e4 (9.5 h) | 4.4e4 (12.3 h) |
| same, plain RV32IM fallback | 1 | 1.38e8 | | |
| int8 weights (2 int4 planes), EXT | 1 | 2.41e8 | | |
| int4 EXT | 129 | 1.55e8 | about 4.0e4 (11 h) | about 5.1e4 (14 h) |

- Attention is now SMAQA int8 KV with EXP2: 1.6e5 instr per context token (was 2.3e7), matching my 1.5e5 estimate; the ctx 1024 value is not measured (extrapolation: about 2.9e8).
- Versus my combined projection A (1.10e8 tokens at ctx 1, 5.5-7.2 h): measured instruction count is 1.34e8, and at 1.4 tok/instr that is 1.9e8 tokens, so 9.5-12 h. The gap is the compiler's firmware (acc spills 1.0e8/token in the VDOT kernels, 64-bit dequant epilogue 5e7, RMSNorm 2.4e7, quantize 1.2e7) versus the kernels agent's register-blocked kernel. EXT saves only 2.6 % once merged. Next lever is the blocked kernels (algo commit 158a8f6, kernels K6/K7), which should bring ctx 1 toward 1e8 tokens and 5-7 h. Using the kernels agent's ~1.02 tok/instr for the VDOT part instead of 1.4 lowers the token count further; the build's true trace tokens must be measured with the token-accurate harness.
- Fidelity (R4): int4 per-row RTN agrees with HF on 120/130 positions (92.3 %; float weight-only int4 91.5 %, so the firmware adds little); position 0 (attention sink, logit gap 0.136) and some low-margin positions fail. int8 weights match positions 1-3 but position 0 is wrong even in a float weight-only int8 model; a 12-bit last-layer MLP does not fix it. Per-group scales would cost more than 2x on the VDOT kernel. Argmax-vs-HF is therefore NOT exact for any current build; remedies (QAT/GPTQ, fp activations at the sink, skipping BOS) are unbuilt. Any measured run must be compared to the pinned integer emulator, with HF agreement reported separately (92 % at best today).

## Algorithmic scenario table (algo c1e4ad0, `experiments/circuit/algo/RESULTS.md`; instr/MAC measured in an ISS, bit-exact; totals computed)

Blocked VDOT 12 rows: 0.1588 instr/MAC, 1.02e8 matvec instr, 1.43e8 tokens at 1.4 tok/instr, 7.2 h at 5.5k tok/s (matvec-only plus a flagged 5.4M non-matvec estimate). Hypothetical wider vector instruction (not in the ISA, not built): 32 / 128 / 1024 MACs per instruction = 0.041 / 0.018 / 0.011 instr/MAC, 3.1k to 7.8k s per token. Did not work (measured): exact LM-head pruning (fraction of MACs 1.000), SVD shortlist (worst rank about 149k of 151,936). Algo's note that its count excludes real nonlinear work: the compiler's measured full-model 1.34e8 is the reference.

## Corrected ROM circuit timing (engine bea5d2e, `engine/results/corrected.json`; MEASURED step times, composite time computed)

k=14 pipelines, 110 appended layers: ROM_DOT token 1.078 ms (928 tok/s); other tokens 0.200 ms with a graph-level skip (4,995 tok/s), 0.897 ms with predicate-skip (kernels still launched), 1.078 ms with no skip. Per Qwen3-0.6B token with the memory agent's 2.4-4.6M tokens: 640 to 1,080 s for the matvec part with graph skip; the ~1e8 nonlinear-glue tokens cost 5.6 h with the skip and 30 h without. So variant C needs the graph-level skip and still sits near 5.6 h + matvec; it does not beat build A by much and depends on nonlinear glue tokens, not matvec. The host-select overhead (one-token device-to-host read per step) is not measured.

## Compiled route (stepping stone, NOT a circuit-route tier)

Custom giant outer (1006 layers, D=40775), real fp32 matvecs, host streams weights. Measured 236.3 s / 237.1 s per position dense baseline (maxrel 5.8e-5, argmax == HF); with COO sparse weights, layers cached on GPU and payload caching, 28.8 s steady (39.7 s on the first position, profiling hooks on), 9.3x. Caveat: the caching is simulation-side (precomputed weight/payload caches that a real run of the same graph would not skip), maxrel 2.8e-5, and the block-best result differs from dense by 2.6e-5 rel; headline 268.6 s for the first run is the untuned number. Evidence: `experiments/vslice/PERF_NOTES.md`, `perf/`. Different thing from the table above.

## Combined projection (PROJECTED): blocked or W4A4 matvec + merged nonlinear kernels + attention

NOTE: build A below is superseded by the measured-instruction headline above (1.34e8 instr vs my 1.10e8 tokens projection). Kept for B and C and for the blocked-kernel target. Built from measured per-op token costs, nothing run end to end. Components, tokens per inner token:
- Matvec (kernels agent, exact on Qwenputer `ref.Machine`, token-accurate): blocked VDOT + requant epilogue 0.1765 tok/MAC on layers, 0.1656 on the LM head (fused argmax) = 1.035e8. Proposed VDOT16 (W4A4, K9): 0.0921 / 0.084 = 5.36e7 (modelled on the same Machine, not built in the circuit).
- Nonlinear with custom instructions (nonlinear agent, `res_counts.json`, ext_tokens): 9.57e6 at T=1, 1.02e7 at T=64, 1.47e7 at T=512 (linear in T between); I subtract requant (3.02e6) and unfused argmax (1.5e5) because the K6/K7 kernels already include them.
- Attention score and PV MACs (114,688 x T int8 x int8, not in either table): estimated at 1.325 tok/MAC (SMAQA K2 class). ESTIMATE; no attention kernel is measured.
- Effects here are additive (different instruction sets of one program), not multiplied. Seconds = tokens / rate; B adds x12/11 for one extra chip layer (kernels agent's depth scaling). Rates: 5,562 tok/s is the upstream L4 number the kernels agent used; 4,225 tok/s is our measured fused rate on A100 for the tiny build (11 chip layers) and is the more honest rate here.

| Build | ctx | trace tokens/token | h/token at 5,562 tok/s | h/token at 4,225 tok/s |
|---|---|---|---|---|
| A: blocked VDOT + ext nonlinear + attention | 1 | 1.10e8 | 5.5 | 7.2 |
| A | 128 | 1.31e8 | 6.5 | 8.6 |
| A | 1024 | 2.76e8 | 13.8 | 18.1 |
| B: VDOT16 W4A4 (proposed hardware, A4 quality risk) | 1 | 6.0e7 | 3.3 | 4.3 |
| B | 128 | 8.1e7 | 4.4 | 5.8 |
| B | 1024 | 2.26e8 | 12.3 | 16.2 |

All PROJECTED [needs R2, R4]. Not included: ROM read overhead and address writes (memory agent: unmeasured), fork/join, fidelity fixes for the q4v argmax mismatch, the embedding lookup. Attention dominates at long context (ctx 1024: 56 % of A), so long-context speed is bounded by attention kernels, and W4A4 only helps the weight matvec. Note these combined numbers (5-7 h at ctx 1) are 3-5x below the q4v measured-instruction route (1.7 days) because they assume the merged fixed-point build, which does not exist yet.

Reconciling the two tok/MAC views: the compiler reports 0.176 instr/MAC and applies an assumed 1.19-1.40 tok/instr (0.21-0.25 tok/MAC); the kernels agent measures tokens directly on `ref.Machine` and gets 0.18 for the same kernel structure (about 1.02 tok/instr, since VDOT records are one small token). For VDOT-dominated code use the kernels figure; the 1.19-1.40 ratio is for general C and overstates the matvec by about 20-40 %.

### Variant C, corrected row (memory section 10, 08bceb7; engine sparse timing)

| Item | Value | Status |
|---|---|---|
| ROM banks / layers | 29 banks, 14 pipelines: 29 ROM + 70 + 11 digit-accumulation = 110 appended layers, about 121 per step | built exact, executor time not measured |
| Trace tokens per inner token for matvec | 2.4e6 to 4.6e6 (per-result epilogue about 8 tokens; 1.85e5 ROM_DOT calls) | estimated |
| Engine sparse ROM timing | 0.543 ms/step (1,842 tok/s) for 13 pipelines, 52 sequential kernels on A10G; 0.425 ms stage-batched | MEASURED for 13 pipelines only, timing only (random table). NOT valid for 29 banks / 121 layers |
| With layer skip (appended layers run only on ROM_DOT tokens) | ordinary tokens at 0.2015 ms/step; non-matvec work about 3.4e7 instr (about 4.8e7 tokens) = 2.7 h, plus ROM tokens 2.4-4.6e6 x 0.5-2 ms = 0.3-2.6 h; total about 3 to 5 h | PROJECTED, needs layer-skip, R2, R4 |
| Without layer skip | every token pays about 1 ms+: more than 12 h, loses to build A | PROJECTED |

### Variant C (PROJECTED, earlier notes, superseded by the table above the VDOT/WROM matvec assumption if built): ROM + MAC in appended 4B layers

Memory agent, measured stage by stage in bf16 (CPU, GPU reruns queued; `MEMORY_DESIGN.md` section 8, 58f0b0b): 230 int4 x int8 MACs per word per pipeline (5 sublayers), 13 pipelines = 65 appended layers = 2,990 MACs per position, exact over 64 positions; address from a DeltaNet register costs 1 layer and 0 extra tokens (exact). NOT built: accumulating the digits into CPU rails (estimated 4 layers); the MLP-ROM earlier estimate of 1,150 MACs/layer was about 3x too optimistic and is withdrawn.
- Matvec: about 2.0e5 trace positions per token (4.7e-4 tokens/MAC) versus 1.0e8 for blocked VDOT, a 500x cut in tokens. That is not a 500x cut in time: the chip grows from 11 to about 76-81 layers, and the fused executor's step time scales with layer count (kernels agent's assumption; a ROM layer may be cheaper as a gather, unmeasured), so the rate drops to about 4,225 x 11/76-81 = 570-620 tok/s.
- Everything else stays: nonlinear (ext) 6.4e6 + attention 1.5e5 (ctx 1) or 1.95e7 (ctx 128) tokens, now at the lower rate.
- Result, PROJECTED [needs R2, R4, and the unbuilt accumulation]: ctx 1: 6.8e6 tokens, 3.1-3.3 h/token; ctx 128: 2.7e7 tokens, 12.4-13.3 h/token (worse than A at ctx 128 because attention and nonlinear tokens now pay the slower step). Token count is the wrong metric for this variant; the executor step-time model decides the winner, so the fused executor must be measured with 76+ layers. Also the dense HF path needs about 23 GB of weights (A100).
- Executor cost estimate from the memory agent (nonzero counting, NOT measured, fused executor not run on it): about 305K nonzeros per pipeline, 4.0M for 13 (13x the 298K of the current 11-layer chip); if ROM decode is executed as an address-indexed gather it falls to about 40K per pipeline, 0.5M total (under 2x the chip). A latency-bound model (about 16 us per layer from 5.5k tok/s at 11 layers) gives 81 layers = 1.3 ms/position = about 770 tok/s, consistent with the 570-620 used above. The agent's comparison to \"26 h\" VDOT uses the SMAQA token count (5.2e8); the right baseline is blocked VDOT at 1.0e8 tokens (5.2-7 h), so C's win at ctx 1 is about 2x, not large, and it reverses at ctx 128 unless per-layer latency is cut.
- SUPERSEDED inputs (memory agent, section 10, 08bceb7; still estimates except where noted): a ROM layer has 9,216 address units, so 29 ROM banks are needed (not 13), via 2,300-entry words with a 10-slot select (exact); digit accumulation is built and exact but takes 7-11 layers (not 4); 14 pipelines = 29 ROM + 70 + 11 = 110 appended layers, about 121 per executor step. Token count is set by the per-result epilogue (ROM_DOT 1.4 tokens plus about 6 epilogue instructions, about 8 tokens per result, estimated): 2.4-4.6M tokens per inner token (2.5e-3 to 7.8e-3 tokens/MAC), not 2e5 (that is only the ROM_DOT call count, 1.85e5). Latency-bound matvec time about 290-740 s depending on pipeline count. Decider: if every token pays the 121-layer step, the ~1e8 nonlinear tokens cost about 55 h and C LOSES to build A; only if the executor skips appended layers on non-ROM_DOT tokens (flag-gated) does C reach about 5.5 h + 1.7 h, i.e. no better than A. So the 3.1-3.3 h figure above for C is withdrawn; C is not shown to beat A.
- Consequence for the plan: after C the nonlinear ops and attention dominate (about 100x the matvec), so the next levers are nonlinear/attention instruction counts and executor step time, not matvec kernels.

## Measured components (not full-model tiers)

| Component | Value | Evidence |
|---|---|---|
| One int4-shaped projection 3072x1024 (3.15 M MAC) on our RV32IM emulator, checked vs numpy: plain 7.01, SMAQA 0.913, VDOT 0.176 instr/MAC (kernel only, not run on the circuit CPU; trace tokens/instr 1.19-1.40 is upstream) | M (emulator) | `experiments/circuit/compiler/fw/res_proj_instr_per_mac.json`, 84878e3 |
| fp32 mul as gate netlist, bitsliced verified | 4,332 neurons, depth 124, max rel err 4.0e-7 | `experiments/circuit/nonlinear/res_fp.json` |
| fp32 add as gate netlist | 2,670 neurons, depth 176, max rel err 1.2e-7 | same |
| Qwenputer upstream numbers (stories260K token: 2.43M / 225k / 53k trace tokens; executor 5,562 tok/s batch 1) | upstream, not reproduced here | plan sec. 2 |
| Matvec firmware, exact token costs on Qwenputer `ref.Machine` (bit-exact vs numpy), K=1024: plain C 8.007 tok/MAC, unrolled 4.32, SMAQA 1.325, VDOT one-row 0.708, existing compiler kernel 0.180, blocked VDOT (22 accumulators) 0.163, + requant epilogue 0.1765, LM-head fused argmax 0.1656. Matvec-only at assumed 5,562 tok/s: plain 238 h, SMAQA 39 h, existing VDOT 5.36 h, blocked 5.17 h (glue excluded) | M (reference Machine; seconds assumed) | `experiments/circuit/kernels/res_kernels.json`, `res_projection.json`, README | 1955ed0 |
| Proposed VDOT16+WB16 (W4A4): 0.082 tok/MAC measured on the modelled Machine, hardware estimate +~500 MLP units and 0-1 chip layer; projected ~2.9 h matvec-only; risk: A4 activation quality. VDOTA16 (W4A8) 0.071 tok/MAC but +3 chip layers, net 3.0 h, not worth building (agent's call) | modelled, not built in circuit | same | 1955ed0 |
| Weight memory: ROM in an MLP down_proj is exact (100 % of entries at 1-8 bits, stock-shaped Qwen3.5 MLP sublayer at 0.8B and 4B geometry, max abs err 3e-4 at 8 bits). Recommended design: ROM in ~13 appended layers at 4B (0.8B would need 82). Blocker: npu WROM is 32,640 words with a 15-bit WB index, 0.6B needs ~55M words (26-bit index), so WB must widen or take two tokens. MAC on the read word and address-from-register are untested | M for ROM read only; design PROPOSED | `experiments/circuit/memory/MEMORY_DESIGN.md`, `rom_mlp_results.json` | 1d383f7 |
| Nonlinear kernels on RV32IM: rsqrt 68.7 -> 3 instr, exp2 38 -> 3 instr (extension), rel err 5e-6; RMSNorm scalar part ~6.2k instr (base and ext) | M (emulator) | `experiments/circuit/nonlinear/res_kernels.json` |
| Lane exactness: row-split int8 matvec on 1-16 lockstep lanes equals numpy; bus-limited speedup 1.5x (2 lanes) to 2.5x (16) of matvec phase | M (Qwenputer ref.Machine) | `experiments/circuit/multicore/res_tiny.json` |
| q3.elf (our tiny Qwen3 firmware) on the tuned fused executor, A10G: 4,995 tok/s (0.2002 ms/step), exact token-for-token vs ref.Machine, 708,920 steps | M | `experiments/circuit/engine/README.md`, `results/autotune.json` | 50b5fac |
| Compiled route (different thing): 1 token, 268.6 s untuned, A100 | measured | `experiments/vslice/logs/real_L28.log` |

## Open items and rulings

1. RESOLVED (coordinator): target ISA is RV32IM Qwenputer (plus VDOT/SMAQA) in the real Qwen3.5-0.8B/4B. Compiler lowers to RV32IM firmware; tensor-op `compiler/isa.py` is an intermediate form only.
2. OPEN, pending user ruling: weight memory. Proposed default (memory agent): ROM-in-appended-layers of the outer checkpoint (the weights become parameters of the outer model; the checkpoint is then not exactly stock Qwen3.5); fallback: host streaming weights (labelled bring-up only, same footing as the compiled route). Also needs the WB index widened (15 to 26 bits). Every full-token row depends on it (298 MB int4 vs 261,120 nibbles in state). Marked [needs R2] below.
3. OPEN, pending user ruling: cross-session activation transport. T5, T6 depend on it [needs R3].
4. OPEN, pending user ruling: numeric contract (W4A8 vs W8A8, fixed-point nonlinear ops, meaning of "equals"). All rows depend on it [needs R4]; argmax-vs-HF agreement is reported separately.

Dependency marks: T0-T4 need R2 and R4; T5-T6 need R2, R3, R4.

## How rows get promoted

A row flips from PROJECTED to MEASURED only when `experiments/circuit/integrate` produces a JSON with trace-token count, wall time, and argmax vs the reference emulator for that build, and that JSON is committed. Until a full-size run exists, tiny-model runs are reported in their own table with the model size stated, never in the tier rows.
