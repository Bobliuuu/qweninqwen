> Maintainer working notes kept for the record. The current tier summary is [`docs/TIERS.md`](../TIERS.md).

# Tier leaderboard: Qwen3-0.6B as a program on the RV32IM CPU inside a Qwen3.5-style outer model (a Qwen3_5-architecture model widened to hidden 7680 with 12 added chip layers; see the README)

Maintained by the integration lead. Rule: a number in a **measured** column comes from a committed log/JSON of a run of that exact build; everything else is **projected** and says how it was derived. The final-contract CF+RAM build was run end to end on 2026-11-07 (section below); the tier rows further down are the earlier projections and are kept as written, so treat them as superseded where they disagree with that section. Definition of the goal and the arithmetic: [`CIRCUIT_ROUTE_PLAN.md`](../CIRCUIT_ROUTE_PLAN.md).

Not this: the 268.6 s/token result is the earlier compiled route (custom giant outer, 1006 layers, D=40775), kept in [`older_compiled_route.md`](../older_compiled_route.md). It is a stepping stone, not a circuit-route tier.

## Measured end to end (2026-11-07): final-contract CF+RAM build

One Qwen3-0.6B token (prompt `[785]`, position 0, ctx 1) on the circuit, **measured** in `experiments/circuit/fullrun/logs/cfram_q_vast/` ([`FINAL_VERDICT.json`](../../experiments/circuit/fullrun/logs/cfram_q_vast/FINAL_VERDICT.json)):

| Item | Value |
|---|---|
| Build | WF table via ROW tokens + constants fetch (CF) + circuit-owned remapped RAM, real `in_ram`/`in_drom` decode; int4 weights with layers 2/26/27 at 8-bit weights + 16-bit activations |
| Instructions / trace tokens | 178,066,316 / 199,287,158 (about 1.12 tok per instr), 96 chunks of 2,097,152 steps |
| Result | circuit's own HALT at step 57,705 of chunk 95; word **15846** = Hugging Face Qwen3-0.6B's token |
| Checks | 96/96 jobs clean (0 free mismatches, 0 forced, 0 injections); 95/95 seams exact |
| Time | 6.0 min per chunk on an RTX 4090, 4.9 to 9.7 min on an A100 PCIE, about 9.2 min on a Modal A100-40GB; chunks ran in parallel, so wall time is one chunk when enough GPUs are available |
| Cost | about $10 of rented Vast.ai GPUs (fleet-log estimate) plus Modal A100 time for chunks 32 to 70 |

The projected rows below assumed about 5,000 trace tok/s on one GPU, so a serial run would take about 11 to 13 hours; the parallel run replaces that with wall time of about one chunk.

## Headline (2026-11-06)

Best build with MEASURED instruction counts on real Qwen3-0.6B weights: v2 integer-only int4 VDOT + custom nonlinear instructions: **1.34e8 instr/token at ctx 1, 1.55e8 at ctx 129** (compiler emulator, `experiments/circuit/compiler/RESULTS.md`). Seconds are computed, not measured: **about 8 to 10 hours per token at ctx 1 for int4, but int4 matches HF top-1 on only 37-51 % of natural-text positions; the int8 build that reaches about 80 % costs 2.41e8 instr = 2.7e8 tokens (1.110 tok/instr) = about 15 to 18 h at ctx 1 (15.1 h at 4,963 tok/s, 17.8 h at 4,225)**; for reference the int4 figure is (model-based token count 1.154 tok/instr = 1.57e8 tokens; 7.9 h at 5,500 tok/s, 8.8 h at 4,963, 10.3 h at 4,225; the wider 7.7-12.4 h range below used assumed 1.02-1.40 tok/instr), 9 to 14 h at ctx 129, 17 to 27 h at ctx 1024 (extrapolated at 1.6e5 instr per context token). The literal T0 build is also measured as instructions: 1.10e11 per token, argmax exact vs HF (0.71 to 0.98 years computed). No Qwen3-0.6B token has been run on the circuit executor; fidelity is 92 % top-1 vs HF, not exact (R4).

Range arithmetic (explicit): trace tokens = instr x tok/instr with tok/instr in 1.02 (VDOT-dominated, kernels agent, token-accurate) to 1.40 (gcc-C mix, Qwenputer docs): ctx 1 gives 1.37e8 to 1.88e8 tokens. Rates are MEASURED but on other firmware: 4,963 tok/s (A10G, tuned, stories260k, `engine/results/autotune.json`) and 4,225 tok/s (A100, untuned, our tiny q3). Best = 1.37e8 / 4,963 = 2.76e4 s = 7.7 h; worst = 1.88e8 / 4,225 = 4.45e4 s = 12.4 h. ctx 129: 1.58e8 to 2.17e8 tokens, 8.8 to 14.3 h. ctx 1024: 2.98e8 instr, 3.04e8 to 4.17e8 tokens, 17 to 27 h. Whether the real 0.6B firmware reaches the stories260k-class rate on the 4B chip circuit is unmeasured (chip layers and per-step arithmetic are the same circuit, but the hidden-4096 core was timed only on those two firmwares).

## Leaderboard (last updated 2026-11-06, commit of this file)

s/token = trace tokens per token / trace-tok/s. Rate column for T1+ is Qwenputer's measured executor figure on a stories260K-class program, not yet measured on a Qwen3-0.6B-shaped program.

| Tier | Optimizations included | Instr/token | Trace tok/token | Trace tok/s | Instances | Hardware | s/token | Status | Evidence | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | literal: fp32 soft-float llama2.c-style firmware on RV32IM, real Qwen3-0.6B weights, ctx 1 | 1.10e11 [M: 1.097e11/1.100e11/1.100e11 at 3 positions, 184 instr/MAC, argmax == HF at all 3 incl. position 0] | 1.12e11 - 1.54e11 [computed, 1.02-1.4 tok/instr] | 4,995 [M on q3.elf, tuned executor, A10G, other firmware] | 1 | A10G | 2.25e7 - 3.08e7 s (0.71 - 0.98 yr) [computed from measured instructions] | INSTRUCTIONS MEASURED, seconds computed, not run | `experiments/circuit/compiler/res_full_f32.json`, `RESULTS.md` | 960918b |
| T1 | T0 + fused executor (Triton + CUDA graph) | 4.0e9 [P] | 5.6e9 [P] | 5,562 [M upstream] | 1 | L4 | 1.0e6 (11.6 d) | PROJECTED | plan sec. 5 | e713778 |
| T2 | T1 + SMAQA int8 build (ctx 1 throughout; see context table) (VDOT int4 is T3) | 5.4e8 matvec only [P from M instr/MAC 0.913] | 7.6e8 at 1.4 tok/instr [P] | 5,562 | 1 | L4 | 1.4e5 (39 h) | PROJECTED | plan sec. 3, 5 | e713778 |
| T3 | T2 + VDOT int4 (0.176 instr/MAC) + nonlinear ISA ext + algorithmic cuts | 1.05e8 matvec only [P from M 0.176] + nonlinear, not yet counted | 1.25e8-1.5e8 matvec only [P] (plan said 1.2e8) | 3,300-5,562 | 1 | L4/A10G | 2.3e4 - 4.5e4 (6-12 h), nonlinear extra |  PROJECTED | plan sec. 5 | e713778 |
| T3m | T3 as BUILT: VDOT int4 (compiler kernel, acc spills) + fixed-point RMSNorm/softmax/SwiGLU/RoPE with EXP2/RSQ/SIGM/AMX/ROT16 (`-DEXT`), SMAQA int8 KV attention, real Qwen3-0.6B | 1.346e8 at ctx 1, 1.553e8 at ctx 129 [M: emulator positions 0 and 129 of one run; slope 1.60e5 instr/ctx token] | 1.55e8 at ctx 1 at 1.154 tok/instr [model-based count, not a ref.Machine run] | 4,225-5,500 [M, other firmware] | 1 | A100/L4 | 2.8e4 - 3.7e4 at ctx 1 (7.9 - 10.3 h), 3.2e4 - 4.3e4 at ctx 129 [computed] | INSTRUCTIONS MEASURED, seconds computed, not run. Fidelity vs HF top-1: 51 % (algo 12 prompts) / 37 % (held-out passage) for int4; int8 build (2.41e8 instr) 81 % / 79 % | `experiments/circuit/compiler/res_full_fx_w4_e1_n130.json`, `res_fidelity_prompts_w4.json`, `RESULTS.md` | this commit |
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

Reconciliation of q4v 1.7 days with the T3 projection of 6-12 h: the q4v build has VDOT (and SMAQA/custom class 3.1e8 of 2.6e9 instr over 4 positions) but still runs RMSNorm, softmax, exp, SwiGLU and RoPE as compiled soft-float: `__mulsf3` 1.25e9, `__addsf3`/`__subsf3` 2.5e8 each, `expf` 3.8e7 of 2.6e9 total. The nonlinear agent's fixed-point build with EXP2/RSQ/SIGM/AMX/ROT16 instructions costs 6.1e6 (T=1) to 9.4e6 (T=512) instructions for all nonlinear plus glue, against about 4.9e8 of non-matvec instructions in q4v at ctx 1. UPDATE: this fold-in is DONE (see the merged integer-only build section below and the T3m row); the q4v soft-float numbers in this table are the pre-merge build. Original text: the custom instructions and fixed-point kernels (9096c9d, ef9a958) were not yet in the q4v build. The nonlinear agent's own matvec figure is 9.7e7 (VDOT, 8 rows sharing activations) vs 2.5e8 for single-row VDOT; q4v matvec uses the 0.176 instr/MAC kernel (1.05e8). Expected combined instruction count at ctx 1 is about 1.1e8-1.2e8, i.e. the T3 range, once both are merged; at long context the fixed-point softmax cost (14.4 instr per score element, 448 rows x T) must be added, not the plain soft-float one. PROJECTED until the merged build runs.

Fidelity flag: `res_full_q4v.json` records `argmax_fw` [13, 315, 279, 374] vs `argmax_hf` [15846, 315, 279, 374] on the 4-token prompt: the q4v firmware differs from HF at position 0 (agrees at 1-3). Argmax vs HF is not yet a pass; this belongs under R4 (numeric contract).

## Fidelity rows (open ruling R4: numeric contract)

| Build | Top-1 agreement with HF fp32 | Notes | Evidence |
|---|---|---|---|
| firmware int4 per-row, EXT, real weights | 120/130 positions = 92.3 % | float weight-only int4: 91.5 %, so the firmware adds little beyond int4 RTN | `compiler/RESULTS.md`, `res_full_fx_*.json` |
| float weight-only int8 per-row | 47/48 | position 0 WRONG | `res_fidelity_ablation.json` |
| firmware int4 per-row EXT, algo's 12 diverse prompts | 72/140 = 51 % | teacher-forced, real weights (compiler `res_fidelity_prompts_w4.json`) | compiler RESULTS.md |
| firmware int4 per-row EXT, held-out natural 116-token non-repeated passage | 43/116 = 37 % | same | same |
| firmware int8 per-row (2 planes) EXT, 12 diverse prompts / held-out passage | 114/140 = 81 % / 92/116 = 79 % | matches the algo fake-quant W8A8 81 %; `res_fidelity_prompts_w8.json` | same |
| firmware int8 weights | positions 1-3 match | position 0 WRONG (argmax 14582 vs HF 15846); 12-bit last-layer MLP does not fix it | same |

### Fidelity reconciliation (compiler 120/130 = 92.3 % vs algo 81 % / 61 %): not comparable as reported

| | Compiler (`compiler/fidelity_ablation.py`, `run_full_fx.py`) | Algo (`algo/quant_study.py`) |
|---|---|---|
| Positions | 130 tokens of one prompt: a three-sentence text ("The capital of France is Paris. ...") repeated 40 times; the fidelity ablation uses a 48-token prompt of repeated text | 140 positions over 12 short, diverse prompts (facts, code, French, arithmetic), teacher-forced; many are first-few-token positions with high entropy |
| Weights | int4 per-row RTN (VDOT scale per row), real firmware; float ablation: int8 per-row, int4 per-row, group 64/128 | RTN per group 128 (W8A8 g128, W4A8 g128; also per-row variant), fake-quantised in torch, every linear incl. tied lm_head |
| Activations | firmware: dynamic int8 with fixed-point nonlinears; float ablation: activations fp32 (weight-only) | symmetric per token (absmax) quantised at the input of every linear incl. down_proj, nonlinears fp32 |
| Result | firmware int4 120/130 (92.3 %), float weight-only int4 91.5 % on that prompt but 33/48 (69 %) on the 48-token prompt; int8 weight-only 47/48 | W8A8 81 %, W4A8 g128 61 %, W4A6 54 % |

RESOLVED by the compiler's cross-check (same firmware, algo's prompts): the 92.3 % was a repeated-prompt artifact. Real figures for diverse text: int4 per-row RTN 51 % (37 % on a held-out non-repeated passage), int8 per-row about 80 % (matches the algo's W8A8 81 %). No current build reaches HF-exact argmax on natural text; the fp32 soft-float build is argmax-exact only on a 3-position check (not run on the diverse set). Earlier reading (kept for the record): the two numbers are not in conflict, they test different things. (1) The compiler's 130-token prompt is a heavily repeated text, so most positions are an easy copy of an earlier occurrence; its own 48-token prompt gives only 69 % for float int4 per-row versus 91.5 % on the 130 tokens, which shows the prompt dependence (about 20 points). (2) The algo grid adds per-token int8 activation quantisation at every linear including down_proj, which is where Qwen3's outlier channels hurt; the compiler's weight-only float rows never test it, and its firmware tested activation int8 only on 4 positions for int8 weights. (3) Algo positions are diverse and short, so more top-1 margins are small. Implication for R4: neither number is the contract. We need one pinned recipe measured on the same positions. Proposed comparable recipe: the algo's 12 prompts plus a held-out 128-token natural-text passage (not repeated), teacher-forced, run through (a) the compiler's firmware int4 EXT (the real contract candidate), (b) the algo torch fake-quant at the same settings. Until that runs, quote the firmware figure as "92 % on a repeated-text prompt; 69 % float on a 48-token prompt; diverse-prompt fake-quant W4A8 g128 61 %", and plan for QAT/GPTQ-class weights if the contract needs HF argmax equality. Asked both agents to run the cross-check.

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

Blocked VDOT 12 rows: 0.1588 instr/MAC, 1.02e8 matvec instr, 1.43e8 tokens at 1.4 tok/instr, 7.2 h at 5.5k tok/s (matvec-only plus a flagged 5.4M non-matvec estimate). Hypothetical wider vector instruction (not in the ISA, not built): 32 / 128 / 1024 MACs per instruction = 0.041 / 0.018 / 0.011 instr/MAC, 3.1k to 7.8k s per token. Did not work (measured): exact LM-head pruning (fraction of MACs 1.000), SVD shortlist (worst rank about 149k of 151,936). Caveat from the algo agent: its ISS VDOT/SMAQA semantics were written from the Qwenputer README (npu branch not in its checkout), so its instr/MAC is not a ref.Machine run; the kernels agent's token-accurate ref.Machine figures (0.163-0.1765 tok/MAC) are the ones to trust for tokens. Algo's note that its count excludes real nonlinear work: the compiler's measured full-model 1.34e8 is the reference.

## Corrected ROM circuit timing (engine bea5d2e, `engine/results/corrected.json`; MEASURED step times, composite time computed)

k=14 pipelines, 110 appended layers: ROM_DOT token 1.078 ms (928 tok/s); other tokens 0.200 ms with a graph-level skip (4,995 tok/s), 0.897 ms with predicate-skip (kernels still launched), 1.078 ms with no skip. Per Qwen3-0.6B token with the memory agent's 2.4-4.6M tokens: 640 to 1,080 s for the matvec part with graph skip; the ~1e8 nonlinear-glue tokens cost 5.6 h with the skip and 30 h without. So variant C needs the graph-level skip and still sits near 5.6 h + matvec; it does not beat build A by much and depends on nonlinear glue tokens, not matvec. The host-select overhead (one-token device-to-host read per step) is not measured.

## Compiled route (stepping stone, NOT a circuit-route tier)

Custom giant outer (1006 layers, D=40775), real fp32 matvecs, host streams weights. Measured 236.3 s / 237.1 s per position dense baseline (maxrel 5.8e-5, argmax == HF); with COO sparse weights, layers cached on GPU and payload caching, 28.8 s steady (39.7 s on the first position, profiling hooks on), 9.3x. Caveat: the caching is simulation-side (precomputed weight/payload caches that a real run of the same graph would not skip), maxrel 2.8e-5, and the block-best result differs from dense by 2.6e-5 rel; headline 268.6 s for the first run is the untuned number. CUDA graphs are not working (COO fails in capture; CSR gives all-zero output), the sparse parameters probably conflict with the audit's dense-parameter inventory, and TF32 was rejected (4e-4). Evidence: `experiments/vslice/PERF_NOTES.md`, `perf/`. Different thing from the table above.

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

## Full-trace circuit replay, Qwen3-0.6B firmware (2026-11-06, A10G, bring-up mode)

Measured. One full reference trace of the Qwen3-0.6B RV32IM firmware (162,682,048 trace tokens, 136,877,011 machine instructions, reference argmax 13, `out_hex` 0d000000, exit 0) was replayed on the circuit executor (fused Triton) in 78 chunk rows of about 2.1M steps each (7 waves: six of 12 rows, one of 6), by `experiments/circuit/fullrun/circuit_run.py` and `waves.py`.

| Quantity | Value |
|---|---|
| Steps run on the circuit | 162,682,047 (78 rows, every row ran its whole chunk) |
| Free tokens (circuit's own prediction, compared to reference) | 162,391,103, **0 mismatches** in all 78 rows |
| Forced tokens (host-served, unaddressable-WROM VDOTs) | 290,944; the circuit's raw prediction disagreed on 205,827 of them (expected, it lacks those weights; not errors of the free path) |
| Throughput, A10G | 18,472 to 18,480 agg tok/s with 12 rows, 15,299 with 6 rows; wall 6 x 1,362 s + 822 s = 8,994 s, i.e. about 18,100 tok/s end to end |
| Reference trace generation (same box) | 1,718 to 1,821 s, about 75 to 80k instr/s |

Evidence: `experiments/circuit/fullrun/logs/cr_out/result_*_n12.json`, `result_72_73_74_n6.json`, `progress.log`, `logs/ref785b/summary.json`.

What this does NOT prove:
- Not one continuous run (the 78-row table above). Each row restarts from the reference machine state (registers, pc, counters) at its chunk boundary. The chained runs in the next section remove that restart for the first chunks only (6 and 20 of 78 chained; the other 58-72 chunks are still independent).
- Bring-up mode, not stock: memory is host-served. Before each LOAD/SB/SH/VDOT the host injects the true RAM word or int4 weight cells into the circuit state, the RAM address-range check was removed from the core, and 290,944 VDOT tokens were overridden by the reference token. The circuit's own 32 KB RAM, 128 KB DROM and 32,640-word WROM could not hold the model (open item 2, weight memory, is still unresolved).
- Zero free-token mismatches means the circuit's ALU, MUL/DIV, branch, WB and refresh predictions matched the reference on every step it was allowed to predict. It does not cover load/store/VDOT results beyond what the injected-state arithmetic produced.
- No token was generated by the circuit. The output (argmax 13) comes from the reference machine; the circuit replayed its trace teacher-forced at the host-served positions. It was not compared against a Hugging Face forward pass here, and it is not running inside a stock Qwen3.5 checkpoint.
- Speed is a per-GPU replay rate of the fused executor on 12 parallel rows, not a latency for generating one token: a single sequential token at the measured per-row rate (about 1,540 tok/s per row) would be about 29 h for 162.7M steps.

## Chained circuit replay, chunk-to-chunk without snapshot restore (2026-11-06, A100 `lambda-final`, legacy firmware, prompt 785)

Measured. `experiments/circuit/fullrun/circuit_run.py --chain` runs consecutive chunks as ONE circuit run: one row, one CUDA graph, the head states (registers, pc anchor, `n`, weight-base register `wbase`, `bn` digit file, 32 KB RAM, lag histories) are simply never reloaded between chunks. At every chunk boundary the circuit's head states are read back (`extract()`: pc, `count`, `wbase`, the three register-file copies `regs.rs1/rs2/rd`, `bn1.*`, all 8192 RAM slots) and compared with the reference machine's snapshot at that boundary (`ref_big.py` state pickles; RAM compared slot-by-slot against the reference word at the slot's last-touched address). `--handoff` additionally re-snapshots the circuit FROM THAT READ-BACK STATE (registers, pc, `n`, `wb`, `bn`, RAM words; no reference values) and restarts the next chunk from it. Logs: `experiments/circuit/fullrun/logs/chain/` (`c0_5` chain, `h0_3` handoff, `c0_19` 20-chunk chain), also `~/qweninqwen-work/box-logs/lambda-fr2/`.

| Run | Chunks | Steps on the circuit | Free tokens (circuit's own) | Free mismatches | Forced tokens (host) | Boundaries compared | State differences at any boundary | Rate |
|---|---|---|---|---|---|---|---|---|
| `--chain` | 0-5 (6) | 12,582,913 | 12,561,408 | **0** | 21,504 | 6 (after each of chunks 0-5) | registers 0 of 31 x 3 copies wrong, pc/`n` equal, `wbase` (15 bits) equal, `bn` equal at all 6; RAM: 0 bad among LOAD-last and SW-last slots at all 6, SB/SH-last slots misread by the first decoder (see below), 0 bad at boundary 5 | 5,524 tok/s (1 row, A100) |
| `--handoff` | 0-3 (4), 3 restarts | 8,388,608 | 8,374,655 | **0** | 13,952 | 4 | all equal incl. all 8192 RAM slots (fixed decoder); the 3 restarts from circuit read-back state ran with 0 free mismatches | 5,527 tok/s |
CHAIN20_ROW

First boundary that diverged: none in the chunks run. The state the circuit holds at the end of chunk N (31 registers in all three copies, pc anchor and `n`, the low 15 bits of the weight base, `bn`, every one of 8192 RAM slots touched in the chunk) equals the reference snapshot, so it is a valid start for chunk N+1 without any host-provided state.

What the boundary readback does and does not show (read before quoting):
- Registers, pc, `n` and RAM are circuit-computed, not host-written. The RAM comparison is split by what last touched a slot (chunk 0-5 chain): slots whose last touch was a SW store (3,070 / 1,056 / 1,058 / 1,056 / 650 / 2,304 per boundary) and SB/SH store (224 / 304 / 96 / 688 / 496 / 0) hold the value the circuit stored itself and equal the reference. Slots whose last touch was a LOAD (4,898 - 7,046 per boundary) equal the reference trivially: the host injected that word. So the readback verifies the circuit's store path, not its ability to serve a later load from its own store: in this run every LOAD word was still injected from the reference (the circuit's stores are never read back by the circuit; that closes only in the remapped-firmware runs of `docs/HOST_ROLE_AUDIT.md` section 5).
- Decoder fixes that were needed to read the state (my bugs, recorded so the numbers are not misread): a register bit is spread over several columns and must be summed (first attempt, summing nothing but column 0, showed 23 of 31 registers "wrong" with low 15 bits zero); the RAM word is block A (cols 2..33, SW data / injected word) plus block B (cols 34..65, the SB/SH merged word). The first 6-chunk log (`logs/chain/c0_5`) was produced before the block-B fix and reports 224 / 299 / 96 / 670 / 496 / 0 "bad" RAM slots; all of them are SB/SH-last slots and were decoder misses (the `--handoff` log, produced after the fix, shows 0 bad on the same chunks 0-3 and the same reference). Registers, pc, `n`, `wbase` and `bn` comparisons are unaffected.
- The weight base is only 15 bits in the circuit (`S.WP_BITS = 15`, the WB token carries `wb mod 2^15`). The reference holds the full 27-bit pointer (e.g. 829,249 at the end of chunk 0, 5,503,361 at the end of chunk 5); the high 12 bits are NOT in the circuit and are host state (the host indexes the 596 MB weight file with them when it injects VDOT operands). The 15-bit part matches at every boundary.
- 12.56 M free tokens in 6 chunks with zero mismatches, in one continuous run, shows no carry-over bug across 5 chunk boundaries; it says nothing about the other 72 chunks.

Host roles that remain in the chained run (counted on chunks 0-5, `ref785t`):
- VDOT weight operands: 830,886 / 973,376 / 890,930 / 877,129 / 889,055 / 1,043,648 int4 8-cell words injected from the 596 MB `wrom.bin` into the WROM head state (88 - 89 % of all injections), PLUS the 12 high bits of the weight pointer, which the circuit never holds.
- LOAD/SB/SH RAM words: 165,777 / 146,998 / 169,501 / 141,712 / 169,644 / 150,890 per chunk (the legacy-firmware run; the remapped-firmware run removes these, see `docs/HOST_ROLE_AUDIT.md` section 5, but it was not chained here). The circuit's own SW/SB/SH stores are correct but never consumed by a later circuit load.
- Forced tokens: 3,200 / 3,840 / 3,456 / 3,456 / 3,456 / 4,096 VDOT records whose weight cells fall in the 128 unaddressable WROM words are overridden by the reference token (circuit raw prediction disagrees on 99 %).
- The injection schedule (which step, slot, value) is precomputed from the reference trace by step index, as before.
- Chunk-start state for chunk 0 only (registers, pc): host snapshot of the reference at step 0. Chunks 1..5 start from the circuit's own state.
- `in_ram := 1` bring-up patch is still in the core (legacy firmware); the RAM address-range check is not real here.
- Not chained: the other 72 chunks (58 beyond the 20-chunk run), the remapped-firmware run (`ramown`), and an end-to-end token from the circuit.

## Circuit-owned RAM replay (2026-11-06, A100, remapped firmware; supersedes the "memory is host-served" caveat for RAM only)

Measured; details and caveats in `docs/HOST_ROLE_AUDIT.md` section 5. The earlier "injects LOAD/SB/SH" description overstated the memory role: of 1.12 M injections per chunk, 147 k (13 %) were LOAD/SB/SH and 973 k (87 %) were VDOT weight cells.

| Quantity (per 2.1 M-step chunk, chunks 0-2, prompt 785) | Before (legacy firmware, `in_ram := 1`) | Now (RAM-minimal firmware, real `in_ram`, snapshot RAM image restored) |
|---|---|---|
| LOAD/SB/SH injections into circuit RAM | 147 k - 170 k | **0** |
| Filtered-only (legacy firmware, ablation) | 14.3 k - 17.8 k (90 % fewer; all aliasing of the 4 MB map into 8192 slots) | n/a |
| DROM-class constants (blob) injected | in the above | 6.1 k - 8.7 k |
| VDOT weight injections | 0.83 - 0.97 M | 0.88 - 0.96 M |

CIRCUIT_PLACEHOLDER

Host roles remaining on the circuit side: VDOT weight operands (99 % of the remaining injections), 6-9 k blob constants per chunk (DROM-class, patched `in_drom`), chunk-start register/pc/RAM-image restore (chunks independent), the precomputed VDOT/blob schedule, and 0.17 % forced unaddressable-WROM tokens. Removed: all LOAD/SB/SH of RAM. Only chunks 0-2 of 67 were run on the remapped firmware.

## Open items and rulings

1. RESOLVED (coordinator): target ISA is RV32IM Qwenputer (plus VDOT/SMAQA) in the real Qwen3.5-0.8B/4B. Compiler lowers to RV32IM firmware; tensor-op `compiler/isa.py` is an intermediate form only.
2. OPEN, pending user ruling: weight memory. Proposed default (memory agent): ROM-in-appended-layers of the outer checkpoint (the weights become parameters of the outer model; the checkpoint is then not exactly stock Qwen3.5); fallback: host streaming weights (labelled bring-up only, same footing as the compiled route). Also needs the WB index widened (15 to 26 bits). Every full-token row depends on it (298 MB int4 vs 261,120 nibbles in state). Marked [needs R2] below.
3. OPEN, pending user ruling: cross-session activation transport. T5, T6 depend on it [needs R3].
4. OPEN, pending user ruling: numeric contract (W4A8 vs W8A8, fixed-point nonlinear ops, meaning of "equals"). All rows depend on it [needs R4]; argmax-vs-HF agreement is reported separately.

Dependency marks: T0-T4 need R2 and R4; T5-T6 need R2, R3, R4.

## How rows get promoted

A row flips from PROJECTED to MEASURED only when `experiments/circuit/integrate` produces a JSON with trace-token count, wall time, and argmax vs the reference emulator for that build, and that JSON is committed. Until a full-size run exists, tiny-model runs are reported in their own table with the model size stated, never in the tier rows.
