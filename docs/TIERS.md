# Tier leaderboard: Qwen3-0.6B as a program on the RV32IM CPU inside a real Qwen3.5

Maintained by the integration lead. Rule: a number in a **measured** column comes from a committed log/JSON of a run of that exact build; everything else is **projected** and says how it was derived. Nothing on the circuit route has been run end to end yet, so every full-model row below is PROJECTED. Definition of the goal and the arithmetic: [`CIRCUIT_ROUTE_PLAN.md`](CIRCUIT_ROUTE_PLAN.md).

Not this: the README's 268.6 s/token result is the earlier compiled route (custom giant outer, 1006 layers, D=40775). It is a stepping stone, not a circuit-route tier.

## Leaderboard (last updated 2026-11-06, commit of this file)

s/token = trace tokens per token / trace-tok/s. Rate column for T1+ is Qwenputer's measured executor figure on a stories260K-class program, not yet measured on a Qwen3-0.6B-shaped program.

| Tier | Optimizations included | Instr/token | Trace tok/token | Trace tok/s | Instances | Hardware | s/token | Status | Evidence | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | literal: plain C int8 RV32IM, stock HF modules | 4.0e9 [P] | 5.6e9 [P] | 91-250 [M upstream, HF path] | 1 | L4 | 2.2e7 - 6.1e7 (0.7-1.9 yr) | PROJECTED | plan sec. 3, 5 | e713778 |
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
| tiny Qwen3-architecture firmware, 16 generated tokens | dim 64, 8 heads / 4 kv, head size 8, hidden 128, 1 layer, vocab 128, random weights | Qwenputer fused circuit executor (checkpoint size and GPU not stated in the log; asked e2e) | 708,920 steps (reference 708,921 tokens, 529,035 instr: 1.34 tok/instr) = 44.3k trace tokens per generated token | 4,225 tok/s (0.237 ms/step) = 10.5 s per tiny token [M] | fused trace "exact" vs reference emulator; HF check of the same firmware at f32 rel err 6.3e-7 (compiler `res_tiny.json`) | `experiments/circuit/e2e/logs/fused_q3_16tok.log` | eb4f4f3 |

This is the first MEASURED circuit-route run. It validates the pipeline (firmware -> executor -> exact trace) at toy size only. Nothing here is a Qwen3-0.6B number.

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

## Measured components (not full-model tiers)

| Component | Value | Evidence |
|---|---|---|
| One int4-shaped projection 3072x1024 (3.15 M MAC) on our RV32IM emulator, checked vs numpy: plain 7.01, SMAQA 0.913, VDOT 0.176 instr/MAC (kernel only, not run on the circuit CPU; trace tokens/instr 1.19-1.40 is upstream) | M (emulator) | `experiments/circuit/compiler/fw/res_proj_instr_per_mac.json`, 84878e3 |
| fp32 mul as gate netlist, bitsliced verified | 4,332 neurons, depth 124, max rel err 4.0e-7 | `experiments/circuit/nonlinear/res_fp.json` |
| fp32 add as gate netlist | 2,670 neurons, depth 176, max rel err 1.2e-7 | same |
| Qwenputer upstream numbers (stories260K token: 2.43M / 225k / 53k trace tokens; executor 5,562 tok/s batch 1) | upstream, not reproduced here | plan sec. 2 |
| Nonlinear kernels on RV32IM: rsqrt 68.7 -> 3 instr, exp2 38 -> 3 instr (extension), rel err 5e-6; RMSNorm scalar part ~6.2k instr (base and ext) | M (emulator) | `experiments/circuit/nonlinear/res_kernels.json` |
| Lane exactness: row-split int8 matvec on 1-16 lockstep lanes equals numpy; bus-limited speedup 1.5x (2 lanes) to 2.5x (16) of matvec phase | M (Qwenputer ref.Machine) | `experiments/circuit/multicore/res_tiny.json` |
| Compiled route (different thing): 1 token, 268.6 s untuned, A100 | measured | `experiments/vslice/logs/real_L28.log` |

## Open items and rulings

1. RESOLVED (coordinator): target ISA is RV32IM Qwenputer (plus VDOT/SMAQA) in the real Qwen3.5-0.8B/4B. Compiler lowers to RV32IM firmware; tensor-op `compiler/isa.py` is an intermediate form only.
2. OPEN, pending user ruling: weight paging (M1) vs embedding-ROM (M2). Every full-token row depends on it (298 MB int4 vs 261,120 nibbles in state). Marked [needs R2] below.
3. OPEN, pending user ruling: cross-session activation transport. T5, T6 depend on it [needs R3].
4. OPEN, pending user ruling: numeric contract (W4A8 vs W8A8, fixed-point nonlinear ops, meaning of "equals"). All rows depend on it [needs R4]; argmax-vs-HF agreement is reported separately.

Dependency marks: T0-T4 need R2 and R4; T5-T6 need R2, R3, R4.

## How rows get promoted

A row flips from PROJECTED to MEASURED only when `experiments/circuit/integrate` produces a JSON with trace-token count, wall time, and argmax vs the reference emulator for that build, and that JSON is committed. Until a full-size run exists, tiny-model runs are reported in their own table with the model size stated, never in the tier rows.
