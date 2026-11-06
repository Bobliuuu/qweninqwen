# Tier leaderboard: Qwen3-0.6B as a program on the RV32IM CPU inside a real Qwen3.5

Maintained by the integration lead. Rule: a number in a **measured** column comes from a committed log/JSON of a run of that exact build; everything else is **projected** and says how it was derived. Nothing on the circuit route has been run end to end yet, so every full-model row below is PROJECTED. Definition of the goal and the arithmetic: [`CIRCUIT_ROUTE_PLAN.md`](CIRCUIT_ROUTE_PLAN.md).

Not this: the README's 268.6 s/token result is the earlier compiled route (custom giant outer, 1006 layers, D=40775). It is a stepping stone, not a circuit-route tier.

## Leaderboard (last updated 2026-11-06, commit of this file)

s/token = trace tokens per token / trace-tok/s. Rate column for T1+ is Qwenputer's measured executor figure on a stories260K-class program, not yet measured on a Qwen3-0.6B-shaped program.

| Tier | Optimizations included | Instr/token | Trace tok/token | Trace tok/s | Instances | Hardware | s/token | Status | Evidence | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | literal: plain C int8 RV32IM, stock HF modules | 4.0e9 [P] | 5.6e9 [P] | 91-250 [M upstream, HF path] | 1 | L4 | 2.2e7 - 6.1e7 (0.7-1.9 yr) | PROJECTED | plan sec. 3, 5 | e713778 |
| T1 | T0 + fused executor (Triton + CUDA graph) | 4.0e9 [P] | 5.6e9 [P] | 5,562 [M upstream] | 1 | L4 | 1.0e6 (11.6 d) | PROJECTED | plan sec. 5 | e713778 |
| T2 | T1 + SMAQA int8 build (VDOT int4 is T3) | 5.4e8 matvec only [P from M instr/MAC 0.913] | 7.6e8 at 1.4 tok/instr [P] | 5,562 | 1 | L4 | 1.4e5 (39 h) | PROJECTED | plan sec. 3, 5 | e713778 |
| T3 | T2 + VDOT int4 (0.176 instr/MAC) + nonlinear ISA ext + algorithmic cuts | 1.05e8 matvec only [P from M 0.176] + nonlinear, not yet counted | 1.25e8-1.5e8 matvec only [P] (plan said 1.2e8) | 3,300-5,562 | 1 | L4/A10G | 2.3e4 - 4.5e4 (6-12 h), nonlinear extra |  PROJECTED | plan sec. 5 | e713778 |
| T4 | T3 + persistent kernel, faster GPU | same | 1.2e8 | 1.0e4-1.6e4 [E] | 1 | L40S/A100/H100 | 7.5e3 - 1.2e4 | PROJECTED | plan sec. 5 | e713778 |
| T5 | T4 + parallel instances / lanes (needs ruling on cross-session transport) | same | 1.2e8 | 2.1e4-3.8e4 agg. [M for independent sessions only] | 8-32 | 1 GPU | 3.2e3 - 5.7e3 | PROJECTED | plan sec. 5 | e713778 |
| T6 | T5 on several GPUs, layer pipelining | same | 1.2e8 | >=1e5 agg. [speculative] | many | 4-8 GPU | ~1.2e3 | PROJECTED | plan sec. 5 | e713778 |

Instr/token = trace tok/token / 1.4 tok per instruction (Qwenputer range 1.0-1.55). [P] projected, [E] estimate, [M] measured by someone else upstream (Qwenputer repo), not by us.

#Matvec-only instruction totals use 596.0 M MACs/token x measured instr/MAC: plain 4.18e9, SMAQA 5.44e8, VDOT 1.05e8. Plan estimates (5.6e9, 5.2e8, 1.2e8 trace tokens) were roughly right for plain, optimistic by 1.5x for SMAQA, close for VDOT. Attention and nonlinear ops are not included yet.

## Combination rules

- Effects multiply only when they act on different factors: instruction count (compiler, kernels, algo, nonlinear), trace tokens per instruction (ISA fusion), and tokens per second (engine, hardware, lanes). Within the same factor they overlap (for example nonlinear ISA and kernels both cut instructions) and must be re-measured jointly, not multiplied.
- Lanes (T5) multiply the rate only if the work is divisible without a serial dependency; layers are serial, so lane speedup applies within a layer's matvec rows, and the per-layer synchronisation cost is unmeasured.
- Stories260K scaling to Qwen3-0.6B assumes cost proportional to parameters; plan sec. 3 gives caveats (LM head 26 % of MACs, nonlinear ops not proportional). Treat as +-2x until the compiler agent's measured instr/MAC replaces it.

## Measured components (not full-model tiers)

| Component | Value | Evidence |
|---|---|---|
| One int4-shaped projection 3072x1024 (3.15 M MAC) on our RV32IM emulator, checked vs numpy: plain 7.01, SMAQA 0.913, VDOT 0.176 instr/MAC (kernel only, not run on the circuit CPU; trace tokens/instr 1.19-1.40 is upstream) | M (emulator) | `experiments/circuit/compiler/fw/res_proj_instr_per_mac.json`, 84878e3 |
| fp32 mul as gate netlist, bitsliced verified | 4,332 neurons, depth 124, max rel err 4.0e-7 | `experiments/circuit/nonlinear/res_fp.json` |
| fp32 add as gate netlist | 2,670 neurons, depth 176, max rel err 1.2e-7 | same |
| Qwenputer upstream numbers (stories260K token: 2.43M / 225k / 53k trace tokens; executor 5,562 tok/s batch 1) | upstream, not reproduced here | plan sec. 2 |
| Compiled route (different thing): 1 token, 268.6 s untuned, A100 | measured | `experiments/vslice/logs/real_L28.log` |

## Open items and rulings

1. RESOLVED (coordinator): target ISA is RV32IM Qwenputer (plus VDOT/SMAQA) in the real Qwen3.5-0.8B/4B. Compiler lowers to RV32IM firmware; tensor-op `compiler/isa.py` is an intermediate form only.
2. OPEN, pending user ruling: weight paging (M1) vs embedding-ROM (M2). Every full-token row depends on it (298 MB int4 vs 261,120 nibbles in state). Marked [needs R2] below.
3. OPEN, pending user ruling: cross-session activation transport. T5, T6 depend on it [needs R3].
4. OPEN, pending user ruling: numeric contract (W4A8 vs W8A8, fixed-point nonlinear ops, meaning of "equals"). All rows depend on it [needs R4]; argmax-vs-HF agreement is reported separately.

Dependency marks: T0-T4 need R2 and R4; T5-T6 need R2, R3, R4.

## How rows get promoted

A row flips from PROJECTED to MEASURED only when `experiments/circuit/integrate` produces a JSON with trace-token count, wall time, and argmax vs the reference emulator for that build, and that JSON is committed. Until a full-size run exists, tiny-model runs are reported in their own table with the model size stated, never in the tier rows.
