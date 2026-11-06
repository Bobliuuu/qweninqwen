# Tier leaderboard: Qwen3-0.6B as a program on the RV32IM CPU inside a real Qwen3.5

Maintained by the integration lead. Rule: a number in a **measured** column comes from a committed log/JSON of a run of that exact build; everything else is **projected** and says how it was derived. Nothing on the circuit route has been run end to end yet, so every full-model row below is PROJECTED. Definition of the goal and the arithmetic: [`CIRCUIT_ROUTE_PLAN.md`](CIRCUIT_ROUTE_PLAN.md).

Not this: the README's 268.6 s/token result is the earlier compiled route (custom giant outer, 1006 layers, D=40775). It is a stepping stone, not a circuit-route tier.

## Leaderboard (last updated 2026-11-06, commit of this file)

s/token = trace tokens per token / trace-tok/s. Rate column for T1+ is Qwenputer's measured executor figure on a stories260K-class program, not yet measured on a Qwen3-0.6B-shaped program.

| Tier | Optimizations included | Instr/token | Trace tok/token | Trace tok/s | Instances | Hardware | s/token | Status | Evidence | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | literal: plain C int8 RV32IM, stock HF modules | 4.0e9 [P] | 5.6e9 [P] | 91-250 [M upstream, HF path] | 1 | L4 | 2.2e7 - 6.1e7 (0.7-1.9 yr) | PROJECTED | plan sec. 3, 5 | e713778 |
| T1 | T0 + fused executor (Triton + CUDA graph) | 4.0e9 [P] | 5.6e9 [P] | 5,562 [M upstream] | 1 | L4 | 1.0e6 (11.6 d) | PROJECTED | plan sec. 5 | e713778 |
| T2 | T1 + VDOT/SMAQA int4 build | 3.7e8 [P] | 5.2e8 [P] | 5,562 | 1 | L4 | 9.3e4 (26 h) | PROJECTED | plan sec. 3, 5 | e713778 |
| T3 | T2 + QwenPU vector unit + nonlinear ISA ext + algorithmic cuts | <=8.6e7 [P] | <=1.2e8 [P] | 3,300-5,562 | 1 | L4/A10G | 2.2e4 - 3.7e4 (6-10 h) | PROJECTED | plan sec. 5 | e713778 |
| T4 | T3 + persistent kernel, faster GPU | same | 1.2e8 | 1.0e4-1.6e4 [E] | 1 | L40S/A100/H100 | 7.5e3 - 1.2e4 | PROJECTED | plan sec. 5 | e713778 |
| T5 | T4 + parallel instances / lanes (needs ruling on cross-session transport) | same | 1.2e8 | 2.1e4-3.8e4 agg. [M for independent sessions only] | 8-32 | 1 GPU | 3.2e3 - 5.7e3 | PROJECTED | plan sec. 5 | e713778 |
| T6 | T5 on several GPUs, layer pipelining | same | 1.2e8 | >=1e5 agg. [speculative] | many | 4-8 GPU | ~1.2e3 | PROJECTED | plan sec. 5 | e713778 |

Instr/token = trace tok/token / 1.4 tok per instruction (Qwenputer range 1.0-1.55). [P] projected, [E] estimate, [M] measured by someone else upstream (Qwenputer repo), not by us.

### Combination rules

- Effects multiply only when they act on different factors: instruction count (compiler, kernels, algo, nonlinear), trace tokens per instruction (ISA fusion), and tokens per second (engine, hardware, lanes). Within the same factor they overlap (for example nonlinear ISA and kernels both cut instructions) and must be re-measured jointly, not multiplied.
- Lanes (T5) multiply the rate only if the work is divisible without a serial dependency; layers are serial, so lane speedup applies within a layer's matvec rows, and the per-layer synchronisation cost is unmeasured.
- Stories260K scaling to Qwen3-0.6B assumes cost proportional to parameters; plan sec. 3 gives caveats (LM head 26 % of MACs, nonlinear ops not proportional). Treat as +-2x until the compiler agent's measured instr/MAC replaces it.

## Measured components (not full-model tiers)

| Component | Value | Evidence |
|---|---|---|
| fp32 mul as gate netlist, bitsliced verified | 4,332 neurons, depth 124, max rel err 4.0e-7 | `experiments/circuit/nonlinear/res_fp.json` |
| fp32 add as gate netlist | 2,670 neurons, depth 176, max rel err 1.2e-7 | same |
| Qwenputer upstream numbers (stories260K token: 2.43M / 225k / 53k trace tokens; executor 5,562 tok/s batch 1) | upstream, not reproduced here | plan sec. 2 |
| Compiled route (different thing): 1 token, 268.6 s untuned, A100 | measured | `experiments/vslice/logs/real_L28.log` |

## Open conflicts and questions (integration lead)

1. Several ISAs exist: `compiler/isa.py` is a tensor-op program (MATVEC, RMSNORM, SOFTMAX...), `algo/iss/rvsim.c` is an RV32 simulator, `engine/qwentpu` is Qwenputer's RV32 circuit. The tier table assumes the RV32IM + VDOT/SMAQA CPU. The compiler agent must say whether its ISA lowers to RV32IM instructions on that CPU or replaces it (a replaced ISA is a different outer-model compile and not Qwenputer-style).
2. Weight memory (298 MB int4 vs 261,120 nibbles in state) is unresolved; every full-token projection assumes M1 paging, which needs a user ruling.
3. T5/T6 need a ruling on cross-session activation transport.

## How rows get promoted

A row flips from PROJECTED to MEASURED only when `experiments/circuit/integrate` produces a JSON with trace-token count, wall time, and argmax vs the reference emulator for that build, and that JSON is committed. Until a full-size run exists, tiny-model runs are reported in their own table with the model size stated, never in the tier rows.
