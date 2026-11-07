# Tiers: Qwen3-0.6B as a program on the RV32IM circuit inside a Qwen3.5-based outer model

One measured result, and the roadmap of faster builds. Definition of the goal and the arithmetic: [`CIRCUIT_ROUTE_PLAN.md`](CIRCUIT_ROUTE_PLAN.md). Detailed maintainer notes: [`history/TIERS_working_notes.md`](history/TIERS_working_notes.md).

## Measured end to end (2026-11-07)

One Qwen3-0.6B token (prompt `[785]`, position 0, ctx 1) on the circuit, measured in `experiments/circuit/fullrun/logs/cfram_q_vast/` ([`FINAL_VERDICT.json`](../experiments/circuit/fullrun/logs/cfram_q_vast/FINAL_VERDICT.json)):

| Item | Value |
|---|---|
| Build | weights and constants as `ROW` fetches from the outer model's own embedding rows, circuit-owned RAM, real `in_ram`/`in_drom` decode; int4 weights with layers 2/26/27 at 8-bit weights + 16-bit activations |
| Instructions / trace tokens | 178,066,316 / 199,287,158 (about 1.12 tokens per instruction), 96 chunks of 2,097,152 steps |
| Result | the circuit's own HALT at step 57,705 of chunk 95; word **15846** = Hugging Face Qwen3-0.6B's token |
| Checks | 96/96 jobs clean (0 free mismatches, 0 forced, 0 injections); 95/95 seams exact |
| Time | 6.0 min per chunk on an RTX 4090, about 9.6 min on a Modal A100-40GB; chunks ran in parallel, so wall time is about one chunk given enough GPUs; about 12 GPU-hours of job time in total |
| Cost | about $10 of rented Vast.ai GPUs (fleet-log estimate) plus Modal A100 time for chunks 32 to 70 |

## Roadmap

Seconds in the table are computed or projected, not run, unless stated. Instruction counts are from the compiler emulator.

| Tier | Build | Status |
|---|---|---|
| T0 | literal fp32 soft-float firmware, 1.10e11 instructions/token | instructions measured; 0.7 to 1 year/token computed |
| T1 | T0 + fused executor (Triton + CUDA graph) | projected (4.0e9 instructions) |
| T2 | + `SMAQA` int8 matvec | projected (5.4e8 instructions, matvec only) |
| T3 | + `VDOT` int4, fixed-point nonlinears, algorithmic cuts | instructions measured at 1.346e8 (ctx 1); seconds computed |
| **T3m-run** | the T3 firmware with `ROW` fetches, circuit-owned RAM and real decode, run end to end | ✅ **measured**, section above |
| T4 | + persistent kernel, faster GPU | projected |
| T5 | + in-forward-pass lanes (SIMT) | projected, needs the 4B base |
| T5b | + independent sessions splitting matvec rows | projected, needs cross-session transport |
| T6 | T5 on several GPUs, layer pipelining | speculative |
