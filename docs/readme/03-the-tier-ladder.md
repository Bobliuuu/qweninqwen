<p align="center">
  <img src="docs/assets/circuit_tiers.svg" alt="Instructions per token on a log scale for tiers T0 to T3m and the final run; measured tiers solid, projected tiers hatched" width="100%">
</p>

The project is organised as tiers: each is a harder claim or a faster build than the one before. T0 to T6 are the roadmap from the planning documents ([`docs/TIERS.md`](docs/TIERS.md), [`docs/CIRCUIT_ROUTE_PLAN.md`](docs/CIRCUIT_ROUTE_PLAN.md)); instruction counts below are from the compiler emulator, and seconds are computed, not run, unless the row says measured.

| Tier | Build | Status | Evidence |
|---|---|---|---|
| **T0** | literal: fp32 soft-float firmware on RV32IM, real Qwen3-0.6B weights, ctx 1: 1.10e11 instructions/token | instructions **measured** (argmax == HF at 3 positions, including position 0); seconds **computed** at 0.71 to 0.98 years/token, not run | [`compiler/res_full_f32.json`](experiments/circuit/compiler/res_full_f32.json) |
| **T1** | T0 + fused executor (Triton + CUDA graph) | the fused executor runs the measured build below; the fp32 firmware at this tier is a **projection** (4.0e9 instr) | [`TIERS.md`](docs/TIERS.md) |
| **T2** | + `SMAQA` int8 matvec build | **projected** (5.4e8 instr, matvec only) | [`TIERS.md`](docs/TIERS.md) |
| **T3** | + `VDOT` int4 + fixed-point nonlinears + algorithmic cuts | **projected** for the matvec part (1.05e8 instr) | [`TIERS.md`](docs/TIERS.md) |
| **T3m** | T3 as built (compiler kernel) | instructions **measured**: 1.346e8 at ctx 1; seconds **computed** (7.9 to 10.3 h) | [`compiler/RESULTS.md`](experiments/circuit/compiler/RESULTS.md) |
| **T3m-run** | the T3m-style int4 firmware with weights and constants as `ROW` fetches, circuit-owned RAM, real address decode, layers 2/26/27 at higher precision, **run end to end** | ✅ **reached, measured**: one token, 96 chunks, 199,287,158 trace tokens, 96/96 clean, 95/95 exact seams, word 15846 | [`FINAL_VERDICT.json`](experiments/circuit/fullrun/logs/cfram_q_vast/FINAL_VERDICT.json) |
| **T4** | + persistent kernel, faster GPU | 📋 plan, projected | [`TIERS.md`](docs/TIERS.md) |
| **T5** | + in-forward-pass lanes (SIMT) | 📋 plan, projected (needs the 4B base; bounded gain) | [`multicore/res_model.json`](experiments/circuit/multicore/res_model.json) |
| **T5b** | + independent sessions splitting matvec rows | 📋 plan, projected (needs cross-session data transport, a ruling is open) | [`scaleout/model_out.json`](experiments/circuit/scaleout/model_out.json) |
| **T6** | T5 on several GPUs, layer pipelining | 📋 plan, speculative | [`TIERS.md`](docs/TIERS.md) |

**Where the measured run sits.** The final build is a T3m-class firmware: int4 `VDOT` weights, fixed-point nonlinears in integer code, executed on the fused executor. It adds weights and constants as `ROW` fetches, circuit-owned RAM, real address decode and the precision design above. Its measured cost is 199.3M trace tokens (more than the 1.55e8 T3m model, because of the table fetches and the higher-precision layers). A single GPU would need about 9.6 hours at the 4090's 6.0 min per chunk, or 15.4 hours at the Modal A100's 9.6 min per chunk (computed from the measured chunk times), the same order as the T3m projection. The measured run used **about 12 GPU-hours of job time** in total, spread over up to a dozen GPUs.

The parallel run splits the trace **in time**, not the matvecs, so it is a different parallelisation from T5, T5b and T6. It works because every seam between jobs is checked for exact state equality.

### The host-role ladder

The second axis is the host's share. Each step below moved a role into the circuit; the evidence column points at the run or document.

| Step | What moved into the circuit | Evidence |
|---|---|---|
| H1 weights as `ROW` fetches | 74.5M weight words from the outer model's own embedding rows | 84 chunks, 174,823,694 free tokens, 0 mismatches ([audit](docs/HOST_ROLE_AUDIT.md)) |
| H2 circuit-owned RAM | RAM loads and stores handled by the circuit's 32 KB map | [`experiments/circuit/ramown`](experiments/circuit/ramown/) |
| H3 constants as rows | the firmware's blob constants (`CF`) | [`docs/CONST_FETCH_PATH.md`](docs/CONST_FETCH_PATH.md), chunks 0 to 2: 0 injections of any kind |
| H4 real address decode | the `in_ram` / `in_drom` bring-up patch removed | [`ramown/realdecode`](experiments/circuit/ramown/realdecode/) |
| H5 look-ahead removed | whole row table, the circuit's own `HALT`, chunk 0 from the ELF alone | [`docs/HOST_ROLE_AUDIT_CFRAM.md`](docs/HOST_ROLE_AUDIT_CFRAM.md) |
| H6 table inside the checkpoint | the table is `embed_tokens.weight` of the merged checkpoint | [stock-module check](#stock-module-check) |
| H7 precision design | layers 2/26/27 at 8-bit weights and 16-bit activations: the token equals the real model's | [precision results](#precision-results) |
