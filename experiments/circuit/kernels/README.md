# Firmware kernels for the Qwenputer RV32IM CPU: instructions and TRACE TOKENS per MAC

Goal: the cheapest inner loop for int4-weight x int8-activation matvec (>95% of Qwen3-0.6B's 596M MACs/token).
Everything here is run on the Qwenputer reference `Machine` (npu branch, the exact trace the compiled layers must emit), so
the token counts are what the circuit would emit, not an estimate. Every kernel is bit-exact vs a numpy reference
(`run_all.py` asserts it). Setup: `export QWENPUTER_NPU=<qwenputer npu-branch>/experimental/riscv; python run_all.py; python project.py`.

Files: `harness.py` (assemble + run + per-opcode token accounting), `kernels.py` (existing ISA: RV32IM+Zba, SMAQA, VDOT/WB),
`ext.py` + `kernels_ext.py` (PROPOSED instructions, modelled on the same Machine), `run_all.py` -> `res_kernels.json`,
`project.py` -> `res_projection.json`.

## 1. The cost model that drives everything (from `ref.py Machine.record`)
* Cost is TOKENS, not instructions. An instruction that writes rd != x0 emits `value_record(v)`: **1 token if -2^14 <= v < 2^14, 2 if |v| < 2^29, else 3**.
  rd = x0, not-taken branch, SW: 1 token. Taken branch / jal / jalr: 1 (J). SB / SH: 4 tokens (!). Every 63rd record without a taken jump adds 1 refresh token.
* **MUL / MULH cost exactly an ADD** (one value record; the 5-stage multiplier lives inside the layers). No multi-token penalty for multiplies. DIV is 4+ tokens.
* So small values are cheap and packed words are expensive: a `lw` of 4 packed int8 costs ~2.0 tokens on real (heavy-tailed, absmax-quantised) activations and 2.8 on uniform random bytes; pointers must stay < 16384 (use negative loop indices / small RAM addresses; DROM addresses at 0x20000 cost 2 tokens per pointer bump).
* Consequence: SWAR/lane tricks that create big words lose most of their instruction savings (K1b: 2.0x fewer instructions than K1, only 1.2x fewer tokens).
* VDOT/SMAQA: one record = value_record(accumulator). The accumulator stays < 2^14 for K=1024 rows on real data, so VDOT is ~1.0 token for 8 MACs = the 0.125 tok/MAC floor of the current ISA.
  The WROM index of a VDOT is `x[rs1] + (its PC word index - the WB's PC word index) - 1`: weights are streamed *by code position*, so VDOTs must follow a `WB` consecutively with the loads placed BEFORE the WB (a gap would waste WROM words).

## 2. Results (K = 1024, int4 weights, int8 acts; "real" = heavy-tailed activations; uniform in `res_kernels.json`)
| kernel | what it does | instr/MAC | **tok/MAC** | vs K0 |
|---|---|---|---|---|
| K0 | plain C: lb,lb,mul,add,2 ptr++,bne | 7.007 | **8.007** | 1x |
| K1 | unroll x16, immediate offsets (lb,lb,mul,add) | 4.195 | **4.320** | 1.9x |
| K1b | plain RV32IM, **2 rows per mul** (weights packed w0 + w1<<16, 16-bit lanes, flush/32, exact) | 2.130 | **3.508** | 2.3x |
| K2 | SMAQA int8 weights (lw w, lw a, smaqa) | 0.804 | **1.325** | 6.0x |
| K3 | VDOT, one row at a time (acts reloaded every row) | 0.444 | **0.708** | 11x |
| K5 | VDOT, structure of the existing compiler kernel (`compiler/fw`: 64 acts in 16 regs, 4 rows/WB, acc round-trips memory) | 0.177 | **0.180** | 44x |
| K4 RB=8,C=4 | VDOT register-blocked, 8 accumulators | 0.174 | 0.209 | |
| K4 RB=16,C=4 | 16 accumulators, 8 act regs | 0.151 | 0.170 | |
| K4 RB=22,C=2 | **22 accumulators in registers over the whole K loop**, 4 act regs, 1 WB per 2 chunks, neg-index pointer | 0.150 | **0.163** | 49x |
| K4 RB=22 + WB per chunk | gap-free alternative (extra addi) | 0.156 | 0.170 | |
| K6 | K4 (RB=20) + **fused requant epilogue** y = clamp((acc*m_r)>>S, +-127), pack 4 int8/word | 0.159 | **0.1765** | |
| K7 | K4 (RB=22) + **LM-head**: fused scale + argmax, no store, no softmax | 0.152 | **0.1656** | |

What mattered, in order: (1) VDOT (45x); (2) hold the accumulators in registers across all of K and block rows so the act loads
are amortised (K3 0.708 -> K4 0.163; the existing kernel's per-pass accumulator load/store costs ~0.017 tok/MAC); (3) negative-index loop
counter (no end register, 1-token pointers) so 22 of 31 registers are accumulators (`abs_ptr=True` variant handles K > 2k, e.g. down_proj K=3072: 0.163); (4) loads before the WB, one WB per 2 chunks;
(5) argmax is free: ~3 instr per 128 VDOTs (`ble acc,best` is a taken branch in the common case = 1 token). Floor of this ISA = 0.125; K4 sits 30% above it
(loads 0.023, loop/WB/refresh ~0.012, requant epilogue 0.010).
Zba/Zbb do not help: unrolled loops have no address arithmetic left. Early cut for the LM head with this ISA: the blocked kernel needs weights in code order, so survivors cost WB+addr per VDOT
(~0.5 tok/MAC, K3-style 0.71): row pruning only pays if it removes > 3x of the MACs *at block granularity*; coordinate with `algo/lmhead_prune.py`.

## 3. Projected seconds/token (Qwen3-0.6B: 440M layer MACs + 156M head MACs; `project.py`, 5,562 tok/s = Qwenputer fused executor on L4 at 11 chip layers; step time scaled by chip layers)
Matvec only; the nonlinear glue (RMSNorm/RoPE/softmax/SiLU/attention, `experiments/circuit/nonlinear`) is extra.

| build | tok/MAC | tokens/token | s/token | h/token |
|---|---|---|---|---|
| T0 plain C int8 (K0) | 8.007 | 4.8e9 | 8.6e5 | 238 (9.9 days) |
| plain, unrolled (K1) | 4.320 | 2.6e9 | 4.6e5 | 129 |
| plain, lane trick (K1b) | 3.508 | 2.1e9 | 3.8e5 | 104 |
| SMAQA (K2) | 1.325 | 7.9e8 | 1.4e5 | 39 |
| VDOT, existing compiler structure (K5) | 0.180 | 1.07e8 | 1.93e4 | 5.36 |
| **VDOT blocked + requant / argmax (K6/K7)** | 0.174 | 1.04e8 | 1.86e4 | **5.17** |
| PROPOSED VDOT16 W4A4 (K9), 12 layers | 0.090 | 5.4e7 | 1.05e4 | 2.9 |
| PROPOSED VDOTA W4A8 (K8), 14 layers | 0.142 | 8.5e7 | 1.94e4 | 5.4 (no gain: depth eats it) |
| PROPOSED VDOTA16 W4A8 (K10), 14 layers | 0.079 | 4.7e7 | 1.07e4 | **3.0** |

So pure firmware tuning buys 1.04x over the existing VDOT kernel on tokens/MAC (0.180 -> 0.1736 incl. requant; 0.163 raw matvec vs 0.180 = 9.6%).
The existing ISA is within 30% of its own 0.125 floor; **the only big lever left is MACs per record token.**

## 4. Proposed new instructions (modelled + bit-exact kernels in `ext.py`/`kernels_ext.py`)
Hard constraint that rules out the obvious ideas: every register write must appear as a value token (the regfile heads are written from the embedding of the value token), so
there is no hidden accumulator, no "two results per token", no packed-lane accumulator trick that beats 1 token per rd (checked: 2 rows packed as acc0 + acc1<<16 yields a 2-3 token record, no gain).
Gain must come from more MACs per instruction. Candidates, measured:

1. **VDOT16 (W4A4)**: `rd += sum_{k<16} int4(act_k) * w_k`, acts = 8 nibbles of rs1 and 8 of rs2, 16 weights = WROM words (WP, WP+1), plus `WB16`. Same register-blocked kernel: **0.082 tok/MAC, 2.0x fewer tokens than K4**, bit-exact.
   Hardware (from `core.py` VDOT: L6 products, L7 splits, L8 column carries, L9-10 lookahead): products are one-hot nibble tables, 15 units each: 16 lanes = ~240 units (VDOT's 8 int8 lanes use ~240 too, 2 nibble products each); sums of 16 products span ~1,900 > one byte-level split, so +1 split stage; WROM keeps its 17 heads, each query returns two adjacent slots: +8 value dims (+~16 gated z dims), no new key heads; read ports unchanged (regs). Estimate: **+~500 MLP units (<6% of one sublayer), +0-1 chip layer (+0-9% step time)**. Risk: A4 activations (needs rotation/smoothing; check against `algo/quant_study.py`).
2. **VDOTA16 (W4A8)**: `rd += sum_{k<16} int8(mem[rs1+imm+k]) * w_k`: activations read straight from RAM (16 bytes, aligned), no `lw`, no register blocking, row-streaming kernel. **0.071 tok/MAC (2.3x vs K4, 2.5x vs existing K5)**, no quality change.
   Hardware: RAM must return 16 bytes per instruction: bank the 32 KB RAM 4-way (word index mod 4) so the same row address queries 4 banks in parallel (key-head count unchanged; +3x32 residual dims of read data; stores/lw gate one bank). But the acts arrive at L8 (after the address adder and the two RAM sublayers), so products/splits/columns/lookahead move to L9-L13: **+~3 chip layers (+27% step time) and ~+500 units**. Net with depth: 3.0 h vs 5.2 h. The 8-wide VDOTA (K8, 0.134) gains only 18% in tokens and loses it all to depth: do not build it.
3. Small: `QPK rd,rs1,rs2` = `rd = (rd >> 8) | (clamp8((rs1*rs2) >> S) << 24)` replaces ~8 epilogue instructions per output (0.010 -> ~0.004 tok/MAC, ~3%); `MINMAX` (Zbb min/max) alone gives half of it.
Recommended order: VDOT16 first (cheapest depth, 2x), then VDOTA16 only if the RAM-bank/depth cost holds.

## 5. Caveats / things the ISA + memory agents must know
* **WROM capacity and WP width**: the NPU WROM holds 32,640 words = 261K int4 weights, `WB` carries a 15-bit word index. Qwen3-0.6B needs ~55M words (26-bit index). All kernels above assume weights are streamed by PC from a big WROM; that memory (and a >15-bit WB payload, e.g. a 2-token WB) is the real blocker, independent of the kernel.
* Token counts use the reference Machine's `value_record`; real activations are modelled (Laplace, absmax-quantised, 2 outlier channels); numbers for uniform bytes are in `res_kernels.json` (<= +8% tokens).
* Accumulator tokens: if a row's partial sums exceeded 2^14 the VDOT record is 2 tokens; K=1024 real-ish data stays 1-token. Layers with huge outlier activations (down_proj) may cost up to ~+10%.
* Executor rate: 5,562 tok/s is the upstream L4 stories260K number (11 chip layers); measured rates for this build are in `experiments/circuit/scaleout`.
