# Nonlinear ops and ISA extensions on the Qwenputer RV32 core (Qwen3-0.6B as firmware)

Target machine: the Qwenputer RV32IM+Zba core compiled into stock Qwen3.5 layers (`~/qwenos-work/qwenputer`, docs `experimental/riscv/docs/rv32.md`),
DeltaNet state = RAM/ROM/regs, one trace token ~ one executor step. The `npu` branch (`~/qwenos-work/qp-npu`) adds `WB`/`VDOT` (8 int8 x 8 int4 from
a weight memory) and `SMAQA` (4 int8 products). Qwen3-0.6B is firmware on that core. This directory prices every non-matmul op of one token in
instructions and trace tokens, and designs/verifies cheap custom instructions for them. Nothing here was compiled into DeltaNet layers (see "Not done").

## What was built and measured
* `rvsim.py`: RV32IM+Zba interpreter (loads the .text of a clang `--target=riscv32` object; no linker needed) with the Qwenputer `VDOT/WB/SMAQA` semantics
  (from `ref.py`) and the extensions below; counts dynamic instructions AND Qwenputer trace tokens (record sizes from rv32.md: rd write = 1/2/3 tokens by
  15-bit chunks, branch/jump/NT = 1, SW = 1, SB/SH = 3, +3 per DIV, +1 R-refresh per 63 instructions).
* `kernels.c` (build with `build.sh`): fixed-point firmware kernels, each in a plain-RV32IM version and an extension version; `test_kernels.py` runs them in the
  interpreter, checks results vs float/numpy references, and writes `res_kernels.json` (per-element slopes between two sizes). Run on the Mac (CPU seconds).
* `ext.py`: bit-exact integer reference semantics of the extensions (tables + interpolation). `counts.py` -> `res_counts.json`: per-token element counts x measured slopes.
* `netlist.py`, `fp.py`, `test_fp.py`, `res_fp.json`: an earlier gate-level (AND/OR/XOR) fp32 add/mul netlist, bitsliced-simulated on the A10G (kept; superseded by the ISA route).

## The proposed instructions (custom-1, opcode 0x2b)
| op | semantics | verified (interpreter vs float) |
|---|---|---|
| `EXP2 rd,rs1` | 2^(rs1/2^16) in Q24, 256-entry table + linear interp, saturating | max rel err 5.1e-6 (soft deg-4 polynomial: 8.3e-6) |
| `RSQ rd,rs1` | (rs1/2^16)^-1/2 in Q24; priority encoder + parity shift + 512-entry table + interp | max rel err 5.6e-6 over ms in [2^-8,2^14] (soft 3-Newton: 3.7e-5) |
| `SIGM rd,rs1` | sigmoid(x) Q16, 768-entry table over [-12,12) + interp | max abs err 2.6e-5 (1.7 LSB) |
| `AMX rs1,rs2` / `AMXR rd,rs1` | running (max,argmax) in 2 hidden registers; first max wins; AMXR reads index and optionally resets | argmax == numpy.argmax in 15 random trials |
| `ROT16 rd,rs1,rs2` | RoPE pair rotate: packed int16 (x1,x2) x packed Q15 (cos,sin) -> packed int16 | max abs err 1.3 LSB of int16 (identical to the soft kernel) |
Scores/log2e: the 1/sqrt(hd)*log2(e) factor is folded into the q weights (free), so softmax needs only EXP2 on (score - max).

## Per-element cost (measured in the interpreter; instructions / trace tokens, includes loads, stores and loop overhead; 8x unrolled)
| op | plain RV32IM | with extension | speedup (instr) |
|---|---|---|---|
| rsqrt scalar (RMSNorm) | 68.7 / 110 | 3 / 4 (incl. ret) | 23x |
| exp2 scalar | 38 / 73 | 3 / 4 | 13x |
| softmax element (exp+sum+scale passes) | 50.5 / 93 | 14.4 / 22.5 | 3.5x |
| SwiGLU element (silu(g)*u) | 61.0 / 119 | 10.6 / 18.3 | 5.7x |
| RoPE pair | 23 / 42 | 8 / 16 | 2.9x (3 ld/st dominate) |
| RMSNorm sumsq / scale*gain per element | 4.4 / 6.4 and 7.6 / 10.3 | unchanged | - |
| argmax element, logits in RAM | 4.1 / 4.3 | 3.25 / 3.4 | 1.3x |
| argmax element fused into the lm_head epilogue | ~2.5 (estimate) | 1.0 (estimate) | 2.5x |
| residual add / requant to int8 | 4.6 / 8.7 | unchanged | - |
| matvec int8 MAC, plain | 4.5 / 4.8 | SMAQA 0.875 | 5x |
| matvec int4 MAC, plain (unpack nibbles) | 5.94 / 6.06 | VDOT 1 row 0.42; VDOT with 8 rows sharing activations 0.162 | 14x / 37x |
(`res_kernels.json` has every number.)

## Per-token totals for Qwen3-0.6B (`res_counts.json`, element counts from the config: 28 layers, d=1024, 16q/8kv x128, I=3072, V=151936)
matvec: 595,984,384 MACs/token (440.4M layers + 155.6M lm_head). Elements/token: RMSNorm 144,384 (+729 rsqrt scalars), SwiGLU 86,016, RoPE 43,008 pairs
(+64 sin/cos), residual 57,344, requant 201,728, dequant 344,064 (ESTIMATE 3 instr each), softmax 448 rows x T, argmax 151,936.
| scenario | instructions/token | trace tokens/token |
|---|---|---|
| matvec, plain RV32IM int8 | 2.68e9 | 2.87e9 |
| matvec, plain RV32IM int4 | 3.54e9 | 3.61e9 |
| matvec, SMAQA int8 | 5.2e8 | 5.7e8 |
| matvec, VDOT int4 (1 row) | 2.5e8 | 2.6e8 |
| matvec, VDOT int4 (8 rows share activations) | 9.7e7 | 9.9e7 |
| all nonlinear + glue, plain, T=1 / T=512 | 1.13e7 / 2.28e7 | 1.95e7 / 4.08e7 |
| all nonlinear + glue, with extensions, T=1 / T=512 | 6.1e6 / 9.4e6 | 9.6e6 / 1.47e7 |
So with a fast matvec (VDOT x8, 9.7e7) the plain nonlinear part is 10-19% of the instructions (and 16-29% of the tokens because those values are wide) and the
extensions bring it to 6-9% (T=512). Dominant leftovers after the extensions: RMSNorm sumsq+scale (1.7e6), requant+dequant glue (2.8e6), softmax at long T, residual (2.7e5).
At the executor's measured ~3.7k instr/s (L4, rv32.md) one token of this program is still ~7 h of stream time at 1e8 instructions: matvec dominates until it gets another ~100x.

## State / layer budget of the extensions (ESTIMATES from rv32.md; not built in the IR)
* Tables are ROM (like the DROM): EXP2 257x4 B = 1 KB, SIGM 769x4 B = 3 KB, RSQ 513x4 B = 2 KB = ~6 KB = about one DeltaNet key head (the 128 KB DROM uses 16 heads, 8 KB each).
  They must be read by a dedicated head keyed on a one-hot of the top index bits (not by a load instruction, which would cost instructions).
* Datapath: index one-hot decode (1 MLP stage, ~256-768 units), table read (1 head, 2-hot query {i,i+1}), interpolation = a small multiply (diff x 8-bit fraction;
  nibble-product columns like the existing multiplier stages s1..s3), result mux. Rough size 1.5-2.5k MLP units per table op; critical path starts after the regfile read (L5):
  index L6, read L7, interpolate L8-L10, output L11-L12 => probably +1 layer (11 -> 12) for the table ops unless fused with the existing multiplier stages.
  RSQ also needs a priority encoder (threshold ladder, 1 stage) and a variable shift (existing shifter, 2 stages) => it is the deepest.
* `ROT16`: 4 parallel 16x16 products = ~4x a 16x16 slice of the existing 5-stage multiplier (L5-L9), roughly +5k units spread over L5-L9 (sublayers there are at 6.6-8.1k of 9.2k units now), so it likely forces wider MLPs or +1 layer.
* `AMX/AMXR`: no table. Two extra rows (MAXV, MAXI) in the register head (rows 32 -> 34, plus a read address for AMXR), the compare is the existing signed compare (SLT, L7);
  the update decision is carried by the output token exactly as taken/not-taken branches and SW/SB tokens do (new families UPD/NOUPD, 1 token each), so the lagged write (key from t-1,
  beta from token t) works with no extra layer. Budget: ~2 register rows + ~100 MLP units + 2 token families. Cheapest and the only one needed to make 151,936-way argmax cost ~1 instr per logit.
* All of this must be re-fit to Qwen3.5-0.8B (one value head per key head; the core uses 2 -> ROM needs twice the key heads; see below).

## Honest caveats / not done
* The extensions are specified and verified as ISA semantics in an interpreter and the firmware that uses them is verified against float references. They are NOT implemented in the
  DeltaNet layer compiler (`chipformer/q35`); instruction-count savings assume each op is one instruction with the stated trace-token record. No layer/head count here has been through the IR.
* VDOT/SMAQA/WB semantics are re-implemented from `ref.py`'s docstring in `rvsim.py`; the weight-memory pointer model is simplified (consecutive 8-weight groups).
* Token counts come from my model of the P2 trace format (rv32.md), not from `chipformer.rv.ref`; tokens/instruction here is 1.1-2.0 (wide fixed-point values cost 2-3 tokens), higher than the 1.2-1.5 quoted for C programs.
* Fixed-point ranges: RMSNorm assumes post-sum mean-square within Q16 range and |x| < ~128; the real Qwen3 residual stream has outliers; formats/scales need a calibration pass (see algo/quant_study).
* The WROM holds 32,640 words x 8 int4 = 261k weights; Qwen3-0.6B has 596M. Weight storage/streaming (the other agents' work) is the real blocker, not the nonlinears.
* Sizing flag for the outer model: rv32.md says the Qwenputer P3 merge has hidden size 3968 and the standalone P2 core needs 2,816 dims (2,679 IR dims, overlayable only in part); this is wider than a stock small
  Qwen3.5 (the 0.8B has 1 value head per key head and the core needs 2 - rv32.md "Not done in P3"). "Stock size 0.8B/4B" and "exactly like Qwenputer" are not the same thing: needs a decision.

## Cross-check against the compiler agent's measured full-model run (`experiments/circuit/compiler/res_full_q4v.json`)
Their firmware (`fw/qwen3.c`, int4 VDOT matvec, glue in picolibc soft-float) executed Qwen3-0.6B for 4 positions in 2.60e9 instructions (6.5e8 per position). Their per-function
breakdown shows the glue is almost all soft-float: `__mulsf3` 1.25e9, `__addsf3` 2.5e8, `__subsf3` 2.5e8, `__divsf3` 7.2e7, `expf` 3.8e7, compares/conversions ~1.4e8 (matvec itself 5.8e7 = 2%).
So ~85-90% of their current instruction count is exactly the nonlinear/glue work priced here: replacing it with the fixed-point kernels of this directory is 1.1e7 (T=1) to 2.3e7 (T=512) instructions
per token without extensions and 6e6 to 9e6 with them, i.e. ~30-60x fewer than their ~4.6e8 soft-float glue instructions per position (my T=1..512 vs their prompt positions 0..3). That swap has NOT been run end to end
(accuracy of an all-fixed-point forward vs HF is untested; their run's position-0 argmax already differs from HF in q4v mode).
