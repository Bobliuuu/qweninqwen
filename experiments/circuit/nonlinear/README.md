## Drop-in header for the firmware (`nl_fixed.h`) and verification on real activations
`nl_fixed.h` (single source of truth, `static inline`, no libc/relocations) = every kernel below in three build modes: default plain RV32IM, `-DNL_EXT` (custom-1 instructions via `.insn`, works in gcc >= 10 binutils and clang),
`-DNL_EXT_EMU` (the same instruction semantics in C, `nl_ext_ref.h`, for host tests). Formats are documented at the top of the header; entry points: `nl_rms_inv_q24`, `nl_rms_scale`, `nl_softmax_q16`,
`nl_swiglu_q16`, `nl_rope_pairs`, `nl_sincos_q15`, `nl_argmax_begin/feed/end`, `nl_requant_i8`, `nl_add`. An emulator that wants the instructions includes `nl_ext_ref.h` (opcode 0x2b, f3 = 0 EXP2, 1 RSQ, 2 SIGM, 3 AMX, 4 AMXR, 5 ROT16).
Checks: `test_ext_c.c` (C reference vs `ext.py`, 20,000 vectors, 0 mismatches), `test_emu_vs_sim.py` (header built for host with EXT_EMU vs RV32 EXT in the interpreter: 200 kernel calls, 0 mismatches),
`test_real.py` (below).
**Real-activation error vs HF fp32** (`capture_acts.py` on the A10G box: Qwen3-0.6B, 64-token prompt, layers 0,1,7,14,21,27; `res_real.json`, `test_real.log`; worst over layers; soft and EXT builds agree except where noted):
| op | max abs err / max abs ref | rel L2 | note |
|---|---|---|---|
| RMSNorm d=1024 (in/post norm, with gain) | 1.1e-4 .. 5.0e-3 (layer 0: 5e-3) | 5e-4 .. 5.6e-3 | layer 0 activations have rms 0.027, so Q12 (2.4e-4) is too coarse there: use a per-layer input scale (Q16) for layers 0-1 |
| q/k RMSNorm d=128 | 2.7e-4 .. 1.6e-3 | 3.4e-4 .. 1.2e-3 | |
| softmax (rows of 10, 32, 64 scores, 16 heads) | 1.6e-5 .. 3.1e-5 abs prob | 1.9e-4 .. 4.3e-4 | needed a Newton step on the 1/sum reciprocal (1e-3 error before) |
| SwiGLU d=3072 | 1.1e-5 .. 2.5e-4 | 1.8e-5 .. 2.5e-3 | 64-bit-safe products: gate/up reach hundreds (massive activations) |
| RoPE (pairs, int16 with per-layer power-of-two scale) | 1.7e-4 .. 6.0e-4 | 3.8e-4 .. 1.4e-3 | |
Real-model findings that changed the kernels: the residual stream has massive activations (|x| up to 6.5e3, rms 26 from layer 7 on), which overflowed the first 32-bit sum-of-squares and `x<<8` products; the header now accumulates mean-square in 64 bits (mul/mulh) and
multiplies with mul/mulh (+~40-60% instructions on RMSNorm/SwiGLU vs the first version). Not yet validated: an all-fixed-point end-to-end forward (error compounding over 28 layers), long contexts (T > 64), the 4 final-layer logits.

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
| `RSQ rd,rs1` | (rs1/2^16)^-1/2 in Q24; priority encoder + parity shift + 2x513-entry table (parity x mantissa) + shift-friendly interp | max rel err 7.4e-6 over ms in [2^-8,2^14] (soft 3-Newton: 3.7e-5) |
| `SIGM rd,rs1` | sigmoid(x) Q16, 768-entry table over [-12,12) + interp | max abs err 2.6e-5 (1.7 LSB) |
| `AMX rs1,rs2` / `AMXR rd,rs1` | running (max,argmax) in 2 hidden registers; first max wins; AMXR reads index and optionally resets | argmax == numpy.argmax in 15 random trials |
| `ROT16 rd,rs1,rs2` | RoPE pair rotate: packed int16 (x1,x2) x packed Q15 (cos,sin) -> packed int16 | max abs err 1.3 LSB of int16 (identical to the soft kernel) |
Scores/log2e: the 1/sqrt(hd)*log2(e) factor is folded into the q weights (free), so softmax needs only EXP2 on (score - max).

## Per-element cost (measured in the interpreter; instructions / trace tokens, includes loads, stores and loop overhead; 8x unrolled)
| op | plain RV32IM | with extension | speedup (instr) |
|---|---|---|---|
| rsqrt scalar (RMSNorm) | 68.7 / 110 | 3 / 4 (incl. ret) | 23x |
| exp2 scalar | 38 / 73 | 3 / 4 | 13x |
| softmax element (exp+sum+scale passes) | 50.5 / 92.5 | 14.3 / 22.8 | 3.5x |
| SwiGLU element (silu(g)*u, 64-bit-safe products) | 65.2 / 126 | 14.6 / 25.2 | 4.5x |
| RoPE pair | 22.6 / 48 | 4.6 / 10.2 | 4.9x |
| RMSNorm mean-square (exact 64-bit accumulation) / scale*gain per element | 7.4 / 14.2 and 13.6 / 24.3 | unchanged | - |
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
| all nonlinear + glue, plain, T=1 / T=512 | 1.32e7 / 2.47e7 | 2.40e7 / 4.52e7 |
| all nonlinear + glue, with extensions, T=1 / T=512 | 7.7e6 / 1.10e7 | 1.33e7 / 1.86e7 |
So with a fast matvec (VDOT x8, 9.7e7) the plain nonlinear part is 12-20% of the instructions (and 20-32% of the tokens, wide values cost 2-3 tokens) and the
extensions bring it to 7-10% (T=512). Dominant leftovers after the extensions: RMSNorm mean-square+scale (3.0e6), requant+dequant glue (2.8e6), softmax at long T (3.3e6 at T=512), residual (2.7e5).
At the executor's measured ~3.7k instr/s (L4, rv32.md) one token of this program is still ~7 h of stream time at 1e8 instructions: matvec dominates until it gets another ~100x.

## State / layer budget of the extensions (ESTIMATES from rv32.md; not built in the IR)
* Tables are ROM (like the DROM): EXP2 257x4 B = 1 KB, SIGM 769x4 B = 3 KB, RSQ 2x513x4 B = 4 KB = ~8 KB = about one DeltaNet key head (the 128 KB DROM uses 16 heads, 8 KB each).
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

## Attention dot products (QK scores and probability-weighted V), hd=128, int8 K/V (`kernels3.c`, `res_attn.json`)
Not nonlinear, but it is the only other per-token cost that grows with context T and it turns out to be larger than the nonlinears. Instructions per cached position per (layer, q head), 128 MACs each:
| kernel | instr / position | tokens / position | instr per MAC |
|---|---|---|---|
| scores, plain RV32IM (lb,lb,mul,add) | 620 | 749 | 4.8 |
| scores, SMAQA, 4 keys share each q-word load | 81.75 | 181 | 0.64 |
| PV, plain (int16 Q15 prob x int8 V) | 747 | 1208 | 5.8 |
| PV, SMAQA with int8 probabilities, V stored transposed (t contiguous) | 88 | 194 | 0.69 |
Both SMAQA kernels are bit-exact vs numpy int dot products. Per token (448 = 28 layers x 16 q heads rows): SMAQA 448 x 170 x T = 7.6e4 x T instructions (T=64: 4.9e6, T=512: 3.9e7, T=2048: 1.6e8); plain 6.1e5 x T (T=512: 3.1e8).
So past T ~ 250 the attention dot products exceed the whole nonlinear budget, and past T ~ 1300 they exceed an 8-row VDOT matvec (9.7e7). Accuracy caveat: int8 probabilities (per-row scale) give 1.4-2.1% relative L2 error on the PV output
(real probs from the 6 captured layers, random V) versus 1.8e-4..4.6e-4 for int16 probabilities, so the SMAQA PV path needs a hi/lo byte split (2 x 88 instr) or a check against HF before use.
No cheap new instruction was found for these: a fused "16-byte dot from RAM" needs 4+4 word reads per instruction and each RAM read port is a separate copy of the RAM key heads (state x4), so I did not propose one;
KV bytes (57 KB per position for int8 K,V over 28 layers) also exceed the 32 KB RAM, which is the real constraint (KV cache placement is outside this directory).
