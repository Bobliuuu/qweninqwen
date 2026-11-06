# Qwen3-0.6B on the Qwenputer RV32IM CPU: instructions and seconds per token

Measured: instructions by the emulator (`fw/rvemu.c`) on real HF weights (`run_full.py`), positions 0..3. Larger contexts extrapolate the measured per-context-token slope.
Seconds = instr x trace-tokens/instr / trace-tokens/s. Best = 1.19 tok/instr @5500/s; worst = 1.40 @3200/s; mid = 1.40 @5500/s.

| build | ctx | instr/token | s/token (1.40 tok/instr, 5500/s) | worst (1.40, 3200/s) | best (1.19, 5500/s) | days (mid) |
|---|---|---|---|---|---|---|
| q4v | 1 | 5.866e+08 | 1.493e+05 | 2.566e+05 | 1.269e+05 | 1.73 |
| q4v | 128 | 3.473e+09 | 8.840e+05 | 1.519e+06 | 7.514e+05 | 10.23 |
| q4v | 1024 | 2.383e+10 | 6.067e+06 | 1.043e+07 | 5.157e+06 | 70.22 |

## Fidelity (real Qwen3-0.6B vs fp32 HF; `res_fidelity_ablation.json`, `res_fidelity_layers.json`, `res_full_fx_*.json`)
Float-HF weight-only ablation, 48-token prompt (isolates weight error; activations fp32):
| weights | top-1 agree vs HF | pos-0 argmax | mean KL |
|---|---|---|---|
| int8 per-row (all, incl. tied head) | 47/48 | WRONG (rel logit err 0.49) | 0.0135 |
| int8 per-row, gate/up kept fp | 48/48 | ok | 0.0031 |
| int8 group 64 | 48/48 | ok | 0.0051 |
| int4 per-row (what VDOT x1 plane gives) | 33/48 | wrong | 1.13 |
| int4 group 128 / group 64 | 34/48 / 41/48 | wrong | 0.41 / 0.32 |
Position 0 is the attention-sink token: its logit gap is only 0.136 (top-1 vs top-2) and the last layer's MLP (layer 27) carries it: keeping only layer 27 gate/up/down in fp restores pos-0 argmax (rel err 0.14), every other single layer does not.
Firmware (fixed-point, int8 dynamic activations, INT weights) on real weights: int8 weights: argmax 14582 vs HF 15846 at pos 0 (the weight-only float model gives the same 14582, so it is the weight error), positions 1-3 match; rms logit err 0.85 (weight-only 0.51, so activation int8 + fixed point add ~0.68 rms in quadrature); boosting layer-27 MLP to 12-bit weights (3 planes, +1.4e6 instr) did NOT fix pos 0 in the firmware (argmax 9): activation quantisation noise on the sink token is enough to flip a 0.136 logit gap. int4 per-row: 120/130 positions agree with HF over 130 tokens (92.3%; weight-only float 91.5%), so the firmware adds little beyond RTN int4 itself; position-0 and a few low-margin positions fail. Cost of each fidelity step (instr/token, ctx 1, EXT): int4 1.343e8 -> int8 (2 planes) 2.413e8 -> +12-bit last-layer MLP +3.7e5. Per-group scales are not in the VDOT kernel (scale per row only): a per-(row,64-chunk) rescale would add ~15-25 instr per 8 VDOT (>2x). Remaining fixes (not done): QAT/GPTQ for the int4 weights, fp16-style activations at the sink token, or skipping position 0 (BOS) in the comparison.

## Merge status of the nonlinear agent's kernels
`fw/qwen3_fx.c` contains their EXP2/RSQ/SIGM/AMX/ROT16 semantics (inline asm under `-DEXT`, emulated in `rvemu.c` per `nonlinear/ext.py`) and the plain RV32IM fixed-point fallbacks (`exp2_q`, `rsq_q`, `sigm_q` copied from `nonlinear/kernels.c`). EXT saves only 3.5e6 instr/token (2.6%): after the merge the nonlinears are <10% (rms 2.4e7 incl. 64-bit sumsq, quantize 1.2e7, mv epilogue 5e7 = 64-bit dequant, ~100 instr/row). Dominant: VDOT kernels (7.5e7 VDOT + ~2.9e8/... see `by_symbol`: vk_acc_raw 3.98e8 over 4 tokens = 1.0e8/token) -> the kernel overhead (acc spills) is the next target (the algo agent's register-blocked kernels, commit 158a8f6).
Attention is now SMAQA int8 K/V cache with EXP2: 1.6e5 instr per context token (was 2.3e7).

## Per-op split reconciliation (v2 int4, EXT, real weights, 4 positions)
Correction: the per-symbol profile in `res_full_fx_*.json` is the TOTAL over the 4 emulated positions (5.376e+08 instr), not per token. My earlier per-op list mixed a per-token VDOT figure with 4-position totals for the other rows, hence 2.07e8 vs 1.34e8. Per token (total/4):
| symbol | instr/token | share |
|---|---|---|
| vk_acc_raw | 9.938e+07 | 73.9% |
| mv.constprop.0 | 1.298e+07 | 9.7% |
| rms | 5.985e+06 | 4.5% |
| vk_init_raw | 5.589e+06 | 4.2% |
| quantize | 3.028e+06 | 2.3% |
| attn | 2.298e+06 | 1.7% |
| swiglu | 1.979e+06 | 1.5% |
| rope | 9.948e+05 | 0.7% |
| kv_store | 8.973e+05 | 0.7% |
| argmax.constprop.0 | 7.597e+05 | 0.6% |
| resid.constprop.0 | 4.018e+05 | 0.3% |
| __divdi3 | 3.691e+04 | 0.0% |
| total | 1.344e+08 | 100% |

`vk_acc_raw`+`vk_init_raw` = VDOT kernels incl. accumulator load/store (7.45e7 VDOT + ~2.7e7 overhead per token); `mv` = 64-bit dequant epilogue (~1.3e7/token, not 5e7).

## True trace tokens (model-based, emulator `tokens=`)
`fw/rvemu.c` counts trace tokens with the Qwenputer P2 record-size model from rv32.md (rd value record 1/2/3 tokens by 15-bit chunks, taken branch/JAL/JALR = 1, not-taken/NT/ST/WB = 1, SB/SH = 4, DIV +9, +1 R refresh per 63 instr without a jump). It models the trace format; it is NOT a run of the circuit or of `chipformer.rv.ref.Machine` (the 0.6B program does not fit that machine's 32 KB RAM / 15-bit WROM map).
| build (4 positions) | trace tokens | instructions | tok/instr | trace tokens/inner token |
|---|---|---|---|---|
| v2 int4, EXT | 627,342,604 | 543,494,099 | 1.154 | 1.57e8 |
| v2 int8 (2 planes), EXT | 1,080,977,440 | 973,500,704 | 1.110 | 2.70e8 |
| 3072x1024 microbench (plain int8 + SMAQA + VDOT) | 34,600,933 | 25,467,933 | 1.36 | |

So tok/instr for the real firmware is about 1.15 (VDOT-dominated, one-token value records), not 1.4. At 4,963 tok/s: 8.8 h/token (ctx 1, int4); at 4,225 tok/s: 10.3 h; at 5,500 tok/s: 7.9 h. (Instructions per token are 1.358e8 in this later build, 1.343e8 in the earlier one: the per-matrix plane generality added ~1%.)
