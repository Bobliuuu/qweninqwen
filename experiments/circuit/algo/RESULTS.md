# Algorithmic tier: instructions per token of the Qwen3-0.6B firmware on the Qwenputer-style RV32 CPU (outer = real Qwen3.5-0.8B / 4B)

Everything measured here ran on Modal (logs in `logs/`, JSON in `results/`, runs in `/RUNS.md`). Code: `iss/` (RV32IM instruction-set simulator with the NPU-branch
instructions SMAQA and VDOT, C matvec kernels compiled with riscv gcc), `cost_model.py`, `capacity.py`, `quant_study.py`, `lmhead_prune.py`.
The compiler agent's full-model counts (experiments/circuit/compiler) were not in the repo when this was written; I use the exact MAC count of Qwen3-0.6B
(595,984,384 = 28 layers x 17,... + lm_head 155,582,464) and a flagged estimate for the non-matvec work (5.4 M instr: norms, RoPE, SiLU, attention at ctx 128, argmax).
Trace tokens per instruction: 1.4 (Qwenputer measured 1.19-1.55); executor 5,500 trace tokens/s (L4, fused executor, Qwenputer README).

## 1. Matvec: instructions per MAC (measured, bit-exact vs numpy int64, 192x1024 matvec)
| kernel | instr/MAC | exact |
|---|---|---|
| plain C, int8 weights (Qwenputer "plain C" analogue) | 8.009 | yes |
| int4 packed, scalar nibble unpack in C | 7.008 | yes |
| int8, SMAQA (4 MAC/instr), one row | 1.758 | yes |
| int8, SMAQA, 4 rows share the activation load | 1.005 | yes |
| int4, VDOT (8 MAC/instr), one row | 0.633 | yes |
| int4, VDOT, 2 rows x unroll 2 (activations held in registers) | 0.3175 | yes |
| int4, VDOT, 4 rows x unroll 4 | 0.2066 | yes |
| int4, VDOT, 8 rows x unroll 4 | 0.1669 | yes |
| **int4, VDOT, 12 rows x unroll 2 (weight stream interleaved offline)** | **0.1588** | yes |
Theoretical floor for 8 MAC/VDOT: 0.125. The gain is pure code generation: hold the 8 activations in two registers and issue one VDOT per output row, accumulators in
registers, weights laid out in stream order by the host loader (data movement only). Plain C 8.0 instr/MAC reproduces Qwenputer's stories260K plain-C figure
(2.43 M tokens / 260 K MAC = 9.3 tokens/MAC at ~1.2 tokens/instr).

## 2. Implied cost per inner token of Qwen3-0.6B (cost_model.py)
| idea | instr/MAC | instructions | trace tokens | seconds @5.5k tok/s |
|---|---|---|---|---|
| plain C int8 | 8.009 | 4.90e9 | 6.86e9 | 1,246,000 (14 days) |
| int4 scalar | 7.008 | 4.29e9 | 6.00e9 | 1,091,000 |
| int8 SMAQA, 4 rows | 1.005 | 6.19e8 | 8.67e8 | 157,500 |
| int4 VDOT 1 row | 0.633 | 3.92e8 | 5.49e8 | 99,800 |
| **int4 VDOT, 12-row blocked** | **0.159** | **1.02e8** | **1.43e8** | **26,100 (7.2 h)** |
| hypothetical 32-MAC vector dot (QwenPU-style) | 0.041 | 3.1e7 | 4.3e7 | 7,800 |
| hypothetical 128-MAC | 0.018 | 1.6e7 | 2.3e7 | 4,150 |
| hypothetical 1024-MAC | 0.011 | 1.2e7 | 1.7e7 | 3,090 (floor: the 5.4 M non-matvec instructions) |
Total speedup from firmware alone: 48x (8.01 -> 0.159). It is nowhere near interactive: 26,000 s/token. Getting to seconds needs (a) a wide vector/matvec
instruction (hundreds of MAC per trace token) AND (b) a faster executor than 5.5k tokens/s, or the frame-streaming route (vslice, 265 -> ~29 s/token measured).

## 3. Ideas that did NOT work (measured)
* **Rigorous exact pruning of the 151,936-way head** (partial dot + Cauchy-Schwarz tail bound, stages 64..512 dims): fraction of MACs = 1.000 (`logs/algo_lmhead_*.log`).
  Exact argmax gives no saving on the tied 1024-dim head.
* **Approximate shortlist** (rank-r SVD sketch, then exact on the top-k): on 140 calibration positions the true argmax's worst sketch rank is ~149,000 of 151,936 at every r<=512
  (median 130 at r=512), so k would need to be ~the whole vocabulary: no saving (`logs/algo_lmshort_*.log`). Argmax-only already avoids the softmax (never computed in the plan; 0.6 M instr).
* **Plain per-token absmax activation quantisation is too lossy for Qwen3-0.6B**, even at 8 bits: W8A8 argmax agrees with fp32 HF on only 81% of 140 teacher-forced positions
  (mean relative logit error 14%), W4A8 g128 61%, W4A6 54%, W3 24%, W2 0% (`logs/algo_quant1_*.log`). So int4 weights and int8 activations as in Qwenputer's NPU are NOT sufficient
  for argmax == HF without better quantisation (per-group activation scales, outlier handling, GPTQ/AWQ-style weights); the second grid (weight-only, per-group activations, mixed
  down_proj precision) was queued on Modal when this was written (app ap-tHRNr6w2lbtiE9EimUFLVs, results will be appended to `logs/` and RUNS.md).
  Consequence: the cost rows above assume the quantisation question is solved at int4/int8; the per-instruction counts do not depend on it, exactness does.

## 4. Weight memory (capacity.py; configs from Hugging Face)
Qwen3-0.6B needs 595,984,384 weights = 298 MB at int4, 2,282x Qwenputer's 261,120 int4 weights.
| | Qwen3.5-0.8B | Qwen3.5-4B |
|---|---|---|
| hidden / layers (GDN + full) / FFN | 1024 / 24 (18+6) / 3584 | 2560 / 32 (24+8) / 9216 |
| DeltaNet state floats, all GDN layers | 4.72 M | 12.58 M |
| int4 weights if state stores 1 bit/float (Qwenputer layout) | 1.18 M (0.2% of inner) | 3.15 M (0.5%) |
| int4 weights if state stores 24-bit integers per fp32 float | 28.3 M (4.8%) | 75.5 M (12.7%) |
| frames (inputs_embeds, 24 bits/float) to stream the whole int4 model once | 97,000 frames of 3 KB | 38,800 frames of 7.7 KB |
| FFN weights as a ROM, 6 int4 per fp32 weight | 1.59 G int4 (needs 9.0 of 24 layers' FFNs) | 13.6 G int4 (needs 1.4 of 32 layers' FFNs) |
Options: (1) state-as-RAM cannot hold the model on either size (best case 5-13%), so weights must be paged: (2) stream from the host as data in inputs_embeds frames
(~40-100k frames per token, ~8-30 s at the executor speeds above, exactly the vslice mechanism; the host only transports); (3) weights-in-checkpoint ROM: 4B can hold the full
int4 model in 2 layers' worth of FFN matrices, 0.8B needs ~9 layers (and then the CPU's ~11 layers do not fit alongside), but this bakes inner weights into the outer checkpoint
(conflicts with the audit's "inner weights not baked" rule) and needs an address-decode + readout path that is not built; (4) hidden-size changes (trimming/enlarging) are out
under the "real fixed size" rule. Recommendation: Qwen3.5-4B + frame streaming for weights, VDOT-class wide instructions for compute.
