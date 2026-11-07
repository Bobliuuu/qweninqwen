# Quantisation study (history)

Detail kept for the record. The current result and design are in the [README](../../README.md).

## Variants studied

Prompt `[785]`, real Qwen3-0.6B. The first rows are Hugging Face; the others are the firmware run in the compiler-level emulator. Position 0 is an attention sink: the layer-2 MLP writes a massive channel (about 6,540) that stays in the residual through layer 25, and layer 27 writes about 4,872.

| Variant | argmax | rank of 15846 | cos(logits, HF) | top-1 gap | instr/token | weight stream |
|---|---|---|---|---|---|---|
| HF fp32 | **15846** (`' Question'`) | 0 | 1 | 0.136 | | |
| HF bf16 | **15846** | 0 | | 0.125 | | |
| HF fp32 + int4 per-row weights (fake-quant) | 454 | | | 0.059 | | |
| HF fp32 + int4 weights + int8 activations (fake-quant) | 838 | | | 0.041 | | |
| Firmware, int4 everywhere (old contract) | **13** | 38,852 | 0.296 | 0.167 | 161,511,713 | 74,498,048 |
| Layers 2/26/27: 8-bit weights, int8 activations | 9 | 14 | 0.808 | 0.247 | 170,497,018 | 80,396,288 |
| Layers 2/26/27: int4 weights, 16-bit activations | 408 | 12,077 | 0.339 | 0.148 | 171,138,874 | 74,498,048 |
| **Layers 2/26/27: 8-bit weights + 16-bit activations (final)** | **15846** | 0 | 0.917 | 0.970 | 189,109,345 (+17%) | 80,396,288 (+7.9%) |
| Layers 2/3/26/27: 8-bit + 16-bit | 15846 | 0 | 0.916 | 0.930 | 198,308,671 | 82,362,368 |
| Layers 2/26/27: 12-bit weights + 16-bit activations | 15846 | 0 | 0.978 | 0.940 | 207,079,555 | 86,294,528 |

What it shows: both finer weights and 16-bit activations at those three layers are needed (each alone fails with 9 or 408), and the final contract's top-1 gap (0.97) is wide, so the answer is not a coin flip. In the layer-by-layer study, quantising only layer 2, 26 or 27 flips the answer (838, 3405, 12) while quantising any one of the other 25 layers keeps 15846. Sources: [`quantfix/results/`](../../experiments/circuit/fullrun/quantfix/results/) (firmware rows), [`diag_quant/REPORT.md`](../../experiments/circuit/fullrun/diag_quant/REPORT.md) (the layer study), [`logs/final_token/hf_785.json`](../../experiments/circuit/fullrun/logs/final_token/hf_785.json) (HF rows). The HF fake-quant rows are torch simulations, not the firmware.


## Firmware contracts compared


| | Old contract | Final contract |
|---|---|---|
| Model | real Qwen3-0.6B, 28 layers, prompt `[785]`, ctx 1 | same |
| Weights | int4 everywhere | int4, except layers **2, 26 and 27 at 8-bit weights** |
| Activations | int8 | int8, except layers **2, 26 and 27 at 16-bit** |
| Instructions (compiler-level count) | 161,511,713 | **189,109,345** (+17%) |
| Weight stream | 74,498,048 words (595,984,384 weights) | **80,396,288 words** (643,170,304 weights, +7.9%) |
| CF+RAM trace (this run's reference) | 157,076,451 tokens, 138,513,317 instructions, 75 chunks | **199,287,158 tokens, 178,066,316 instructions, 96 chunks** |
| Answer for `[785]` | 13 | **15846** |

The compiler-level counts and the CF+RAM trace counts come from different builds and are not comparable across the two rows. Sources: [`quantfix/REPORT.md`](../../experiments/circuit/fullrun/quantfix/REPORT.md), [`quantfix/results/`](../../experiments/circuit/fullrun/quantfix/results/).


## FAQ note

**Why 15846, and why was it 13 before?**
15846 is what Hugging Face's Qwen3-0.6B produces for this prompt (fp32 and bf16). The earlier firmware used int4 weights and int8 activations everywhere. It matched its own emulator exactly, but quantisation changes the answer at position 0, where a massive activation (about 6,540 at layer 2) sits in the residual. Raising layers 2, 26 and 27 to 8-bit weights and 16-bit activations gives 15846 ([study above](#quantisation-study)).


