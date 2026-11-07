# Why the firmware says 13 ('.') and HF says 15846 (' Question') for prompt [785]

Independent float re-implementation of the firmware numeric contract (`compiler/gen_fx.py`, `compiler/fw/qwen3_fx.c`) at position 0, CPU on Modal (no GPU), scripts `diag_quant.py`, `diag_quant2.py`, `diag_quant3.py`, results `diag_quant*.json`.
At position 0 (T=1) softmax == 1, attention output == v, so q/k/RoPE do not matter. The contract: per-row symmetric RTN weights (int4 qmax 7; int8 = two int4 planes, qmax 119), lm_head = tied embedding quantised, input embedding NOT quantised, dynamic per-tensor int8 activations before q/k/v, o, gate/up, down, lm_head, v int8 per kv head.

## Verdict: quantisation, not a bug. Specifically 4 layers (2, 3?, 26, 27) that carry the massive activation.
- My fp32 re-implementation == HF (argmax 15846, max logit diff < 1e-2): the pipeline is understood correctly.
- The same emulator running the fp32 soft-float C (res_full_f32.json) gives 15846 at all 3 positions == HF. So ISA/emulator/C structure are right; only quantisation changed the answer.
- Firmware W8 gives 14582, identical to HF with W8 weights only (my sim: 14582). Firmware W8 behaves like pure weight quantisation.
- Soft-float glue (res_full_q4v.json) and fixed-point nonlinears (res_full_fx_w4_e0/e1) both give 13: the nonlinear approximations are not the cause.
- Position 0 has a massive activation: the layer-2 MLP down_proj writes a ~6540 channel that stays in the residual through layer 25 (xmax 6543 at layers 2..25), and layer 27's down_proj writes ~4872. Everything else is O(1). Residual never exceeds Q16 (32768), so no overflow bug.

## Which layers break it (float sim, W4 weights + int8 acts at all sites)
Quantise ONLY one layer: layers 0,1,3..25 each keep argmax 15846 (cos 1.000). Layer 2 alone -> 838 (cos 0.80), layer 26 alone -> 3405, layer 27 alone -> 12. All 28 layers -> 661 (cos 0.754). `diag_quant2.json`.

| variant (W4 weights + per-tensor int8 acts unless stated) | argmax | rank of 15846 |
|---|---|---|
| fp32 (HF == mine) | 15846 | 0 |
| W4 weights only | 454 | 12990 |
| W8 weights only | 14582 | 1 |
| fp weights, int8 acts only | 3988 | 10 |
| firmware contract W4 (float sim) | 661 | 47427 |
| firmware contract W8 (float sim) | 9 | 14 |
| W4, act-quant ONLY at gate/up | 13 | 58751 |
| W4, all layers fp EXCEPT 2,3,26,27 (weights+acts fp there) | 15846 | 0 (cos 0.994) |
| W4, W8 weights at 2,3,26,27, act int8 there | 25715 | 11 |
| W4, W8 weights at 2,3,26,27, act OFF there | 15846 | 0 |
| W4, W8 weights at 2,26,27, act OFF there | 15846 | 0 |
| W4, fp down_proj only at 2,27 | 25 | 1399 |
| W8 everywhere except fp at 2,27 | 38297 | 1 |

Reading: the answer needs BOTH finer weights AND finer activations at layers 2, 26, 27 (per-tensor int8 on a vector with one 6540 channel zeroes everything else). W8 weights with int8 acts there still fails.

## Why 13 specifically
13 is one member of a cloud of wrong answers (454, 661, 838, 408, 284, 25, 13, ...). With int4 the final hidden state has cosine 0.75 to fp32 and top-1 flips on tiny numeric differences: my float sim reproduces 13 only in subsets (e.g. act-quant at gate/up only), not for the full contract (661). The exact integer firmware lands on 13; I did NOT bit-reproduce it (that needs the integer rounding chain), so "13 exactly" is not independently confirmed, only "something in this cloud".

## What would give the real token (untested in the firmware)
Per-layer exceptions at layers 2, 26, 27 (maybe 3): int8 weights (firmware already has `boost` per (layer, matrix) up to 12 bits) AND 16-bit activations (APL=2) at those layers. APL is currently a global define; making it per-layer is a firmware change, ~3 of 28 layers => roughly +10% instructions. Needs a new firmware + reference (~33 min CPU/lambda-final reference run) and a replay.

Caveat: float simulation of the contract, not bit-exact; single prompt token; no firmware variant was run.
