# Final-token check, prompt [785] ('The')

## HF Qwen3-0.6B (real model) vs the firmware numerics

| run | argmax | text | gap top1-top2 | rank of token 13 | rank of 15846 |
|---|---|---|---|---|---|
| HF fp32 | 15846 | ' Question' | 0.136 | 22538 | 0 |
| HF bf16 | 15846 | ' Question' | 0.125 | 24099 | 0 |
| HF fp32 + int4 per-row weights only (= firmware weight contract) | 454 | 'up' | 0.059 | 109 | 12850 |
| HF fp32 + int4 weights + int8 per-token activations at every linear (closest to firmware W4A8, exact nonlinears) | 838 | 'ark' | 0.041 | 28 | 52394 |

Single-token-prompt scan (first 3000 ids): int4 weights agree with fp32 argmax on 0.0%, int4+int8 activations on 0.0%.

## Circuit read-out

Circuit's own tail decoded: word=13 bytes=[13, 0, 0, 0] halt_at=1887187; reference tail word=13; circuit == reference: True; free mismatches in this chunk: [0].
