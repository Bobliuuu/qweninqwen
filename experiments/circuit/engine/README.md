# Qwenputer executor on an A10G (gpuB)
Env: A10G 23GB, torch 2.14.1+cu130, transformers 5.19.0, triton 3.8.0; qwenputer branch `npu` (cb1c969 main / 2717232 npu), fused Triton executor `fusedexec.FreeRun3`,
stories260k.elf (12,834,869 trace tokens / 10,021,206 instr per 60-token story; reference text reproduced). Every run below is checked token-for-token against the reference machine prefix.
The executor runs the compiled circuit (the chip layers' sparse weights) in fp32 Triton kernels, not the HF Qwen3.5 module stack; the 16 GB Qwenputer-npu checkpoint was downloaded but the HF-path (stock Qwen3_5ForCausalLM, ~250 tok/s on L4) was NOT measured here.
| config | trace tok/s | notes |
|---|---|---|
| fused, B=1, default warps | 3,350 (0.2985 ms/step) | exact for 300k steps (storyrun) and 100k (bench_exec); kernel time = wall time (profile: _head3 57%, _ffn3 38%, decode 5%), so launch overhead is NOT the bottleneck; K steps/graph (K=4,16): 3,353 / 3,368 (no gain) |
| fused, B=1, head 32 warps / ffn 4 warps | 4,350 (0.230 ms/step) | +30%, exact (tune_warps.py grid: results/tune_warps.log) |
| batch B=2/4/8/16/32/64 (default warps) | 6.3k/10.6k/14.6k/17.7k/20.4k/21.5k aggregate | all rows exact; saturates ~21.5k tok/s (6.4x); per-row rate falls to 335 tok/s at B=64 |
| batch with 32 head warps | 10.8k (B=4) ... 17.0k (B=64) | worse than default warps for B>=4: use 32 warps only for B=1 |
Implied cost of a Qwen3-0.6B token with the compiler agent's q4v firmware (experiments/circuit/compiler/res_full_q4v.json: 586.6M instr at pos 0, 694M at pos 3; Qwenputer 1.19-1.55 trace tokens/instr => ~0.7-1.1e9 trace tokens):
one CPU at 4,350 tok/s: ~1.6e5-2.5e5 s (45-70 h) per token; perfectly 64-way split at 21.5k tok/s aggregate: ~3.3e4-5e4 s (9-14 h); splitting needs the token-bus/multicore design (experiments/circuit/multicore).
`bench_batched.py` (separate, earlier): qwentpu signed INT8xINT4 tile circuit compiled to a stock Qwen3.5-style model, B independent traces per forward, exact: 8x2 tile 41k MAC/s at B=256 on the A10G (vs 14.7 MAC/s baseline); not the Qwenputer route.

## Per-sublayer autotune (B=1), results/autotune.{json,log}
`autotune_layers.py`: CUPTI kernel durations over a grid of triton num_warps, best warps chosen per sublayer (ffn 4/8, head 8/16/32) and for _decode_embed (16 warps: 14.2 -> 3.9 us).
Per-step kernel sum 297 -> ~190 us; **0.2015 ms/step = 4,963 tok/s at B=1, exact token-for-token vs the reference (20k steps of stories260k)** (baseline 3,350; +48%).
Timeline (prof_timeline.py): 21 kernels/step, every kernel 4.5-21 us with zero gaps in the graph; ~4.5 us is the per-kernel floor, so a megakernel could save at most ~2 us x 21.
