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

## ROM+MAC pipelines (memory/mac_layer.py, MEMORY_DESIGN s.8): step time on gpuB (A10G), results/mac_exec.{json,log}
`mac_exec_bench.py`: exact sparsity structure of mac_layer.run(K=230, H=2560, I=9216): ROM 9216 units (gate nnz 29, down 230 scattered only for active units = sparse ROM gather, 2.12M nnz),
PRODUCT 1840, DIGITIZE 5152, LOSUM 368 units; new Triton kernel `_ffn_mo` (multi-output CSR down, predicated scatter). TIMING ONLY (random table, relu-gated kernel arithmetic; values are checked by mac_layer.py, not here; DeltaNet-read layers modelled as 13 x 11 us = measured head-kernel time).
| treatment (13 pipelines x [DN read, ROM, PRODUCT, DIGITIZE, LOSUM]) | added ms/step | total with chip step (0.2015 ms) | tok/s |
|---|---|---|---|
| sparse, 52 kernels sequential | 0.198 + 0.143 (DN) | **0.543 ms** | **1,842** |
| sparse, each stage launched once for all 13 pipelines (grid.y) | 0.080 + 0.143 | 0.425 ms | 2,353 |
| dense HF bf16 (stock Qwen3_5MLP x 52, no DN layers: lower bound) | **17.4 ms** | >17.6 ms | <57 |
Per-pipeline kernels: ROM 7.3 us, PRODUCT 4.3, DIGITIZE 4.9, LOSUM 4.4 (floor ~4.4 us per kernel). Memory: sparse ROM+MAC state 231 MB for 13 pipelines (CSR fp32/int32); dense 52 MLPs 6.9 GB bf16 (3.68 B params) + the 13 DN layers + the 4B base 8.4 GB: tight on 23 GB.
Implied: 2.0e5 positions per Qwen3-0.6B token for the MACs (MEMORY_DESIGN s.7) x 0.543 ms = **~108 s** (sparse) / ~85 s (stage-batched) versus >= 3,500 s dense; nonlinear glue not included. Caveat: the 13 pipelines share one residual in the real circuit, so the stage-batched number assumes independent residual regions (not designed).

## Batch scaling with tuned warps (ffn 4, head 8), results/exec_batch_wh8.log
Aggregate tok/s, all rows exact: B=4 11.9k, B=16 19.0k, B=64 **21.95k** (343 tok/s per row). B=1 per-sublayer tuned: 4,963.
Not done: megakernel (timeline shows every kernel at its ~4.5 us floor with zero inter-kernel gaps in the graph, so fusing launches saves at most ~2 us x 21 = ~40 us of 201 us; not attempted), q3 firmware on the A10G (no riscv64 gcc / q3.elf on gpuB; e2e agent's A100 run: 4,225 tok/s untuned), HF-path check.

## q3.elf (e2e agent's tiny Qwen3 firmware, 16 tokens) on gpuB with the tuned executor, results/q3_tuned.{json,log}
`run_tuned.py q3.elf`: 708,920 trace steps (529,035 instructions) in 141.9 s = **4,995 tok/s (0.2002 ms/step), exact token-for-token vs ref.Machine** (first_mismatch null).
The e2e agent's untuned A100 run was 4,225 tok/s. => 14.2 s per tiny-model token (45,056 int4 weights) at 44.3k trace tokens/token.
