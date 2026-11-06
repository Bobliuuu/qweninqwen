# Measured outer step throughput (A100 40GB shared, torch 2.14 cu130, transformers 5.20.2, NO fla/causal_conv1d kernels -> torch reference GDN)
Outer = stock Qwen3_5ForCausalLM, 24 layers (23 GDN 16x128x128 + 1 full-attn stub), D=4096, inter 8192, 3.39B params (random).
| dtype | chunked prefill 2048 steps | single-step decode |
| fp32 | 0.923 s -> 2219 steps/s | 25 steps/s |
| bf16 | 0.494 s -> 4145 steps/s | 24 steps/s |
Implication: outer steps must be batched as chunked prefill rounds; per-token decode is 90-170x slower.
Rough paging bound (untested design): 494M weights / 2048 floats per step = 241k write steps -> ~110 s (fp32) / ~58 s (bf16)
per inner token IF frame carries one 2048-float payload per step and reads/rounds add <2x. Needs fla kernels for more.
Bit-circuit baseline (T0/T1, ref_copy parallel_complete.py rerun here, contended CPU BF16): 60-195 MAC/s (ref recorded 458-488);
real 896-weight row: 46 s (19 MAC/s) => 494M MAC ~ 1e6-8e6 s/token.
