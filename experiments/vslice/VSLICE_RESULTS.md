# vslice toy (debug vehicle ONLY, not the result): random Qwen3-arch d=128, 4q/2kv heads x32, inter 256, 2 layers, vocab 512
Run: python toy_test.py N  (A100; ssh box ~/qiq/true/vs_pkg/trueqiq).  Outer: stock Qwen3_5ForCausalLM, 34 layers (33 GDN + 15..: 2 full-attn), D=5641, 2.02B params fp32.
Result: 4 prompt + 4 generated tokens; outer logits vs HF max rel err 5e-7..1.9e-6 every step; generated tokens identical (389,408,375,424).
1530 outer steps/inner token, ~12-14 s/token on A100 (stock forward, torch GDN).
Host: embedding row, flags, one-hot addresses, silu-pre-inverted weight payloads, folded norm gains W*diag(g), q/k-norm gain payload, position_ids, argmax/append.
In outer weights/state: all norms, matvec (GDN state banks), registers, SwiGLU, products, RoPE+softmax+KV (outer full attention), LM head logits.
Bugs found: (1) section-5 depth-ordering error (registers must sit between producer/consumer; unrolled pipeline), (2) shared AO slot leaked from inactive attention layers,
(3) attention scale forgot outer hdo^-1/2, (4) pure-GDN stack crashes in mask builder, (5) transparent norm needs BALLAST>=2^27, (6) norm/XS poison avoided by GDN gated-RMSNorm heads.
