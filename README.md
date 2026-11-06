# qweninqwen

The smallest Qwen actually running inside a Qwen.

**Inner:** Qwen3-0.6B (all 28 layers, real weights). **Outer:** the real pretrained Qwen3.5-0.8B.
One merged Qwen3.5 checkpoint (7.56B params, mostly structural zeros): hidden = 1024 outer + 1024 inner + 1 flag (padded to 2080), 24 original layers + 28 appended full-attention layers.
Inner tokens are vocab rows >= 248320. The stock `Qwen3_5ForCausalLM` forward code computes the inner model; nothing is patched. Everything is done in weights.

Approach differs from the bit-level circuit route in the reference zip (about a year per token): inner tensors are embedded block-diagonally, and Qwen3.5 quirks are cancelled by weight choices (see `src/nest2.py` docstring: norm rescale, q/k-norm padding, folded attention scale, shared RoPE grid 10^(-k/32), gate saturation via a flag dim).

## Results
- fp32 (CPU): inner mode max |dlogit| 0.008 vs Qwen3-0.6B, identical argmax and 20-token greedy text. Outer mode max |dlogit| 0.05 vs untouched Qwen3.5-0.8B, identical text.
- bf16 (A100): 2/5 prompts byte-identical over 48 tokens; others diverge at near-tie tokens (bf16 rounding), text stays fluent and correct. See `demo_gpu_results.json`.
- Mutation: zeroing inner MLPs of the merged model breaks inner output.
- Speed: 11.4 tok/s nested vs 27.7 tok/s plain (HF, A100, no fused kernels).

## Caveats
Outer mode relies on outer top logit > 0 (inner vocab logits are 0 there). Norm eps differs slightly per subspace (source of the small logit diffs). Stage 1 (`src/nest.py`): Qwen2.5-0.5B compiled into a Qwen3.5-architecture shell, no pretrained outer.

Run: `python src/test_stage2.py fp32 cpu` (needs ~32GB RAM) or `python src/demo_gpu.py` (A100).
