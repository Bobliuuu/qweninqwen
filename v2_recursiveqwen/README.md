# v2 prototype: multi-model hosting ("recursiveqwen")

Still the *hosted* kind of nesting (see top-level README): several inner models share the same appended layers of one Qwen3.5-0.8B checkpoint, each in its own hidden slice, heads and MLP neurons.

- per-model RoPE tables via `rope_type="longrope"` config values (no shared integer grid limit)
- per-model qk-norm slots, one flag dim per hosted model, foreign vocab masked in `lm_head`
- verified: Qwen3-0.6B + Qwen3-0.6B-Base in one checkpoint (15.6B params, 52 layers); full-depth fp32 Qwen3-0.6B argmax 22/22, greedy identical 3/3; truncated multi-model checks incl. Qwen3-1.7B. Full-depth Base/outer and bf16 GPU checks were not completed.
- params grow ~cubically in the number of hosted models (`params.py`); recursion (Qwen-in-Qwen-in-Qwen) not done.

Files: `merge2.py` (builder), `test2.py` (checks), `params.py` (analytic param counts).
