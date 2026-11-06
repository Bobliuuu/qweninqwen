# Variant 0 honesty audit: real 28-layer Qwen3-0.6B driver (test_vs2.py real151936_L28 1)

Box gpuB (A10G), see env.txt (transformers 5.20.2; 5.22.1 is not installable). Command in cmd.txt. Runtime 06:45-07:26 UTC (6 positions, ~41 min).

HostGuard (--entry layer, --fast): FAIL, 32 violations, all in the host decode of the 36 output rails (test_vs2.py):
- 6x host-math-on-outer-output + 6x host-math: `lg = codes @ amo` (host matvec over the outer's emit row, line 83)
- 6x host-math: `lg.max()` (display of top code score, line 85)
- 14x argmax-on-non-logits: `int(lg.argmax())` on the host-decoded scores (8 outer-derived; 6 are `rr.argmax()`, the HF reference argmax, a false positive of the metric)
No violations anywhere else: 0 host readbacks, 0 host math on outer activations between layers. Declared exempt scopes: see audit0.log.
Layer-major executor: no Qwen3_5ForCausalLM exists, so run_audit's --check-outer found no outer (outer_checks = []); audit/outer_check_layers.py
applies the same checks per stock Qwen3_5DecoderLayer (result: outer_layers0.* if present; the run was still in progress when this was committed).
Numerics on this box differ from the A100 log: maxrel 1.8e-2, 3.3e-1, 1.7e-1, 1.2e-1, 1.6e-1, 6.8e-2 for positions 0-5; argmax matches HF at 0-4,
MISMATCH at position 5 (outer 11, HF 13). The unguarded control (control_unguarded.log, control_notf32.log with TF32 off) reproduces 1.77e-2 / 3.32e-1 bit-for-bit,
so the guard does not change numerics; cause (A10G / torch 2.14.1 / transformers 5.20.2 vs A100 run) not isolated.
static_scan.txt: 949 findings over experiments/vslice, 33 non-exempt HOT/MATMUL/HF: `codes @ amo` plus math.sqrt/log on config constants.

## Outer-is-real per layer (outer_layers0.json/.log, audit/outer_check_layers.py): FAIL
All 1006 layers are exact stock Qwen3_5DecoderLayer, no hooks/overrides, pristine library code, pip RECORD hashes ok (14 files). 85 layers FAIL the
"data-bearing parameter" rule (>64 distinct nonzero values): 57 NG (norm-gain Newton stage) layers with mlp.gate_proj/down_proj (2705/2704 distinct values)
and 28 attention layers (A) with self_attn.q_proj/k_proj. The A layers are the compiled inner q_norm/k_norm gains (inner weights, read from the checkpoint by build_attention);
the NG ones are compiled constants (hinge/Newton schedule). Fingerprint search of raw + gain-folded inner matrices in outer params: 0 hits (60,313 windows), but 1-D gains are not searched.
