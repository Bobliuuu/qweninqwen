# audit variant 1 (host RNG offset 7919) - real 28-layer Qwen3-0.6B driver, box gpuA (A10G)

Driver: `experiments/vslice/test_vs2.py real151936_L28 1` (6 positions of "The capital of France is", teacher-forced prompt then 1 generated token).
Guard: `audit/run_audit.py --variant 1 --mode log --fast --entry layer` (the driver is a layer-major executor over stock `Qwen3_5DecoderLayer`, so `--entry layer`).
Files: `audit1.json/.log` (report), `res_vs2_real151936_L28.variant1.json` (driver result), `control_demo_*` (positive/negative control), `cmd.txt`, `env.txt`, `run_variant.sh`, `nofla_launch.py`.

## HostGuard result: FAIL (same 26 violations as variant 0, all in the host decode of the 36 rails)
| count | kind | site (test_vs2.py in `run`) | what it is |
|---|---|---|---|
| 6 | host-math + host-math-on-outer-output | `mv` line 83 | `lg = codes @ amo`: host matrix-vector product of the 36 argmax "rails" read from the outer residual stream with a 36 x 151,936 code table |
| 6 | host-math | `max` line 85 | `lg.max()` on that result (display of the top code score) |
| 12 + 2 | argmax-on-non-logits | lines 88, 86 | host `argmax` of the decoded `lg` (token id appended/fed back, and the printed outer argmax) |
These are exactly the "host evaluates the outer lm_head through its 36 non-zero columns" that FULL_TOKEN_RESULT.md discloses; the guard counts it as host math on outer output.
So the claim "outer argmax" is NOT guard-clean: block-level maxima and the 36 rails come out of the outer model, but the last decode + argmax is on the host.
No other violation: 0 host readbacks of non-logits, no forbidden/gray op outside declared scopes, no outer-output fed back by host math, no foreign code inside outer forwards (stack check is off with `--fast`, see below).
Not run: `--check-outer` (no `Qwen3_5ForCausalLM` is ever called by this driver, it calls stock decoder layers directly, so "no Qwen3_5ForCausalLM forward observed" and the exit code is 1 regardless of violations). The outer-is-real check for this driver is therefore NOT covered by this run.
`--fast` disables the per-op python stack walk (foreign-code-in-forward and stale-depth detection are off).

Stream accounting (variant 1): 17,444 outer layer calls, 27,590,982 frame positions, 4.50e12 bytes of frames streamed (4.50e12 in arg '0', 3.3e8 in arg '1', plus position_ids/cache_position/attention_mask), 9,641,017 ATen ops inside outer forwards, 0 host readbacks.

## Declared (exempt) scopes - reviewer must read these
CLI (`cmd.txt`): weight_prep `vs2.py:tile_payload` (4,843,620 ops: silu pre-inversion, fp64 pad/transposition of inner weights), weight_prep `test_vs2.py:weights` (57 ops: gains folded into W), build `vs2.py:compile_entry` (962,815 ops: writes the constant outer structure), build `vs2.py:build_layout`/`make_plan` (0 ops), build `engine.py:<module>` (4 ops, import-time SILU_MIN), transport `vs2.py:dense_job` (13,644 ops: builds weight frames).
In-code (committed in 8b6cf3e): reference (load HF Qwen3-0.6B; HF reference forward for the error metric; block-max vs HF metric), build (stock Qwen3_5DecoderLayer / rotary construction), transport (causal mask from indices, rail codebook from vocab ids), validation_read (print).
`weight_prep`/`build` are checked by the guard to never touch outer-derived tensors (none did). `reference` scopes are not provenance-checked (README known hole 3).

## Positive control
`control_demo_cheat.log`: demo_cheat_driver.py -> 1 violation (host `bmm`), AUDIT RESULT FAIL. `control_demo_good.log`: PASS. (CPU, fla off.)

## Token-independence: see COMPARE section appended below / in the run report.

## Token-independence / compare (status at hand-off)
`compare_runs.py audit0.json audit1.json` (gpuB variant 0 vs gpuA variant 1) printed `rounds: 0 vs 0 ... STREAM TOKEN-INDEPENDENT`. THIS VERDICT IS VACUOUS: `hostguard.py` records stream digests only when `entry != "layer"`, and this driver runs under `--entry layer`, so 0 rows were compared. No black-box token-independence evidence comes from compare_runs for this driver. (Output: `compare_audit0_vs_audit1.txt`.)
Indirect evidence: a variant-0 run of the same command on gpuA (same box, same scopes) gave step-0 maxrel 1.0397381993243471e-04, bit-identical to variant 1 step 0, i.e. host RNG offset does not change the outer computation on one box. (Full gpuA variant-0 result pending; the run was still going at hand-off, files will be under `audit/results/variant0_gpuA/`.)
UNEXPECTED: the committed gpuB variant-0 result has maxrel 1.8e-2 .. 3.3e-1 and a wrong last-position argmax (11 vs HF 13), while gpuA gives 1.0e-4 .. 3.2e-4 and argmax == HF at all 6 positions (A100 original: 7e-6 .. 5.8e-5). Cause not isolated (different physical box / environment; not host RNG, per the gpuA variant-0 step-0 match).
