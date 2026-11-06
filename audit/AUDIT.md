# AUDIT.md — read-only audit of agents' code (agent F)   snapshot 2026-11-05 ~23:00
Method: HostGuard dynamic runs on Mac CPU where feasible (trueqiq_state/test_tiled.py, trueqiq_alt/t_glue.py), `compare_runs.py`
token-independence, `static_scan.py` on trueqiq_state / trueqiq_run / trueqiq_ops / trueqiq_alt, and line-by-line reading of
engine.py, tiled.py, stream.py, run_token.py, run_proj.py, ops.py, mlpmul.py, glue.py, machine.py. GPU-only drivers (run_token.py,
run_proj.py, bench_*) were read, not executed.  Severity: HIGH = invalidates an end-to-end claim if presented as such;
MED = must be disclosed in the tier report; LOW/INFO = note.

## Verdict
Nothing found that violates the rule INSIDE what is claimed today (T1 matvec / op exactness): no host matmul of activations,
no host softmax/norm/RoPE on activations, no baked inner weights in outer parameters (the compilers write only 0/+-1/scale
selectors and constants). BUT no driver yet demonstrates a T2/T3 chain: every activation entering the machine is placed in a
frame by the host (H1), so current results are strictly T1/teacher-forced and must be labelled so.

## HIGH
**H1. Host injects activations in every current driver (token-independence FAILS, measured).**
`python run_audit.py --variant 0/1 test_tiled.py` + `compare_runs.py`: 16 of 1024 streamed rows differ (the READ frames). Places:
- trueqiq_state/engine.py `read_frames(X)` (docstring "TEST HARNESS") -> tiled.py:`project` (X slot of READ frames), run_block.py:`matvec_pass`, test_tiled/test_down/test_accuracy.
- trueqiq_run/run_token.py (`Xp[:, :k] = X`, X = `torch.randn(1,k)*3`, "harness-fed random ... NOT yet a chained token"), run_proj.py (same), stream.py `FrameAssembler.read(X)`.
- trueqiq_alt/glue.py `run`: `FA[:, S['X']] = x.reshape(nt, C)` (round A); mlpmul.py `frames`: `F[:, X] = x.expand(T, -1)` — the host BROADCASTS x into every weight step; the 0.6-1.1 s/token and ~1e9 MAC/s figure (FINDINGS F1) assumes this host copy is free; glue.py is the first attempt to replace it by in-model attention (x tile selected by code match) but x itself is still host-written once.
- trueqiq_ops/test_*.py: inputs are HF-inner activations/logits written by the host (test_argmax.py: `frame[0,:V,K.start] = lg` where lg = HF inner logits; test_attn.py: q/k/v captured from HF hooks; test_norm_swiglu.py: HF hidden_states). Legit op tests, but teacher-forced.
Required statement in reports: "T1, teacher-forced; activations in frames are host/HF-supplied". For T2/T3 compare_runs.py must say TOKEN-INDEPENDENT.

**H2. BALLAST / norm-transparency constants are inconsistent across agents (integration blocker, accuracy-relevant).**
machine.py BALLAST=1e4 (INTERFACE: "1e4"); engine.py README requires B>=1e6 (uses 1e6 in drivers); ops.py uses 2^27 and shows 1e4 gives 3e-5..5e-4 error;
glue.py/mlpmul.py use 1e6. All share slot 0 of one Layout: a single value must be chosen; outer-norm error |x|^2/(2B^2) is activation-dependent (this is a real
approximation, not a host trick, but it means "exact" claims hold only for |x| << B). fp32 + B=2^27 also makes payload precompensation (stream.py `g` loop) differ.

## MED (must be disclosed; none is a rule violation if token/activation-independent)
**M1. Weight-only nonlinear host math ("silu pre-inversion").** engine.py `silu64`/`silu_inv` (60-iter fp64 bisection, `torch.exp`, `where`), `write_frames` shift `s=(-0.27-c.min).clamp`, and stream.py `build_matrix_payload` (same math vectorised). Streamed payload != raw checkpoint values (silu^-1(w+s)). Allowed by INTERFACE s2 only as weight reformatting: it depends on weights only (verified: no activation touches it; HostGuard run with `weight_prep` scope shows no outer-data taint). Report must say "weights streamed after silu-inversion", not "raw weights". Contrast: trueqiq_alt streams RAW weights (odd-silu trick) — strictly cleaner for the claim.
**M2. Host emulation of the OUTER RMSNorm to precompensate weight frames.** engine.py `_norm_comp` (fixed point, 30 iters, `sqrt`) and stream.py:41-43. Weight/flag-only frames, so token independent; the READ frames are NOT compensated (honest residual error |x|^2/2B^2). Disclose as `weight_prep`.
**M3. Two implementations of the payload builder.** stream.py:`build_matrix_payload` (used by run_token/run_proj on GPU) vs engine.py:`write_frames` (used by test_tiled/tiled). I verified bit equality on CPU/fp64 (nL=2,H=16, 130x1500 incl. a negative-shift row): max abs diff 0.0. Re-run on GPU/fp32 before quoting run_token numbers (stream.py uses `out_dtype=float32`, torch.sqrt on GPU).
**M4. Gain folding = host multiplication weight x weight.** ops/README: "gain g folded into consuming matvec weights W' = W diag(g)" (also Qwen2.5 biases placed by host in destination slot; embedding row E[tok] transported). Weight-only and allowed, but they are host arithmetic on inner weights: list them under weight_prep in the T3 report.
**M5. Validation reads hide behind forward hooks.** run_token/run_proj/tiled register `layers[-1].register_forward_hook(lambda ...: cap['o'] = o)` and read `out[...][spec.Y].double()`; ops tests use hooks on HF (reference) and `output_hidden_states`. Observer-only (my outer_check classifies them as warn), and the values are used for error tables only. In a chained run the same read-back must not be fed to the next round (HostGuard `outer-output-fed-back` enforces it).
**M6. Driver bypasses the CausalLM wrapper.** `model.model(inputs_embeds=...)` (tiled.py, run_token.py, run_proj.py, test_attn.py): the stock `Qwen3_5TextModel` is called, `lm_head` (vocab=8 zeros) and the outer logits path are never used => no outer argmax/append loop is exercised anywhere yet except ops/test_argmax.py (4 positions, Mac CPU, logits host-injected, see H1). HostGuard accepts both entry classes; outer_check resolves the wrapper via gc.
**M7. Module-function rebinding.** trueqiq_state/gdn_impl.py `set_impl` rebinds `modeling_qwen3_5.torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` (to the unwrapped function or back); stream.py/run_token.py `NOFLA` sets `sys.modules['fla']=None`. Still stock code, but report which kernel ran (torch reference vs fla triton) and run outer_check with `allow_swaps` so the pristine-code comparison is explicit.
**M8. `make_outer2` (ops.py) builds its own config** (`rope_parameters`, `partial_rotary_factor=(rot+0.5)/hd`, `mrope_section`, `linear_*` head dims 1x8, `mtp_num_hidden_layers=0`) separate from machine.make_outer. Config-only, legal, but integration must adopt ONE constructor; two outers with different RoPE config cannot share a model.
**M9. Compile-time code that runs host math on constants** (not inner data): ops.py `rms_scale`, `transparent`, `compile_attention` (bf16 grid search `rnd`), engine.py `compile` (`math.sqrt(D)`, `G=Bg*Rq/(sqrt(dv)*SILU_MIN)`), glue.py `Matvec.__init__` (hundreds of index/arith ops), engine.py import-time `SILU_MIN` (module level constant from torch.exp). These write only structure/constants: `outer_check` shows 0 data-bearing params in B's compiled bank (nonzeros are 0/+-1/KB/Rq/Bg style constants with <=64 distinct values) — re-run with `inner=` once the final compiled model exists.

## LOW / INFO
- L1. machine.py `make_outer` has a dead `... if False else dict(...)` expression (harmless, but confusing; `attn_output_gate` etc. never applied).
- L2. Hash/provenance: run_token/run_proj/stream.py insert `~/qiq/run` and `~/work/run` ahead of repo paths; the box may run different copies than the repo. `run_audit.py` writes `source_sha256` of every non-library module imported — put it in the tier report.
- L3. `gpu_bench.py`/`bench_*` measure steps/s with synthetic frames: valid for throughput only; they carry no correctness claim.
- L4. test accuracy numbers (1e-6..1e-7) are vs fp64/torch fp32 matmul on the SAME host-supplied X; they measure the machine's arithmetic, not an inner-model result.
- L5. argmax head (ops.py `compile_argmax_head`): needs V+1 = 151,937 outer positions per inner token with the logit of token j in the K slot of position j; those logits must come from in-model matvec (B's lm_head tiles) — currently host-injected (H1). Cost accounting for T3 must include these steps.
- L6. HostGuard blind spots to keep in mind when reading new code: `ndarray @ ndarray`, C extensions, and any pre-computation of activations from FIXED inputs that does not depend on RNG (would pass token-independence; read all `weight_prep`/`reference` scopes for tensors that are not weights).

## Recommended audit command lines (no code edits needed)
```bash
A=trueqiq_audit; PY=python
# B (CPU, ~4 s)
cd trueqiq_state && $PY $A/run_audit.py --mode log --check-outer \
  --scope weight_prep:engine.py:write_frames --scope weight_prep:engine.py:init_frames --scope weight_prep:engine.py:_norm_comp \
  --scope weight_prep:engine.py:tile_weights --scope build:engine.py:compile --scope build:engine.py:__init__ \
  --scope transport:engine.py:read_frames --scope transport:engine.py:new_frames --scope reference:test_tiled.py:'<module>' test_tiled.py
# B2 (CPU)
cd ../trueqiq_alt && $PY $A/run_audit.py --mode log --check-outer --scope build:glue.py:__init__ --scope transport:glue.py:code \
  --scope transport:glue.py:fwd --scope transport:glue.py:frames_B --scope reference:t_glue.py:'<module>' t_glue.py 16 8 2 2 cpu
```
Measured now: B test_tiled with those scopes -> HostGuard PASS + outer check PASS (32.9M params, 8/17 tensors nonzero, 0 data-bearing) (only scopes listed; 2 rounds, 1025 steps, 13.1 MB streamed) but token-independence FAIL (H1).
B2 t_glue (C=16,O=8,R=2,nt=2) with scopes -> HostGuard PASS, outer check PASS, 3 rounds, 11 steps, 760 outer ATen ops, 0 non-logits readbacks; x host-injected (H1).
