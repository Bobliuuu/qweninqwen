# trueqiq_audit — honesty & verification harness (agent F)

Claim being protected: the FULL real inner Qwen runs INSIDE a stock `Qwen3_5ForCausalLM` outer model. The host may only stream
weights / tokens / embeddings / addresses / flags / position ids, and do outer argmax + append. No host matmul, attention, norm,
activation, softmax, RoPE, residual add or sampling of the inner model (INTERFACE.md s2).

Run everything with the repo venv: `python` (torch 2.15, transformers 5.22.1).

| tool | file | what it proves |
|---|---|---|
| HostGuard | `hostguard.py`, `run_audit.py` | dispatch-level (every ATen op) classification host vs inside-outer-forward; taint tracking of outer activations; readback/feedback/numpy/math guards; per-run report (ops, bytes, rounds, exempt scopes) |
| token-independence | `run_audit.py --variant N` + `compare_runs.py` | the streamed inputs are bit-identical when host RNG changes => no activation data is injected by the host |
| outer-is-real | `outer_check.py` | exact class, no monkeypatch (code objects vs a pristine re-exec of the installed source, pip RECORD hashes), no hooks that modify, params inventory, inner weights NOT baked (scale/transposition-invariant fingerprint search) |
| reference kit | `compare.py` | per-op / per-layer / logits table vs HF inner with tolerances, first-divergence localisation |
| static scan | `static_scan.py` | AST scan for `@`, HOT math calls, HF-inner use, and `with reference/transport/...` scopes |
| tests | `test_hostguard.py` (22 pos/neg), `test_outer_check.py` (13), `python compare.py` (selftest) | the harness itself catches cheats |

## 1. Auditing any agent's driver (5 commands)
```bash
cd <driver dir>      # so its relative imports work
A=trueqiq_audit; PY=python
# 1 dynamic guard (log mode lists everything; use --mode raise to stop at first violation; --fast = no per-op stack walk, for GPU/big runs)
$PY $A/run_audit.py --mode log --check-outer --out audit0.json  driver.py [driver args]
# 2 token-independence (two runs, different host randomness; compare streams)
$PY $A/run_audit.py --variant 1 --out audit1.json driver.py [args];  $PY $A/compare_runs.py audit0.json audit1.json
# 3 static scan
$PY $A/static_scan.py -v .
```
Exit code 0 only if: no violations, an outer model was actually run, outer checks pass.

### What a driver must declare (3 context managers, no-ops when no guard is active)
```python
from hostguard import reference, weight_prep, transport, validation_read, build   # sys.path.insert(0, '.../trueqiq_audit')
with reference("HF inner / fp64 matmul for error metric"):  ...   # ANY op allowed; counted + listed (reviewer reads it)
with weight_prep("silu-inverse + norm precompensation of weights"): ...  # token-independent math on INNER WEIGHTS only; may not touch outer outputs
with transport("pad/tile/quantise"): ...                         # gray elementwise ops (add/mul/where...) for reformatting only
with validation_read("read Y slot for the error table"): ...     # .numpy()/.tolist() of outer tensors; never fed back
with build("compile norm gains / selectors"): ...                # outer-parameter compilation (constants)
```
You may also declare scopes WITHOUT editing the code under audit: `--scope weight_prep:engine.py:write_frames` (kind:file:func; repeatable).
Declared scopes appear in the report as `EXEMPT ... reviewer must read`: they are the honest list of "places the host computes".

### What HostGuard classifies (see hostguard.py FORBIDDEN / GRAY / MOVE)
* outside the outer forward: FORBIDDEN (mm, softmax, exp, rsqrt, silu, pow, mean/sum, sdpa, conv, topk, random ...) -> violation;
  GRAY (add/sub/mul/div/where/compare...) -> violation unless `transport()`; MOVE (view/index/cat/copy/to/embedding/zeros...) allowed;
  unknown ops -> violation; argmax only on tensors derived from outer logits (its output is then "clean": token id).
* inside the outer forward (exact class `Qwen3_5ForCausalLM`, or its stock `.model` TextModel): everything counts as outer ops, but a
  math op whose python stack has a non-library frame (a forward hook that computes) is a violation.
* taint: every tensor produced inside the outer forward is tainted; taint follows move ops and in-place writes (incl. via views).
  Host math on a tainted tensor, `.numpy()/.tolist()/.item()` of non-logits, or streaming a tainted tensor into the next outer
  forward (`outer-output-fed-back`) are violations. Argmax-of-logits token ids are the only sanctioned way back in.
* numpy/math compute functions called from user code, a second (non-outer) HF model forward on the host, stale-depth after exceptions.

### Known holes (be honest about them in reports)
1. `ndarray @ ndarray`, C extensions, `torch.compile`d blobs, and python scalar arithmetic are invisible to dispatch -> `static_scan.py`.
2. Clean tensors have no provenance: the guard cannot tell "weights from safetensors" from "host-computed activations written into a
   frame". The black-box answer is `compare_runs.py` (stream must not depend on host RNG) plus reading every declared scope.
   A deterministic host-side pre-computation of activations from fixed inputs would pass compare_runs: review `weight_prep` scopes.
3. `reference()` scope allows everything and is not provenance-checked: keep it to metric/reference code (static_scan lists it).
4. Digest check covers inputs_embeds/ids/position_ids/cache_position only (not past_key_values contents).

## 2. Tier report requirement
Every T1/T2/T3 report MUST include the filled `AUDIT_TEMPLATE.md` (HostGuard PASS + declared scopes, token-independence result,
outer-is-real result, reference table, stream accounting, and "what is host-injected"). Reports without it are "unaudited".
Tier definitions used by the template: T1 = single op/matvec exactness (teacher-forced inputs allowed, must say so);
T2 = full layer chained inside the model (no host activations between ops); T3 = full inner model, tokens out via outer argmax.

## 3. Reference kit
```python
from compare import hf_trace, compare_traces, table, all_pass, Tol
with reference(): ref = hf_trace(hf_inner_fp32, input_ids)        # canonical names: embed, L{i}.attn_norm|q|k|v|attn_out|resid_mid|mlp_norm|gate|up|down|out, final_norm, logits
cand = {...}                                                       # same names, read from outer slots (validation_read)
rows = compare_traces(ref, cand, Tol(atol=1e-4, rtol=1e-3, per={'logits': (1e-2, 1e-3), r'L\d+\.attn_out': (1e-3, 1e-3)}))
print(table(rows))     # max_abs, rel_l2, scale_err (max_abs/max|ref|), cosine, PASS/FAIL, argmax agreement & top-5 for logits, FIRST DIVERGENCE
```
## 4. Outer-is-real
```python
from outer_check import check_outer
r = check_outer(model, inner='/path/model.safetensors' or hf_model or state_dict,
                declared_constants={r'model\.layers\.\d+\.input_layernorm\.weight': 'transparent norm gain B/sqrt(D)-1'},
                allow_swaps={'torch_chunk_gated_delta_rule': 'gdn_impl.set_impl torch reference kernel'})
print(r['text']); assert r['PASS']
```
FAIL conditions: subclass / foreign module class; instance `forward` override or non-observer hook; class/module-level function whose
code differs from a pristine re-exec of the installed modeling source; installed transformers files differ from pip RECORD;
non-plain Parameters; any parameter with > `distinct_thresh` (64) distinct nonzero values that is not declared (data-bearing);
fragments of the inner weights found in any outer parameter (8-wide windows, scale- and transpose-invariant).
