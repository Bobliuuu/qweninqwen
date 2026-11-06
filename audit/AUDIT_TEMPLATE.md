# Audit block (copy into every tier report; fill every line; "n/a" must be justified)

Tier: T1 / T2 / T3      Agent:      Date:      Commit/hashes: (report `source_sha256` from audit0.json)
Inner model + dtype:                  Outer: Qwen3_5ForCausalLM  D=   layers=   (types)   params=   dtype=
Hardware (Mac CPU / A100):

## 1 HostGuard  (`run_audit.py --mode log --check-outer --out audit0.json driver.py`)
- Result: PASS / FAIL   violations: N (kinds: ...)
- Outer rounds (forward calls): ...   outer steps (positions): ...   bytes streamed: ... (inputs_embeds ..., ids ..., position_ids ...)
- ATen ops inside outer forward: ...    host ops (allowed data movement): ...    host readbacks of non-logits: ...
- Declared exempt scopes (paste from report, each with a one-line justification):
  | kind | where | what it computes | token/activation independent? |
  |---|---|---|---|
- Outer argmax used for: (tokens emitted / none)    Host-appended tokens: ...

## 2 Token-independence (`--variant 1` + `compare_runs.py`)
- Streamed rows compared: ...   differing: ...   => STREAM TOKEN-INDEPENDENT / DEPENDS ON HOST DATA
- If differing: list exactly which inputs are host-injected (activations X, logits, ...). Such a result is T1-teacher-forced only.

## 3 Outer is real (`outer_check.check_outer(model, inner=...)`)
- PASS / FAIL;  transformers version/hash-verified files: ...;  pristine-code-diff: none / list (declared swaps)
- Params: nonzero tensors N/M;  data-bearing (declared, with reason): ...;  inner fingerprint hits: none / list
- Hooks present: none / observer-only (list)

## 4 Reference comparison (`compare.py` table, paste)
- Ops compared: ...   tolerances: ...   worst scale_err: ... at ...   first divergence: none / ...
- Logits argmax agreement: .../...   top-5 overlap: ...   tokens generated vs HF greedy: identical for N tokens

## 5 Honest statement
- What ran inside the outer: ...
- What the host computed (every item from section 1 scopes + static_scan findings): ...
- What is NOT yet done / scaled down / teacher-forced: ...
- Throughput numbers: per outer step ... s; per inner token ... s; extrapolation basis (measured vs assumed): ...
