# Updating the launch material when results land

All launch visuals are generated from one status switchboard and the committed logs.

1. Edit `STATUS` at the top of `docs/launch/make_media.py` (values `'done'` or `'in progress'`), then run
   `python docs/launch/make_media.py` (PNG + GIF + MP4; the MP4 step takes about 2 minutes of CPU on a laptop, run it under `nice`).
   This refreshes: tier ladder, tier climb GIF, tier flow PNG and `.mmd`, hero GIF footer, results/caveat panels, both videos and square cuts.
2. Edit the hand-written places listed below. Only flip a row when a result file is committed; cite its path.
3. Add provenance rows in `docs/launch/PROVENANCE.md`. No row, no claim.

## Per result

### Free-running generation vs HF greedy
- `make_media.py`: `STATUS['free_running'] = 'done'`. Tier T3 real flips to achieved (together with `hostguard`).
- README: status board row "Free-running generation"; tier table row T3 real ("Pending: free-running generation"); top "Read this first" box ("not free-running generation"); "What is verified today" row 1 and "Known open items" ("free-running generation"); "Status by tier (detail)" T3 real row.
- Add the real numbers (tokens generated, match count, time) to the README results and, if good, a new row in `demo_table`/`results_table` scenes; update `res_*` source in PROVENANCE section 2.
- `TWEET.md`: post 1 ("Teacher-forced, not free-running yet"), post 6 caveat (1). Rewrite post 5 with the generated continuation.
- If it does NOT match HF, record the first diverging position instead; do not flip the status.

### HostGuard audit of the real driver
- `make_media.py`: `STATUS['hostguard'] = 'done'` (only if the audit passes; otherwise leave and document findings).
- README: status board row; tier table T3 real; "Read this first"; "How to audit" last paragraph ("The chained toy driver ... has not been run under HostGuard"); "Known open items"; "What is verified today" row 1 ("HostGuard audit pending").
- `TWEET.md`: post 6 caveat (2). `audit/AUDIT.md` link goes in the status row.

### Speedup work
- `make_media.py`: `STATUS['speedup'] = 'done'`. Update `CIRCUIT_S`-style numbers only from new logs: add the new per-token time as a separate bar in `sc_cost` (do NOT overwrite the 268.6 s baseline; label the executor variant and the log path).
- README: status board row; tier T4 row; hero table "per token" cell; cost model section ("Not yet measured: tuned kernels"); `Status by tier (detail)` T4.
- `TWEET.md`: main post (268.6 s), post 5, post 6 caveat (4). Update `square()` footer in `make_media.py` ("one token: 268.6 s").

### Tier changes in general
- Tier table rows live in `tier_rows()` in `make_media.py` and in the README tier table; keep the two in sync. The mermaid block in the README must be re-pasted from `docs/assets/flow_tiers.mmd` after re-running.

### Circuit-route tier measured (Qwenputer-style CPU in a real Qwen3.5)
Source of truth: `docs/TIERS.md`; a row flips only from a committed `experiments/circuit/integrate/results/*.json` with `measured: true`.
- README: status board row "Circuit route"; "Status by tier (detail)" circuit-route paragraph; headline box only if a full Qwen3-0.6B token ran.
- `make_media.py`: add the new tier as its own rung/bar (do not reuse the compiled-route T-names; label it "circuit T<n>"); regenerate tier ladder and climb GIF, add the measured s/token as a separate bar in `sc_cost`.
- `TWEET.md`: add one line with tier, model size actually run (tiny/0.6B), trace tokens, s/token; keep "projected" wording for unmeasured tiers.
- `PROVENANCE.md`: row with the result JSON path.

Current state (2026-11-06): the only measured circuit-route run is the tiny-model fused-executor run (eb4f4f3, `docs/TIERS.md` tiny-model table). Do not feature it as a tier-ladder rung; if media mention it, label it "toy model, 16 tokens". Qwen3-0.6B instruction counts are measured but s/token is computed from assumed rates and must be drawn hatched/labelled "projected". Tier media must state context length (ctx 1 vs 128 vs 1024).

Circuit route outer model: the Qwenputer core requires 2 value heads per key head, so the base is Qwen3.5-4B, not 0.8B; any media saying "0.8B/4B" for the circuit route must say 4B. The combined 5.5-7 h/token (ctx 1) figure in `docs/TIERS.md` is PROJECTED (matvec token costs measured on the reference machine, nonlinear and attention parts from separate kernels, no merged build); show it hatched, never as a result bar. Weight memory (ROM in appended layers vs host streaming) is an open ruling and changes what "weights stay in the model" means in the media.
