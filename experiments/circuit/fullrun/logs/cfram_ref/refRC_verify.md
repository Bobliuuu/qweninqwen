# refRC (CF+RAM reference) completeness check, 2026-11-06

Source: `modal volume get qfr-data cfram/refRC` (452 files, 763 MB), CPU check only, no GPU.

- Chunks present: `state/ramimg` 0..75, `tok/kind/inj/stats` 0..74. State 75 is the final state (halted=True).
  Chunks **64 and 72 are present with all of tok/state/kind/inj/ramimg/stats**; only `stats_0063` is absent (cosmetic).
  (A report that 64 and 72 were missing did not reproduce; I cannot say whether they were synced after that listing, or the listing was truncated.)
- True chunk count: **75 (indices 0-74)**, `summary.json`: chunks 75, trace_tokens 157,076,451, instructions_machine 138,513,317, argmax 13, out_hex 0d000000, exit_code 0, fault null.
- Sum of len(tok_k) over 0..74 = 157,076,451 = summary trace_tokens. `state_0075.steps` = 138,513,317 = summary instructions_machine.
- Continuity: `tok_start[k] + len(tok_k) == tok_start[k+1]` for every k = 1..74 (exact). k=0 differs by 1 because `tok_start[0]=1` (the GO token precedes chunk 0).
- All `inj_*.npy` (0..74) are empty; `kind` length == `tok` length for every chunk; every `ramimg_k` has shape (8192,).
- Chunk 35 has 2,097,159 tokens (record boundary overshoot), others 2,097,152, last chunk 1,887,188.
- The pickle field `n` is an intra-record counter, not an instruction count (not monotone); not used for the check.
- lambda-final `~/cfram/refRC` and `~/fr2/refRC` are empty; `~/ramown/refRC` (339 files) is a different reference. lambda-final disk is 100% full (4.5 GB free) and its GPU was at 100% utilisation at check time, so no regeneration was attempted (not needed).
