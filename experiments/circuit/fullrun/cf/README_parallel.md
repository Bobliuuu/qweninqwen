# Parallel final-path run (CF+RAM, any firmware)

One A100 job per chunk, each started from the reference start-state of its chunk (chunk 0: the program alone), each saving the **circuit's own end state**. `collect_seams.py` then checks that job k's end state
**exactly equals** the seed job k+1 starts from. All 74 seams exact + every job clean (0 free mismatches, 0 forced tokens, 0 injections, no host write path in the graph) => by induction equal to one unbroken run
(deterministic circuit: identical state in => identical tokens out), provided state not in the seed (`bn`) does not influence later tokens (reported per seam, `bn_bad_informational`).

Flags every job uses (final path): `REALMAP=1 REALDECODE=1` (real in_ram/in_drom decode, no bring-up patch), `--chain --save-state`, `--fulltable` (whole row table, no reference window), `--rowtab-ckpt` (row table read from the
merged checkpoint's `embed_tokens.weight`), `--stop-on-halt` (the circuit's own HALT), `--no-host-writes` (asserts 0 injections / 0 forced tokens and removes the injection hook and token override from the CUDA graph),
`--ramimg` (RAM seed), `--reset0` on chunk 0 (boot from the program alone). The last chunk's result JSON has `readout`: the circuit's own BYTE tokens before HALT decoded to the final word.

## Commands (every path is a parameter)
```
CFRAM_MAX_CONTAINERS=25 modal deploy cf/modal_cfram_par.py
python cf/spawn_cfram_parallel.py --build-cache --root R --elf E --refdir D --ckpt C --rowtab T --rowemb EMB     # ONCE per table (~20 min, 1 A100); jobs fail fast without a matching cache
python cf/spawn_cfram_parallel.py --n N --root R --elf E --refdir D --ckpt C --rowtab T --rowemb EMB             # detached; survives the laptop
python cf/collect_seams.py --n N --root R --refvol-dir <D relative to qfr-data> --expect-word W                  # ONE verdict
modal app stop -y <app id>                                                                                       # when done
```
Paths: `E D T EMB` on volume qfr-data (`/data/...`), `C` on volume qfr-wfhf (`/wf/...`; `--ckpt ''` reads the table from `T` instead), outputs on qfr-out `/R/chunk_NNNN`.

Current CF+RAM firmware: `--n 75 --root cfram_par --elf /data/cfram/ramcf.elf --refdir /data/cfram/refRC --ckpt /wf/ckpt_cf --rowtab /data/rowtab_cf.npy --rowemb /data/cfram/rowemb_cf.npy`, collect `--refvol-dir cfram/refRC --expect-word 13`.
New firmware (quantfix), e.g.: `--n <N> --root cfram_q_par --elf /data/cfram_q/<fw>.elf --refdir /data/cfram_q/<ref> --ckpt /wf/<ckpt> --rowtab /data/<table>.npy --rowemb /data/cfram_q/rowemb_q.npy`, collect `--refvol-dir cfram_q/<ref> --expect-word 15846`.

The reference dir must hold, for k=0..N: `tok_k kind_k state_k ramimg_k inj_k` (and `state_N`, `ramimg_N` = final). Chained alternative (serial, one GPU, ~9.3 min/chunk): `modal run --detach cf/modal_cfram_chain.py --chunks 0-<N-1> --reset0 --elf ... --refdir ... --ckpt ... --rowtab ... --rowemb ...`.
Time: ~10 min per chunk + ~1-2 min startup; with all N chunks in parallel the wall time is one chunk (~12 min) if Modal grants N A100s, else N/concurrency waves. GPU-hours ~ N x 0.17.
