# Quantfix step 2: the FINAL-contract CF+RAM stack (layers 2/26/27: 8-bit weights + 16-bit activations)

Everything is under NEW paths; `/data/cfram`, `/data/w785`, `/data/rowtab_cf.npy` were never written (qfr-data), nor `qfr-wfhf:/ckpt_cf`.

## Paths (volume qfr-data unless noted)
| what | path |
|---|---|
| firmware inputs (cfg.h with APLN/boosts, blob.bin, wrom.bin 80,396,288 words) | `/data/cfram_q/cfg/` |
| CF layout (cf_cfg.h with per-layer tables, cfetch.S, `rowtab_cf.npy` 168,684 x 3,840, cf_layout.json) | `/data/cfram_q/cf/` |
| firmware ELF (CF+RAM, EXT off, 8,440 B text, bss 30,284 B) | `/data/cfram_q/ramcf_q.elf` (md5 20928a5000874ae6a07b141a358dcc72) |
| reference trace (96 chunks 0..95, final state_0096/ramimg_0096) | `/data/cfram_q/refQ/` |
| row-embedding cache | `/data/cfram_q/rowemb_q.npy` |
| merged Qwen3.5-4B checkpoint (hidden 7680, vocab 680,128, table 2.59 GB bf16) | qfr-wfhf `/ckpt_cf_q` |

## What changed vs the 13-token stack
* `gen_fx_q.gen(..., wbits=4, ext=False, boost={(l,'proj'):8 for l in (2,26,27)}, aplmap={2:2,26:2,27:2})` (EXT OFF: the circuit firmware uses plain RV32IM nonlinears; the first gate had EXT=1, so it was re-run EXT-off: control 13, boosted **15846 rank 0**, cos 0.917, top1-top2 gap 0.965, `results/*_noext.json`).
* `fw_qwen3_ram_cf_q.c`: per-layer `apl_cur = APLN[l]`; 16-bit activations as two signed base-256 planes (hi plane = the existing XQB/xqi buffers, **lo plane written in place into the dead `xb` buffer**, so no new activation buffers); `mv_rows` loops over activation planes (`accs_[APL*PMAX][TILE]`, PMAX=2) and combines exactly like `qwen3_fx_q.c`; `quantize_gqa` has the same 16-bit branch. TWOPASS is disallowed with APL 2 (`#error`).
* RAM: the real 32 KB map overflowed by 336 B with TILE=256; **`-DTILE=128`** (accumulator tile 2 KB instead of 4 KB, `hut` 512 B) fits: bss 30,284 B, the reference run's `ram_max_ea` is 32,764 (< 32,768, no fault).
* `cf_gen_q.py`: per-layer weight-row strides (4,096 rows for 1-plane layers, 8,192 for layers 2/26/27), `CF_LBASE`/`CFW_SHIFT` as per-layer tables in `cf_cfg.h`, row id mapping `new = r + KH + layer*KL`. Backward-compat control: on the OLD cfg (`/data/w785`) it reproduces `/data/rowtab_cf.npy` bit for bit (md5 be83f3a9...), with identical layout fields and macros.
* Table: 156,396 -> **168,684 rows** (+12,288 = 3 layers x 4,096); well inside the 262,144-id ROW family of the reference token numbering.

## Results
| check | result |
|---|---|
| EXT-off compiler-level gate (rvemu, real firmware C) | control 13; 8-bit + 16-bit acts at 2/26/27 -> **15846** (rank 0) |
| reference trace (`cf/ref_chain_cf.py`, Modal CPU, 30 min) | **argmax 15846 (out_hex e63d0000)**, exit 0, no fault; 96 chunks; 199,287,158 trace tokens; 178,066,316 instructions; blob loads 0; **0 injections in all 96 chunks** (stats aggregate in `logs/refQ/`) |
| hfcheck, stock Qwen3_5 modules + merged checkpoint, chunk 0, 130,000 steps, program-only start | **0 mismatches** over full vocab 680,128 incl. 2,320 ROW + 23,817 VDOT tokens, 0 injections, min top1-top2 margin 22,528 |
| table+1 negative control | diverges at step 41 (F0(0) vs F0(17), right after WB, ROW(0)), 20 mismatches |
| replay smoke, final-path flags (real decode, --fulltable, table from the checkpoint, --no-host-writes), chunks 0,1,2 as 3 parallel single-chunk jobs on A100 (549 s each) | each: rc 0, 2,097,152 free tokens, **0 mismatches, 0 forced, 0 injections**; seams 0->1, 1->2, 2->3 **exact** (regs x1..x31, anchor, n, wb15, 8x8 latch, all 8,192 RAM words). The collector prints FAIL only because chunk 2 is not the last chunk (no HALT/final word in a 3-chunk subset): artifact of the subset, not of the run (`logs/smoke_par/`) |

## Cost / size deltas
instructions 138.5M -> 178.1M (+28.6%); trace tokens 157.1M -> 199.3M; chunks 75 -> **96**; table 2.40 GB -> 2.59 GB bf16 (hidden 7680). A full parallel replay is ~96 chunks x ~9.6 min = ~15.4 A100-hours (vs ~12 for 75 chunks).

## IMPORTANT: Modal workspace GPU cap
Modal reports `Your workspace's concurrency limits are reached (at 10 gpus)`: only 10 GPUs run at once workspace-wide (the tenpct/other projects share them). A 96-chunk parallel run therefore runs in waves of at most 10 (fewer while other projects hold GPUs): >= 96/10 x ~10 min = ~1.6 h wall time best case, not ~12 min. Raising the limit needs Modal.

## Not done / caveats
* One prompt token at position 0 (ctx 1), as before. 15846 is HF fp32's answer for this prompt (top-1 gap in the new firmware 0.965 vs 0.18 before); other prompts not tested.
* qwen3_fx_785_cf.c (plain non-RAM CF reference firmware) was NOT ported (not part of the final stack).
* The hfcheck covers chunk 0 only (130k of 199M steps), as before.
* hfwf `check_wf.py` runs with the REALMAP core patch (`in_drom := 1 - in_ram`), as in the previous CF check; the replay path uses REALDECODE (real decode).
