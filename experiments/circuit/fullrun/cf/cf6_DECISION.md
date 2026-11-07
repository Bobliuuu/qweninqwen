# 6-bit table packing with CF+RAM: not done (decision 2026-11-06)

**Result: not combined, no GPU run.** 6-bit removes no host role; it only shrinks the table, and the user ruled the 2.38 GB bf16 table (hidden 7680, 4-bit payload 0.73 GB) fine.

What combining would need (all in code that CF+RAM hard-codes to 480-word rows / 4-bit nibbles):
1. `cf_gen.py`: `RL = 480` -> 640 (weight rows per layer LSTR 4096 -> 3072; `LM_WB // RL == NL * LSTR` assert changes); `put()` writes constants as 8 four-bit nibbles per word (+0x88888888 bias) and would need the inverse of `wf_table.build_table(bits=6)` (6 entries/word, 6th carries 2 bits).
2. Firmware sources `fw_qwen3_ram_cf.c` / `qwen3_fx_785_cf.c`: `g % 480`, `g / 480` -> 640; constants-fetch kernel `cfetch.S` regenerated; vdk with `WF_ROW_WORDS=640` / `-DROWW=640`.
3. New reference trace (`ref_big_cf.py` on lambda-final, the WF one took 1962 s wall) and new `rowtab6_cf.npy`; replay chunks 0-2, then re-verify (negative control) before use.
4. 6-bit unpack needs 13+ chip layers vs 12 (HOST_ROLE_AUDIT): deeper outer model, and the in-checkpoint check (hfwf) was only done for 4-bit.

Existing evidence for 6-bit alone (no CF, no RAM): WF firmware chunks 0-47, 0 free mismatches, negative control diverges (logs/wf6, commit 7f4c3e1).

Revisit only if table size becomes binding. Final run should use the 4-bit CF+RAM firmware (`ramcf.elf`, `refRC`, `rowtab_cf.npy`).
