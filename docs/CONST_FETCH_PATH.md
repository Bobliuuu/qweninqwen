# Constants-fetch path: the ~625 MB read-only blob as table rows of the outer model (no host injection, no DROM table)

Tags: **[V]** verified (log cited), **[M]** measured, **[H]** still host-assisted. Written to be read by a strict reviewer.
Context: `HOST_ROLE_AUDIT.md` sec. 5.4 and 7, `WEIGHT_FETCH_PATH.md`, `RAM_FIT.md`. Code: `experiments/circuit/fullrun/cf/`.

## 0. Result in one paragraph

Every constant the Qwen3-0.6B firmware used to read from the 624.6 MB blob at 0x10000000 (embedding row of the inner token, RMSNorm gains, per-row matvec scales, rope row) is now stored as extra ROWS of the same embedding-row table that already holds the int4 weights, and is fetched by the firmware with the same `WROW` instruction (ROW token, embedding row r, DeltaNet latch). No new circuit logic: a 32-bit constant lives in one 8-cell VDOT word and is read back with four ordinary VDOTs whose int8 activations are (1,16) on one cell pair, so the circuit's own VDOT multiplier returns the constant exactly. Blob/DROM-class injections: **0** (was 6-9 k words per 2.1 M-step chunk on the remapped firmware, 0.5-0.7 M loads per token in total). The table grows from 155,205 to 156,396 rows (+1,191 rows, +9.8 MB at hidden 4096 bf16 versus 1.27 GB for the weights).

## 1. What was read from the blob (measured, `ramown/ADDR_MAP.md` sec 3, firmware source)

| Item | Words | Used by |
|---|---|---|
| embedding row of token 785 | 1,024 | `x[i] = emb[i]` |
| RMSNorm gains: 2 per layer (1,024) + q/k norm (128 each) + final | 28 x 2,304 + 1,024 | `rms()` |
| per-row matvec scales (q 2048, k 1024, v 1024, o 1024, gate 3072, up 3072, down 1024 per layer) + LM head | 28 x 12,288 + 151,936 | matvec epilogue |
| rope row, position 0 | 64 | `rope()` |
| total distinct 32-bit constants | **562,624** | |

(The ctx-1 RAM firmware of `ramown/` never calls q/k/rope: 469,376 loads, 410 K of them scales. The legacy WF firmware reads the 562,624 above.)

## 2. Why rows, and not the circuit DROM table

The circuit DROM is the program-constant memory of the chip: 32,768 words (128 KB) of compiled head state. The blob constants are 562,624 words = 2.25 MB distinct (1.88 MB on the RAM firmware), 17x the DROM, and an earlier bring-up (`HOST_ROLE_AUDIT.md` 5.4) simply wrote them into the DROM head state from the host, which is a host role. Growing the DROM that far means 17x more key heads in a chip layer. The WROW path already exists, is verified, and has 4.6 M spare entries: the constants need 1,191 more rows.

## 3. Design

**Storage.** Constant c (uint32) is stored as the 8 nibbles u_k (k = 0 low) of c' = (c + 0x88888888) mod 2^32. One VDOT word per constant, 480 constants per 3840-cell row. (`cf/cf_gen.py`, table `rowtab_cf.npy`.)

**Fetch (`cfetch.S`, generated).** For a window of 8 constants: `wb` so that the `wrow` sits at WP = 8 win + 511 (the instruction after it has WP = 0 mod 8, so the latch word index of the j-th following instruction is j mod 8), `wrow row`, then four rounds of 8 VDOTs with int8 activations (1,16) on cell pair q = 3, 2, 1, 0 (rs1/rs2 = 0x10010000 / 0x00001001 and the zero register). Each VDOT j mod 8 = i returns d_q = (u_2q - 8) + 16 (u_2q+1 - 8) = byte_q - 136 for word i; between rounds `slli acc, acc, 8`. After the last round acc_i = sum_q d_q 256^q = c' - 0x88888888 = c (mod 2^32). Eight accumulators, eight `sw` into the destination. 84 instructions per 8 constants (10.5 per constant). Integer arithmetic only, mod 2^32, the circuit's existing VDOT, ADD/SLLI, SW.

**Row-id layout (so every 2.1 M-step chunk touches a contiguous slice of about 5 K rows, as in the WF run).** Layer stride in the weight stream is exactly 4096 rows (1,966,080 words), so constants rows are interleaved at layer boundaries without splitting any matvec: `[3 head rows: emb row + rope] [layer 0: 4096 weight rows + 31 constant rows] ... [layer 27] [3 final-norm rows] [40,517 LM weight rows] [317 LM-scale rows]`. Weight row id = g/480 + shift (shift = 3 + 31 l, LM: +874), emitted by `cf_gen.py` into `cf_cfg.h`. The firmware `qwen3_fx_785_cf.c` reads no address >= 0x10000000 any more; constants are fetched into small scratch buffers (a norm gain vector of 1,024 words, scales 4,096 at a time) right before use.

**Port to the 32 KB RAM firmware.** `cf/fw_qwen3_ram_cf.c` (new file, `ramown/fw_qwen3_ram.c` is untouched) fetches scales per 256-row tile and gains per 256-element tile: +512 RAM words, .bss 29,776 B, 7,764 of 8,192 words with the stack (95 %).

## 4. Verification

RESULTS_PLACEHOLDER

## 5. What is still host-assisted (exact)

HOSTLIST_PLACEHOLDER
