# Weight-fetch path: VDOT weight operands from the outer model's own parameters

Tags: **[V]** verified (log cited), **[M]** measured, **[D]** designed, not built or not run, **[H]** host-assisted. Written to be read by a strict reviewer.
Ruling used (owner): an outer Qwen3.5-4B (Qwenputer merge, 11 chip layers in front of Qwen's 32) whose PARAMETERS include a 0.9-1.3 GB inner-weight table is acceptable, provided nothing is host-served.
Code and logs: `experiments/circuit/weight_fetch/` (`wf_npu.patch` against Qwenputer-npu `2717232` `experimental/riscv`, `test_rv_wfetch.py`, `bench_wf_exec.py`, `wf_stats.py`, `logs/`).
Context: `HOST_ROLE_AUDIT.md` sec. 3 (embedding-row ROM, `hold_weights/embed_rom.py`), `TIERS.md`.

## 0. Result in one paragraph

A new instruction `WROW rs1` makes the CPU emit a token `ROW(r)`. The embedding row of that token IS table row r (a parameter of the outer model: `ROW_CELLS` packed weight entries as token features, no data from the host). Inside the compiled chip layers, at the position of the `ROW` token, one MLP selects the 8W entries of the window the next VDOTs will use, a second MLP unpacks them to bits (4-bit or 6-bit entries), a DeltaNet head latches the bits, and the existing VDOT multiplier consumes them in place of the WROM read. The WROM (17 key heads, 32,640 words loaded by the host) is removed from the circuit. **[V]** The compiled circuit (11 chip layers + 1, IR evaluator with its bf16/silu exactness assertions) predicts every token of every test trace exactly like the reference emulator: one VDOT, W-word windows, and chunks of up to 600 VDOTs, at 4 and 6 bits per entry, 2048 and 3840 entries per row. Cost [M]: +1 chip layer (12 vs 11; 13 for 6-bit at W>=4), and ROW tokens: 1.0-1.1 extra trace tokens per VDOT at W=1, 0.125-0.13 at W=8, 0.07-0.10 at W=16 (baseline is 1.30 tokens per VDOT in the blocked kernel).

## 1. Address path (what exists, what each stage is)

```
CPU register x[rs1] --WROW--> circuit exit writes code bits (family ROW + 18 payload bits)
   --> output head (tied) picks token ROW(r)  [this is the "address decoding": the ROW row of the tied vocab]
   --> next position embeds row r:  rowcell[0..ROW_CELLS) features (the table row)           [parameters, gather]
   --> L5 MLP  SELECT : entry ci of window = sum_win AND(WP[SB-1:logW]==win) * rowcell[win*EPW*W+ci]
   --> L6 MLP  UNPACK : the EB bits of each entry (threshold units), regrouped into 32 bits per word
   --> L7 attn LATCH  : DeltaNet head, write beta=[ROW], read gated by WP[logW-1:0] -> 8 weights u (4 bits each)
   --> existing VDOT multiplier (w = u - 8), L7+ ... exit
```
`WP = D + w` is the existing weight address of VDOT (D from `WB`, w the PC word). At the ROW position the circuit already computes the WP of the next instruction; its low `logW` bits pick the word of the latched window, its high bits pick the window (`ROW_SLOT_BITS` = 9 for 480-word rows). The firmware generator therefore lays the stream out by code position (as the WROM layout already does), with one row of W words per `wrow`.

### 1.1 Why no untied lm_head
`HOST_ROLE_AUDIT.md` assumed an untied head with 259 K address-decoding rows (+1.33 GB). Not needed: the chip's exit already writes a code (5 family bits + payload bits, `CODE_AMP*(2*bit-1)` in the token's embedding row) and the tied head scores tokens by code agreement; the final-norm gains make the head read only the code dims. A ROW token's embedding row carries BOTH the code dims (address decoding) and the cell features (data). So the table rows are the address-decoding rows: **vocab growth = number of table rows, parameters = rows x hidden x 2 B, once.** Payload widened from 16 to 18 bits (`NPAY`, +2 code dims) so that up to 262,144 rows are addressable. [D for the HF forward: the code mechanism is the existing one and was verified here only through the IR's code-bit decode and the fused executor, not through a full-vocab HF argmax over the new rows; the full-vocab head grows from 478,656 to ~634 K rows (4-bit) so the HF head GEMV reads about 32 % more per text token; the fused executor decodes code bits and does not pay it.]

Cheaper alternatives considered: (a) split the row id over two tokens or two smaller tables: needs a second token per fetch (+1 trace token), no saving since the table rows are needed anyway as embeddings; (b) key-match attention (qwenFS style): needs a full-attention layer over a KV store holding the table, i.e. the table would be KV (token-bound fill), the opposite of what a parameter table gives. Not pursued.

### 1.2 Table size and vocab (arithmetic [E], matches the 0.9-1.3 GB band)
Payload 596.0 M int4 weights = 74.5 M VDOT words. A row is as wide as the hidden size, so bytes = rows x hidden x 2; use as many cells per row as hidden allows (about hidden minus the code dims).

| Packing | entries/word | entries total | rows at 3840 entries/row | bytes at hidden 4096 (3968) | vocab: 478,656 + rows |
|---|---|---|---|---|---|
| 4-bit (nibble per entry) | 8 | 596.0 M | 155.2 K | 1.27 GB (1.23) | 633.9 K |
| 6-bit (6 entries per 32-bit word, 36 bits) | 6 | 447.0 M | 116.4 K | 0.95 GB (0.92) | 595.0 K |
(Exact 6-bit packing of 3 nibbles per 2 entries would be 0.884 GB, `embed_rom.py`; the per-word packing used here costs 89 % efficiency but keeps the VDOT word aligned. Note the 2048-entry rows of the first tests would need 291 K rows, over 2^18; 3840 entries/row is the intended size.)
Where the 3840 cells live in the merged model's 4096-wide residual (the Qwen dims 0..2559 are zero for chip tokens and unread by the chip layers; the rest are chip slots) is **[D]**: in the standalone IR they are separate feature dims (`dims` 6,865 vs 2,959). The IR check that every residual value stays below 2^28 (cells <= 63, 3840 of them, are ~2^20 against the 2^56 ballast scale) passed, so the chip RMSNorm treats them as in the existing design (ballast-dominated norm), not as the pinned-energy scheme of `embed_rom.py`.

## 2. Exactly which layers and parameters are new

| Item | 4-bit | 6-bit | Status |
|---|---|---|---|
| uop `WROW` (decode, opcode 0x0B f3=6 f7=1), exit case `ROW(x[rs1][17:0])`, `NPAY` 16 to 18, token family `ROW` (FINAL) | yes | yes | [V] |
| SELECT (L5 MLP): ROW_CELLS units | 3,840 | 3,840 | [V] |
| UNPACK (L6 MLP, thresholds): units for W=1 / 8 / 16 | 458 / 3,370 / 6,698 | 1,482 / 11,562 / 23,082 | [V] (stats, `logs/wf_stats.log`) |
| LATCH (L7 attn): key heads for W=1 / 8 / 16 | 1 / 2 / 3 | 1 / 2 / 3 | [V] |
| Removed: WROM (17 key heads, 32,640 words, host-loaded) | -16 net heads | | [V] |
| Chip depth, W=1 / 4 / 8 / 16 / 32 (baseline 11) | 12 / 12 / 12 / 13 / 13 | 12 / 13 / 13 / 14 / 17 | [V] (build) |
| Outer parameters: ROW rows in the tied embedding | +155.2 K rows (1.27 GB) | +116.4 K rows (0.95 GB) | [D] sizes, rows built only for 2-6 test rows |
| Total chip units (baseline 45,691) W=8 | 52,901 (+7,210) | 61,093 (+15,402) | [V] |

Depth note: the compiler places nodes greedily, so the fetch nodes are created right after `WP` is known (before the multiplier stages take the free MLP units) to get 12 instead of 15 layers.

## 3. What was verified (all [V], logs in `experiments/circuit/weight_fetch/logs/`)

Method: `core.build(wrow=W)` compiles the circuit; the reference `ref.Machine` (with the same table) produces the trace; `ir.evaluate(check=True)` runs the circuit teacher-forced with all of the IR's exactness assertions (every materialised value bf16, silu ranges, one-hot keys, ballast absorption), and the circuit's predicted next token must equal the reference's at every position. Negative control: a table shifted by one makes the check fail. No host data enters the circuit: the WROM tables are not loaded at all; the cells come only from the ROW token's embedding features.

| Test (`test_rv_wfetch.py`) | Config | Result |
|---|---|---|
| ONE VDOT word fetched+unpacked+used (`test_weight_fetch`, W=1, real Qwen3-0.6B int4 nibbles from `wrom.bin` for the rows, random activations/accumulators incl. extreme values, a row past the table) | 4-bit, 2048 and 3840 entries/row | exact, `test_rv_wfetch_cells{2048,3840}.log` |
| Window of W=8 words, up to 5 VDOTs after one `wrow` | same | exact |
| Negative control (table +1) | W=1, 8 | fails as it must |
| N-VDOT chunks with a pure-Python oracle (accumulator vs `ref.vdot` on the chosen weights, independent of the table layout code): N=20, 300 (W=1), 64, 300 (W=8), 96, 600 (W=16) | 4-bit | exact over 99 to 833 positions |
| Same suite at 6 bits/entry, 3840 entries (640 words/row) | 6-bit | 16/16 exact, `test_rv_wfetch_6bit_cells3840.log` |
| Whole suite again with `kernel=True` (every head read, including the latch, replayed the way the HF torch kernels compute it, traces < 1500 tokens) | 4-bit and 6-bit, 3840 entries | 16/16 each, `test_rv_wfetch_kernel_{4bit,6bit}_cells3840.log` |

Unpack: at 4 bits the "unpack" is the nibble to bit-plane extraction; at 6 bits it is the 6-bit entry to bits, with the 32-bit word spanning 6 entries (the top 4 bits of the sixth entry unused). Both verified.

### What is NOT verified / host-assisted (be explicit)
- Teacher-forced IR evaluation, not the merged HF checkpoint: the fetch layers were not merged into the real Qwen3.5-4B checkpoint, and `hfcheck` (HF modules) was not run on them. The row cells' placement in the 4096-wide residual and the layer-0 norm on a 3840-cell row are [D].
- The fused-executor free-running check (sec. 5) uses a gather of the ROW embedding from a table of 64 rows (`WF_ROW_SIZE=64`), not the 155 K-row table.
- Only a handful of table rows exist in the tests (2 to 6 rows, real nibbles for the single-word tests, random for the chunk tests). Whole-table readback is the `embed_rom.py` result (different norm regime); the full 155 K-row table with the chip's ballast norm was not built.
- The CPU still gets its activations, RAM words and registers the way `HOST_ROLE_AUDIT.md` describes (LOAD results injected in `circuit_run.py`): this work removes only the weight host-role. Slot/layout generation for a real firmware (one `wrow` per W words, ROW row ids) is done for the synthetic chunks only (`chunk()` in the test); the compiler (`experiments/circuit/compiler`) does not emit it yet.
- The layout constraint is real: the window is `WP[SB-1:logW]` at the ROW and the word is `WP mod W`, so the weight stream is laid out by code position, and any non-VDOT word inside a window wastes that slot. Table density in the synthetic chunks was 0.4 to 0.8 (rows allocated vs ideal); a layout optimiser could raise it, not done.

## 4. Step overhead (trace tokens per VDOT) [M]

Extra tokens per VDOT = (ROW tokens + row-register changes + padding) / VDOTs, measured on the same VDOTs as WROM-mode programs (`CHUNK` lines of the logs):

| W | N VDOTs | trace tokens WF / WROM | extra tokens per VDOT |
|---|---|---|---|
| 1 | 300 | 785 / 449 (3840 cells) | 1.01-1.12 |
| 8 | 300 | 492 / 454 | 0.127 |
| 16 | 600 | 833 / 775 (3840) or 816 / 775 (2048) | 0.068-0.097 |
In the blocked VDOT kernel (22 VDOT words + 2 loads, 0.163 tok/MAC = 1.30 tokens per VDOT [TIERS]) one ROW per window gives about 0.14 (W=8) and 0.09 (W=16). The ROW token itself is 1 token; a row-register `addi` (1-2 tokens) happens once per row (every 480 words), negligible.

## 5. Projection: tokens per instruction and s/token [E from the [M] inputs]

Inputs [M]: 74.5 M VDOT words per inner token (596 M MACs / 8); baseline v2 int4 VDOT build 1.34e8 instr, 1.55e8 tokens (1.154 tok/instr, `TIERS.md`); 4,963 tok/s tuned A10G / 4,225 tok/s untuned A100 on the 11-layer executor.

| W (4-bit, 3840 entries/row) | extra tok/VDOT [M] | tokens/inner token (baseline 1.55e8) | chip layers | step-time factor [E: linear in layers, NOT measured] | s/token at 4,963 tok/s (baseline 3.12e4 s = 8.7 h) |
|---|---|---|---|---|---|
| 1 | ~1.05 | 2.33e8 (+50 %) | 12 | 1.09 | 5.1e4 s = 14.2 h |
| 8 | 0.13 | 1.65e8 (+6 %) | 12 | 1.09 | 3.6e4 s = 10.0 h (+15 %) |
| 16 | ~0.085 | 1.61e8 (+4 %) | 13 | 1.18 | 3.8e4 s = 10.6 h (+22 %) |
**Executor measurement [M] (A100, fused FreeRun3, untuned, 3840 entries/row, `logs/bench_wf_exec_cells3840.log`)**: free-running, the circuit consuming its own tokens with the ROW embedding gathered from a 64-row table, is token-exact against the reference over the whole trace for all six runs (baseline WROM and WF, W=1/8/16 chunks of 300/300/600 VDOTs). ms/step: baseline 11 layers 0.242-0.243 (4,115-4,127 tok/s); WF W=1 0.278 (x1.145, 12 layers); W=8 0.265 (x1.091, 12 layers); W=16 0.273 (x1.125, 13 layers). Measured factors match the linear-in-layers estimate for W=8 and are lower than it for W=16. Using them: W=8 gives 1.65e8 tokens x 1.091 = 10.0 h at 4,963 tok/s (+15 %), W=16 1.61e8 x 1.125 = 9.8 h (+13 %, replaces the 10.6 h above), W=1 2.33e8 x 1.145 = 14.9 h. Caveat: the ROW table in this run has 64 rows (the gather is not the 155 K-row table).

## 6. Open items, in order
1. Merge the fetch layers and ROW rows into the real checkpoint and run `hfcheck`/`npucheck` (HF modules) on a ROW-bearing trace; place the 3840 cells in the 4096-wide residual.
2. Compiler support: emit `wrow` + window layout (and the optimiser for table density).
3. Remove the per-VDOT WP coupling to the PC with an auto-increment `WROWN` (design: a row-register head + adder, would make the ROW token the only overhead and free the layout from code positions) [D, not built].
4. Full-width table (155 K rows) built and read back through the chip's first layer, and the HF full-vocab argmax over the new rows.
