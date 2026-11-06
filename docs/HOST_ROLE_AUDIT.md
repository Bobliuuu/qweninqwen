# Host-role audit of `circuit_run.py`, and the cheapest honest home for the 0.6B weights

**UPDATE 2026-11-06 (section 5 at the end): the RAM half of the LOAD/SB/SH host role has been removed on the remapped firmware, and the section 2 attribution of the injections was wrong. Read section 5 first; sections 1-4 are kept as the baseline they describe.**

Tags: **[M]** measured (file cited), **[R]** read from code, **[E]** estimate. Written to be read by a strict reviewer. This is not the compiled route and the run below is not a completed circuit-route run.

## 1. What `circuit_run.py` is

A **lock-step verification of the compiled CPU circuit against a software emulator that is allowed to supply all memory**. It is not a self-sustained run of Qwen3-0.6B on the circuit.

- The "outer model" executed is the Qwenputer **compiled chip layers** (hidden-4096 layout of the Qwen3.5-4B merge) run by the fused Triton executor (`fusedexec.FreeRun3`), not the HF `Qwen3.5-4B` forward pass [R]. Executor = "same arithmetic, not same kernels" (TIERS.md). The HF-module check exists only for the tiny firmware (`e2e/hfcheck.py`).
- The circuit core is **patched** for bring-up: the RAM address-range check is replaced by `in_ram := one` so out-of-range model addresses do not fault [R, lines 14-17]. It is not the shipped core.
- The "reference" is `ref_big.py`: Qwenputer's Python `ref.Machine` with enlarged memories (RAM 4 MB, DROM 624.6 MB, WROM 74.5 M words), which **has already executed the entire inner token**: 162,682,048 trace tokens, 136.9 M instructions, 78 chunks, argmax 13 (`logs/ref785b/summary.json`) [M].

## 2. Host-served vs computed in the circuit

Evidence: `logs/cr_out/result_*.json`, `logs/waves_cr_out.log`, code lines cited.

| Item | Who produces it | Strict-reviewer label |
|---|---|---|
| Next-token argmax, embedding of the circuit's own token, all register/ALU/MUL/DIV/branch/jump/PC/WB logic, load merge and sign-extension, VDOT arithmetic on its WROM cells, register write-back | the circuit (every token not masked is the circuit's argmax fed back) [R] | computed in the forward pass |
| **Every LOAD/SB/SH result word AND every VDOT weight word** ("injections per row" in `waves_*.log`, about 1.0-1.2 M per 2.1 M-step segment, is the SUM of both; section 5 shows 13 % of it is LOAD/SB/SH and 87 % is VDOT weights) | host: `ref.Machine`'s memory word, written into the 32 KB RAM head state (slot = word index mod 8192) before the step [R: `_inject`, `ty==1`] | **HOST DECODE (oracle memory read).** Includes all inner activations, KV, accumulators spilled to RAM, and the weight/table blob in DROM. Anything the circuit stored earlier is never read back; the reference's value is. So an error cannot propagate through memory. |
| **Every VDOT weight operand** (8 int4 cells from the 596 MB `wrom.bin`) | host, written into WROM head state at WP mod 2^15 [R: `ty==2`] | **HOST DECODE (weights supplied from outside the model).** The WROM state holds 32,640 words = 261,120 nibbles, 0.044 % of 595,984,384. |
| VDOT records whose WP mod 2^15 falls in the 128 unaddressable words | host: the **whole record's tokens (including the dot result) are replaced by reference tokens** (`MASK`) [R]. 3,200-4,096 steps per 2.1 M-step row, about 0.17 % [M] | **HOST DECODE (output forced).** Raw circuit prediction disagrees with the reference on 205-4,067 of ~3,500 forced tokens per row. |
| **Injection schedule** (which step, which slot, which value) | host, precomputed from the reference trace (`inj_*.npy`) by step index, not derived from the circuit's own addresses [R] | **HOST DECODE (look-ahead).** The host knows the future and the circuit's own address computation is not what selects the data. |
| Start of each 2.1 M-step segment: registers, PC anchor, `n`, weight base | host: snapshot of the reference machine at the chunk boundary [R: `machine()`, `core.snapshot`] | **HOST DECODE (state restore).** 60 of 78 chunks were run, in 5 batches of 12, each starting from a reference snapshot. They are independent, not chained circuit-to-circuit. Run covers 125.6 M of 162.7 M tokens (77 %), 0 free-token mismatches, 99.83 % of steps free [M]. |
| Harness | host: loop, block replay of the CUDA graph, comparison | allowed (README transport items) |

Bottom line: [M] 0 mismatches over 125.6 M free tokens is a real result about the **control/ALU/VDOT-datapath of the circuit**: given correct memory operands it reproduces the emulator's trace exactly. It is not evidence that the circuit can hold or fetch the model's data, and the data-dependent part of Qwen3-0.6B (every value that goes through memory) is host-supplied. A strict reviewer would call this "host decode of all memory" and would not accept it as "Qwen3-0.6B run inside Qwen3.5".

What closes the gap, in order: (1) a weight home inside the outer parameters (section 3) so VDOT operands stop being injected; (2) circuit-owned RAM: loads read what the circuit's own stores wrote (needs the real 32 KB RAM plus a spill target for activations/KV, plan sec. 6 puts KV in outer full-attention KV); (3) chain chunks circuit-to-circuit with no snapshot; (4) remove the `in_ram` patch; (5) retire the forced VDOT records.

## 3. Holding all 596 M int4 weights: options with byte counts

Payload: 595,984,384 nibbles (`wrom.bin`) = **297,992,192 B**. Outer = Qwen3.5-4B geometry (H 2560, I 9216, 24 DeltaNet + 8 attention layers, 24 DeltaNet layers x 32 value heads x 128 x 128 = 12.58 M state entries; `memory/MEMORY_DESIGN.md`).

| Home | Capacity at stock size | Bytes needed | Verdict |
|---|---|---|---|
| DeltaNet state loaded once | 12.58 M entries; at 8 bits/entry (bf16 integer ceiling) 12.6 MB; realised Qwenputer density 0.7-1.6 MB | 298 MB payload | **24x short at the ceiling, 190-420x at realised density.** Cannot be fixed by loading; only by 18 K+ extra value heads (36 GB checkpoint). Keep for registers, RAM, small constants. |
| Constants in appended MLP layers (memory agent's ROM, option d) | 29 banks x 9,216 words x 2,300 weights/word (one weight per bf16 entry) | per bank gate+up+down = 3 x 9216 x 2560 x 2 B = 141.6 MB; 29 banks **4.11 GB** (down_proj alone 1.37 GB). 13.8x the payload | Works as storage (exactness measured by memory agent), but 29 extra layers plus read, select, MAC and accumulate layers (110 appended layers in the current plan). Heaviest and deepest. |
| **Rows of an appended embedding table, row fetched by token id (option M2)** | 2300 data dims + filler per 2560-wide row | **1.327 GB bf16 at 4 bits/entry (259,124 rows, 4.45x payload); 0.884 GB at 6 bits/entry (172,750 rows, 2.97x)** | **Cheapest honest storage**: no extra depth, one parameter table, exact by construction. Prototype below. |
| (stock 0.8B) embedding | 254 M entries | 596 M nibbles do not fit at 4 bits/entry (needs >= 596 M entries) | Use 4B. |

All of these put the weights in the **outer model's parameters** (checkpoint grows from 8.4 GB to 9.7-9.8 GB for the embedding route). That is the same relation the program ROM has to the chip; it is not stock-Qwen3.5-with-a-CPU-in-it, and nothing at stock size can be (state is 24x too small). State this ruling openly.

### Prototype (measured): `experiments/circuit/hold_weights/embed_rom.py`

The real `wrom.bin` (all 595,984,384 nibbles of the Qwen3-0.6B int4 firmware weights) is packed into a bf16 embedding table, looked up for every row, passed through a Qwen3.5-style RMSNorm (fp32, `(1+w)` gain), and compared with the original stream. Filler dims (greedy sum-of-squares, entries <= 255) pin sum(x^2) to C = H x 4^m so mean(x^2) is exactly a power of two, `rsqrt` is exactly 2^-m, and the norm is a pure power-of-two scale. A10G, bf16, whole-table check, logs in `hold_weights/logs/`:

| Config | Rows | Table | Inexact entries after norm | Symbol mismatches | Stream identical to `wrom.bin` |
|---|---|---|---|---|---|
| H 2560, 2300 data dims, 4 bit/entry | 259,124 | 1.327 GB | 0 | 0 | yes |
| H 2560, 2300 data dims, 6 bit/entry (1.5 nibbles/entry) | 172,750 | 0.884 GB | 0 | 0 | yes |
| H 1024 (0.8B width), 900 data dims, 4 bit | 662,205 | 1.356 GB | 0 | 0 | yes (storage fits in bytes, but 0.8B vocabulary 254 M entries is too small) |

What this does and does not show:
- Shows: all 0.6B weights fit in an outer-geometry embedding table, 1.33 GB at 4 bits/entry (0.88 GB at 6), and a row read through the stock norm returns exactly the stored integers. 6 bits/entry is the practical ceiling for exact fp32 energy sums (< 2^24); 8 bits needs a different norm-pinning scheme.
- Does **not** show: (a) the CPU can emit the row's token id: that needs the **lm_head to be untied** with 259 K (or 173 K) address-decoding rows, +1.33 GB, and the full-vocab argmax per step grows (the fused executor evaluates only a candidate set, which is itself not the stock forward) [E]; (b) a MAC consuming the row at 2,300 entries/position (memory agent: 230 MACs per 5-layer pipeline, residual-width limited, 185 K ROM_DOT calls per inner token); (c) the 6-bit unpack layer; (d) 4B vocabulary growth from 263,092 to about 522 K (19 bits/token id) in the circuit's token alphabet [E]. These are the next experiments; none is built.

## 4. Recommendation

1. Adopt the embedding-row ROM as the weight home (1.33 GB, 0.88 GB with a 6-bit unpack), outer = 4B, untied head. Next experiment: row fetch from a CPU register (print the row id), then one MAC pipeline on a fetched row, then VDOT-from-ROM replacing the injection of `ty==2`. That removes the weight host-role; the memory (LOAD) host-role remains and is the larger gap (about half of all steps).
2. Until (1) and the RAM spill exist, label every `circuit_run.py` number "oracle-memory lock-step verification of the CPU datapath", never "Qwen3-0.6B on Qwenputer".
3. Ruling needed from the owner: is an outer checkpoint whose parameters include a 0.9-1.3 GB weight table acceptable? If not, no stock-size option exists (state is 24x short).


## 5. Update: the RAM role is removed from the host (remapped firmware), measured 2026-11-06

Tags as above. Code: `experiments/circuit/ramown/ref_ram.py` (reference run that also models the circuit's own 8192-slot RAM and writes per-policy injection files), `experiments/circuit/fullrun/circuit_run.py` (`--ramimg`, `--inj`, `REALMAP=1`), firmware `ramown/fw_qwen3_ram.c` + `fw_rom.ld` (other agent; working set 18.2 KB peak live, `ramown/logs/addr_ramfw_ctx1.txt`). Logs: `ramown/logs/` (stats per chunk, circuit results).

### 5.1 Correction to section 2

The "about 1.0-1.2 M injections per 2.1 M-step segment" are LOAD/SB/SH **plus** VDOT. Counted separately on chunk 1 of the legacy (4 MB-map) firmware [M, `ramown/logs/ref0/stats_0001.json`]: 146,998 RAM-type (LOAD/SB/SH, 13 %) and 973,376 VDOT weight injections (87 %). The statement "roughly one step in two is a LOAD injection" was wrong; the dominant host role is the VDOT weight operand, not memory.

### 5.2 The five gaps and what was done

| Gap | Fix | Status |
|---|---|---|
| (b) chunk starts with empty circuit RAM | `ref_ram.py` dumps the RAM image at each chunk start (`ramimg_NNNN.npy`, the 4 MB image folded into 8192 slots); `circuit_run.py --ramimg` passes it to `core.snapshot` via `m.ram` | done; restored words are really used on the remapped firmware (chunk 1: 7,212 loads answered by the restored image) |
| (c) SB/SH merge needs the old word | same restore + the circuit's own slot; the simulation injects the old word only if the circuit's slot differs from the true word | done; on the remapped firmware 4,096 / 6,144 / 1,024 SB/SH per chunk, 0 old-word injections |
| (d) schedule injects loads the circuit already answers | `ref_ram.py` simulates the circuit's RAM slot array (SW writes, SB/SH merge, injection overwrites) and keeps only loads whose slot would hold the wrong word | done (policy F) |
| (a) `in_ram := 1` aliases 4 MB into 8192 slots | not fixable by harness changes on the legacy firmware (14,306 of 14,307 remaining injections on chunk 1 are aliasing). Fixed by the remapped firmware: with the real 32 KB map `in_ram` is **not patched**; `REALMAP=1` only widens `in_drom` (see 5.4) | done for the remapped firmware |
| firmware working set > 32 KB | other agent's `fw_qwen3_ram.c` (tiled matvec, streaming LM-head argmax, ctx-1 attention without KV) | landed, bit-exact vs the original on rvemu |

### 5.3 Measured injection counts (chunks 0, 1, 2 of the 785-prompt trace; per 2.1 M-step chunk)

Legacy firmware, 4 MB map, `in_ram := 1` (simulation of the circuit RAM; policy F = filtered; the circuit run of F on chunk 1 is in 5.5):

| Chunk | LOAD/SB/SH injected, old schedule | filtered (F) | drop | of the remaining: blob / RAM-region | cause of every remaining one |
|---|---|---|---|---|---|
| 0 | 165,777 | 17,336 | 89.5 % | 8,510 / 8,826 | aliasing (15,144), cold (2,192) |
| 1 | 146,998 | 14,307 | 90.3 % | 6,144 / 8,163 | aliasing (14,306), cold (1) |
| 2 | 169,501 | 17,754 | 89.5 % | 6,464 / 11,290 | aliasing (17,754) |

Remapped firmware, real 32 KB map, **no `in_ram` patch** (simulation `ramown/logs/refR2/`, circuit run in 5.5):

| Chunk | LOAD/SB/SH old schedule | RAM-region injections now | loads served by circuit's own RAM | loads served from the restored image | blob (DROM-class) words injected | VDOT weight injections |
|---|---|---|---|---|---|---|
| 0 | 159,229 | **0** | 155,133 loads + 4,096 SB/SH | 0 | 8,700 | 879,204 |
| 1 | 170,588 | **0** | 164,444 + 6,144 | 7,212 (+287 zero-init) | 6,093 | 955,804 |
| 2 | 147,487 | **0** | 146,463 + 1,024 | 2,659 | 7,218 | 877,702 |

Per-chunk host-served total on the remapped firmware (chunk 1): 955,804 VDOT weight cells + 6,093 blob words = 961,897, against 1,120,374 before (the old LOAD/SB/SH 146,998 plus VDOT 973,376). Real-DROM rodata loads (59-70 per chunk) are answered by the circuit's own DROM table.

### 5.4 What the circuit run is (and the one core patch left)

- RAM: `in_ram` is the shipped expression (EA[15:32] == 0), no aliasing, no fault masking; loads/stores in 0x100-0x8000 are the circuit's own.
- The firmware also reads constants from a 624 MB blob at `0x10000000` (embedding row, norm weights, matvec scales, final norm). A real core faults on that address. For the run, `in_drom := not in_ram`, so those loads go to the circuit's DROM table (slot = word index mod 32768) and the host writes the blob word into the DROM head state before the load (injection type 3, `circuit_run.py`). That is a **constants/weight-memory host role, same class as the VDOT weight injection**, not a RAM role. The simulation checks that these injections never overwrite a real-DROM rodata word that is later read (0 `drom_need_real`).

### 5.5 Circuit results (A100, `ssh lambda-final`)

RESULTS_PLACEHOLDER

### 5.6 What remains host-served after this change

1. **VDOT weight operands**, 0.88-0.96 M injections per chunk (about 99 % of all remaining injections), plus 128 unaddressable WROM words (forced tokens, 0.17 %). Unchanged. Weight home = embedding-row ROM (section 3), prototype only.
2. **DROM-class constants** (the blob: embedding row, norms, scales), 6-9 k words per chunk, injected into the circuit's DROM head state. Needs a ROM home (same decision as 1).
3. **Chunk-start state**: registers, pc, `n`, `wb` (and now the RAM image) are restored from the reference machine at each chunk boundary; chunks are still independent, not chained circuit to circuit.
4. **Look-ahead**: the injection schedule (which VDOT / blob word at which step) is still precomputed from the reference trace. The RAM schedule is now empty on the remapped firmware, so for RAM the question is moot.
5. Unaddressable-WROM VDOT forcing (290,944 tokens over the full run).

Removed: the LOAD/SB/SH host role for RAM (stack, activations, accumulators, SB/SH merge words) on the remapped firmware, in the measured chunks. Not shown: that the other 75 chunks behave the same (only chunks 0-2 were simulated/run), that the remapped firmware's full-run answer equals the legacy one is the other agent's claim (argmax 13 in `ramown` reference run), and the patched `in_drom` is still a bring-up patch.

## 6. Update: chunks chained circuit-to-circuit (legacy firmware), measured 2026-11-06

Code: `experiments/circuit/fullrun/circuit_run.py --chain | --handoff`; reference additions in `ref_big.py` (per-chunk `bn`, and per RAM slot the last-touched word address, its value and whether a load, SB/SH or SW touched it). Logs: `experiments/circuit/fullrun/logs/chain/`. Results table and caveats: `docs/TIERS.md`, section "Chained circuit replay".

What changed for section 2, row "Start of each 2.1 M-step segment": for the chained runs the host restores state only at step 0 of the first chunk. After that the registers, pc anchor, `n`, the 15-bit weight base, `bn` and the 32 KB RAM are whatever the circuit holds. They were read back from the circuit's head states at every chunk boundary and equal the reference snapshot (0 of 31 registers wrong in each of the three register-file copies; pc, `n`, `wbase`, `bn` equal; all 8192 RAM slots equal after the SB/SH block-B decoder fix). `--handoff` restarted the circuit from only that read-back state at three boundaries and ran 8.4 M steps with 0 free-token mismatches. So "chunks are independent snapshots" is no longer true for the chained chunks; it remains true for the unchained ones.

The answer to "do circuit stores ever get read back": in this firmware, no. The stores are right (SW-last and SB/SH-last RAM slots equal the reference at every boundary), but every LOAD word is injected from the reference before the load, so a wrong store could not change a later load. The boundary readback is what shows the stores are right; the loads would not have.

Still host-served in the chained run (counts per 2.1 M-step chunk, chunks 0-5): VDOT weight words 0.83-1.04 M (88-89 % of injections) and the upper 12 bits of the 27-bit weight pointer (the circuit's `wbase` is 15 bits; the reference's full pointer, e.g. 5,503,361 at the end of chunk 5, exists only in the host); LOAD/SB/SH RAM words 142-170 k (removed only on the remapped firmware of section 5, not chained here); 3,200-4,096 forced VDOT records (0.15-0.20 %); the injection schedule; chunk 0's initial registers/pc; the `in_ram := 1` patch on the legacy firmware. Not chained: 58+ of 78 chunks, and the remapped-firmware run.


## 7. Update: the VDOT weight host role is removed in the full-token replay (WROW embedding-row path), measured 2026-11-06

Tags: **[V]** verified (log cited), **[H]** host-assisted. Code: `experiments/circuit/compiler/fw/gen_vdk_wf.py` (kernel) + `fullrun/fw/qwen3_fx_785.c` (the C source that built the w785 firmware, with a `WF` call-site), `fullrun/build_fw.sh` (`WF=1`), `fullrun/wf_table.py` (row table), `fullrun/ref_big.py` (`WF=1`), `fullrun/circuit_run.py` (`WF=1`), `fullrun/modal_wf.py`, `launch_wf_groups.py`, `collect_wf.py`. Logs: `fullrun/logs/wf/` (per chunk stdout/json, `summary.json`, `ref_summary.json`, smoke, negative control).

What runs: the SAME Qwen3-0.6B int4 token (prompt 785, argmax 13) as the baseline replay, with firmware regenerated so every 8-word VDOT window is preceded by `wrow` (`vk_*_raw` in `vdk_wf.S`: per 32-word iteration `wb slot; wrow row; 8 vdot; wrow row; 8 vdot; wrow; 8 vdot; wrow; 8 vdot`, slot = global word index mod 480, row = index div 480, computed by the firmware). The 155,205-row table (`wf_table.build_table`, 3840 4-bit entries per row = 480 words, 1.27 GB at hidden 4096 in bf16) is built from `wrom.bin`; its layout follows from the kernel (window = slot div 8, latch position p = (1 + k + window mod 4) mod 8). The circuit is `core.build(wrow=8)` (12 chip layers, 4-bit, 3840 cells, W=8) with the same bring-up edit as before (`in_ram := one`). The circuit run injects NO weight word and forces NO VDOT record (`--novdot` implied, asserted: the WF reference has 0 type-2 injections). The fused executor embeds each ROW token from the table row (only the rows a chunk's trace touches are materialised: 1.3-2.0 K rows per chunk, patched into `fastexec.FreeRun` at run time).

Verified [V]:
- Reference: `ref_big.py` WF=1 on lambda-final: 174,823,695 trace tokens, 151,156,591 instructions, 9,312,256 ROW tokens (exactly 0.125 per VDOT word: 74,498,048 words / 8), 84 chunks, argmax 13, exit 0 (`logs/wf/ref_summary.json`). Baseline legacy trace: 162,682,048 tokens. So the WF trace is +7.5 % tokens. The first 300 K VDOTs read the same 8 cells through the ROW/latch path as through the WROM (equality check on lambda-final, 2026-11-06).
- Circuit replay on Modal (L40S, 8 chunks batched as rows per GPU, ~29.8 K tok/s aggregate; A10 B=1 2.8 K tok/s): all 84 chunks, 174,823,694 free tokens (every token after GO is the circuit's own prediction, 0 forced), **0 free-token mismatches in every chunk** (`logs/wf/summary.json`, `logs/wf/chunks/*`).
- Negative control: the same chunk 1 with every table entry +1 diverges at step 61,870, the first VDOT after the first ROW token, 23,409 mismatches in 100 K steps (`logs/wf/negcontrol_chunk1_tableplus1.stdout`).
- Host injections left per chunk: only type-1 RAM/blob words, 8.6 K to 160 K per chunk, 12,169,430 in total (7.0 % of steps). Before: 74.5 M weight-word injections plus the same RAM words (88 % of injections).

Still host-assisted [H] (do not call this "Qwen in Qwen"):
1. LOAD/SB/SH words (RAM and the 624 MB blob/rodata reads) are injected from the reference into the circuit's aliased RAM head state before each load (legacy 4 MB-map firmware; the remapped-firmware RAM removal of section 5 is a different firmware and is NOT combined with WF here). The circuit RAM address check is removed (`in_ram := one`).
2. Chunk start: each of the 84 chunks restarts from the reference snapshot (registers, pc anchor, n, 15-bit weight base, `bn`, and now the 8-word latch window). This run is per-chunk, not one chained circuit run; the latch window at a chunk start is host-seeded.
3. The table is not inside a merged Qwen3.5-4B checkpoint: it is a 155 K-row array loaded by the executor harness (IR/fused executor, not the HF modules; `hfcheck` was not run on ROW tokens; cell placement in the 4096-wide residual and the full-vocab HF argmax over the new rows remain open, `WEIGHT_FETCH_PATH.md` sec 6). Per run only the touched rows are materialised.
4. The injection schedule and the pass/fail comparison come from the reference trace; DROM-class constants are inside item 1.
5. 4-bit entries used (the verified fused free-run configuration); the 6-bit/0.95 GB packing needs 13+ layers and was not run through the fused executor.
Removed host roles: the VDOT weight operands (all 74.5 M words), the unaddressable-WROM forcing, and the upper 12 bits of the 27-bit weight pointer for weights (only WP[8:0] is used now; the row id is a firmware register).
