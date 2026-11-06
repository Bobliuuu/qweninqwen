# Host-role audit of `circuit_run.py`, and the cheapest honest home for the 0.6B weights

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
| **Every LOAD/SB/SH result word** (about 1.0-1.2 M injections per 2.1 M-step segment, i.e. roughly one step in two; `waves_*.log` "injections per row") | host: `ref.Machine`'s memory word, written into the 32 KB RAM head state (slot = word index mod 8192) before the step [R: `_inject`, `ty==1`] | **HOST DECODE (oracle memory read).** Includes all inner activations, KV, accumulators spilled to RAM, and the weight/table blob in DROM. Anything the circuit stored earlier is never read back; the reference's value is. So an error cannot propagate through memory. |
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
