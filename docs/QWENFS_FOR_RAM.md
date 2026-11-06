# Can the qwenFS technique make RAM (and the weight table) circuit-owned?

Tags: **[M]** measured (file cited), **[R]** read from code, **[E]** estimate with derivation. Sources read: `docs/HOST_ROLE_AUDIT.md`, `docs/TIERS.md` (full-trace replay), `experiments/circuit/fullrun/{circuit_run,ref_big}.py`, Qwenputer `chipformer/rv/core.py` + `spec.py`, `experiments/circuit/memory/MEMORY_DESIGN.md`, and the qwenFS clone `~/qweninqwen-work/qwenFS-ref` (`README.md`, `qfs/circuit.py`, `qfs/merge.py`, `qfs/disk.py`; read only, nothing of it was executed).

## Verdict (short)

1. **RAM: the Qwenputer core already has circuit-owned RAM with the semantics we want, and it is cheaper than the qwenFS log.** `core.py` L7-8 `ram{h}` tables are DeltaNet one-hot-keyed rows: SW/SB/SH overwrite the addressed row in place (exact latest-write-wins, constant size), loads read it back. The qwenFS attention-log approach would be a strictly worse RAM (KV grows with every store, never overwrites). Do not port it for RAM.
2. **The replay injects because of capacity and bring-up convenience, not because the mechanism is missing**: 32 KB (8192 words) of real RAM vs a 4 MB model address space aliased mod 8192, a 624.6 MB read-only blob in DROM, chunks that start without RAM contents, and an `in_ram := 1` patch. Section 3.
3. **Weights as a qwenFS-style attention-log "file": not feasible.** 149 M log tokens, 4.9 TB of KV (610 GB if only one attention layer holds the log), about 0.4 s per lookup on an A100; qwenFS's own limits are 31x too small; and qwenFS pages bytes in through the host. The audit's embedding-table option (1.33 GB in parameters, exact, measured) is far better. Section 7.
4. What qwenFS *is* useful for: a tiny prototype (below) confirms exact LWW lookup over an attention log works in fp32 and degrades under bf16-rounded scores. That is the only qwenFS-style piece with any role: an unbounded-capacity spill target for words that fall out of the 32 KB window, and only if bounded in stores and not host-paged. Section 6.

## 1. qwenFS primitives and what maps to LOAD/STORE [R]

qwenFS = Qwen3.5-2B, 8 hand-set layers in front of the 24 stock ones (residual widened 2048 to 4096, `qfs/merge.py`), ids/versions are FS tokens. Layer list (`qfs/circuit.py`):

| qwenFS layer | Type | Job | Maps to |
|---|---|---|---|
| P, 1, 2, 4, 5 | DeltaNet | lag registers, counters (offset counter decaying 2^(-1/16)), path slots, tables (version/base/path, 128 rows each), holds | our DeltaNet registers; **not** RAM-relevant (Qwenputer has these) |
| 3 lookup | full attention | path at GO matched against paths at earlier write/read acks, "not found" = fallback key scoring between exact and near miss | **LOAD address match**: address bits as dual-rail key, exact match by softmax |
| 6 seek | full attention | (file, base, word) rows and link rows, +-1 values averaged | not needed |
| 7 copy | full attention | read ack (index, version, offset) against stored text keyed the same; value = token id bits | **LOAD data return**: key = address, value = 32 data bits; and the STORE side is simply "the token is in the log at its position" |
| MLPs | MLP | parse, pick forced tokens, exit code written into the id dims of the tied lm_head | forced output token (we already have this: the core emits value tokens) |

Exactness tricks that transfer: ballast dim so RMSNorm is a constant scale (we already use BALLAST), dual-rail bits, a "fallback" key sitting between exact and near-miss scores, gains ~16-50 on q/k norms to make softmax one-hot. Key limits that do not transfer: keys are (index 7 bit, version 6 bit, offset 11 bit); there are no duplicate keys because versions are unique, so qwenFS never needs *latest-write-wins over equal keys*. A RAM log needs it (same address written many times). Recency has to be added as extra key dims (prototype below).

Where the stores go: a STORE = append one position to the copy layer's KV. There is no "overwrite". Reads pick the newest by a recency term.

## 2. Capacity vs our needs

Our geometry (Qwen3.5-4B, real `config.json` fetched from HF [M]): 32 layers, 8 full-attention, `num_key_value_heads` 4, `head_dim` 256, H 2560, 24 DeltaNet layers with 16 key / 32 value heads, 128 x 128.
**KV per token, all 8 full layers: 8 x 2 x 4 x 256 x 2 B = 32,768 B (32 KB) [E from config]; one full-attention layer: 4,096 B.**

qwenFS own capacity [R README]: 128 files, 63 segments, ~590 tokens per segment, so at most about 4.76 M stored tokens (README says ~37k per file); 2B config.

Our 0.6B firmware needs at ctx 1 [E: from `fw/qwen3_fx.c` static arrays with DIM 1024, QD 2048, KVD 1024, HID 3072, 28 layers, VOCAB 151,936; shape constants assumed from the 0.6B config, not rebuilt]:

| Item | Bytes |
|---|---|
| activation vectors x, xb, q32, k32, v32, ao, hg, hu (int32) | 72 KB |
| qh/kh/xq/xql | 12 KB |
| inner KV at ctx 1: `kq` int8 28 KB + transposed V `vt` 112 KB (pad 4) + max tables | about 142 KB (the compact form is 56 KB/token, MEMORY_DESIGN) |
| **working set without logit buffers** | **about 215 KB = 6.6x the 32 KB RAM [E]** |
| logit accumulators `tmp`, `acc1/2`, `accs_[6]` (VOCAB int32 each) | 5.4 MB total static image, why `ref_big` gave the model a 4 MB RAM [E]; streaming argmax would avoid most of it |
| live addresses | 215 KB = about 54,000 words if everything is touched; exact live count over the trace is **not measured** (the `inj_*.npy` files are on the A10G box, which timed out; see section 8) |

Stores in the full trace: ST0 5,127,500 + ST1 5,100,092 + SS0 202,560 + SS1 201,216 = **10.63 M store records** [M `logs/ref785b/summary.json`]. Injected loads/VDOTs: ~1.0-1.2 M per 2.1 M-step segment, about 86 M over 78 segments [M, from the audit's counts; split RAM-hit vs DROM-hit not measured].

KV growth per store if RAM were a qwenFS log [E]:

| Scope | Per store | 10.63 M stores | vs A100 40 GB |
|---|---|---|---|
| one log layer (qwenFS paging style: other layers' cache empty) | 4 KB | 43.5 GB | 1.09x (over) |
| all 8 full layers (log tokens flow through the normal stack) | 32 KB | 348 GB | 8.7x |
| one layer, one KV head | 1 KB | 10.9 GB | fits |

Even the best case is a growing log read in full at each load (section 5 cost model): 10.9 GB / 1.555 TB/s = 7 ms per load, times about 86 M loads = **6.0e5 s (7 days) per inner token**, against about 2.7 h at the current replay rate. So a store-log RAM is too slow before it is too big. The DeltaNet RAM costs 0 extra bytes and 0 extra steps.

## 3. Does the existing Qwenputer RAM already do this, and why does the replay still inject?

Yes for the semantics [R `core.py` L821-855, L1260-1270]:
- `RAM_WORDS = 1<<13` (32 KB), 32 tables `ram{h}` (`RAM_WORDS>>8`), each 128 rows x 2 value heads x 32 value bits. Word index `wi = EA[14:2]`; key head = (EA[14], EA[5:2]); value head EA[6]; row EA[13:7].
- STORE: the key row is `andpre(row lits + head lits + is_store)`; beta = `ST0/ST1/SS0/SS1`, value = SW data from t-1 or the SB/SH merged word from the record's V tokens (blocks A and B). Delta rule with one-hot key and beta 1 **overwrites** the row: exact LWW.
- LOAD: query row one-hot, output = 32 bits into `mem`, then byte/half select and sign-extension (`LW/LB/LBU/LH/LHU`, L909-928). `oor_ld` faults for out-of-range.
- Snapshots restore it from `m.ram` (L1260). The tiny-firmware e2e uses it end to end [audit].

Why `circuit_run.py` injects every LOAD/SB/SH anyway [R]:
1. **Address space.** ref_big relocates a 4 MB RAM to 0x1000000 (1 M words); the circuit has 8192 words and the `in_ram := one` patch makes any address commit to slot `(ea>>2) & 0x1FFF`. 128 model words alias to one slot, so unmodified circuit loads would read a neighbour's value.
2. **DROM blob.** The 624.6 MB weight/table blob (`DROM_BASE 0x10000000`) is loaded with `LW/LB`; the real DROM is 128 KB. Those loads are out of range by 4,900x.
3. **Chunk starts have no RAM.** `machine(stt)` rebuilds a Machine from the program with the snapshot `x, anchor, n, wb` only; `ram`/`bn` in the pickle are ignored, so in a chunk, every word stored before the boundary is absent from the circuit.
4. **SB/SH merge** needs the old word, same cause.
5. **Convenience/look-ahead.** The injection schedule is precomputed per step (`inj_*.npy`) and written for every load, including those whose word the circuit itself stored earlier within the chunk and that would have been correct.

So (1)-(4) are capacity/state, (5) is a shortcut. A refinement that needs no new mechanism: inject **only** loads whose address is outside the circuit's live window or never stored in this chunk, and let the rest read the circuit's own RAM; that quantifies the "circuit-owned fraction" honestly. Needs the `inj` data (not available now, section 8).

## 4. What would a qwenFS-style spill give that DeltaNet RAM does not?

Unbounded capacity in principle (log growth) for the 215 KB - 4 MB window, at an O(N) read per load, plus 4 KB-32 KB KV per store. Per-store cost kills it (section 2). Cheaper alternatives for the same gap, none host-assisted:
- **More RAM tables**: DeltaNet `ram{h}` cost is 128 rows x 2 x 32 bits per 256 words (1 KB). 4 MB = 128x more key heads (4096 heads); each key head costs ~2 MB of q/k/v projection (MEMORY_DESIGN section 2) = 8 GB. Not stock-sized.
- **Hybrid (recommended)**: keep the 32 KB DeltaNet RAM for the hot window (stack, activation vectors); put the **inner KV cache** (which only grows by one entry per token, ctx 1 here) in outer full-attention KV (MEMORY_DESIGN section 6). Large scratch (logit accumulators) is eliminated by streaming argmax in firmware (firmware change, not host).
- If a log is wanted: bound it (evict via a fixed-size ring: address = slot, equal to the existing table) which is exactly the DeltaNet table again.

## 5. Concrete design and step estimate

Goal: LOAD/STORE results come from the circuit's own RAM; remaining host-served items (DROM blob, WROM weights) are labelled separately.

1. **Measure first (0.5 d).** From ref_big trace: histogram of load addresses by region (RAM <32 KB window, RAM beyond, DROM, WROM), distinct live words, stores per word, reuse distance. Script sketch in section 8; needs the `inj_*.npy` or a re-run of ref_big (about 30 min on a box). Decides everything below.
2. **Firmware relink to a 32 KB map (2-3 d).** Streaming argmax (no VOCAB buffers), int8 activations in place, KV out of RAM. Confirms the working set can be at most 32 KB. If it cannot, step 3 widens RAM.
3. **RAM widening only if needed (1-2 d).** `RAM_WORDS` is a spec constant; each 32 KB costs 32 tables and is a core rebuild (the `RAM_WORDS>>8` loop) plus `fusedexec` retune. Target 64-128 KB.
4. **Remove the RAM injection (2 d).** Drop `in_ram := one`, restore real faults, restore `m.ram` into the circuit at chunk start (`core.snapshot` already reads `m.ram`, so pass the snapshot's `ram`), keep `inj` only for DROM loads (static data). Run the existing 60-chunk replay; expect "0 free mismatches" now **includes memory**. Cost: re-run ~2.5 h on A10G.
5. **DROM (the 624.6 MB blob)** is read-only and is the same problem as the weights. It needs the weight home of the audit (embedding-row ROM) or stays host-served (labelled).
6. **Chain chunks circuit-to-circuit (2 d)** after 4, so RAM carries.

Total about 8-11 working days, dominated by firmware shrinkage; the DeltaNet RAM needs no new circuit layers.

## 6. Prototype (measured): exact LWW lookup over an attention log

`experiments/circuit/ramown/lww_attn_log.py`, log `logs/lww_attn_log_mac_cpu.log` (small CPU job on the Mac, torch). One log entry per store; key = 13 address bits dual-rail + recency digits (base 128, exact in bf16); query weights M = 8N on address bits and 8 x 128^j on the digits; value = 32 data bits; softmax; sign of the weighted bits is the loaded word; checked against a dict of latest writes. Repeated addresses (256-word space with up to 16,384 stores = 64 writes/word):

| N stores | 256-word space, fp32 scores | same, scores rounded to bf16 |
|---|---|---|
| 64 | 57/57 | 63/67 |
| 1,024 | 252/252 | 231/251 |
| 4,096 | 256/256 | 169/256 |
| 16,384 | 256/256 | 77/256 |

(8192-word space: fp32 exact in all rows, 441/441 at N=16,384; bf16 421/444.) So **the technique works exactly with fp32 score accumulation, and fails in the bf16-rounded-score path as the log grows** (recency margin vanishes under 8-bit mantissa). Real Qwen3.5 attention computes scores in fp32 inside fused kernels but eager bf16 does not; the margin must be checked in the real HF layer, not claimed here. Not run on the A10G (box unreachable, section 8). Not tested: real merged layer, head_dim 256, GQA, RoPE.

## 7. Scope extension: the 596 M-nibble weight table as a qwenFS-style read-only log

Payload 595,984,384 nibbles = 297,992,192 B (`wrom.bin`) [M, audit].

**(a) Tokens.** 4 nibbles per token (16 bits, ids < 65,536 of the 248,320 vocab): **148,996,096 tokens** [E]. Max integer packing is 17 bits/token (131,072 ids), 4.25 nibbles, 140.2 M tokens; information-theoretic log2(248,320) = 17.92 bits gives 133.0 M tokens. Each token also needs an address key (position in the log).

**(b) KV per stored token, Qwen3.5-4B** [E from real config: 8 full layers, 4 KV heads, head_dim 256, bf16]:

| Log scope | per token | total (148,996,096 tokens) | vs A100-40GB |
|---|---|---|---|
| all 8 full layers | 32,768 B | **4.88 TB (4.44 TiB)** | 122x |
| one layer (qwenFS-style: only the copy layer holds the file) | 4,096 B | **610 GB** | 15.3x |
| one layer, one KV head x 256 (custom, non-stock geometry) | 1,024 B | 152.6 GB | 3.8x |
| qwenFS own max file capacity | n/a | 128 x 63 x 590 = 4.76 M tokens | the weights are **31x** larger than the whole qwenFS capacity |

**(c) Per-step attention cost** [E, memory-bound]: a lookup reads the whole layer's K once: 610 GB / 1.555 TB/s (A100-40GB HBM) = **0.39 s per lookup, 2.5 lookups/s**; best custom case 152.6 GB gives 98 ms (10/s). Compute 2 x N x d is small next to this. One inner token needs 74.5 M VDOT weight words (8 nibbles = 2 tokens each) = about 149 M token lookups, so about 5.8e7 s (1.8 years) per inner token at 0.39 s, or 1.5e7 s (170 days) in the custom case. Compare the current replay: 18.1 k tok/s end to end [M]. Even sharded over N GPUs the floor is bandwidth-per-GPU: 122 A100s still give 3.2 ms per lookup, about 5 days per inner token, and they are not available in any stock box. Approximate/segmented attention would not be exact softmax.

**(d) qwenFS's disk role.** `qfs/disk.py` stores paths and token ids on disk (3 bytes/token, about 447 MB for 149 M tokens), mounts tables/lookups, and **the runtime pages a segment's bytes in when the model emits that segment's read ack** ("like a disk controller"; the pager rebuilds copy rows with the model's own projections). The model chooses which segment; the host fetches and installs the bytes. For weights that is exactly the audit's "host streaming weights" (the `ty==2` WROM injection, labelled HOST DECODE): the weights are **host-served**, not circuit-owned. It does not count as circuit. Also 148,996,096 / 590 = 252,500 segments exceed the 63-segments x 128-files address space, so even the ack addressing does not fit.

**(e) Versus the audit's embedding-table option** [M `hold_weights/`]: 0.884 GB (6 bit/entry) to 1.327 GB (4 bit/entry) in the **parameters**, exact (0 symbol mismatches over all 595,984,384 nibbles on A10G), row fetch O(1) by token id, no growth, no host. Ratios: the log is 460x larger (610 GB vs 1.327 GB), costs O(N) per lookup instead of O(1), requires pre-filling 149 M positions by tokens (the token channel is 18 bits/position, MEMORY_DESIGN section 3), and in its useful form needs host paging. The log wins nothing; its only advantage, "no checkpoint change", is false since the data would have to be loaded as 4.9 TB of KV or paged by the host. Recommendation stands: embedding-row ROM, untied head, next experiment = row-fetch from a CPU register.

## 8. Honest limits and what is not done

- **A10G not used.** `ssh ubuntu@<gpu-box>` timed out (box stopped or address changed; ~/.ssh/config not re-checked). The prototype ran on the Mac CPU (tiny torch job, safe). Not measured: the load-address histogram by region (needs `inj_*.npy` / a re-run of `ref_big.py`), live word count, a run in the real HF layer. All "86 M loads" and "live words" figures are derived from the audit's injections-per-row, not from addresses.
- Static RAM figures use my reading of `qwen3_fx.c` array shapes with 0.6B config constants (DIM 1024, HID 3072, 28 layers, 8 KV heads x 128); the actual `cfg.h` used for the run was not re-opened. Treat 215 KB as +-30%.
- qwenFS code was only read. It is built on Qwen3.5-2B with its own tied-head exit code and custom KV cache (`disk.py` `attach/mount` override sequence length bookkeeping); nothing here claims it merges into 4B unchanged.
- "Host-assisted" in qwenFS: disk paging (runtime loads bytes on a read ack), the chat loop's sampling and token decode, and the monitor panel. The circuit part is parsing, lookup, table update, forced tokens.
