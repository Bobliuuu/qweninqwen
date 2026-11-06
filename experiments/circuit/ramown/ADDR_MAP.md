# ADDR_MAP: Qwen3-0.6B ctx-1 firmware in the core's real 32 KB RAM

All numbers **[M]** = measured on the `rvemu_mt` emulator (instrumented copy of `compiler/fw/rvemu.c`, logs in `ramown/logs/`) unless tagged **[E]** (estimate) or **[R]** (read from code).
Workload: the ref785b token (prompt `[785]`, w4, EXT off, 1 inner token, argmax 13). Machine: lambda-final (A100 host CPU); nothing ran on the Mac.

Files: `fw_qwen3_ram.c` (firmware), `fw_rom.ld` (real chipformer map), `fw_build.sh`, `fw_rebuild.sh` (everything from scratch), `fw_rvemu_mt.c` + `fw_addr_analysis.py` (trace + histogram + live words),
`fw_verify.py` (bit-exact check vs the original firmware), `fw_ref32.py` (ref.Machine with the real 8192-word RAM; replaces `ref_big.py`'s 4 MB RAM).

## 1. Baseline, measured (original `qwen3_fx.c`, flat map, `logs/addr_baseline_ctx1.txt`)

| quantity | measured |
|---|---|
| instructions / token | 157,063,952 (emulator count) |
| data LOAD instrs / STORE instrs | 11.76 M / 10.63 M (10,630,383 stores = the 10.63 M store records of `ref785b`) |
| of the loads, blob (DROM-region model data) | 684,416; rodata 2,519 (so ~94 % are RAM loads) |
| static `.bss` | 4,531,696 B (4.3 MB) |
| distinct RAM words ever touched | 357,643 words = 1.43 MB |
| ... of which LM-head buffers `accs_` + `tmp` (VOCAB int32 each) | 303,872 words (85 %) |
| ... everything else | 53,771 words = **210 KB** (the earlier hand estimate was 215 KB: within 2.5 %) |
| peak LIVE words (stored value still to be read), exact sweep | 152,222 words = 594.6 KB (LM head) |
| stack | 48 words |

Dominant buffers by traffic (loads+stores): `accs_` 9.3 M ld / 9.3 M st (all of it the VDOT accumulators), `xq`, `hg`, `tmp`, `xb`, `q32`, `qh`, `x`.
KV at ctx 1 is small in traffic (`kq`+`vt` = 86 K ld) but 142 KB of footprint (`vt` is padded to 4 tokens: 114,688 B).

## 2. What changed in the layout (all exact: bit-identical logits)

1. **LM head in tiles with a running argmax** (TILE=256 rows): no `tmp`/`accs_` of VOCAB words (saves 1.2 MB of bss, 303,872 words).
2. **Tiled matvec** (`mv_rows`): the VDOT kernel runs on a row tile of 256 rows (the weights are chunk-major, a row tile is a contiguous run inside each 64-column chunk). Accumulator buffer 1 KB instead of VOCAB*4.
3. **ctx==1 attention needs no K/V cache and no q/k** (`attn_single`): softmax over one key is exactly 1, so the integer pipeline reduces to `o[j] = ((w8*vq[j])*G)>>24` per KV group (same for the 2 heads of a group). q_proj, k_proj, q_norm, k_norm, RoPE, kv_store are dead and removed; `v32` is overwritten in place by the attention output. Gated by `MAXPOS==1`; the generic path (ctx>1) is kept and verified (section 4).
4. **Fused epilogues**: o_proj and down_proj accumulate straight into `x` (no `xb` round trip / `resid`); gate/up/swiglu per tile; int8 activations quantised in place (down_proj input reuses `hg`).
5. `ao`, `q32`, `k32`, `qh`, `kh`, `acc1/2`, `xql`, `tmp` gone; rodata (`SOFF`,`WBW`,`PLN`,`NORM_OFF`,`__clz_tab`, 3,144 B) is in the **real 128 KB DROM at 0x20000** (`fw_rom.ld` = chipformer `env/rom.ld`), code in the real ROM at 0x40000.

## 3. Address map (default build, `fw_build.sh CFG out.elf`)  [M, from `nm`]

| address | size | symbol | role | words |
|---|---|---|---|---|
| 0x00000100 | 1,024 B | `hut` | "up" tile / LM-head logit tile (256 words) | 256 |
| 0x00000500 | 1,024 B | `accs_` | VDOT accumulator tile | 256 |
| 0x00000900 | 1,088 B | `xqi` | int8 input of every DIM-wide matvec (1024 + 64 pad) | 272 |
| 0x00000d40 | 4,096 B | `v32` | V (1024 words), then attention output in place | 1024 |
| 0x00001d40 | 12,288 B | `hg` | swiglu result int32 (3072 words), later its bytes are the int8 input of o_proj/down_proj | 3072 |
| 0x00004d40 | 4,096 B | `xb` | rmsnorm output | 1024 |
| 0x00005d40 | 4,096 B | `x` | residual stream | 1024 |
| 0x00006d40 | 4 B | `vk_x` | VDOT kernel activation pointer | 1 |
| 0x00006d50 | 11 B | (compiler local) | | 3 |
| 0x00006d5b | | end of bss | `.bss` = 27,995 B | **6,999 words** |
| 0x00007ec0..0x00008000 | 320 B | stack (SP_INIT 0x8000) | deepest touched 0x7ec0 | 80 |
| 0x00020008.. | 3,144 B | `.rodata` in DROM | SOFF/WBW/PLN/NORM_OFF/clz tab | read-only, not RAM |
| 0x00040000.. | 8,068 B | `.text` | | ROM |
| 0x10000000.. | 624,576,256 B | model blob (embeddings, norm gains, per-row scales, rope) | static model data, still host-served (see 5) | not RAM |

**RW footprint = 27,995 B bss + 320 B stack (deepest touched 0x7ec0) = 28,315 B = 7,079 words of 8,192 (86.4 %).** 4,768 B lie between bss end (0x6d5b) and the stack top, 320 B of it used.
Measured over the whole run (`logs/addr_new785.txt`): 7,514 distinct words touched, 6,984 stored, 530 read-only (rodata); **peak live (stored values) 4,657 words = 18.2 KB**; no access outside [0x100, 0x8000) except blob/rodata loads.

LOAD/STORE histogram (new, per token, `logs/addr_new785.txt`): 10,695,804 load instrs = 10,226,428 RAM (95.6 %) + 469,376 blob (4.4 %) (+3,503 rodata loads, in DROM); 9,166,022 stores, all RAM. Traffic: `accs_` 7.94 M ld / 7.94 M st (86 % of stores), `hg` 545 K / 315 K, `hut` 238 K / 238 K, `xqi` 411 K ld, `x` 174 K / 58 K.
The 469,376 blob loads are exactly: 28 x (9,216 row scales + 2,048 norm gains) + 151,936 LM scales + 1,024 embedding row + 1,024 final-norm gain.

## 4. Verification  [M]  (`fw_verify.py`, `logs/verify_*.json`)

New vs original source on the **same cfg.h / blob / wrom**, both built with a logits dump, every one of the 151,936 logits per position compared:

| case | argmax old / new | logits identical | instr old | instr new | ratio |
|---|---|---|---|---|---|
| ctx 1, EXT off (real 32 KB map) | 13 / 13 | **yes (0 diffs)** | 157,063,956 | 126,952,443 | 0.808 |
| ctx 1, EXT on | 13 / 13 | **yes** | 153,533,256 | 123,965,267 | 0.807 |
| ctx 1, `-DTWOPASS` | 13 / 13 | **yes** | 157,063,956 | 172,400,901 | 1.098 |
| 4 tokens (generic path, flat map), default | [13,315,279,11] both | **yes** | 614,928,594 | 606,665,667 | 0.987 |
| 4 tokens generic, `-DTWOPASS` | same | yes | | 776,539,297 | 1.263 |
| 4 tokens generic, `-DTILE=128` | same | yes | | 616,509,946 | 1.003 |

(DUMP builds include the per-tile logit writes; the instruction counts of the non-DUMP production build are: 123,910,899 instr/token for the default, versus 157,063,952 baseline = **-21.1 %**.)
**Caveat on `res_full_fx_w4_e1_n4.json`:** that file lists argmax_fw `[13,315,279,374]`; the *current checked-in* `qwen3_fx.c` itself gives `[13,315,279,11]` for the same 4 prompt tokens (old == new above), so the json was produced by an older firmware revision. Position 0 (the inner token) matches (13) and the new firmware equals the current original bit for bit.
The instruction drop comes from removing q_proj/k_proj/q-norm/k-norm/RoPE/KV work that cannot influence a ctx-1 output; the generic path is unchanged in cost (+ about 0.2 % for tiling at TILE=128).

Full reference-machine run with the REAL 8192-word RAM (`fw_ref32.py`, faults if any RAM access leaves it): see section 7.

## 5. What is and is not circuit-owned after this change

- RAM LOAD/SB/SH: 95.6 % of loads and every store are in the 32 KB window; nothing needs a 4 MB alias or `in_ram := 1`. The circuit's own stores can answer them, **provided the RAM image is restored at chunk starts** (`fw_ref32.py` writes `ramimg_NNNN.npy`, 8192 words, plus the machine state).
- The remaining 469,376 loads (4.4 %) read the 624.6 MB blob (static model constants). They are host-served exactly as before and must **not** be injected into RAM slots: with the blob word index `& 0x1FFF` these would overwrite live RAM words. `inj_*.npy` from `fw_ref32.py` contains only these plus the VDOT rows (type 2); the circuit agent should route type-1 injections of address >= 0x10000000 to the DROM table state, not a `ram{h}` slot (the `slot` column is `(ea>>2)&0x1FFF`, kept for format compatibility only).
  Scales are 410 K of the 469 K; they could be folded into weights or re-derived (not done).
- Weights (VDOT rows) are unchanged: host-served WROM injection.

## 6. Minimum achievable and what bigger contexts need

| build | bss | + stack | RW words | % of 8,192 | instr/token (emulator) |
|---|---|---|---|---|---|
| baseline | 4,531,696 B | 192 B | 1,133,000 | 13,800 % | 157.06 M |
| default (`TILE=256`) | 27,995 B | 320 B | 7,079 | 86 % | 123.91 M |
| `-DTWOPASS` (gate/up/swiglu recomputed twice, int8 buffer instead of 12 KB `hg`) | 19,536 B | 320 B | 4,964 | 61 % | 169.36 M (+7.8 % vs baseline, +36.7 % vs default) |
| TWOPASS, measured peak live | | | 3,140 words | 38 % | |

Measured distinct stored words: 6,984 (default), 4,944 (TWOPASS). **Floor for this design [E]**: `x` 4 KB + `xqi` 1 KB + int8 buffer 3 KB + three 128-word tiles 1.5 KB + stack 0.3 KB, with `v32` recomputed per group and `xb` fused into the quantiser, about 10 KB (2,500 words); every further halving costs extra passes (V recompute, rmsnorm recompute), not implemented.
Tiles: the three tile buffers cost 3 x TILE x 4 B; `-DTILE=128` verified; kernel call overhead rises about 0.2 % per halving.

**Context > 1 does not fit any RAM of this family.** Per extra token the K/V cache is `kq` 28,672 B + `vt` 28,672 B (compact; the allocated `vt` is padded to 4 tokens = 114,688 B) + `kmx`/`vmx` 1,792 B = **59,136 B/token (57.75 KB)**, i.e. 1.8 x the whole 32 KB RAM per token. The generic path at ctx 1 (no q/k skipping) is 196 KB [E: default + q32 8K + k32 4K + ao 8K + qh 4K + kh 2K + kq 28K + vt 112K + 1.8K]. Next sizes: 256 KB RAM (8 x the current 32 tables) for generic ctx 1-4; the right answer for ctx > 1 is the KV in outer full-attention KV (docs/QWENFS_FOR_RAM.md section 4, MEMORY_DESIGN section 6), RAM kept at 32 KB.

## 7. Full reference-machine run (real 8192-word RAM)
See `logs/ref32_785/summary.json` (added when the run finishes).
