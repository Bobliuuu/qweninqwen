# Provenance of the README visuals

Every number, bar and frame in `docs/assets/circuit_*` is read from a file in this repository or from the run's reference trace. Anything schematic is marked **schematic** here and in the image. Nothing is simulated. Paths are relative to the repo root; `FR` = `experiments/circuit/fullrun`.

Rebuild: `python3 docs/assets/viz/make_static.py` (SVG/PNG) and `python3 docs/assets/viz/make_anim.py` (GIF/MP4); needs Pillow, numpy, ffmpeg and `rsvg-convert`. The trace reduction (`viz/prep_data.py`, output `viz/one_token_data.json`) needs the 96 reference token files:
`modal volume get qfr-data cfram_q/refQ/tok_NNNN.npy <dir>` for NNNN = 0000..0095 and `modal volume get qfr-data cfram_q/cf/cf_layout.json`, then `python3 docs/assets/viz/prep_data.py <dir> cf_layout.json docs/assets/viz/one_token_data.json`.

## Shared inputs
| File | Used for |
|---|---|
| `FR/logs/cfram_q_vast/seams_0-95_20261107_004845.json` | per-chunk steps, job seconds, mismatches / forced / injections, seam exactness, the chunk-95 read-out (`rows[95].readout`: halt step 57,705, bytes `[230, 61, 0, 0]`, word 15846) |
| `docs/demo/data/FINAL_VERDICT.json`, `docs/demo/data/fleet_result.json` | which chunks ran on Modal (32 to 70) and which Vast GPU ran each other chunk |
| `qfr-data:cfram_q/refQ/tok_0000..0095.npy` (via `viz/one_token_data.json`) | the real reference trace: total 199,287,158 trace tokens (equals `summary.json` `trace_tokens`), token families, ROW ids, chunk 0 composition, an 18-token window, the chunk-95 tail |
| `qfr-data:cfram_q/cf/cf_layout.json` | row-id layout: head rows 0 to 2; per layer l, `layer_weight_rows[l]` weight rows (4,096; 8,192 for layers 2, 26, 27) followed by 31 constant rows; 3 final-norm rows; LM-head weight rows from `LM_ROW0` = 127,850 |
| `qfr-data:cfram_q/refQ/summary.json` | 178,066,316 instructions, 199,287,158 trace tokens, argmax 15846 |

**Derived, not logged by the run:** the layer a `ROW(r)` token belongs to is computed from the row id with `cf_layout.json` (a row in layer l's block means the circuit is fetching layer l's weights or constants). The share of the trace per layer is the span between the first `ROW` token of one layer and the first of the next. The "~3.1x" is layer 2's span (13,319,745 tokens) over the median plain-layer span (about 4.3M); layers 26 and 27 are about the same as layer 2.

## Visuals
| Asset | Real data | Schematic / animation-only |
|---|---|---|
| `circuit_architecture.svg/.png` | 12 chip layers + Qwen's 32, hidden 7680, vocab 680,128, 168,684 rows x 3,840 cells, 2.59 GB (`quantfix/STEP2_REPORT.md`); 199,287,158 tokens / 178,066,316 instructions (`summary.json`); layers 2/26/27 at 8-bit weights + 16-bit activations (`quantfix/REPORT.md`); 15846 = `' Question'` (`logs/final_token/REPORT.md`); 0/0/0 and 96 chunks, 95 seams (seams JSON) | **schematic**: box sizes, positions and the arrows |
| `circuit_hero.gif/.mp4` | layer segment widths (real share of the trace), 199,287,158 counter end value, 168,684 rows, 12 + 32 layers, the token chips in the stream (real names from chunk 0 steps 98,170 to 98,187), 15846 | **schematic layout**; the counter counting up is the animation (marked "sped up"); highlighting of circuit blocks and table rows is decorative |
| `circuit_one_token.gif/.mp4` | ch 1 total, 96 chunks of 2,097,152 steps, layer widths; ch 2 chunk 0 composition in 160 slices (`one_token_data.json` `bins0`, families V/F0/F1/F2/BYTE vs ROW/WB vs the rest); ch 3 the 18 real tokens at chunk 0 steps 98,170 to 98,187 including `WB(31579)` and `ROW(822)` (822 is a layer-0 weight row: ids 3 to 4,098); ch 4 per-layer spans, LM head = 26% of the run; ch 5 the last 14 tokens of chunk 95 (halt at step 57,705 of 57,706 tokens), bytes `[230, 61, 0, 0]` = 15846, `' Question'` | the zoom transitions and reveal speeds; bracket captions are explanations, not logged events |
| `circuit_seam_grid.gif/.mp4` | per-chunk job seconds (5.4 to 9.2 min; chunk 95 is 15 s), seam exactness (95/95), counters 96/96, 0/0/0 | **replay**: all 96 jobs are drawn as if started together, each cell filling for its real job time. The real run was staggered across Modal and Vast, and per-chunk completion wall-clock times were not logged. The snake layout is only a layout |
| `circuit_quantfix.gif/.mp4` | `FR/diag_quant/diag_quant2.json` (`W4+act ONLY layer k`: `cos_final`, `argmax` for k = 0..27); `FR/quantfix/results/{ctrl_w4,b8_apl1,w4_apl2only,b8_apl2,b12_apl2}.json` (`argmax_fw`, `instr`); HF argmax from `diag_quant.json`; "+17.1% instructions" = 189,109,345 / 161,511,713 - 1 | none (note: these are emulator runs of the real integer firmware at the compiler level, one prompt token; `quantfix/REPORT.md` rounds the instruction delta to +19%, the exact counts give +17.1%) |
| `circuit_tiers.svg/.png` | instructions per token: T0 1.10e11 (measured), T1 4.0e9, T2 5.4e8 (matvec only), T3 1.05e8 (matvec only) (projected), T3m 1.346e8 (measured), all from `docs/TIERS.md`; Final 1.78066316e8 (`summary.json`, measured) | the log axis starting at about 3e7 is a drawing choice; instruction counts differ by firmware (see TIERS.md derivations) |
| `circuit_heatstrip.svg/.png` | per-chunk job minutes and machine (seams JSON, `FINAL_VERDICT.json`, `fleet_result.json`); medians per machine over chunks 0 to 94 | job time, **not billed time**; colour scale spans the min and max of chunks 0 to 94 |
| `circuit_metrics.svg/.png` | sums over the seams JSON (96/96 clean, 95/95 exact, 0/0/0, 11.4 h = sum of job seconds); 168,684 rows; 2.59 GB; 6.0 min per chunk on an RTX 4090 and about $0.05 (fleet log, `FR/logs/cfram_q_vast/status.txt`); 57,705 (halt step) | "about $10" is the fleet-log estimate for the Vast GPUs, **not billing**, and excludes Modal A100 time for chunks 32 to 70 |
| `circuit_fleet.svg/.png` | `FR/logs/cfram_q_vast/status.txt`: chunks done and boxes up over time; restart and vanish events from the same file | counts only the Vast fleet manager's own bookkeeping; chunks 32 to 70 (Modal) were imported at the end and are not in the curve; the cause of the vanished instances is unknown |

## Limits shown in the visuals
One prompt token at position 0 (ctx 1 firmware). The 96 jobs were seeded from the reference start state and checked for exact end-state equality at all 95 seams. The executor is the fused Triton kernel; the stock-module check covers chunk 0 (130,000 steps, with a table+1 negative control). The outer checkpoint carries the 2.59 GB table.
