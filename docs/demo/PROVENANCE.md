# Provenance of `demo.mp4` / `demo_silent.mp4` / `demo.gif` / `stills/`

`make_demo.py` builds everything. It reads the numbers from the files below and **asserts** the key ones (it refuses to build if the
verdict, the 13 / 15846 answers, the halt step 57,705, 95 exact seams, the 0 / 0 / 0 counters or the 199,287,158 token total differ).
Paths are relative to the repo root. Style: white, one accent (indigo) plus greys, soft shadows, SF type, emojis only in the title/end cards.
The scenes reuse the README animations (`docs/assets/viz/make_anim.py`, see `docs/assets/PROVENANCE.md`), re-rendered at their native 1920x1080,
and the static figures in `docs/assets/` animated with Ken-Burns moves and reveals.

## Scenes (96.6 s, 1080p, 30 fps; 0.6 s cross-dissolves, two slide transitions)
| # | Scene | Length | Built from | Data-driven or schematic |
|---|---|---|---|---|
| 1 | Hero: "A real Qwen3-0.6B token, computed inside another Qwen." | 10 s | `make_anim.hero` | Layout is **schematic**; layer widths, counts (199,287,158 trace tokens, 168,684 table rows) and the answer are real |
| 2 | Title card | 4.5 s | `make_demo.py` | text only; repo URL |
| 3 | Architecture: the host shrinks to boot / GO / loop / ecall while table, RAM, VDOT and argmax glow | 12.5 s | `docs/assets/circuit_architecture.png` (Ken-Burns pan/zoom, host card scaled down, glow rectangles drawn over the boxes) | The figure is **schematic** (box sizes not to scale); the glow and the host shrink are **animation only**. The numbers in the figure are real (see its caption and `docs/assets/PROVENANCE.md`) |
| 4 | How one token is generated | 24 s | `make_anim.one_token` (data: `docs/assets/viz/one_token_data.json`, from the CF+RAM reference `logs/refQ/`) | Layer shares of the trace and the 18 shown trace tokens are real (chunk 0 steps 98,170 to 98,187 etc., see the asset provenance); the pacing is animation |
| 5 | Why the first answer was 13, and the fix | 13 s | `make_anim.quantfix` | Real: `diag_quant/diag_quant2.json` (cosine and answer per quantised layer), `quantfix/results/*.json` (answers 15846 / 13 / 9 / 408, +17.1% instructions) |
| 6 | The answer rolls 13 to 15846 | 5.5 s | `make_demo.py` | **Animation**: the rolling number is a count between two measured values (13 = `quantfix/results/ctrl_w4.json`, 15846 = `b8_apl2.json` = HF fp32 in `diag_quant/diag_quant.json`); the video says so on screen |
| 7 | Run the 96 chunk jobs in parallel | 11 s | `make_anim.seam_grid` | Real per-job durations and counters from `logs/cfram_q_vast/seams_0-95_20261107_004845.json`, replayed as if all jobs started together (it says so on screen) |
| 8 | The whole run in numbers: 199,287,158 / 96 of 96 / 95 of 95 / 0, 0, 0 + per-chunk job time | 9 s | counters in `make_demo.py`; strip: `docs/assets/circuit_heatstrip.png` | Counters count up to the logged totals (count-up easing is animation); the strip is real per-chunk job time |
| 9 | How it was run: fleet timeline | 6.5 s | `docs/assets/circuit_fleet.png` | Real fleet-manager status lines (`logs/cfram_q_vast/status.txt`), including the restarts and vanished boxes |
| 10 | End card | 6 s | `make_demo.py` | text only: 96 / 96, 95 / 95, final word 15846, repo URL |

Lower-third captions (scenes 3, 8, 9) carry the scope line: **"One token at position 0 · parallel jobs seeded from the reference, verified at 95 exact seams"**.

## Where each number comes from
| Number | Source |
|---|---|
| 96 / 96 chunks clean, 95 / 95 seams exact, 0 mismatches, 0 forced tokens, 0 host injections | `experiments/circuit/fullrun/logs/cfram_q_vast/seams_0-95_20261107_004845.json` (`verdict`, `rows[*].free_mismatches / forced_tokens / injections / seam.exact`; summed and asserted in `make_demo.py`); `docs/demo/data/FINAL_VERDICT.json` |
| 199,287,158 trace tokens | `docs/assets/viz/one_token_data.json` `total_tokens` (= `quantfix/logs/refQ/summary.json`), asserted equal to the constant in `make_anim.py` |
| chunk 95: 57,707 steps, HALT at step 57,705, answer bytes [230, 61, 0, 0] = 15846, tail equals the reference | `seams_0-95_...json` `rows[95]` (asserted) |
| 15846 = `' Question'` = Qwen3-0.6B's own answer for prompt 785 ("The") | `experiments/circuit/fullrun/logs/final_token/REPORT.md`, `diag_quant/diag_quant.json` (`hf_argmax`) |
| 13 = the first (int4) firmware's answer; 9 / 408 for one fix knob alone; +17.1% instructions per token | `quantfix/results/{ctrl_w4,b8_apl1,w4_apl2only,b8_apl2}.json` (exact counts 161,511,713 and 189,109,345; `quantfix/REPORT.md` rounds this to +19%) |
| per-layer cosine, answers 838 / 3405 / 12 at layers 2 / 26 / 27 | `diag_quant/diag_quant2.json` |
| per-machine medians in the strip | median of `rows[*].seconds` per machine (job time, not billed time) |

## Not shown in the video, but true (they are in the README and `docs/HOST_ROLE_AUDIT_CFRAM.md`)
- The executor is a fused Triton kernel, not the stock HF forward. HF-module checks cover chunk 0 only (`quantfix/results/hfcheck_cfq_pos_130k.json`: 130,000 steps, 0 mismatches, 2,320 ROW tokens; the table+1 control `hfcheck_cfq_neg.json` diverges at step 41).
- The table lives in a bigger checkpoint (hidden 7680, 168,684 rows). A decoded copy used by the Vast jobs was checked bit-identical to the checkpoint table (md5 `5072e52537e840ae8541941a85721360`, from the agent's check log, not re-run for the final firmware's Modal chunks).
- The firmware is single-position (ctx = 1).
- A full HF check of 199,287,158 steps is infeasible (about 155 steps/s per the sampled-check agent's report).

## Audio
Generated locally by `make_demo.py` with numpy (slow chords, soft pulse at 72 bpm, quiet chimes at the transitions); no samples, no third-party music. `demo_silent.mp4` has the same picture without it.

## Privacy scan
Frames, data copies and this file contain no IP addresses, machine hostnames, `user@host` strings or keys. `data/` holds only `FINAL_VERDICT.json`.
