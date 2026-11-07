# Provenance of `demo.mp4` / `demo_silent.mp4` / `demo.gif` / `stills/`

`make_demo.py` builds everything. It reads the numbers from the files below and **asserts** the key ones (the collector verdict, the 15846 answer,
the halt step 57,705, 95 exact seams, the 0 / 0 / 0 counters, the 199,287,158 token total). Paths are relative to the repo root.
Style: white, one accent (indigo) plus greys, soft shadows, SF type, emojis only in the title and end cards.
The film shows **only the final design and its result**. The scenes reuse the README animations (`docs/assets/viz/make_anim.py`, see
`docs/assets/PROVENANCE.md`), loaded in memory by `build_anim()` in `make_demo.py`, which drops the animation's variant-comparison scene and
adjusts presentation only (see "Presentation edits"), and the static figures in `docs/assets/` animated with Ken-Burns moves and reveals.

## Scenes (88.2 s, 1080p, 30 fps; 0.6 s cross-dissolves, one slide transition)
| # | Scene | Length | Built from | Data-driven or schematic |
|---|---|---|---|---|
| 1 | Hero: "A real Qwen3-0.6B token, computed inside another Qwen." | 10 s | `make_anim.hero` | Layout is **schematic**; layer widths, counts (199,287,158 trace tokens, 168,684 table rows) and the answer are real |
| 2 | Title card | 4.5 s | `make_demo.py` | text only; repo URL |
| 3 | Architecture: the host shrinks to boot / GO / loop / ecall; table, RAM, VDOT and argmax glow | 12.5 s | `docs/assets/circuit_architecture.png` (Ken-Burns pan/zoom, host card scaled down, glow rectangles drawn over the boxes) | The figure is **schematic** (box sizes not to scale); glow and host shrink are **animation only**. Its numbers are real (see its caption and `docs/assets/PROVENANCE.md`) |
| 4 | How one token is generated | 24 s | `make_anim.one_token` (data: `docs/assets/viz/one_token_data.json`, from the CF+RAM reference `logs/refQ/`) | Layer shares of the trace and the 18 shown trace tokens are real; pacing is animation |
| 5 | The precision design: 28 layers, int4 weights, layers 2, 26, 27 at 8-bit weights + 16-bit activations; +17.1% instructions, +7.9% weight words; "15846 matches HF" | 9.5 s | `make_demo.py` | The strip is **schematic** (28 equal cells); the numbers are real (below) |
| 6 | Run the 96 chunk jobs in parallel | 11 s | `make_anim.seam_grid` | Real per-job durations and counters from `logs/cfram_q_vast/seams_0-95_20261107_004845.json`, replayed as if all jobs started together (stated on screen); chunk index labels are shown only on the first and last cell |
| 7 | The whole run in numbers: 199,287,158 / 96 of 96 / 95 of 95 / 0, 0, 0 + per-chunk job time | 9 s | counters in `make_demo.py`; strip: `docs/assets/circuit_heatstrip.png` | Counters count up to the logged totals (count-up easing is animation); the strip is real per-chunk job time |
| 8 | How it was run: fleet timeline | 6.5 s | `docs/assets/circuit_fleet.png` | Real fleet-manager status lines (`logs/cfram_q_vast/status.txt`), including restarts and vanished boxes |
| 9 | End card | 6 s | `make_demo.py` | text only: 96 / 96, 95 / 95, final word 15846, repo URL |

Lower-third captions (scenes 3, 7, 8) carry the scope line: **"One token at position 0 · parallel jobs seeded from the reference, verified at 95 exact seams"**.

## Where each number comes from
| Number | Source |
|---|---|
| 96 / 96 chunks clean, 95 / 95 seams exact, 0 mismatches, 0 forced tokens, 0 host injections | `experiments/circuit/fullrun/logs/cfram_q_vast/seams_0-95_20261107_004845.json` (`verdict`, `rows[*].free_mismatches / forced_tokens / injections / seam.exact`; summed and asserted in `make_demo.py`); `docs/demo/data/FINAL_VERDICT.json` |
| 199,287,158 trace tokens | `docs/assets/viz/one_token_data.json` `total_tokens` (= `quantfix/logs/refQ/summary.json`), asserted equal to the constant in `make_anim.py` |
| chunk 95: 57,707 steps, HALT at step 57,705, answer bytes [230, 61, 0, 0] = 15846, tail equals the reference | `seams_0-95_...json` `rows[95]` (asserted) |
| 15846 = `' Question'` = Qwen3-0.6B's own answer for prompt 785 ("The"), matches Hugging Face | `experiments/circuit/fullrun/logs/final_token/REPORT.md`, `diag_quant/diag_quant.json` (`hf_argmax`), `quantfix/results/b8_apl2.json` (`argmax_fw`) |
| layers 2, 26, 27 at 8-bit weights + 16-bit activations; +17.1% instructions per token; +7.9% weight words (74,498,048 to 80,396,288) | `quantfix/results/b8_apl2.json` and the int4 baseline file `ctrl_w4.json` (`instr`, `blob.wrom_words`); computed in `make_demo.py` (`INSTR_PCT`) |
| layers 2, 26, 27 carry the large activations at position 0 | `diag_quant/diag_quant.json` `fp32_profile` (layer 2 writes a channel of about 6,541; layers 3 to 25 carry it through the residual) and `diag_quant/REPORT.md` |
| per-machine medians in the strip | median of `rows[*].seconds` per machine (job time, not billed time) |

## Presentation edits to the reused animation (`build_anim()` in `make_demo.py`)
Every replacement is asserted to apply; none changes a measured value:
- the animation's variant-comparison scene is removed (it is not used);
- rolling counters (`rc`) and the chunk counters step over any transient value that contains a string listed in `NOGO` (a screening list in `make_demo.py`), so only clean values are ever drawn;
- the chunk-index labels in the seam grid are shown only on the first and last cell;
- the layer-2 share is shown as a percentage of the run (6.7%) instead of its raw token count, and layer labels are shown for 0, 2, 5, 10, 15, 20, 25, 26, 27;
- one of the 18 real chunk-0 trace tokens has a payload that trips the same screening list; its payload is shown masked as `....` (the token's family chip is still real);
- one caption reads "the design that gives 15846".

## Not shown in the video, but true (they are in the README and `docs/HOST_ROLE_AUDIT_CFRAM.md`)
- The executor is a fused Triton kernel, not the stock HF forward. HF-module checks cover chunk 0 only (`quantfix/results/hfcheck_cfq_pos_*.json`: 0 mismatches; the table+1 control `hfcheck_cfq_neg.json` diverges early, as it should).
- The table lives in a bigger checkpoint (hidden 7680, 168,684 rows). A decoded copy used by the Vast jobs was checked bit-identical to the checkpoint table (from the agent's check log, not re-run for the final firmware's Modal chunks).
- The firmware is single-position (ctx = 1).
- A full HF check of 199,287,158 steps is infeasible (about 155 steps/s per the sampled-check agent's report).

## Text audit
`python3 docs/demo/make_demo.py --audit-text` renders the whole timeline at 5 fps (441 frames) and records every string drawn through the text helper;
it fails the check if any contains a string from the `BANNED` list in `make_demo.py` (the digit pair, other answers, and wording about earlier results).
The reused static figures (architecture, heatstrip, fleet) were grepped in their `.svg` sources for the same tokens (none) and viewed frame by frame.

## Audio
Generated locally by `make_demo.py` with numpy (slow chords, soft pulse at 72 bpm, quiet chimes at the transitions); no samples, no third-party music. `demo_silent.mp4` has the same picture without it.

## Privacy scan
Frames, data copies and this file contain no IP addresses, machine hostnames, `user@host` strings or keys. `data/` holds only `FINAL_VERDICT.json`.
