# Launch media: provenance ledger

Every number or frame shown in the launch media, and where it comes from. Rule: no row, no claim. All measured numbers are read by `make_media.py` from the JSON files below at render time (not typed in), except where marked "text constant".

Executors: all numbers are from the real Qwen3-0.6B run in a stock `Qwen3_5ForCausalLM` (fp32, layer-major executor, torch reference GDN, shared A100, Lambda box) unless marked otherwise. The videos are **replays**: no frame is a live recording; the demo is a time-lapse of a ~270 s per position run.

## 1. Source data

| Id | File | Content |
|---|---|---|
| D1 | `experiments/vslice/logs/res_vs2_real151936_L28_tok1.json` | headline single-token run, step 0: sec 268.62, build_s 13.88, run_s 241.94, maxrel 5.805e-5, outer_argmax 15846, hf_argmax 15846, frames 293,068 (same run printed in `real_L28.log` line 50) |
| D2 | `experiments/vslice/logs/res_vs2_real151936_L28.json` | 6-position teacher-forced run: tokens [785, 6722, 315, 9625, 374, 12095, 13], per step sec/build_s/run_s/maxrel/argmax; layers 1006, D 40775 (log: `real_L28_gen.log` lines 8, 11, 14, 17, 20, 23) |
| D3 | `experiments/vslice/logs/real_L28.log` line 49 / `real_L28_gen.log` line 7 | rails line `[-0.0, 1.0, 1.0, -0.0, 1.0, -0.0, -0.0, 1.0]`, top code score 18.000669 (step 0) |
| D4 | `experiments/vslice/FULL_TOKEN_RESULT.md` | planner counts: 291,072 write + 1,996 control = 293,068 frames; 595,984,384 weights; prompt text |
| D5 | README "Cost model" and `docs/make_figures.py` (`cost_model`, circuit rows) | circuit route: 494e6 MACs / 14.71 MAC/s = 3.4e7 s. **ESTIMATE (extrapolation), matvecs only, not a run.** Base data `experiments/bench/circuit_baseline/` |
| D6 | README tier table / "Status by tier" | tier definitions and statuses (T0-T4) with evidence links |

## 2. Numbers by value

| Shown value | Where shown | Source |
|---|---|---|
| 15846 == 15846 | hero, tournament, result, demo row 0 | D1 (`outer_argmax`, `hf_argmax`), also D2 step 0 |
| 5.8e-5 | hero, result, demo | D1 `maxrel` 5.805e-5 |
| 268.6 s | hero, result, cost, square cut footer | D1 `sec` 268.62 |
| 13.9 s build + 242 s run | cost chart | D1 `build_s`, `run_s` |
| 6 / 6 positions | hero, result, demo | D2: all six `outer_argmax == hf_argmax` (asserted in make_media.py at render time) |
| argmax per position 15846, 315, 279, 374, 12095, 13 | demo table, result | D2 `outer_argmax` |
| input ids 785, 6722, 315, 9625, 374, 12095 | demo table | D2 `tokens[0..5]` |
| errors 5.8e-5, 6.9e-6, 1.8e-5, 1.4e-5, 3.2e-5, 1.05e-5 (printed as 5.8e-5, 6.9e-6, 1.8e-5, 1.4e-5, 3.2e-5, 1.1e-5) | demo table, results table | D2 `maxrel` |
| times 300.7, 268.4, 264.6, 279.0, 274.8, 272.3 s | demo table, cost chart | D2 `sec` (position 0 includes 26 s build in this run, hence it differs from D1's 268.6 s) |
| 6.9e-6 to 5.8e-5; 265 to 301 s | result, demo final | min/max of D2 |
| 1006 outer layers, D = 40775 | title, idea, pipeline, hero | D2 `layers`, `D` |
| 293,068 frames (291,072 + 1,996) | pipeline, demo intro | D2 `frames`, D4 |
| 28 layers, d=1024, vocab 151,936, 594 blocks of 256 | idea, tournament | README architecture; FULL_TOKEN_RESULT.md; 594*256 = 152,064 >= 151,936 (text constants) |
| 18-bit code = 10 block bits + 8 local bits, 36 rails, score 18.0007 | tournament | README "full-vocabulary LM head"; D3 (18.000669) |
| block 61, local 230 | tournament | **derived**: 15846 = 61*256 + 230 (token id and block size 256). Not read from a log. |
| first 8 rails 0 1 1 0 1 0 0 1 | tournament | D3 (the log prints 8 values; -0.0 shown as 0) |
| 595,984,384 weights | state memory footer | D4 |
| 3.4e7 s circuit route | cost chart (red bar, labelled ESTIMATE) | D5 |
| ~4.5 minutes per token | result caveat, tweet | 268.6 s (D1) / 4.5 min = 270 s |
| tier names/statuses | ladder, climb GIF, flow_tiers, explainer tier scene | D6; pending rows from `STATUS` in make_media.py |
| "in progress" for free-running, HostGuard, speedup | everywhere | no result files exist yet for the real driver; `RUNS.md` / `PERF_NOTES.md` show speedup runs but they are not claimed |

## 3. Frame and clip ledger

| File | Scene | Type | Numbers from |
|---|---|---|---|
| `assets/hero.gif`, `hero_still.png` | slivers sweep, then result | **illustration** (70 slivers stand for 1006 layers) | D1, D2 |
| `assets/idea.gif` | inner/outer boxes | illustration | D2, D4 |
| `assets/state_memory.gif`, `state_memory_toy.png` | 4x5 matrix, write then read | **illustration with random toy numbers** (seeded RNG 7); the matvec shown is computed with numpy in the script | none measured |
| `assets/pipeline.gif`, `pipeline_flow.png` | op blocks lit in turn; layer counter | illustration; the counter is animation progress, not a measurement | counts from D2 |
| `assets/tournament.gif`, `tournament.png` | 594 bars, winner, code | **illustration** (random bar heights, seeded RNG 3; winner bar forced to be tallest) | D1, D3, derived block/local |
| `assets/tiers_climb.gif`, `tiers_ladder.png`, `flow_tiers.png/.mmd` | tier staircase | schematic of status | D6, STATUS |
| `assets/flow_architecture.*`, `flow_token.*` | block diagrams | schematic | README architecture |
| `assets/results_table.png` | 6-row table | **data** | D2 |
| `assets/cost_chart.png` | per-position bars + log comparison | data (blue/yellow); circuit bar is an estimate | D1, D2, D5 |
| `assets/architecture.png`, `state_memory.png`, `unrolled_pipeline.png`, `lm_head.png`, `cost_model.png`, `comparison.png` | earlier hand-drawn figures, converted from SVG | schematics; numbers as quoted in README | `docs/make_figures.py` |
| `video/explainer_16x9.mp4`, `explainer_square.mp4` | scenes: title (4 s), idea (9), state memory (12), pipeline (11), tournament (12), tier ladder (9), verified + caveats (9) | illustrations except the final scene, which is data | as above |
| `video/demo_16x9.mp4`, `demo_square.mp4` | intro (6 s), positions 0-5 (5 s each; progress bar is a time-lapse of the recorded `sec`), result (8 s) | **replay of recorded results, not live** | D2 |

The prompt words (The, capital, of, France, is) and their ids 785, 6722, 315, 9625, 374 are from `real_L28_gen.log` line 2 and the prompt in `FULL_TOKEN_RESULT.md`. Token ids beyond the prompt are shown as ids only (no decoded text is claimed).

## 4. Not shown, not claimed

Free-running generation vs HF greedy; HostGuard audit of the real driver; any speedup result (see `PERF_NOTES.md`, `RUNS.md`); anything from the toy beyond its label in the README.

## 5. Reproduce

`python docs/make_figures.py` (static PNG, needs rsvg-convert) and `python docs/history/launch/make_media.py` (PNG, GIF, MP4; Pillow, numpy, matplotlib, ffmpeg). The rendering host was a Mac; no GPU is needed.
