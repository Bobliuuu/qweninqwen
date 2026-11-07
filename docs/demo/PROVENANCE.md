# Provenance of every number in `demo.mp4` / `demo.gif`

`make_demo.py` reads each value below from the listed file at build time and asserts the key ones
(it refuses to build if the verdict, the 15846/13 answers, the halt step, 95 exact seams or 0/0/0 counters differ).
Paths are relative to the repo root unless they start with `docs/demo/data/` (small copies of files that live outside the repo in `~/qweninqwen-work/cfram_q_vast_out/`).
Nothing in the video is simulated; the only synthetic thing is the **animation order** of the chunk grid (chunk number order, because per-chunk completion wall-clock times were not logged).

Footer on every frame: *Visualization of a real run | one token at position 0 | parallel jobs seeded from the reference, verified at 95 exact seams*.

## Scene 1: title
| Shown | Source |
|---|---|
| "one Qwen3-0.6B token computed by a circuit living in the weights of Qwen3.5-4B" | project definition: `docs/CIRCUIT_ROUTE_PLAN.md` (stock Qwen3.5 outer, Qwen3-0.6B inner program), `docs/HOST_ROLE_AUDIT_CFRAM.md` |
| 96 chunk-jobs, 95 exact seams, 0 host injections | `experiments/circuit/fullrun/logs/cfram_q_vast/seams_0-95_20261107_004845.json` (`verdict`, `rows[*]`) |

## Scene 2: why the first answer was wrong
| Shown | Source |
|---|---|
| 28 bars: cosine of the final hidden state vs fp32 when ONLY layer k is quantised (int4 weights, int8 activations) | `experiments/circuit/fullrun/diag_quant/diag_quant2.json`, keys `W4+act ONLY layer 0..27`, field `cos_final` |
| red bars / wrong answers 838 (layer 2), 3405 (layer 26), 12 (layer 27) | same file, field `argmax` (a bar is red when `argmax != 15846`) |
| first firmware answer 13 `'.'`, rank of 15846 = 38,852 | `quantfix/results/ctrl_w4.json` (`argmax_fw`, `rank_15846_in_fw`); the `'.'` text is from `diag_quant/REPORT.md` title |
| HF fp32 answer 15846 `' Question'` | `diag_quant/diag_quant.json` (`hf_argmax`); token text from `logs/final_token/REPORT.md` and `diag_quant/REPORT.md` |
| "layer 2 writes a ~6541 channel" | `diag_quant/diag_quant.json`, `fp32_profile[2].dmax` = 6540.8 (rounded) |
| "the circuit reproduced this" | the CF+RAM circuit run before the fix reproduced the emulator's 13 (`quantfix/REPORT.md`, control row; `diag_quant/REPORT.md`). The pre-fix 13-token circuit run itself is described in the repo history, not re-plotted here |

All under `experiments/circuit/fullrun/`. Prompt = the single token 785 ("The").

## Scene 3: the fix
| Shown | Source |
|---|---|
| 5 variants, argmax, cos(logits, HF), rank of 15846, top-1 gap | `quantfix/results/{ctrl_w4,b8_apl1,w4_apl2only,b8_apl2,b12_apl2}.json` (`argmax_fw`, `cos_logits_fw_vs_hf`, `rank_15846_in_fw`, `gap_fw_top1_top2`) |
| "+17.1% instructions per token (161,511,713 -> 189,109,345)" | same JSONs, `instr[0]` of `ctrl_w4` and `b8_apl2`. **`quantfix/REPORT.md` says "+19%"** (it rounds 0.160G / 0.19G); the video uses the exact counts |
| "+7.9% weight words (74,498,048 -> 80,396,288)" | same JSONs, `blob.wrom_words` |
| "weights alone give 9, activations alone give 408" | `b8_apl1.json` (9), `w4_apl2only.json` (408) |

Caveat shown in the data: these are emulator runs of the real integer firmware at the compiler level (`quantfix/REPORT.md`), single prompt token.

## Scene 4: the run
| Shown | Source |
|---|---|
| 96 cells, 2,097,153 steps per chunk, chunk 95 = 57,707 steps | `seams_0-95_20261107_004845.json` `rows[*].steps_run` |
| cell color = machine | Modal for chunks 32-70: `docs/demo/data/FINAL_VERDICT.json` `modal_chunks`. Vast GPU for the other 57: `docs/demo/data/fleet_result.json` `done[chunk].gpu` (RTX 4090 / A100 PCIE / L40S). Modal GPU type was not recorded |
| counters: chunks verified, seams exact, free-token mismatches, forced tokens, host injections | running sums over `rows[*]` (`clean`, `seam.exact`, `free_mismatches`, `forced_tokens`, `injections`); final values 96/96, 95/95, 0, 0, 0 |
| per-machine median minutes (RTX 4090 5.5, A100 PCIE 9.2, L40S 6.1, Modal 9.2) | median of `rows[*].seconds` per group, chunks 0-94 (chunk 95 is the 15 s HALT chunk and is excluded; stated on screen). Note: the fleet manager's own status lines quote 6.0 min for 4090s and 4.9-9.7 for A100s at different moments; the video uses the final per-chunk values |
| 199,287,253 steps, 11.4 GPU-hours | sum of `rows[*].steps_run` and sum of `rows[*].seconds` (41,143 s) = job time, **not billed time or cost** |

## Scene 5: the last chunk
| Shown | Source |
|---|---|
| last 24 circuit tokens, HALT | `seams_0-95_...json` `rows[95].readout.tail_names_circuit` |
| steps 57,642..57,706, halt_step 57,705 | `rows[95].readout.from_step/to_step/circuit_halt_step` |
| answer bytes [230, 61, 0, 0] -> 230 + 61 x 256 = 15846 | `rows[95].readout.circuit_bytes`, `circuit_word` (arithmetic done on screen) |
| "tail equals the reference trace: True", 0/0/0 | `rows[95].readout.tail_equals_reference`, `free_mismatches`, `forced_tokens`, `injections` |
| collector verdict line | `seams_0-95_...json` `verdict` (the trailing "0s to collect" is trimmed) |
| 15846 = ' Question' = Qwen3-0.6B's own answer | `logs/final_token/REPORT.md`, `diag_quant/diag_quant.json` |

## Scene 6: host roles and limits
| Shown | Source |
|---|---|
| 0 injections / 0 forced / 0 mismatches in all 96 chunks | `seams_0-95_...json` |
| table 168,684 rows, hidden 7680, HF-module check chunk 0: 130,000 steps, 0 mismatches, 2,320 ROW tokens; table+1 control diverges at step 41 | `quantfix/results/hfcheck_cfq_pos_130k.json` (`table_rows`, `steps_checked`, `n_mismatch`, `n_row_tokens_checked`), `quantfix/results/hfcheck_cfq_neg.json` (`first_mismatch.step`) |
| "the 96 jobs loaded a bit-identical decoded copy" | `docs/demo/data/verify_ckpt_vs_npy.log` (checkpoint table vs `rowtab_cf.npy`, identical, md5 5072e525...) for the Vast jobs; shape 168,684 x 3,840 is the final-contract table; see Open points |
| chunk 0 boots from the program alone; real in_ram/in_drom decode | `docs/HOST_ROLE_AUDIT_CFRAM.md` rows 8, 5; `ramown/logs/realdecode/`, `logs/cfram_reset0/` |
| allowed by the plan: boot, GO token, embed-and-append loop, ecall I/O; reference comparison is the test | `docs/HOST_ROLE_AUDIT_CFRAM.md` rows 10-12, 17 |
| limits: ctx=1, parallel seeded jobs, fused Triton executor, bigger checkpoint, 4-bit | `docs/HOST_ROLE_AUDIT_CFRAM.md` rows 5, 9, 15, 16; seam count from `rows[*].seam.exact` |
| "199M steps full HF check infeasible" | 199,287,253 steps (above) at the measured HF-module rate of about 155 steps/s (agent report for `hfwf/hfcheck_sampled.py`; that sampled check was stopped partway on the old firmware and is not claimed) |

## Scene 7: end card
`seams_0-95_20261107_004845.json` (`verdict`, `ok`), `docs/demo/data/FINAL_VERDICT.json`.

## Open points (not sourced from a data file, said plainly)
- The `hfcheck` result is for chunk 0 only. The 'same table in the checkpoint' statement for the final run rests on `verify_ckpt_vs_npy.log`, which compares the checkpoint table (168,684 x 3,840) with `rowtab_cf.npy`; the log's timestamp (23:14 on 2026-11-06) is from the Vast agent's check, run before the final jobs.
- The per-chunk wall-clock completion times were not logged, so the grid does not claim an order of completion.
- 'Real Qwen3-0.6B says 15846' is HF fp32/bf16 for this one prompt token ("The" at position 0); not a statement about other prompts.
