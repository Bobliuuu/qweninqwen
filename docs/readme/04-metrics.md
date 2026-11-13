<p align="center">
  <img src="docs/assets/circuit_metrics.svg" alt="The run in numbers: final word 15846, 96 of 96 jobs clean, 95 of 95 seams exact, 0 mismatches, 0 forced tokens, 0 host injections, 199,287,158 trace tokens, 178,066,316 instructions" width="100%">
</p>

All numbers below come from committed files. Sources are linked in each table.

### Run totals

| Quantity | Value | Source |
|---|---|---|
| Chunks | 96 (chunk size 2,097,152 steps) | [`seams_0-95_….md`](experiments/circuit/fullrun/logs/cfram_q_vast/seams_0-95_20261107_004845.md) |
| Trace tokens | 199,287,158 | [`quantfix/STEP2_REPORT.md`](experiments/circuit/fullrun/quantfix/STEP2_REPORT.md) |
| Instructions | 178,066,316 | same |
| Final chunk | halts at step 57,705 | [`FINAL_VERDICT.json`](experiments/circuit/fullrun/logs/cfram_q_vast/FINAL_VERDICT.json) |
| Total job time | about 12.1 GPU-hours (sum of the 96 recorded job times) | [`fleet_result.json`](experiments/circuit/fullrun/logs/cfram_q_vast/fleet_result.json) |
| Where chunks ran | Vast.ai: chunks 0 to 31 and 71 to 95 (57); Modal A100-40GB: chunks 32 to 70 (39) | [`FINAL_VERDICT.json`](experiments/circuit/fullrun/logs/cfram_q_vast/FINAL_VERDICT.json) |

### Correctness

| Check | Result |
|---|---|
| Jobs clean | **96 / 96** |
| Free-token mismatches (circuit token != reference token) | **0** |
| Forced tokens | **0** |
| Host injections (weight, RAM, blob words) | **0** in all 96 chunks; the injection hook is not in the graph (`--no-host-writes`) |
| Seams exact | **95 / 95** |
| Final word from the circuit's own output tokens | **15846** (reference 15846), read from the tokens before `HALT` |
| Chunk 0 | boots from the firmware ELF alone |

<!-- VISUAL:quant-chart -->
<p align="center">
  <img src="docs/assets/circuit_precision.gif" alt="Precision design: int4 weights with layers 2, 26 and 27 at 8-bit weights and 16-bit activations; the circuit's answer 15846 matches Hugging Face" width="900">
  <br><sub>🎞️ <a href="docs/assets/circuit_precision.mp4">mp4</a></sub>
</p>

### Precision results

Prompt `[785]`, real Qwen3-0.6B.

| Model | argmax | cos(logits, HF) | top-1 gap | instr/token | weight stream |
|---|---|---|---|---|---|
| Hugging Face fp32 | **15846** (`' Question'`) | 1 | 0.136 | | |
| Hugging Face bf16 | **15846** | | 0.125 | | |
| **Circuit firmware** (int4; layers 2/26/27 at 8-bit weights + 16-bit activations) | **15846** (rank 0) | 0.917 | 0.970 | 189,109,345 | 80,396,288 words |

The firmware's top-1 gap (0.97 in its logits) is wide, so the answer is not a coin flip. Sources: [`quantfix/results/b8_apl2.json`](experiments/circuit/fullrun/quantfix/results/b8_apl2.json), [`logs/final_token/hf_785.json`](experiments/circuit/fullrun/logs/final_token/hf_785.json).

### Stock-module check

The merged checkpoint (hidden 7680, 168,684 table rows, vocabulary 680,128) was run through the HF `Qwen3_5` module classes on chunk 0, teacher-forced with the reference tokens, with the argmax taken over the **full vocabulary**:

| | Positive check | Negative control (every table entry +1 mod 16) |
|---|---|---|
| Steps checked | 130,000 | stopped after 20 mismatches |
| `ROW` tokens / `VDOT` tokens | 2,320 / 23,817 | |
| Mismatches | **0** | first at **step 41** (right after `WB`, `ROW(0)`) |
| Injections | 0 | |
| Source | [`hfcheck_cfq_pos_130k.json`](experiments/circuit/fullrun/quantfix/results/hfcheck_cfq_pos_130k.json) | [`hfcheck_cfq_neg.json`](experiments/circuit/fullrun/quantfix/results/hfcheck_cfq_neg.json) |

The negative control shows the pass depends on the table contents held in the checkpoint's parameters.

### Speed and cost

Time per chunk from the 96 recorded job times ([`fleet_result.json`](experiments/circuit/fullrun/logs/cfram_q_vast/fleet_result.json)):

| Hardware | Chunks | Mean time per chunk | Rate | Cost per chunk |
|---|---|---|---|---|
| RTX 4090 (Vast.ai) | 50 | 5.7 to 6.0 min (range 5.5 to 7.2) | about $0.47 to $0.50/h | about $0.05 |
| A100 PCIE (Vast.ai) | 6 | 8.0 min (range 4.9 to 9.7; includes the short last chunk) | about $0.58 to $0.70/h | about $0.06 to $0.11 |
| L40S (Vast.ai) | 1 | 8.8 min | about $0.61/h | about $0.09 |
| A100-40GB (Modal) | 39 | 9.6 min (range 9.5 to 10.2) | n/a | n/a |

The whole run cost about **$10 of rented Vast.ai GPUs** (an estimate from the fleet log, not from billing). The chunks on Modal ran on its A100s and are not priced here. The fleet used 11 RTX 4090s, an A100 PCIE and an L40S at its peak, about $6.6 per hour; three instances disappeared mid-run, and the fleet finished on the remaining nine RTX 4090s.

### Wall-clock timeline

All times are EST, 2026-11-06 to 2026-11-07.

| Time | Event |
|---|---|
| 21:13 | new reference trace started (about 30 min): argmax 15846, 96 chunks, 0 injections |
| 21:19 | stock-module check and negative control on the new checkpoint |
| 21:38 | row-embedding cache built (about 20 min, once per table) |
| 22:31 | 96 jobs queued on Modal (10-GPU workspace cap, shared with other work) |
| 23:09 | Vast.ai fleet rented |
| 23:41 | first Vast chunk finished (A100 PCIE, 9.7 min) |
| 23:59 | chunk 95 finished on the hub: **`HALT` at step 57,705, word 15846** |
| 00:48 | collector verdict **PASS**; all rented instances destroyed; Modal app stopped |

<!-- VISUAL:fleet-timeline -->
<p align="center">
  <img src="docs/assets/circuit_fleet.svg" alt="Chunks completed by the Vast.ai fleet manager over time, with boxes up and the points where the manager restarted" width="100%">
</p>

<p align="center">
  <img src="docs/assets/circuit_heatstrip.svg" alt="Job time in minutes for each of the 96 chunks, coloured by minutes, with the machine that ran each chunk" width="100%">
</p>
