| | |
|---|---|
| Outer model | a Qwen3_5-architecture model (HF `Qwen3_5ForCausalLM` classes, widened to hidden 7680 with 12 added chip layers and an extended 680,128-token vocabulary; derived from the Qwen3.5-4B layer design), with an RV32IM CPU + vector-dot unit compiled into its weights (merged checkpoint: 12 chip layers + Qwen's 32) |
| Inner model | real Qwen3-0.6B as a C firmware, prompt token `[785]` ("The"), position 0 |
| Run size | **96 chunks**, 199,287,158 trace tokens, 178,066,316 instructions |
| Ending | the circuit's own `HALT` at step 57,705 of chunk 95; final word **15846** |
| Hugging Face Qwen3-0.6B (fp32 and bf16) | **15846** (`' Question'`), the same token |
| Jobs | **96 / 96 clean**: 0 free-token mismatches, 0 forced tokens, 0 host injections |
| Seams | **95 / 95 exact**: each job's own end state equals the next job's start state (registers, pc, anchor, `n`, `wb`, latch, all 8,192 RAM words, `bn`) |
| Chunk 0 | boots from the **firmware ELF alone** (no reference state) |

The verdict is in [`FINAL_VERDICT.json`](experiments/circuit/fullrun/logs/cfram_q_vast/FINAL_VERDICT.json) and [`seams_0-95_20261107_004845.md`](experiments/circuit/fullrun/logs/cfram_q_vast/seams_0-95_20261107_004845.md); every command and run is logged in [`RUNS.md`](RUNS.md).
