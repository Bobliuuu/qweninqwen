# Route history


| Route | What it was | Per token | Host role |
|---|---|---|---|
| Compiled route ([older README](../older_compiled_route.md)) | custom giant outer (1006 layers, D=40775, 293,068 frames) doing real fp32 matvecs; the real Qwen3-0.6B at one position, 6 teacher-forced positions | **268.6 s** on a shared A100 | host streamed the weights and addresses |
| Circuit, literal (T0) | fp32 soft-float firmware on the circuit | 1.10e11 instructions, computed 0.71 to 0.98 years | not run |
| Circuit, weight fetch (WF) | weights as `ROW` fetches; replay from per-chunk reference snapshots | 84 chunks, 174,823,694 free tokens, 0 mismatches | RAM and blob words still host-injected (7% of steps) |
| Circuit, CF+RAM, int4 everywhere | constants as rows, circuit-owned RAM | 75 chunks, 157,076,451 tokens; word **13** | 0 injections |
| **Circuit, final contract** | layers 2/26/27 at 8-bit weights + 16-bit activations | **96 chunks, 199,287,158 tokens; word 15846; about 12 GPU-hours of job time** | 0 injections, real decode |

