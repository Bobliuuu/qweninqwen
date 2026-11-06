# Qwen3-0.6B as RV32IM firmware on the Qwenputer CPU: instructions and seconds per token

Instruction counts: `fw/rvemu.c` on real Qwen3-0.6B weights (`run_full.py`, `run_full_fx.py`). Seconds = instr x trace-tokens/instr / trace-tokens/s.
Columns: best = 1.19 tok/instr @5500/s; mid = 1.40 @5500/s (Qwenputer fused executor on an L4); worst = 1.40 @3200/s. VDOT results carry 8 products per instruction.

| build | ctx | instr/token | s/token best | mid | worst | note |
|---|---|---|---|---|---|---|
| T0a fp32 soft-float llama2.c (MEASURED) | 1 | 1.097e+11 | 2.374e+07 | 2.793e+07 | 4.800e+07 |  |
| v1 int4 VDOT + soft-float glue | 1 | 5.866e+08 | 1.269e+05 | 1.493e+05 | 2.566e+05 |  |
| v2 fixed-point int8 weights, plain RV32IM nonlinears | 1 | 2.448e+08 | 5.297e+04 | 6.232e+04 | 1.071e+05 |  |
| v2 fixed-point int8 weights, custom-1 EXT nonlinears | 1 | 2.413e+08 | 5.221e+04 | 6.142e+04 | 1.056e+05 |  |
| v2 fixed-point int4 weights, plain RV32IM nonlinears | 1 | 1.378e+08 | 2.982e+04 | 3.508e+04 | 6.030e+04 |  |
| v2 fixed-point int4 weights, custom-1 EXT nonlinears | 1 | 1.343e+08 | 2.905e+04 | 3.418e+04 | 5.875e+04 |  |
| v2 int4 EXT (MEASURED at position 129) | 130 | 1.553e+08 | 3.360e+04 | 3.953e+04 | 6.794e+04 | per-context-token slope 1.60e+05 instr |

## T0 fp32 soft-float: MEASURED (replaces the extrapolation)
`res_full_f32.json`: real Qwen3-0.6B, fp32 weights, 3 positions in the emulator: 1.097e11 / 1.100e11 / 1.100e11 instr per token (total 3.297e11, alu 2.27e11, load 2.3e10, branch 4.1e10, mul 7.2e9). Argmax 15846, 315, 279 == HF at all 3 positions (fp32 soft-float is argmax-exact, including position 0). 184 instr/MAC. At 1.154 tok/instr (VDOT mix; soft-float mix is closer to 1.2-1.4) and 5,500 tok/s: ~2.3e7 s (0.74 yr); at 1.4 tok/instr and 3,200 tok/s: 5.4e7 s (1.7 yr). So the literal T0 is about one year per token.
