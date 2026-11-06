# Qwen3-0.6B on the Qwenputer RV32IM CPU: instructions and seconds per token

Measured: instructions by the emulator (`fw/rvemu.c`) on real HF weights (`run_full.py`), positions 0..3. Larger contexts extrapolate the measured per-context-token slope.
Seconds = instr x trace-tokens/instr / trace-tokens/s. Best = 1.19 tok/instr @5500/s; worst = 1.40 @3200/s; mid = 1.40 @5500/s.

| build | ctx | instr/token | s/token (1.40 tok/instr, 5500/s) | worst (1.40, 3200/s) | best (1.19, 5500/s) | days (mid) |
|---|---|---|---|---|---|---|
| q4v | 1 | 5.866e+08 | 1.493e+05 | 2.566e+05 | 1.269e+05 | 1.73 |
| q4v | 128 | 3.473e+09 | 8.840e+05 | 1.519e+06 | 7.514e+05 | 10.23 |
| q4v | 1024 | 2.383e+10 | 6.067e+06 | 1.043e+07 | 5.157e+06 | 70.22 |
