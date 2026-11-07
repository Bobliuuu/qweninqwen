# Quantfix: making the emulator's token for prompt [785] equal HF's (15846) by fixing layers 2, 26, 27

Gate (step 1) PASSED at the compiler level: `compiler/fw/qwen3_fx.c` + `gen_fx.py` copies in this dir (`qwen3_fx_q.c`, `gen_fx_q.py`) with PER-LAYER activation precision (`APLN[NL+1]` table, `apl_cur` runtime variable; the original has a global `#define APL`) and the existing per-(layer,matrix) weight-bit `boost`. Real Qwen3-0.6B, prompt [785], real RV32IM firmware (EXT=1, custom nonlinears) built with riscv64-unknown-elf-gcc, run in `rvemu`, CPU only on Modal (`modal_q.py`, results `results/*.json`, also qfr-out /quantfix/).

| variant (W4 elsewhere, ext=1) | emulator argmax | rank of 15846 | cos(logits, HF) | instr/token | WROM words |
|---|---|---|---|---|---|
| ctrl: W4 everywhere, int8 acts (current firmware contract) | **13** (reproduces the CF+RAM answer) | 38,852 | 0.296 | 0.160 G | 74,498,048 |
| layers 2,26,27: int8 weights (8-bit), acts int8 | 9 | 14 | 0.808 | 0.17 G | 80,396,288 |
| layers 2,26,27: W4 weights, 16-bit acts | 408 | 12,077 | 0.339 | 0.17 G | 74,498,048 |
| **layers 2,26,27: 8-bit weights + 16-bit acts** | **15846 (rank 0)** | 0 | 0.917 | 0.19 G (+19 %) | 80,396,288 (+7.9 %) |
| layers 2,3,26,27: 8-bit weights + 16-bit acts | 15846 | 0 | 0.916 | 0.20 G | 82,362,368 |
| layers 2,26,27: 12-bit weights + 16-bit acts | 15846 | 0 | 0.978 | 0.21 G (+31 %) | 86,294,528 |

- The control reproduces 13, so the harness is faithful to the firmware answer the circuit run reproduces.
- Both finer weights AND 16-bit activations at those three layers are needed (each alone fails: 9 / 408). Confirms diag_quant's finding, now in the real integer firmware (not a float sim).
- Margin: top-1 over top-2 in firmware logits 0.97 (8-bit) / 0.94 (12-bit) vs 0.17 for the current firmware; the answer is not a coin flip. Still one prompt token only.
- Recommended: layers 2,26,27 at 8-bit weights + 16-bit acts (smallest change that works). 12-bit has more headroom (cos 0.978) at +31 % instructions.

## Step 2: what regenerating the CF+RAM final firmware needs (NOT done; existing /data/cfram/ramcf.elf, refRC, rowtab_cf untouched)
The final firmware is NOT qwen3_fx.c: it is `cf/fw_qwen3_ram_cf.c` / `cf/qwen3_fx_785_cf.c`, derived from qwen3_fx.c WITHOUT the APL=2 path and with WROW-row weight fetch (`vdk_wf.S`, rows of 480 words). Porting work, in order:
1. `cf/fw_qwen3_ram_cf.c` (+ `qwen3_fx_785_cf.c` for the reference): add `apl_cur` per layer, the second activation plane `xql`, the 16-bit branch of `quantize()`, and `mv()` loops over `ap < apl_cur` (copy from `qwen3_fx_q.c`; the CF `mv(out, srow, soffw, wb, P, ...)` already takes P planes; the activation-plane loop is new). The extra VDOT passes need `xql` as a second `vk_x`.
2. `cf/cf_gen.py`: today layer stride is fixed LSTR = 4096 weight rows (assert) with 4-bit single planes. Layers 2,26,27 with P=2 planes need 2x the rows (8192): make the stride per-layer (or place the extra plane rows after the LM block). Row ids are remapped by `rsh`; update `LM_WB` assert and the cf_cfg.h macros. Table grows by about 5.9 M words = 12.3 K rows of 480 words (156,396 -> ~168.7 K rows, bf16 table 2.40 GB -> ~2.59 GB).
3. Generate firmware inputs with `gen_fx_q.gen(..., wbits=4, boost={(l,'proj'):8 for l in (2,26,27)}, aplmap={2:2,26:2,27:2}, ext=True, dump=False)` for prompt [785], MAXPOS 1 (the cf_gen asserts NTOK==1, ctx 1).
4. Build: `cf/build_fw_ram_cf.sh CFGDIR CFDIR OUT.elf` on a host with riscv64-unknown-elf-gcc + picolibc (Modal CPU image in `modal_q.py` has both; lambda-final disk is 100 % full, do not use it without freeing space).
5. Reference: `cf/ref_big_cf.py` (old run: 1,962 s on lambda-final). Expect ~19 % more trace tokens (157.1 M -> ~187 M, about 89 chunks instead of 75) so the full parallel run costs ~19 % more.
6. Row table `rowtab_cf.npy` from cf_gen; merged checkpoint `hfwf/merge_wf.py` with the bigger table (vocab grows by ~12.3 K rows); re-run the hfcheck + table+1 negative control on chunk 0; re-run the 3-chunk chained + parallel smoke.
All outputs go under NEW paths (/data/cfram_q/..., qfr-wfhf:/ckpt_cf_q); nothing existing is overwritten.

## Caveats
- Compiler-level gate only: single prompt token, firmware = qwen3_fx.c family, not yet the CF/RAM/WROW port; the port can still change the answer if it changes numerics (it should not: CF only changes where weights/constants are fetched from).
- 15846 is HF fp32's answer for this prompt; the new firmware's other positions/prompts are not tested.
- The step-2 regeneration is hours of work plus a new ~33 min reference run and a new full replay; the current CF+RAM run (token 13) is unaffected and can proceed in parallel.
