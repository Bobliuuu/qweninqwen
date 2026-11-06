# Compiler target: RV32IM firmware on the Qwenputer CPU (replaces the earlier tensor-op ISA)

The outer model is a stock Qwen3.5 (0.8B primary, 4B) with the Qwenputer RV32IM CPU compiled into extra layers; DeltaNet state is
RAM/ROM; one trace token ~ 1/1.19..1.40 of an instruction (Qwenputer `docs/rv32.md`). The inner Qwen3-0.6B is a **C program** (`fw/qwen3.c`)
whose weights are **data** (DROM float tables / WROM int4 cells). Compiler = `gen_fw.py` (HF state dict -> `cfg.h`, `blob.bin`, `wrom.bin`)
+ `fw/build.sh` (riscv64-unknown-elf-gcc 10.2, `-march=rv32im -mabi=ilp32 -O2`, picolibc/libgcc soft-float) + `fw/vdk.S` (VDOT kernels).

## Target ISA
| layer | instructions | status |
|---|---|---|
| base | RV32IM (+Zba allowed), ecall putchar (a7=1), exit (a7=93) | exists in Qwenputer main |
| optional ext. (Qwenputer `npu` branch) | `WB rs1` (custom-0 f3=6): set weight-memory base; `VDOT rd,rs1,rs2` (f3=5): rd += 8 int8 (rs1,rs2 bytes) x next 8 int4 WROM cells, base advances 1 word; `SMAQA rd,rs1,rs2` (f3=7): rd += 4 signed byte products | exists on `npu` branch, marked OPTIONAL here |
| profiling only | `ecall a7=2` = mark (emulator records instr count; NOT on the real machine, build without `MARK` for deployment) | emulator only |
`fw/rvemu.c` is an RV32IM+Zba+VDOT/WB/SMAQA emulator (flat 4 GB memory, per-function and per-class instruction counts). Its base semantics follow
`chipformer.rv.ref`; VDOT/WB follow `ref.py` (cell u in low nibble, w = u-8).

## Firmware ABI (`fw/qwen3.c`)
* `forward(tok,pos)`: embed row copy; per layer: RMSNorm, q/k/v projections, per-head q/k RMSNorm (+gain), RoPE (rotate_half, cos/sin table in blob),
  KV append (float, `.bss`), GQA softmax attention (`expf`), o-proj, residual, RMSNorm, gate/up, SwiGLU, down, residual; final RMSNorm, tied lm_head, argmax.
* Modes: `MODE_F32` (fp32 weights, soft-float: the literal T0), `MODE_Q4V` (all projections + lm_head int4 per-row-scale via VDOT, int8 dynamic activation
  quantisation; glue still soft-float).
* Memory map: code @0x100000 (qemu.ld flat map for emulation; on the real machine ROM 256 KB), blob @0x10000000, WROM = `wrom.bin` cell stream
  (chunk-major: matrix, then 64-input chunk, then row, 64 cells = 8 words; one WB per 4 rows).
* Constraint: projection inner dims must be multiples of 64 (true for Qwen3-0.6B: 1024, 2048, 3072).

## Front-end IR -> lowering
The tensor-op view (MATVEC / RMSNORM / ROPE / ATTN / SWIGLU / ARGMAX) is only the shape of `qwen3.c`'s functions (`matvec`, `rmsnorm`, `rope`,
`attn`, `swiglu`, `argmax`); each lowers to RV32IM as: MATVEC -> plain triple loop (7.0 instr/MAC int8) | SMAQA loop (0.91) | VDOT kernel (0.176).
Other agents plug in by replacing a function body (e.g. integer-only glue, SMAQA attention) and re-running `test_tiny.py`.

## Files
`fw/qwen3.c` firmware; `gen_fw.py` weight/cfg emitter; `fw/vdk.S` (from `fw/gen_vdk.py`); `fw/rvemu.c`; `fw/bench_proj.c` micro-benchmark;
`test_tiny.py` (HF vs emulator, tiny config); `run_full.py` (real Qwen3-0.6B); `footprint.py`; results `fw/res_proj_instr_per_mac.json`,
`res_tiny.json`, `res_full_*.json`, `res_footprint.json`.
