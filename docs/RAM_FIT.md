# 0.6B program in the core's real 32 KB RAM (status note for the circuit-side work)

Full details, measured numbers and the address map: `experiments/circuit/ramown/ADDR_MAP.md`.

- Measured (not estimated): the original 0.6B firmware touches 357,643 distinct RAM words (1.43 MB); without the LM-head logit/accumulator buffers it is 53,771 words = 210 KB (the earlier estimate of 215 KB was right). Peak live words 152,222 (LM head).
- New firmware `ramown/fw_qwen3_ram.c`: RW footprint 7,079 words (86 % of 8,192), logits bit-identical to the original on the emulator, 123.9 M instr/token vs 157.1 M. `-DTWOPASS`: 4,964 words, 169.4 M instr.
- Real map (`ramown/fw_rom.ld`): RAM 0x100..0x8000, rodata in the real DROM at 0x20000, code in ROM 0x40000. No 4 MB RAM relocation, no address aliasing, no `in_ram := 1` needed for RAM accesses.
- What the circuit side still has to do: (1) restore `ramimg_NNNN.npy` (8192 words) at chunk start; (2) loads at address >= 0x10000000 (469,376 per token, 4.4 % of loads: model scales, norm gains, embedding row) are static model data and must be served from the DROM/blob state, NOT written into a `ram{h}` slot (slot = (ea>>2)&0x1FFF would clobber live RAM); (3) use `ramown/fw_ref32.py` (real 8192-word RAM reference machine) instead of `fullrun/ref_big.py` to generate the trace for this ELF.
- ctx > 1 does not fit: 57.75 KB of K/V per extra token.
