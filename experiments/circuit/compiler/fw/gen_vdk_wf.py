"""Emit vdk_wf.S: weight-fetch VDOT kernels (raw variants only). Same structure as gen_vdk.py, but the weights are NOT read from the WROM at wb:
each 8-word VDOT group is preceded by `wrow a4` (ROW token = embedding row a4 of the outer model; the circuit latches the 8-word window and the VDOTs consume it).
ABI: a0 = acc ptr, a1 = slot (WP of the first word, 32*j, j in 0..14), a2 = R (rows, multiple of 4), a4 = table row. One iteration = 4 accumulators x 8 VDOT = 32 words = 4 windows (4j..4j+3).
After each iteration slot += 32; at slot == 480 slot = 0 and row += 1 (a table row holds 480 words = 15 iterations).
WP bookkeeping: `wb a1` makes the next instruction's WP = a1; the 4 wrow sit at WP a1, a1+9, a1+18, a1+27 so group g's VDOTs have WP a1+1+9g+k and the window (WP_next[8:3]) is 4j+g."""
import sys
REGS = [f"s{i}" for i in range(12)] + ["t3", "t4", "t5", "t6"]
ACC = ["a5", "a6", "a7", "t0"]
out = ["  .text"]
for init in (True, False):
    f = f"vk_{'init' if init else 'acc'}_raw"
    fn = [f"  .globl {f}", f"  .type {f},@function", f"{f}:", "  addi sp, sp, -48"] + [f"  sw s{i}, {4*i}(sp)" for i in range(12)]
    fn += ["  lui t2, %hi(vk_x)", "  lw t2, %lo(vk_x)(t2)"] + [f"  lw {REGS[i]}, {4*i}(t2)" for i in range(16)] + ["1:"]
    fn += [f"  li {a}, 0" if init else f"  lw {a}, {4*r}(a0)" for r, a in enumerate(ACC)]
    fn += ["  .insn r 0x0B, 6, 0, zero, a1, zero"]   # WB a1
    for a in ACC:
        fn += ["  .insn r 0x0B, 6, 1, zero, a4, zero"]   # WROW a4
        fn += [f"  .insn r 0x0B, 5, 0, {a}, {REGS[2*j]}, {REGS[2*j+1]}" for j in range(8)]   # VDOT
    fn += [f"  sw {a}, {4*r}(a0)" for r, a in enumerate(ACC)]
    fn += ["  addi a0, a0, 16", "  addi a1, a1, 32", "  li t1, 480", "  bne a1, t1, 2f", "  li a1, 0", "  addi a4, a4, 1", "2:", "  addi a2, a2, -4", "  bnez a2, 1b"]
    fn += [f"  lw s{i}, {4*i}(sp)" for i in range(12)] + ["  addi sp, sp, 48", "  ret", f"  .size {f}, .-{f}"]
    out += fn
open(sys.argv[1] if len(sys.argv) > 1 else "vdk_wf.S", "w").write("\n".join(out) + "\n")
