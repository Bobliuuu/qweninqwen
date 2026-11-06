"""Emit vdk.S: the VDOT matvec kernels (same structure as Qwenputer guests/stories260k/gen_vd2.py, np=8 only).
vk_acc(out, wb, rows, S, F): rows of 8 words (64 int4 weights) stored back to back from WROM word wb, 4 rows per WB;
out[r] (+)= x . row r, x = 64 int8 activations at *vk_x (loaded into 16 regs).  scaled: out[r] = mulh(out[r]*S[r], F)."""
import sys
REGS = [f"s{i}" for i in range(12)] + ["t3", "t4", "t5", "t6"]
ACC = ["a5", "a6", "a7", "t0"]
out = ["  .text"]
for init in (True, False):
    for scaled in (True, False):
        f = f"vk_{'init' if init else 'acc'}_{'scaled' if scaled else 'raw'}"
        fn = [f"  .globl {f}", f"{f}:", "  addi sp, sp, -48"] + [f"  sw s{i}, {4*i}(sp)" for i in range(12)]
        fn += ["  lui t2, %hi(vk_x)", "  lw t2, %lo(vk_x)(t2)"] + [f"  lw {REGS[i]}, {4*i}(t2)" for i in range(16)] + ["1:"]
        fn += [f"  li {a}, 0" if init else f"  lw {a}, {4*r}(a0)" for r, a in enumerate(ACC)]
        fn += ["  .insn r 0x0B, 6, 0, zero, a1, zero"]   # WB a1
        for a in ACC: fn += [f"  .insn r 0x0B, 5, 0, {a}, {REGS[2*j]}, {REGS[2*j+1]}" for j in range(8)]   # VDOT
        for r, a in enumerate(ACC):
            if scaled: fn += [f"  lw t1, {4*r}(a3)", f"  mul {a}, {a}, t1", f"  mulh {a}, {a}, a4"]
            fn += [f"  sw {a}, {4*r}(a0)"]
        fn += ["  addi a0, a0, 16", "  addi a1, a1, 32"] + (["  addi a3, a3, 16"] if scaled else []) + ["  addi a2, a2, -4", "  bnez a2, 1b"]
        fn += [f"  lw s{i}, {4*i}(sp)" for i in range(12)] + ["  addi sp, sp, 48", "  ret"]
        out += fn
open(sys.argv[1] if len(sys.argv) > 1 else "vdk.S", "w").write("\n".join(out) + "\n")
