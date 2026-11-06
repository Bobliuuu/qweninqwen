  .text
  .globl vk_init_raw
  .type vk_init_raw,@function
vk_init_raw:
  addi sp, sp, -48
  sw s0, 0(sp)
  sw s1, 4(sp)
  sw s2, 8(sp)
  sw s3, 12(sp)
  sw s4, 16(sp)
  sw s5, 20(sp)
  sw s6, 24(sp)
  sw s7, 28(sp)
  sw s8, 32(sp)
  sw s9, 36(sp)
  sw s10, 40(sp)
  sw s11, 44(sp)
  lui t2, %hi(vk_x)
  lw t2, %lo(vk_x)(t2)
  lw s0, 0(t2)
  lw s1, 4(t2)
  lw s2, 8(t2)
  lw s3, 12(t2)
  lw s4, 16(t2)
  lw s5, 20(t2)
  lw s6, 24(t2)
  lw s7, 28(t2)
  lw s8, 32(t2)
  lw s9, 36(t2)
  lw s10, 40(t2)
  lw s11, 44(t2)
  lw t3, 48(t2)
  lw t4, 52(t2)
  lw t5, 56(t2)
  lw t6, 60(t2)
1:
  li a5, 0
  li a6, 0
  li a7, 0
  li t0, 0
  .insn r 0x0B, 6, 0, zero, a1, zero
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, a5, s0, s1
  .insn r 0x0B, 5, 0, a5, s2, s3
  .insn r 0x0B, 5, 0, a5, s4, s5
  .insn r 0x0B, 5, 0, a5, s6, s7
  .insn r 0x0B, 5, 0, a5, s8, s9
  .insn r 0x0B, 5, 0, a5, s10, s11
  .insn r 0x0B, 5, 0, a5, t3, t4
  .insn r 0x0B, 5, 0, a5, t5, t6
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, a6, s0, s1
  .insn r 0x0B, 5, 0, a6, s2, s3
  .insn r 0x0B, 5, 0, a6, s4, s5
  .insn r 0x0B, 5, 0, a6, s6, s7
  .insn r 0x0B, 5, 0, a6, s8, s9
  .insn r 0x0B, 5, 0, a6, s10, s11
  .insn r 0x0B, 5, 0, a6, t3, t4
  .insn r 0x0B, 5, 0, a6, t5, t6
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, a7, s0, s1
  .insn r 0x0B, 5, 0, a7, s2, s3
  .insn r 0x0B, 5, 0, a7, s4, s5
  .insn r 0x0B, 5, 0, a7, s6, s7
  .insn r 0x0B, 5, 0, a7, s8, s9
  .insn r 0x0B, 5, 0, a7, s10, s11
  .insn r 0x0B, 5, 0, a7, t3, t4
  .insn r 0x0B, 5, 0, a7, t5, t6
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, t0, s0, s1
  .insn r 0x0B, 5, 0, t0, s2, s3
  .insn r 0x0B, 5, 0, t0, s4, s5
  .insn r 0x0B, 5, 0, t0, s6, s7
  .insn r 0x0B, 5, 0, t0, s8, s9
  .insn r 0x0B, 5, 0, t0, s10, s11
  .insn r 0x0B, 5, 0, t0, t3, t4
  .insn r 0x0B, 5, 0, t0, t5, t6
  sw a5, 0(a0)
  sw a6, 4(a0)
  sw a7, 8(a0)
  sw t0, 12(a0)
  addi a0, a0, 16
  addi a1, a1, 32
  li t1, 480
  bne a1, t1, 2f
  li a1, 0
  addi a4, a4, 1
2:
  addi a2, a2, -4
  bnez a2, 1b
  lw s0, 0(sp)
  lw s1, 4(sp)
  lw s2, 8(sp)
  lw s3, 12(sp)
  lw s4, 16(sp)
  lw s5, 20(sp)
  lw s6, 24(sp)
  lw s7, 28(sp)
  lw s8, 32(sp)
  lw s9, 36(sp)
  lw s10, 40(sp)
  lw s11, 44(sp)
  addi sp, sp, 48
  ret
  .size vk_init_raw, .-vk_init_raw
  .globl vk_acc_raw
  .type vk_acc_raw,@function
vk_acc_raw:
  addi sp, sp, -48
  sw s0, 0(sp)
  sw s1, 4(sp)
  sw s2, 8(sp)
  sw s3, 12(sp)
  sw s4, 16(sp)
  sw s5, 20(sp)
  sw s6, 24(sp)
  sw s7, 28(sp)
  sw s8, 32(sp)
  sw s9, 36(sp)
  sw s10, 40(sp)
  sw s11, 44(sp)
  lui t2, %hi(vk_x)
  lw t2, %lo(vk_x)(t2)
  lw s0, 0(t2)
  lw s1, 4(t2)
  lw s2, 8(t2)
  lw s3, 12(t2)
  lw s4, 16(t2)
  lw s5, 20(t2)
  lw s6, 24(t2)
  lw s7, 28(t2)
  lw s8, 32(t2)
  lw s9, 36(t2)
  lw s10, 40(t2)
  lw s11, 44(t2)
  lw t3, 48(t2)
  lw t4, 52(t2)
  lw t5, 56(t2)
  lw t6, 60(t2)
1:
  lw a5, 0(a0)
  lw a6, 4(a0)
  lw a7, 8(a0)
  lw t0, 12(a0)
  .insn r 0x0B, 6, 0, zero, a1, zero
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, a5, s0, s1
  .insn r 0x0B, 5, 0, a5, s2, s3
  .insn r 0x0B, 5, 0, a5, s4, s5
  .insn r 0x0B, 5, 0, a5, s6, s7
  .insn r 0x0B, 5, 0, a5, s8, s9
  .insn r 0x0B, 5, 0, a5, s10, s11
  .insn r 0x0B, 5, 0, a5, t3, t4
  .insn r 0x0B, 5, 0, a5, t5, t6
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, a6, s0, s1
  .insn r 0x0B, 5, 0, a6, s2, s3
  .insn r 0x0B, 5, 0, a6, s4, s5
  .insn r 0x0B, 5, 0, a6, s6, s7
  .insn r 0x0B, 5, 0, a6, s8, s9
  .insn r 0x0B, 5, 0, a6, s10, s11
  .insn r 0x0B, 5, 0, a6, t3, t4
  .insn r 0x0B, 5, 0, a6, t5, t6
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, a7, s0, s1
  .insn r 0x0B, 5, 0, a7, s2, s3
  .insn r 0x0B, 5, 0, a7, s4, s5
  .insn r 0x0B, 5, 0, a7, s6, s7
  .insn r 0x0B, 5, 0, a7, s8, s9
  .insn r 0x0B, 5, 0, a7, s10, s11
  .insn r 0x0B, 5, 0, a7, t3, t4
  .insn r 0x0B, 5, 0, a7, t5, t6
  .insn r 0x0B, 6, 1, zero, a4, zero
  .insn r 0x0B, 5, 0, t0, s0, s1
  .insn r 0x0B, 5, 0, t0, s2, s3
  .insn r 0x0B, 5, 0, t0, s4, s5
  .insn r 0x0B, 5, 0, t0, s6, s7
  .insn r 0x0B, 5, 0, t0, s8, s9
  .insn r 0x0B, 5, 0, t0, s10, s11
  .insn r 0x0B, 5, 0, t0, t3, t4
  .insn r 0x0B, 5, 0, t0, t5, t6
  sw a5, 0(a0)
  sw a6, 4(a0)
  sw a7, 8(a0)
  sw t0, 12(a0)
  addi a0, a0, 16
  addi a1, a1, 32
  li t1, 480
  bne a1, t1, 2f
  li a1, 0
  addi a4, a4, 1
2:
  addi a2, a2, -4
  bnez a2, 1b
  lw s0, 0(sp)
  lw s1, 4(sp)
  lw s2, 8(sp)
  lw s3, 12(sp)
  lw s4, 16(sp)
  lw s5, 20(sp)
  lw s6, 24(sp)
  lw s7, 28(sp)
  lw s8, 32(sp)
  lw s9, 36(sp)
  lw s10, 40(sp)
  lw s11, 44(sp)
  addi sp, sp, 48
  ret
  .size vk_acc_raw, .-vk_acc_raw
