#!/bin/sh
C="/Users/jhome/.swiftly/bin/clang --target=riscv32 -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -mno-relax -c kernels2.c"
$C -DNL_EXT -o k2_ext.o && $C -o k2_soft.o && cc -DNL_EXT_EMU -O1 -fsyntax-only -x c kernels2.c && echo built
