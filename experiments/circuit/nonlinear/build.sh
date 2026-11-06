#!/bin/sh
/Users/jhome/.swiftly/bin/clang --target=riscv32 -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -funroll-loops -mno-relax -c kernels.c -o kernels.o
