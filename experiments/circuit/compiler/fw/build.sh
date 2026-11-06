#!/bin/bash
# build.sh out.elf src... [-Dflags]   (run on a host with riscv64-unknown-elf-gcc + picolibc; e.g. the GPU box)
set -e
D=$(cd "$(dirname "$0")" && pwd); PICO=/usr/lib/picolibc/riscv64-unknown-elf; GCC=riscv64-unknown-elf-gcc
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
OUT=$1; shift
$GCC -march=rv32im -mabi=ilp32 -O2 -isystem $PICO/include -I$D -nostdlib -nostartfiles -static -T$D/env/qemu.ld -o $OUT $D/env/crt0.S "$@" \
  -Wl,--start-group $PICO/lib/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group
