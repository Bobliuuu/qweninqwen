#!/bin/bash
# build_fw.sh CFGDIR OUT.elf [flat|ref]   (needs riscv64-unknown-elf-gcc + picolibc). flat = compiler's qemu.ld (rvemu); ref = enlarged Qwenputer-style map (ref_big.ld), no MARK, putchar ABI.
set -e
H=$(cd "$(dirname "$0")" && pwd); C=$H/../compiler/fw; PICO=/usr/lib/picolibc/riscv64-unknown-elf; GCC=riscv64-unknown-elf-gcc
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
if [ "${3:-ref}" = flat ]; then LD=$C/env/qemu.ld; F=""; else LD=$H/env/ref_big.ld; F="-DNOMARK -DPUTCHAR"; fi
VDK=$C/vdk.S; if [ -n "$WF" ]; then F="$F -DWF"; VDK=$C/vdk_wf.S; fi   # WF=1: weight-fetch firmware (wrow before each VDOT window)
$GCC -march=rv32im -mabi=ilp32 -O2 $F -isystem $PICO/include -I$1 -I$C -nostdlib -nostartfiles -static -T$LD -o $2 $C/env/crt0.S $C/qwen3_fx.c $VDK \
  -Wl,--start-group $PICO/lib/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group
riscv64-unknown-elf-size -A $2 | head -12
