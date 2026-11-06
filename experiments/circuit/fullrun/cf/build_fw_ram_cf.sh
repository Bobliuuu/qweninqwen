#!/bin/bash
# build_fw_ram_cf.sh CFGDIR CFDIR OUT.elf [gcc flags]  : CF port of the RAM-minimal firmware (real 32 KB map, ramown/fw_rom.ld), weights (-DWF) AND constants via WROW.
set -e
H=$(cd "$(dirname "$0")/../../ramown" && pwd); C=$H/../compiler/fw; HC=$(cd "$(dirname "$0")" && pwd); PICO=/usr/lib/picolibc/riscv64-unknown-elf; GCC=riscv64-unknown-elf-gcc
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
CFG=$1; CFD=$2; OUT=$3; shift 3
$GCC -march=rv32im -mabi=ilp32 -O2 -DNOMARK -DPUTCHAR -DWF "$@" -isystem $PICO/include -I$CFG -I$CFD -I$C -nostdlib -nostartfiles -static -T$H/fw_rom.ld -o $OUT $C/env/crt0.S $HC/fw_qwen3_ram_cf.c $C/vdk_wf.S $CFD/cfetch.S \
  -Wl,--start-group $PICO/lib/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group
riscv64-unknown-elf-size -A $OUT | head -8
