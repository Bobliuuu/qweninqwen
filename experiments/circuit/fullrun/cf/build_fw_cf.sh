#!/bin/bash
# build_fw_cf.sh CFGDIR CFDIR OUT.elf   CFGDIR: cfg.h; CFDIR: cf_cfg.h + cfetch.S from cf_gen.py. CF firmware (weights AND constants via WROW), ref_big map, Qwenputer putchar ABI.
set -e
H=$(cd "$(dirname "$0")/.." && pwd); C=$H/../compiler/fw; PICO=/usr/lib/picolibc/riscv64-unknown-elf; GCC=riscv64-unknown-elf-gcc
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
$GCC -march=rv32im -mabi=ilp32 -O2 -DNOMARK -DPUTCHAR -DWF -isystem $PICO/include -I$1 -I$2 -I$C -nostdlib -nostartfiles -static -T$H/env/ref_big.ld -o $3 $C/env/crt0.S $H/cf/qwen3_fx_785_cf.c $C/vdk_wf.S $2/cfetch.S \
  -Wl,--start-group $PICO/lib/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group
riscv64-unknown-elf-size -A $3 | head -12
