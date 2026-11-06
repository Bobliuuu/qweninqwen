#!/bin/bash
# fw_build.sh CFGDIR OUT.elf [extra gcc flags, e.g. -DEXT=1 -DTILE=128 -DPMAX=2 -DDUMP=1]
# RAM-minimal 0.6B firmware (ramown/fw_qwen3_ram.c) linked with the REAL Qwenputer map (fw_rom.ld = chipformer env/rom.ld): RAM 0x100..0x8000 (stack from 0x8000),
# rodata in the 128 KB DROM at 0x20000, code in the ROM at 0x40000.  putchar ABI, no MARK.  Needs riscv64-unknown-elf-gcc + picolibc.
set -e
H=$(cd "$(dirname "$0")" && pwd); C=$H/../compiler/fw; PICO=/usr/lib/picolibc/riscv64-unknown-elf; GCC=riscv64-unknown-elf-gcc
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
CFG=$1; OUT=$2; shift 2
VDK=$C/vdk.S; case " $* " in *" -DWF "*) VDK=${VDKWF:-$C/vdk_wf.S};; esac   # -DWF: weight-fetch kernel (wrow before each 8-word window); optional -DROWW=640 for 6-bit rows
$GCC -march=rv32im -mabi=ilp32 -O2 -DNOMARK -DPUTCHAR "$@" -isystem $PICO/include -I$CFG -I$C -nostdlib -nostartfiles -static -T$H/fw_rom.ld -o $OUT $C/env/crt0.S $H/fw_qwen3_ram.c $VDK \
  -Wl,--start-group $PICO/lib/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group
riscv64-unknown-elf-size -A $OUT | head -8
riscv64-unknown-elf-nm -S -n $OUT | grep -iE " [bBdDrR] " | grep -v "\.L" | awk '{printf "%s %6d B  %s\n",$1,strtonum("0x"$2),$4}' | head -40
