"""Bit-exact check of the RAM-minimal firmware against the original compiler/fw/qwen3_fx.c on the rvemu emulator, on the SAME cfg.h/blob/wrom:
both built with DUMP (all logits written out), compare every logit of every position + argmax, report instr/token for both.
usage: fw_verify.py CFGDIR [EXTRA_GCC_FLAGS...]   (e.g. -DEXT=1 ; -DTILE=128).  Needs riscv64-unknown-elf-gcc, ./rvemu_mt (cc -O2 -o rvemu_mt fw_rvemu_mt.c -lm)."""
import sys, os, subprocess, json, re, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); C = H + '/../compiler/fw'; cfg = os.path.abspath(sys.argv[1]); fl = sys.argv[2:]
emu = os.environ.get('RVEMU', os.path.expanduser('~/ramown/rvemu_mt'))
h = open(cfg + '/cfg.h').read(); V = int(re.search(r'#define VOCAB (\d+)', h).group(1)); NT = int(re.search(r'#define NTOK (\d+)', h).group(1))
def run(elf, extra, tag):
    log = subprocess.check_output([emu, elf, '-b', '0x10000000', cfg + '/blob.bin', '-w', cfg + '/wrom.bin', '-o', f'{cfg}/{tag}.out', '-p'] + extra, text=True)
    return re.search(r'instr=(\d+)', log).group(1), np.fromfile(f'{cfg}/{tag}.out', dtype=np.int32)
# baseline (original source, flat qemu map, MARK ecall off for the NOMARK/PUTCHAR-less comparison: use its own ABI)
PICO = '/usr/lib/picolibc/riscv64-unknown-elf'; GCC = 'riscv64-unknown-elf-gcc'
libgcc = subprocess.check_output([GCC, '-march=rv32im', '-mabi=ilp32', '-print-libgcc-file-name'], text=True).strip()
def cc(out, src, ld, flags):
    subprocess.check_call([GCC, '-march=rv32im', '-mabi=ilp32', '-O2', '-DDUMP=1', *flags, '-isystem', PICO + '/include', '-I' + cfg, '-I' + C, '-nostdlib', '-nostartfiles', '-static', '-T' + ld, '-o', out,
                           C + '/env/crt0.S', src, C + '/vdk.S', '-Wl,--start-group', PICO + '/lib/rv32im/ilp32/libc.a', libgcc, '-Wl,--end-group'])
cc(cfg + '/old_d.elf', C + '/qwen3_fx.c', C + '/env/qemu.ld', fl)
cc(cfg + '/new_d.elf', H + '/fw_qwen3_ram.c', H + '/fw_rom.ld', fl + ['-DNOMARK', '-DPUTCHAR'])
io, o = run(cfg + '/old_d.elf', [], 'old'); inn, n = run(cfg + '/new_d.elf', ['-s', '0x8000'], 'new')
o = o.reshape(NT, 1 + V); n = n.reshape(NT, V + 1)           # old: [argmax][logits]; new: [logits][argmax]
am_o, lg_o = o[:, 0], o[:, 1:]; am_n, lg_n = n[:, V], n[:, :V]
res = dict(cfg=cfg, flags=fl, ntok=NT, argmax_old=am_o.tolist(), argmax_new=am_n.tolist(), logits_identical=bool((lg_o == lg_n).all()), n_logit_diff=int((lg_o != lg_n).sum()),
           instr_old=int(io), instr_new=int(inn), instr_ratio=int(inn) / int(io))
print(json.dumps(res)); sys.exit(0 if res['logits_identical'] and (am_o == am_n).all() else 1)
