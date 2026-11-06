"""Build + run matvec kernels on the ISS and verify bit-exactness against numpy.  Needs riscv gcc (riscv64-unknown-elf-gcc) and gcc for the ISS."""
import numpy as np, subprocess, os, sys, json, re, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.environ.get('RVCC', 'riscv64-unknown-elf-gcc'); OBJCOPY = os.environ.get('RVOBJCOPY', 'riscv64-unknown-elf-objcopy')

def build(variant, rb=4, unr=2, tmp='/tmp/rvk'):
    os.makedirs(tmp, exist_ok=True); elf = f'{tmp}/k{variant}_{rb}_{unr}.elf'; binf = elf[:-4] + '.bin'
    subprocess.check_call([CC, '-march=rv32im', '-mabi=ilp32', '-O2', '-ffreestanding', '-nostdlib', '-fno-pic', '-Wl,-Ttext=0', '-Wl,-e,_start', f'-DVARIANT={variant}', f'-DRB={rb}', f'-DUNR={unr}',
                           os.path.join(HERE, 'kernels.c'), '-o', elf])
    subprocess.check_call([OBJCOPY, '-O', 'binary', elf, binf]); return binf

def sim_bin():
    b = '/tmp/rvk/rvsim'
    if not os.path.exists(b): os.makedirs('/tmp/rvk', exist_ok=True); subprocess.check_call(['gcc', '-O2', os.path.join(HERE, 'rvsim.c'), '-o', b])
    return b

def pack4(W):   # [R,K] int4 -> bytes, low nibble first
    u = (W.astype(np.int64) & 15).astype(np.uint8); return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)

def run(variant, R, K, rb=4, unr=2, seed=0):
    rng = np.random.default_rng(seed); x = rng.integers(-128, 128, K).astype(np.int8)
    four = variant in (3, 4, 5)
    W = rng.integers(-8, 8, (R, K)).astype(np.int8) if four else rng.integers(-128, 128, (R, K)).astype(np.int8)
    ref = (W.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)
    data = bytearray(0x300000); data[0:8] = np.array([R, K], np.uint32).tobytes(); data[0x100:0x100 + K] = x.tobytes()
    if variant == 5:   # stream order: for row group, for k-chunk(8), for r in group: 4 bytes
        P = pack4(W).reshape(R // rb, rb, K // 8, 4).transpose(0, 2, 1, 3); wb = P.tobytes()
    elif four: wb = pack4(W).tobytes()
    else: wb = W.tobytes()
    data[0x10000:0x10000 + len(wb)] = wb
    d = tempfile.mkdtemp(); open(d + '/in.bin', 'wb').write(bytes(data))
    out = subprocess.run([sim_bin(), build(variant, rb, unr), d + '/in.bin', d + '/out.bin'], capture_output=True, text=True, check=True).stdout
    got = np.frombuffer(open(d + '/out.bin', 'rb').read(), np.int32)
    n = int(re.search(r'INSTR (\d+)', out).group(1)); assert np.array_equal(got, ref), (variant, 'MISMATCH')
    return dict(variant=variant, rb=rb, unr=unr, R=R, K=K, macs=R * K, instr=n, instr_per_mac=n / (R * K), exact=True, stats=out.strip())

def main(R=192, K=1024):
    rows = []
    for v, rb, unr in [(0, 1, 1), (1, 1, 1), (2, 4, 1), (3, 1, 1), (4, 1, 1), (5, 2, 2), (5, 4, 2), (5, 4, 4), (5, 8, 2), (5, 8, 4), (5, 12, 2)]:
        r = run(v, R, K, rb, unr); rows.append(r); print('KERNEL ' + json.dumps(r), flush=True)
    return dict(kernels=rows)

if __name__ == '__main__': main()
