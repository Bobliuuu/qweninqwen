"""Kernels for the PROPOSED instructions of ext.py (VDOT16 = W4A4 16-wide, VDOTA = acts straight from RAM, VDOTA16 = W4A8 16-wide from RAM)."""
from kernels import *
from kernels import _exit, _wrom_blocked
from ext import run_ext

class ExtKernel(Kernel):
    def run(self, W, x, ref_y):
        r = run_ext(self.src, self.ram, self.rodata, self.wrom)
        y = np.array([r.sword(self.out + 4 * i) for i in range(self.R)])
        macs = self.R * self.K
        return dict(name=self.name, ok=bool((y == ref_y).all()), instr=r.instr, tokens=r.ntok, macs=macs, ipm=r.instr / macs,
                    tpm=r.ntok / macs, fam=dict(r.fam), res=r)

def nib_bytes(x4):
    x4 = np.asarray(x4) & 15
    return bytes((x4[0::2] | (x4[1::2] << 4)).astype(np.uint8).tolist())          # little-endian nibbles: act k = nibble k

def k9_vdot16_blocked(W, x4, RB=22, C=2):
    """W4A4: 16 MACs per VDOT16.  Same dataflow as K4: C chunks (of 16 acts = 8 bytes) in 2C regs, RB accumulators."""
    R, K = W.shape
    nch = K // 16
    assert R % RB == 0 and nch % C == 0
    acc = REGS[:RB]; actr = REGS[RB:RB + 2 * C]; (wp, ap, op, rc) = REGS[RB + 2 * C:RB + 2 * C + 4]
    ACTEND = ACT + K // 2
    zero = ''.join(f" li {a}, 0\n" for a in acc)
    loads = ''.join(f" lw {actr[2*c]}, {ACTEND + 8*c}({ap})\n lw {actr[2*c+1]}, {ACTEND + 8*c + 4}({ap})\n" for c in range(C))
    vd = ''.join(f" vdot16 {a}, {actr[2*c]}, {actr[2*c+1]}\n" for c in range(C) for a in acc)
    sts = ''.join(f" sw {a}, {4*i}({op})\n" for i, a in enumerate(acc))
    src = f"""_start:
 li {wp}, 0
 li {op}, {OUT}
 li {rc}, {R // RB}
blk:
 li {ap}, {-nch * 8}
{zero}in:
{loads} wb16 {wp}
{vd} addi {wp}, {wp}, {2 * RB * C}
 addi {ap}, {ap}, {8 * C}
 bnez {ap}, in
{sts} addi {op}, {op}, {4 * RB}
 addi {rc}, {rc}, -1
 bnez {rc}, blk
""" + _exit()
    words = []
    for b in range(0, R, RB):
        for g in range(0, nch, C):
            for c in range(C):
                for r in range(RB):
                    seg = W[b + r, (g + c) * 16:(g + c) * 16 + 16]
                    words += [seg[:8], seg[8:]]
    return ExtKernel(f'K9 VDOT16 (W4A4, 16 MAC/instr) blocked RB={RB} C={C}', src, {ACT: nib_bytes(x4)},
                     wrom=pack_wrom_nibbles(np.array(words)), R=R, K=K)

def k8_vdota_row(W, x, wide=False):
    """acts read straight from RAM by the instruction (no LW, no register blocking): one row = WB + K/8 (or K/16) VDOTA(16)."""
    R, K = W.shape
    cw = 16 if wide else 8
    nch = K // cw
    ins = 'vdota16' if wide else 'vdota'
    body = ''.join(f" {ins} s5, {ACT + cw * j}(zero)\n" for j in range(nch))
    wpr = K // 8                                         # WROM words per row
    src = f"""_start:
 li s1, 0
 li s2, {OUT}
 li s0, {R}
row:
 li s5, 0
 {'wba16' if wide else 'wba'} s1
{body} sw s5, 0(s2)
 addi s2, s2, 4
 addi s1, s1, {wpr}
 addi s0, s0, -1
 bnez s0, row
""" + _exit()
    return ExtKernel(f'K{10 if wide else 8} {"VDOTA16 (W4A8, 16 MAC/instr, acts from RAM)" if wide else "VDOTA (W4A8, 8 MAC/instr, acts from RAM)"}',
                     src, xbytes(x), wrom=pack_wrom_nibbles(W.reshape(-1, 8)), R=R, K=K)

if __name__ == '__main__':
    R, K = 220, 1024
    W, x, y = problem(R, K)
    k = k4_vdot_blocked(W, x, 22, 2); r = k.run(W, x, y)
    print(f"{k.name:62s} ok={r['ok']} instr/MAC={r['ipm']:.4f} tok/MAC={r['tpm']:.4f}")
    for k in [k8_vdota_row(W, x), k8_vdota_row(W, x, wide=True)]:
        r = k.run(W, x, y); print(f"{k.name:62s} ok={r['ok']} instr/MAC={r['ipm']:.4f} tok/MAC={r['tpm']:.4f}")
    x4 = np.clip(np.rint(x / 127 * 7), -8, 7).astype(np.int64)      # A4 version of the same activations
    y4 = W @ x4
    k = k9_vdot16_blocked(W, x4, 22, 2); r = k.run(W, x4, y4)
    print(f"{k.name:62s} ok={r['ok']} instr/MAC={r['ipm']:.4f} tok/MAC={r['tpm']:.4f}")
