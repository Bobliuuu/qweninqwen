"""Matvec firmware kernels for the Qwenputer RV32IM(+Zba)+VDOT/SMAQA CPU.  Each builder returns a Kernel that
assembles, runs on the reference Machine, and is checked bit-exact against numpy.  y[r] = sum_k W[r,k] * x[k]
(W int4 in [-8,7], x int8, y int32), R rows total, K columns.

Token-cost facts of the circuit (chipformer/rv/ref.py Machine.record), which drive every design choice:
  * an instruction that writes rd != x0 emits value_record(v): 1 token if -2^14 <= v < 2^14, 2 if |v| < 2^29, else 3.
    rd == x0, not-taken branch, SW: 1 token.  taken branch / jal / jalr: 1 token (J).  SB / SH: 4 tokens.
  * MUL / MULH cost the same as ADD (value record of the result): there is NO extra token cost for multiplies.
  * every 63rd non-jump record costs one extra R (refresh) token.
  * VDOT/SMAQA: one record, value_record(acc).  WB: 1 token.
  So: small values are cheap (pointers < 16384, small accumulators), packed int8 words (lw of 4 bytes) cost ~2.75.
"""
import numpy as np
from harness import *

REGS = ['ra', 'sp', 'gp', 'tp', 't0', 't1', 't2', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
        'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 't3', 't4', 't5', 't6']   # all 31 usable (no calls in kernels)
WBASE = 0x20000      # DROM base: plain-weight kernels read weights from here

def problem(R, K, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.integers(-8, 8, (R, K)); x = rng.integers(-128, 128, K)
    return W, x, W @ x

class Kernel:
    def __init__(self, name, src, ram, rodata=b'', wrom=b'', R=0, K=0, out=OUT):
        self.name, self.src, self.ram, self.rodata, self.wrom, self.R, self.K, self.out = name, src, ram, rodata, wrom, R, K, out
    def run(self, W, x, ref_y):
        r = run(self.src, self.ram, self.rodata, self.wrom)
        y = np.array([r.sword(self.out + 4 * i) for i in range(self.R)])
        ok = bool((y == ref_y).all())
        macs = self.R * self.K
        return dict(name=self.name, ok=ok, instr=r.instr, tokens=r.ntok, macs=macs, ipm=r.instr / macs, tpm=r.ntok / macs,
                    fam=dict(r.fam), res=r)

def _exit(): return "li a7, 93\n li a0, 0\n ecall\n"

def _wrom_blocked(W, R, C, K):
    """weights stream for blocked VDOT: for row-block b, for chunk-group g (C chunks), for chunk c, for row r -> word."""
    nR, nK = W.shape
    out = []
    for b in range(0, nR, R):
        for g in range(0, K // 8, C):
            for c in range(C):
                for r in range(R):
                    out.append(W[b + r, (g + c) * 8:(g + c) * 8 + 8])
    return np.array(out)

def xbytes(x): return {ACT: i8bytes(x)}

# ---------------------------------------------------------------- plain RV32IM, int8 weights
def k0_naive(W, x):
    R, K = W.shape
    src = f"""_start:
 li s0, {R}
 li s1, {WBASE}
 li s2, {OUT}
row:
 li s3, {ACT}
 li s4, {ACT + K}
 li s5, 0
in:
 lb t0, 0(s1)
 lb t1, 0(s3)
 mul t0, t0, t1
 add s5, s5, t0
 addi s1, s1, 1
 addi s3, s3, 1
 bne s3, s4, in
 sw s5, 0(s2)
 addi s2, s2, 4
 addi s0, s0, -1
 bnez s0, row
""" + _exit()
    return Kernel('K0 plain C int8 (lb,lb,mul,add,ptr++,bne)', src, xbytes(x), i8bytes(W), R=R, K=K)

def k1_unroll(W, x, U=16):
    R, K = W.shape
    body = ''.join(f" lb t0, {i}(s1)\n lb t1, {i}(s3)\n mul t0, t0, t1\n add s5, s5, t0\n" for i in range(U))
    src = f"""_start:
 li s0, {R}
 li s1, {WBASE}
 li s2, {OUT}
row:
 li s3, {ACT}
 li s4, {ACT + K}
 li s5, 0
in:
{body} addi s1, s1, {U}
 addi s3, s3, {U}
 bne s3, s4, in
 sw s5, 0(s2)
 addi s2, s2, 4
 addi s0, s0, -1
 bnez s0, row
""" + _exit()
    return Kernel(f'K1 unroll x{U} (imm offsets)', src, xbytes(x), i8bytes(W), R=R, K=K)

# ---------------------------------------------------------------- SMAQA (4 int8 MAC / instr), int8 weights
def k2_smaqa(W, x, U=16):
    R, K = W.shape
    body = ''.join(f" lw t0, {4*i}(s1)\n lw t1, {4*i}(s3)\n smaqa s5, t0, t1\n" for i in range(U))
    src = f"""_start:
 li s0, {R}
 li s1, {WBASE}
 li s2, {OUT}
row:
 li s3, {ACT}
 li s4, {ACT + K}
 li s5, 0
in:
{body} addi s1, s1, {4*U}
 addi s3, s3, {4*U}
 bne s3, s4, in
 sw s5, 0(s2)
 addi s2, s2, 4
 addi s0, s0, -1
 bnez s0, row
""" + _exit()
    return Kernel(f'K2 SMAQA x{U} int8 weights', src, xbytes(x), i8bytes(W), R=R, K=K)

# ---------------------------------------------------------------- VDOT, int4 weights in WROM
def k3_vdot_row(W, x, U=8):
    """one row at a time: acts re-loaded for every row (the naive VDOT port)."""
    R, K = W.shape
    nch = K // 8
    # per iteration: 2U lw, WB, U VDOT.  weights: row-major words.
    regs = REGS[:]
    ld = ''
    for i in range(U):
        ld += f" lw {regs[13+2*i]}, {8*i}(s3)\n lw {regs[14+2*i]}, {8*i+4}(s3)\n"
    vd = ''.join(f" vdot s5, {regs[13+2*i]}, {regs[14+2*i]}\n" for i in range(U))
    src = f"""_start:
 li s0, {R}
 li s1, 0
 li s2, {OUT}
row:
 li s3, {ACT}
 li s4, {ACT + K}
 li s5, 0
in:
{ld} wb s1
{vd} addi s1, s1, {U}
 addi s3, s3, {8*U}
 bne s3, s4, in
 sw s5, 0(s2)
 addi s2, s2, 4
 addi s0, s0, -1
 bnez s0, row
""" + _exit()
    return Kernel(f'K3 VDOT, 1 row (acts reloaded per row), U={U}', src, xbytes(x), wrom=pack_wrom_nibbles(W.reshape(-1, 8)), R=R, K=K)

def k4_vdot_blocked(W, x, RB=16, C=4, wb_per_chunk=False):
    """Row-blocked VDOT: C chunks of 8 acts live in 2C registers; RB accumulators; one WB per C chunks.
    negative-index trick: ap runs -K..0 so pointer values are 1 token; weights word pointer wp is small."""
    R, K = W.shape
    assert R % RB == 0 and K % (8 * C) == 0
    acc = REGS[:RB]; actr = REGS[RB:RB + 2 * C]; (wp, ap, op, rc) = REGS[RB + 2 * C:RB + 2 * C + 4]
    ACTEND = ACT + K
    assert ACTEND < 2040 and len(REGS) >= RB + 2 * C + 4
    zero = ''.join(f" li {a}, 0\n" for a in acc)
    loads = ''.join(f" lw {actr[2*c]}, {ACTEND + 8*c}({ap})\n lw {actr[2*c+1]}, {ACTEND + 8*c + 4}({ap})\n" for c in range(C))
    vd = ''.join(f" vdot {a}, {actr[2*c]}, {actr[2*c+1]}\n" for c in range(C) for a in acc)
    sts = ''.join(f" sw {a}, {4*i}({op})\n" for i, a in enumerate(acc))
    if not wb_per_chunk:
        inner = f"{loads} wb {wp}\n{vd}"
    else:
        inner = loads + ''.join(f" wb {wp}\n" + ''.join(f" vdot {a}, {actr[2*c]}, {actr[2*c+1]}\n" for a in acc) for c in range(C))
    src = f"""_start:
 li {wp}, 0
 li {op}, {OUT}
 li {rc}, {R // RB}
blk:
 li {ap}, {-K}
{zero}in:
{inner} addi {wp}, {wp}, {RB * C}
 addi {ap}, {ap}, {8 * C}
 bnez {ap}, in
{sts} addi {op}, {op}, {4 * RB}
 addi {rc}, {rc}, -1
 bnez {rc}, blk
""" + _exit()
    wr = _wrom_blocked(W, RB, C, K)
    return Kernel(f'K4 VDOT blocked RB={RB} C={C}' + (' WB/chunk' if wb_per_chunk else ''), src, xbytes(x),
                  wrom=pack_wrom_nibbles(wr), R=R, K=K)

if __name__ == '__main__':
    import sys
    R, K = 32, 256
    W, x, y = problem(R, K)
    for k in [k0_naive(W, x), k1_unroll(W, x), k2_smaqa(W, x), k3_vdot_row(W, x), k4_vdot_blocked(W, x)]:
        r = k.run(W, x, y); print(f"{r['name']:55s} ok={r['ok']} instr/MAC={r['ipm']:.3f} tok/MAC={r['tpm']:.3f}")
