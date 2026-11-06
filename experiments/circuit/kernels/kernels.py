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

def problem(R, K, seed=0, dist='real'):
    """dist='uniform': x uniform int8 (token-pessimistic: packed words are ~always 3 tokens).
    dist='real': heavy-tailed activations as after absmax int8 quantisation (Laplace, scale ~10, 2 outlier channels at +-127);
    weights ~ clipped round(N(0,2.5)) int4.  Token counts depend on VALUE SIZE so the data distribution matters."""
    rng = np.random.default_rng(seed)
    if dist == 'uniform':
        W = rng.integers(-8, 8, (R, K)); x = rng.integers(-128, 128, K)
    else:
        W = np.clip(np.rint(rng.normal(0, 2.5, (R, K))), -8, 7).astype(np.int64)
        x = np.clip(np.rint(rng.laplace(0, 10, K)), -127, 127).astype(np.int64)
        x[rng.choice(K, 2, replace=False)] = [127, -127]
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

def k1b_lane_pair(W, x, U=32):
    """plain RV32IM trick: TWO rows per multiply.  Weights of rows (2p, 2p+1) are stored as one word w0 + (w1 << 16); one mul by the
    (shared) activation gives both products in 16-bit lanes; flush every U=32 MACs (|lane| <= 32*8*128 < 2^15 -> exact)."""
    R, K = W.shape
    assert R % 2 == 0 and K % U == 0
    body = ''
    for k in range(U):
        body += f" lw t0, {4*k}(s1)\n lb t1, {k}(s3)\n"
        body += " mul s5, t0, t1\n" if k == 0 else " mul t0, t0, t1\n add s5, s5, t0\n"
    flush = " slli t0, s5, 16\n srai t0, t0, 16\n sub t1, s5, t0\n srai t1, t1, 16\n add s6, s6, t0\n add s7, s7, t1\n"
    src = f"""_start:
 li s0, {R // 2}
 li s1, {WBASE}
 li s2, {OUT}
row:
 li s3, {ACT}
 li s4, {ACT + K}
 li s6, 0
 li s7, 0
in:
{body}{flush} addi s1, s1, {4*U}
 addi s3, s3, {U}
 bne s3, s4, in
 sw s6, 0(s2)
 sw s7, 4(s2)
 addi s2, s2, 8
 addi s0, s0, -1
 bnez s0, row
""" + _exit()
    packed = (W[0::2].astype(np.int64) + (W[1::2].astype(np.int64) << 16)) & 0xFFFFFFFF
    ro = b''.join(int(v).to_bytes(4, 'little') for v in np.asarray(packed.T).reshape(-1) ) if False else b''
    # layout: for pair p, for k: one word  ->  index p*K + k
    ro = b''.join(int(v).to_bytes(4, 'little') for v in packed.reshape(-1))
    return Kernel(f'K1b plain RV32IM, 2 rows per mul (16-bit lanes), flush/{U}', src, xbytes(x), ro, R=R, K=K)

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

def k4_vdot_blocked(W, x, RB=16, C=4, wb_per_chunk=False, abs_ptr=False):
    """Row-blocked VDOT: C chunks of 8 acts live in 2C registers; RB accumulators; one WB per C chunks.
    negative-index trick: ap runs -K..0 so pointer values are 1 token; weights word pointer wp is small."""
    R, K = W.shape
    assert R % RB == 0 and K % (8 * C) == 0
    acc = REGS[:RB]; actr = REGS[RB:RB + 2 * C]; (wp, ap, op, rc) = REGS[RB + 2 * C:RB + 2 * C + 4]
    ACTEND = ACT + K
    if abs_ptr:        # K > ~1.7k: the negative-index trick needs the vector end address < 2048; use a real pointer + one end register
        aend = REGS[RB + 2 * C + 4]; ACTEND = 0
    else:
        assert ACTEND < 2040 and len(REGS) >= RB + 2 * C + 4
    zero = ''.join(f" li {a}, 0\n" for a in acc)
    loads = ''.join(f" lw {actr[2*c]}, {ACTEND + 8*c}({ap})\n lw {actr[2*c+1]}, {ACTEND + 8*c + 4}({ap})\n" for c in range(C))
    vd = ''.join(f" vdot {a}, {actr[2*c]}, {actr[2*c+1]}\n" for c in range(C) for a in acc)
    sts = ''.join(f" sw {a}, {4*i}({op})\n" for i, a in enumerate(acc))
    if not wb_per_chunk:
        inner = f"{loads} wb {wp}\n{vd}"
    else:
        inner = loads + ''.join(f" wb {wp}\n" + ''.join(f" vdot {a}, {actr[2*c]}, {actr[2*c+1]}\n" for a in acc) + f" addi {wp}, {wp}, {RB}\n" for c in range(C))
    src = f"""_start:
 li {wp}, 0
 li {op}, {OUT}
 li {rc}, {R // RB}
{f' li {aend}, {ACT + K}' if abs_ptr else ''}
blk:
 li {ap}, {ACT if abs_ptr else -K}
{zero}in:
{inner}{'' if wb_per_chunk else f' addi {wp}, {wp}, {RB * C}' + chr(10)} addi {ap}, {ap}, {8 * C}
 {f'bne {ap}, {aend}' if abs_ptr else f'bnez {ap}'}, in
{sts} addi {op}, {op}, {4 * RB}
 addi {rc}, {rc}, -1
 bnez {rc}, blk
""" + _exit()
    wr = _wrom_blocked(W, RB, C, K)
    return Kernel(f'K4 VDOT blocked RB={RB} C={C}' + (' WB/chunk' if wb_per_chunk else ''), src, xbytes(x),
                  wrom=pack_wrom_nibbles(wr), R=R, K=K)

def k5_vdk_like(W, x, rows4=4, acts=64):
    """The structure of the existing compiler kernel (compiler/fw/gen_vdk.py): 64 int8 acts in 16 regs, 4 rows per WB
    (32 VDOT), accumulators round-trip through memory every pass.  Re-implemented here to be measured by the same harness."""
    R, K = W.shape
    npass = K // acts
    A = REGS[11:27]                     # 16 act regs
    acc = REGS[27:31]                   # 4 accumulators
    p_acc, p_w, p_cnt, p_pass, p_ap = 'ra', 'sp', 'gp', 'tp', 't0'   # t0..: other fixed regs
    # regs used: ra sp gp tp t0 + s... A = REGS[11:27] = s4..a... ; acc REGS[27:31] = a7? t3..t6
    ld = ''.join(f" lw {A[i]}, {4*i}({p_ap})\n" for i in range(16))
    body = ''
    for r, a in enumerate(acc):
        body += f" lw {a}, {4*r}(t1)\n"
    body += f" wb {p_w}\n"
    for a in acc:
        body += ''.join(f" vdot {a}, {A[2*j]}, {A[2*j+1]}\n" for j in range(8))
    for r, a in enumerate(acc):
        body += f" sw {a}, {4*r}(t1)\n"
    src = f"""_start:
 li {p_w}, 0
 li {p_pass}, {npass}
 li {p_ap}, {ACT}
pas:
{ld} li t1, {OUT}
 li {p_cnt}, {R // rows4}
grp:
{body} addi t1, t1, 16
 addi {p_w}, {p_w}, 32
 addi {p_cnt}, {p_cnt}, -1
 bnez {p_cnt}, grp
 addi {p_ap}, {p_ap}, {acts}
 addi {p_pass}, {p_pass}, -1
 bnez {p_pass}, pas
""" + _exit()
    # OUT must start at zero: the harness RAM is zero-initialised.  weights: pass-major, row, 8 words of the pass's 64 acts
    out = []
    for p in range(npass):
        for r in range(R):
            for w8 in range(8):
                out.append(W[r, p * acts + w8 * 8:p * acts + w8 * 8 + 8])
    return Kernel('K5 VDOT, existing-compiler structure (64 acts in regs, 4 rows/WB, acc in memory)', src, xbytes(x),
                  wrom=pack_wrom_nibbles(np.array(out)), R=R, K=K)

def q_ref(acc, m, S):
    return np.clip((acc * m) >> S, -127, 127)

def k6_vdot_requant(W, x, m, S, RB=24, C=2):
    """K4 + fused epilogue: y_r = clamp((acc_r * m_r) >> S, -127, 127) packed 4 int8 per word (the next layer's activation vector).
    One outer-loop pointer (op) doubles as the m-table / output index; a sentinel byte ends the row-block loop (frees a register)."""
    R, K = W.shape
    assert R % RB == 0 and RB % 4 == 0 and K % (8 * C) == 0
    acc = REGS[:RB]; actr = REGS[RB:RB + 2 * C]; wp, ap, op = REGS[RB + 2 * C:RB + 2 * C + 3]
    assert RB + 2 * C + 3 <= 31
    ACTEND = ACT + K
    MB, SENT, OUTB = 0x1400, 0x1600, 0x1800        # m bytes, sentinel bytes, packed int8 outputs
    tmp = actr + [ap]                              # 5 temps in the epilogue (acts are reloaded next block, ap reset)
    assert len(tmp) >= 5
    t, mr, w, lo, hi = tmp[:5]
    zero = ''.join(f" li {a}, 0\n" for a in acc)
    loads = ''.join(f" lw {actr[2*c]}, {ACTEND + 8*c}({ap})\n lw {actr[2*c+1]}, {ACTEND + 8*c + 4}({ap})\n" for c in range(C))
    vd = ''.join(f" vdot {a}, {actr[2*c]}, {actr[2*c+1]}\n" for c in range(C) for a in acc)
    ep = f" li {lo}, -127\n li {hi}, 127\n"
    stubs = ''
    for i, a in enumerate(acc):
        ep += f" lb {mr}, {MB - OUTB + i}({op})\n mul {t}, {a}, {mr}\n srai {t}, {t}, {S}\n"
        ep += f" blt {t}, {lo}, FL{i}\n bgt {t}, {hi}, FH{i}\nB{i}:\n"
        stubs += f"FL{i}:\n mv {t}, {lo}\n j B{i}\nFH{i}:\n mv {t}, {hi}\n j B{i}\n"
        if i % 4 == 0: ep += f" slli {w}, {t}, 24\n"
        else: ep += f" srli {w}, {w}, 8\n slli {t}, {t}, 24\n or {w}, {w}, {t}\n"
        if i % 4 == 3: ep += f" sw {w}, {i - 3}({op})\n"       # 4 outputs = one word at OUTB + blk*RB + (i-3)
    src = f"""_start:
 li {wp}, 0
 li {op}, {OUTB}
blk:
 li {ap}, {-K}
{zero}in:
{loads} wb {wp}
{vd} addi {wp}, {wp}, {RB * C}
 addi {ap}, {ap}, {8 * C}
 bnez {ap}, in
{ep} addi {op}, {op}, {RB}
 lbu {t}, {SENT - OUTB}({op})
 bnez {t}, blk
""" + _exit() + stubs
    ram = dict(xbytes(x)); ram[MB] = i8bytes(m)
    for j in range(R // RB - 1): ram[SENT + RB * (j + 1)] = bytes([1])
    return Kernel(f'K6 VDOT blocked RB={RB} C={C} + requant/clamp/pack epilogue', src, ram,
                  wrom=pack_wrom_nibbles(_wrom_blocked(W, RB, C, K)), R=R, K=K, out=OUTB)

def k7_vdot_argmax(W, x, m, RB=24, C=2):
    """LM-head: K4 loop + epilogue best = argmax_r (acc_r * m_r) (strictly greater wins -> first max), no store, no softmax.
    taken branch per row (acc <= best is the common case) = 1 token."""
    R, K = W.shape
    acc = REGS[:RB]; actr = REGS[RB:RB + 2 * C]; wp, ap, op = REGS[RB + 2 * C:RB + 2 * C + 3]
    assert RB + 2 * C + 3 <= 31
    ACTEND = ACT + K; MB, SENT = 0x1400, 0x1600
    bestv = actr[0]; mr = actr[1]; t = actr[2]
    zero = ''.join(f" li {a}, 0\n" for a in acc)
    loads = ''.join(f" lw {actr[2*c]}, {ACTEND + 8*c}({ap})\n lw {actr[2*c+1]}, {ACTEND + 8*c + 4}({ap})\n" for c in range(C))
    vd = ''.join(f" vdot {a}, {actr[2*c]}, {actr[2*c+1]}\n" for c in range(C) for a in acc)
    # best value/index must survive the block loop: keep them in memory-free regs -> use gp-like spare: here RB+2C+3 regs used,
    # remaining spare regs hold bestv/bidx
    spare = REGS[RB + 2 * C + 3:]
    bv, bi = spare[0], spare[1] if len(spare) > 1 else None
    assert bi is not None, 'need 2 spare regs: lower RB'
    ep = ''
    for i, a in enumerate(acc):
        ep += f" lb {mr}, {i}({op})\n mul {t}, {a}, {mr}\n ble {t}, {bv}, N{i}\n mv {bv}, {t}\n addi {bi}, {op}, {i}\nN{i}:\n"
    src = f"""_start:
 li {wp}, 0
 li {op}, {MB}
 li {bv}, -2147483648
 li {bi}, 0
blk:
 li {ap}, {-K}
{zero}in:
{loads} wb {wp}
{vd} addi {wp}, {wp}, {RB * C}
 addi {ap}, {ap}, {8 * C}
 bnez {ap}, in
{ep} addi {op}, {op}, {RB}
 lbu {t}, {SENT - MB}({op})
 bnez {t}, blk
 li {t}, {MB}
 sub {bi}, {bi}, {t}
 li {t}, {OUT}
 sw {bi}, 0({t})
""" + _exit()
    ram = dict(xbytes(x)); ram[MB] = i8bytes(m)
    for j in range(R // RB - 1): ram[SENT + RB * (j + 1)] = bytes([1])
    return Kernel(f'K7 LM-head argmax RB={RB} C={C}', src, ram, wrom=pack_wrom_nibbles(_wrom_blocked(W, RB, C, K)), R=R, K=K)

if __name__ == '__main__':
    import sys
    R, K = 32, 256
    W, x, y = problem(R, K)
    for k in [k0_naive(W, x), k1_unroll(W, x), k2_smaqa(W, x), k3_vdot_row(W, x), k4_vdot_blocked(W, x)]:
        r = k.run(W, x, y); print(f"{r['name']:55s} ok={r['ok']} instr/MAC={r['ipm']:.3f} tok/MAC={r['tpm']:.3f}")
