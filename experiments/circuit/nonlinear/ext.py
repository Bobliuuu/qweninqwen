"""Reference semantics (exact integer, bit-for-bit definition) of the proposed custom-1 ISA extensions for the Qwenputer RV32 core.
Each op is one instruction (custom-1, opcode 0x2b, funct3 below); tables are ROM constants held in DeltaNet key heads like the DROM.
  f3=0 EXP2  rd = 2^(rs1/2^16) * 2^24   rs1 signed Q16 (|x|<8 useful), result uint32 Q24 (saturates), 256-entry table + linear interp
  f3=1 RSQ   rd = (m/2^16)^-1/2 * 2^24  rs1 = m uint32 > 0 (Q16 mean-square + eps), priority-encoder + parity shift + 512-entry table + interp
  f3=2 SIGM  rd = sigmoid(rs1/2^16)*2^16  rs1 signed Q16, clamped to [-12,12), 768-entry table + interp, Q16 result
  f3=3 AMX   (rs1 = value int32, rs2 = index): if rs1 > MAXV: MAXV=rs1, MAXI=rs2        (hidden 2-register state, no rd)
  f3=4 AMXR  rd = MAXI; if rs1 != 0: MAXV = -2^31, MAXI = 0                              (read / reset)
  f3=5 ROT16 rs1 = (x2<<16)|x1 (int16 pair), rs2 = (s<<16)|c (Q15 pair); rd = (y2<<16)|y1, y1=(x1 c - x2 s)>>15, y2=(x1 s + x2 c)>>15 (round half up, sat int16)
"""
import numpy as np
M32 = 0xFFFFFFFF
def s32(v): v &= M32; return v - (1 << 32) if v >> 31 else v

EXP_T = [int(round(2.0 ** (i / 256) * (1 << 30))) for i in range(257)]
SIG_T = [int(round((1 / (1 + np.exp(-(-12 + i / 32)))) * (1 << 16))) for i in range(769)]
RSQ_T = [int(round((1 + 3 * i / 512) ** -0.5 * (1 << 30))) for i in range(513)]     # x in [1,4)

def exp2q(x):
    x = s32(x); xi = x >> 16; f = x & 0xFFFF; i = f >> 8; fr = f & 255
    m = (EXP_T[i] * (256 - fr) + EXP_T[i + 1] * fr) >> 8          # Q30 in [1,2)
    sh = 24 - 30 + xi                                              # result Q24
    v = (m << sh) if sh >= 0 else (m >> -sh) if -sh < 40 else 0
    return min(v, M32)
def sigmq(x):
    x = s32(x); x = max(-(12 << 16), min((12 << 16) - 1, x)); u = x + (12 << 16); i = u >> 11; fr = u & 2047
    return SIG_T[i] + (((SIG_T[i + 1] - SIG_T[i]) * fr) >> 11)
def rsqq(m):
    m &= M32
    if m == 0: return M32
    e = m.bit_length() - 1; E = e - 16; par = E & 1; Ee = E - par
    Mp = ((m << 30) >> e) << par                                   # M' in [1,4) as Q30
    u = Mp - (1 << 30); idx = min((u * 512) // (3 << 30), 511)
    fr = u - ((3 << 30) * idx) // 512                              # offset inside the segment, Q30
    t0, t1 = RSQ_T[idx], RSQ_T[idx + 1]
    y = t0 + (((t1 - t0) * fr * 512) // (3 << 30))                 # Q30 of M'^-1/2
    sh = 24 - 30 - Ee // 2
    v = (y << sh) if sh >= 0 else (y >> -sh)
    return min(v, M32)
def rot16(a, b):
    def h(v): v &= 0xFFFF; return v - 65536 if v >> 15 else v
    x1, x2, c, s = h(a), h(a >> 16), h(b), h(b >> 16)
    def sat(v): return max(-32768, min(32767, v))
    y1 = sat((x1 * c - x2 * s + (1 << 14)) >> 15); y2 = sat((x1 * s + x2 * c + (1 << 14)) >> 15)
    return ((y2 & 0xFFFF) << 16) | (y1 & 0xFFFF)

class ExtState:
    def __init__(self): self.maxv = -(1 << 31); self.maxi = 0
    def exec(self, f3, a, b, rd, x):
        if f3 == 0: return "exp2", exp2q(a)
        if f3 == 1: return "rsq", rsqq(a)
        if f3 == 2: return "sigm", sigmq(a) & M32
        if f3 == 3:
            if s32(a) > self.maxv: self.maxv, self.maxi = s32(a), b
            return "amx", None
        if f3 == 4:
            r = self.maxi
            if a != 0: self.maxv, self.maxi = -(1 << 31), 0
            return "amxr", r
        if f3 == 5: return "rot16", rot16(a, b)
        raise RuntimeError(f3)
