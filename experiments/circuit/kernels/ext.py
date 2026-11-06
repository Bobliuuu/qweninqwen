"""PROPOSED new instructions, modelled on top of the Qwenputer reference Machine (custom-2, opcode 0x5B) so that kernels using them
are measured with the same token model.  They are proposals for the ISA agent; each is ONE record = one value_record(rd) like VDOT.
  f3=0 VDOT16 rd,rs1,rs2 : rd += sum_{k<16} int4(act_k) * w_k ; acts = the 8 signed nibbles of rs1 (k<8) and rs2 (k>=8);
                           w_k = cell k of WROM words (WP, WP+1), WP = D + 2*pcidx           (W4A4: 16 MAC / token)
  f3=1 WB16   rs1        : D = x[rs1] - 2*pcidx - 2 (so a VDOT16 right after it reads words x[rs1], x[rs1]+1)
  f3=2 VDOTA  rd,imm(rs1): rd += sum_{k<8} int8(mem8[x[rs1]+imm+k]) * w_k ; w = WROM word WP = wbA + pcidx (as VDOT)   (acts straight from RAM: no LW)
  f3=3 VDOTA16 rd,imm(rs1): rd += sum_{k<16} int8(mem8[x[rs1]+imm+k]) * w_k ; WROM words (WP, WP+1), WP = wbA + 2*pcidx  (W4A8, 16 MAC / token)
  f3=4 WBA    rs1        : sets the WROM base for VDOTA/VDOTA16: wbA = x[rs1] - c*pcidx - c (c = 1 for VDOTA, 2 for VDOTA16; two WBA flavours f3=4/5)
VDOTA reads the next 8/16 bytes of the *RAM* (aligned to 8/16), VDOTs' value record is the accumulator.
"""
from harness import *
from chipformer.rv.ref import Machine, decode, Inst, sx, M32 as _M32

CUS = {'vdot16': 0, 'wb16': 1, 'vdota': 2, 'vdota16': 3, 'wba': 4, 'wba16': 5}
NAMES = {v: k.upper() for k, v in CUS.items()}

def enc(f3, rd, rs1, rs2_or_imm):
    return 0x5B | (rd << 7) | (f3 << 12) | (rs1 << 15) | ((rs2_or_imm & 0xFFF) << 20)

def preprocess(src):
    """expand 'vdota rd, imm(rs1)' style lines into .word encodings."""
    out = []
    for line in src.split('\n'):
        t = line.strip()
        op = t.split(' ')[0].lower() if t else ''
        if op in CUS:
            args = [a.strip() for a in t[len(op):].split(',')]
            f3 = CUS[op]
            if op in ('vdot16',):
                w = enc(f3, asm.REG[args[0]], asm.REG[args[1]], asm.REG[args[2]])
            elif op in ('wb16', 'wba', 'wba16'):
                w = enc(f3, 0, asm.REG[args[0]], 0)
            else:                                   # vdota rd, imm(rs1)
                imm, rs1 = args[1].rstrip(')').split('(')
                w = enc(f3, asm.REG[args[0]], asm.REG[rs1], int(imm))
            out.append(f' .word {w}')
        else:
            out.append(line)
    return '\n'.join(out)

class ExtMachine(Machine):
    wb16 = 0; wba = 0; wba16 = 0
    def record(self):
        if self.n == S.N_MAX:
            return super().record()
        word = self.rom[(self.pc - S.ROM_BASE) >> 2]
        if word & 0x7F != 0x5B:
            return super().record()
        f3, rd, rs1, imm = word >> 12 & 7, word >> 7 & 31, word >> 15 & 31, word >> 20
        rs2 = imm & 31; self.steps += 1
        pcidx = self.anchor + self.n; x = self.x
        cells = lambda wp: self.wrom_cells(wp)
        def mem(a, k):
            return self.ram_byte(a + k)
        if f3 == 0:
            a, b = x[rs1], x[rs2]
            acts = [sx(a >> 4 * k, 4) for k in range(8)] + [sx(b >> 4 * k, 4) for k in range(8)]
            wp = self.wb16 + 2 * pcidx
            w = [u - 8 for u in cells(wp) + cells(wp + 1)]
            v = x[rd] + sum(p * q for p, q in zip(acts, w))
        elif f3 == 1:
            self.wb16 = (x[rs1] - 2 * pcidx - 2) & 0x7FFF; self.n += 1; return [S.tok('WB', self.wb16)]
        elif f3 in (4, 5):
            c = 1 if f3 == 4 else 2
            v = (x[rs1] - c * pcidx - c) & 0x7FFF
            if f3 == 4: self.wba = v
            else: self.wba16 = v
            self.n += 1; return [S.tok('WB', v)]
        elif f3 == 2:
            ea = (x[rs1] + sx(imm, 12)) & _M32; assert ea % 8 == 0
            acts = [sx(mem(ea, k), 8) for k in range(8)]
            w = [u - 8 for u in cells(self.wba + pcidx)]
            v = x[rd] + sum(p * q for p, q in zip(acts, w))
        else:
            ea = (x[rs1] + sx(imm, 12)) & _M32; assert ea % 16 == 0
            acts = [sx(mem(ea, k), 8) for k in range(16)]
            wp = self.wba16 + 2 * pcidx
            w = [u - 8 for u in cells(wp) + cells(wp + 1)]
            v = x[rd] + sum(p * q for p, q in zip(acts, w))
        self.n += 1
        if rd:
            x[rd] = v & _M32; return S.value_record(v)
        return [S.tok('NT')]

def run_ext(src, ram_bytes=None, rodata=b'', wrom=b'', max_records=50_000_000):
    import harness
    old = harness.ref.Machine
    harness.ref.Machine = ExtMachine
    try:
        return harness.run(preprocess(src), ram_bytes, rodata, wrom, max_records)
    finally:
        harness.ref.Machine = old
