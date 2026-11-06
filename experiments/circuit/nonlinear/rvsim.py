"""Minimal RV32IM(+Zba) interpreter that counts dynamic instructions and Qwenputer trace tokens, with the Qwenputer NPU ops
(VDOT/WB/SMAQA, semantics from chipformer/rv/ref.py) and this directory's proposed extensions (ext.py).  Loads the .text of a clang
-c object (no relocations: kernels take all pointers as arguments)."""
import struct, sys
from collections import Counter
import ext
M32 = 0xFFFFFFFF
def s32(x): x &= M32; return x - (1 << 32) if x >> 31 else x

def elf_text(path):
    d = open(path, "rb").read()
    shoff, = struct.unpack_from("<I", d, 32); shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 46)
    secs = []
    for i in range(shnum):
        nm, ty, fl, ad, off, sz, lk, inf, al, es = struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize); secs.append((nm, ty, off, sz, lk, es))
    stroff = secs[shstrndx][2]
    def name(o): e = d.index(b"\0", stroff + o); return d[stroff + o:e].decode()
    text = None; symtab = None
    for s in secs:
        if name(s[0]) == ".text": text = d[s[2]:s[2] + s[3]]
        if s[1] == 2: symtab = s
    syms = {}
    if symtab:
        strsec = secs[symtab[4]]
        for k in range(symtab[3] // 16):
            nmo, val, sz, info, oth, shn = struct.unpack_from("<IIIBBH", d, symtab[2] + 16 * k)
            e = d.index(b"\0", strsec[2] + nmo); syms[d[strsec[2] + nmo:e].decode()] = val
    relocs = [name(s[0]) for s in secs if name(s[0]).startswith(".rela") or name(s[0]).startswith(".rel")]
    return text, syms, [r for r in relocs if "text" in r and True]

class Sim:
    def __init__(self, text, mem=1 << 22, wrom=b""):
        self.code = {i * 4: struct.unpack_from("<I", text, i * 4)[0] for i in range(len(text) // 4)}
        self.mem = bytearray(mem); self.x = [0] * 32; self.pc = 0
        self.cnt = Counter(); self.tokens = 0; self.n = 0; self.since_refresh = 0
        self.wrom = wrom; self.wp = 0; self.ex = ext.ExtState()
    def w32(self, a, v): struct.pack_into("<I", self.mem, a, v & M32)
    def r32(self, a): return struct.unpack_from("<I", self.mem, a)[0]
    def put_i32(self, a, arr):
        for i, v in enumerate(arr): struct.pack_into("<i", self.mem, a + 4 * i, int(v))
    def get_i32(self, a, n): return list(struct.unpack_from("<%di" % n, self.mem, a))
    def put_bytes(self, a, b): self.mem[a:a + len(b)] = b
    def call(self, entry, args, max_instr=50_000_000):
        self.x = [0] * 32
        for i, v in enumerate(args): self.x[10 + i] = v & M32
        self.x[2] = len(self.mem) - 16; self.x[1] = 0xFFFFFFF0; self.pc = entry
        self.cnt.clear(); self.tokens = 0; self.n = 0; self.since_refresh = 0
        while self.pc != 0xFFFFFFF0:
            self.step(); assert self.n < max_instr
        return s32(self.x[10])
    def _tok_rd(self, v):
        v = s32(v)
        return 1 if -(1 << 14) <= v < (1 << 14) else (2 if -(1 << 29) <= v < (1 << 29) else 3)
    def step(self):
        ins = self.code[self.pc]; x = self.x
        op = ins & 0x7F; rd = (ins >> 7) & 31; f3 = (ins >> 12) & 7; rs1 = (ins >> 15) & 31; rs2 = (ins >> 20) & 31; f7 = ins >> 25
        npc = self.pc + 4; wr = None; tok = None; name = None
        a = x[rs1]; b = x[rs2]
        if op == 0x37: wr = ins & 0xFFFFF000; name = "lui"
        elif op == 0x17: wr = (self.pc + (ins & 0xFFFFF000)) & M32; name = "auipc"
        elif op == 0x6F:
            imm = ((ins >> 31) << 20) | (((ins >> 12) & 0xFF) << 12) | (((ins >> 20) & 1) << 11) | (((ins >> 21) & 0x3FF) << 1)
            if imm >> 20: imm -= 1 << 21
            wr = npc; npc = (self.pc + imm) & M32; tok = 1; name = "jal"
        elif op == 0x67:
            imm = ins >> 20; imm -= (1 << 12) if imm >> 11 else 0
            wr = npc; npc = (a + imm) & ~1 & M32; tok = 1; name = "jalr"
        elif op == 0x63:
            imm = ((ins >> 31) << 12) | (((ins >> 7) & 1) << 11) | (((ins >> 25) & 0x3F) << 5) | (((ins >> 8) & 0xF) << 1)
            if imm >> 12: imm -= 1 << 13
            sa, sb = s32(a), s32(b)
            t = [a == b, a != b, False, False, sa < sb, sa >= sb, a < b, a >= b][f3]
            name = "branch"; tok = 1
            if t: npc = (self.pc + imm) & M32
        elif op == 0x03:
            imm = ins >> 20; imm -= (1 << 12) if imm >> 11 else 0; ad = (a + imm) & M32; name = "load"
            if f3 == 0: v = self.mem[ad]; wr = v - 256 if v >> 7 else v
            elif f3 == 1: v = struct.unpack_from("<h", self.mem, ad)[0]; wr = v
            elif f3 == 2: wr = self.r32(ad)
            elif f3 == 4: wr = self.mem[ad]
            else: wr = struct.unpack_from("<H", self.mem, ad)[0]
        elif op == 0x23:
            imm = ((ins >> 25) << 5) | ((ins >> 7) & 31); imm -= (1 << 12) if imm >> 11 else 0; ad = (a + imm) & M32; name = "store"
            if f3 == 0: self.mem[ad] = b & 255; tok = 3
            elif f3 == 1: struct.pack_into("<H", self.mem, ad, b & 0xFFFF); tok = 3
            else: self.w32(ad, b); tok = 1
        elif op == 0x13:
            imm = ins >> 20; imm -= (1 << 12) if imm >> 11 else 0; name = "alu"
            if f3 == 0: wr = a + imm
            elif f3 == 1: wr = a << (imm & 31)
            elif f3 == 2: wr = int(s32(a) < imm)
            elif f3 == 3: wr = int(a < (imm & M32))
            elif f3 == 4: wr = a ^ (imm & M32)
            elif f3 == 5: wr = (s32(a) >> (imm & 31)) if (ins >> 30) & 1 else (a >> (imm & 31))
            elif f3 == 6: wr = a | (imm & M32)
            else: wr = a & (imm & M32)
        elif op == 0x33:
            sa, sb = s32(a), s32(b)
            if f7 == 1:
                name = "mul" if f3 < 4 else "div"
                if f3 == 0: wr = a * b
                elif f3 == 1: wr = (sa * sb) >> 32
                elif f3 == 2: wr = (sa * b) >> 32
                elif f3 == 3: wr = (a * b) >> 32
                elif f3 == 4: wr = -1 if b == 0 else (int(sa / sb) if not (sa == -(1 << 31) and sb == -1) else sa)
                elif f3 == 5: wr = M32 if b == 0 else a // b
                elif f3 == 6: wr = sa if b == 0 else (sa - sb * int(sa / sb) if not (sa == -(1 << 31) and sb == -1) else 0)
                else: wr = a if b == 0 else a % b
            elif f7 == 0x10: name = "alu"; wr = b + (a << {2: 1, 4: 2, 6: 3}[f3])
            else:
                name = "alu"
                if f3 == 0: wr = a - b if f7 == 0x20 else a + b
                elif f3 == 1: wr = a << (b & 31)
                elif f3 == 2: wr = int(sa < sb)
                elif f3 == 3: wr = int(a < b)
                elif f3 == 4: wr = a ^ b
                elif f3 == 5: wr = (sa >> (b & 31)) if f7 == 0x20 else (a >> (b & 31))
                elif f3 == 6: wr = a | b
                else: wr = a & b
        elif op == 0x0B:      # custom-0: Qwenputer NPU ops
            name = {5: "vdot", 6: "wb", 7: "smaqa"}[f3]
            if f3 == 6: self.wp = a; wr = None; tok = 1
            elif f3 == 5:
                cells = [self.wrom[8 * (self.wp) + k] & 15 if 8 * self.wp + k < len(self.wrom) else 0 for k in range(8)]; self.wp += 1
                av = [(((a if k < 4 else b) >> (8 * (k % 4))) & 255) for k in range(8)]; av = [v - 256 if v >> 7 else v for v in av]
                wr = x[rd] + sum(av[k] * (cells[k] - 8) for k in range(8))
            else:
                av = [((a >> (8 * k)) & 255) for k in range(4)]; bv = [((b >> (8 * k)) & 255) for k in range(4)]
                av = [v - 256 if v >> 7 else v for v in av]; bv = [v - 256 if v >> 7 else v for v in bv]
                wr = x[rd] + sum(p * q for p, q in zip(av, bv))
        elif op == 0x2B:      # custom-1: proposed extensions
            name, wr = self.ex.exec(f3, a, b, rd, x)
            if wr is None: tok = 1
        else: raise RuntimeError(f"op {op:#x} at pc {self.pc:#x}")
        self.cnt[name] += 1; self.n += 1; self.since_refresh += 1
        if wr is not None:
            if rd != 0: x[rd] = wr & M32; self.tokens += self._tok_rd(wr) if tok is None else tok
            else: self.tokens += 1
        else:
            self.tokens += 1 if tok is None else tok
            if op == 0x63 and tok == 1 and npc == self.pc + 4: pass
        if name == "div": self.tokens += 3                  # QS + QD digits (avg 4.1 tokens per DIV total)
        if self.since_refresh >= 63: self.tokens += 1; self.since_refresh = 0
        x[0] = 0; self.pc = npc
