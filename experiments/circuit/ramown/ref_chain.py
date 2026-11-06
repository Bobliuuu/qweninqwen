"""ref_chain.py (derived from fw_ref32.py): reference trace for the CHAINED circuit run on the remapped firmware (REALMAP). Differences vs fw_ref32: blob/DROM-class loads are emitted as type-3 DROM-table injections (slot = word index mod 32768, only when the circuit DROM word would differ; per-chunk pristine mirror UNION persistent mirror, so the file is valid for both restarted and chained chunks), RAM touches record only real-RAM addresses plus the touching op (0 load, 1 SB/SH, 2 SW), VDOT rows type 2, no RAM injection at all.
Original docstring: fw_ref32.py: the RAM-minimal 0.6B firmware (ramown/fw_qwen3_ram.c, REAL Qwenputer map: RAM 0x0..0x8000 = 8192 words, rodata in the real 128 KB DROM at 0x20000, code in ROM at 0x40000)
on the software reference machine.  RAM is the REAL 8192 words (a RAM access outside faults, so a clean exit proves the working set fits); the only enlarged memories are the read-only model blob
(624.6 MB at 0x10000000, loaded by the driver) and the WROM.  inj_*.npy therefore contains ONLY blob loads (static model data: scales, norm gains, embedding row, rope) and VDOT weight rows;
no RAM word is injected (RAM loads/SB/SH are answered by the machine's RAM = by what the program stored).  Also dumps the 8192-word RAM image at every chunk start (ramimg_NNNN.npy) and
RAM stats in summary.json.  Derived from fullrun/ref_big.py.   usage: fw_ref32.py ELF BLOB WROM OUTDIR [QP_RISCV_DIR]
--- original ref_big docstring ---
Full Qwen3-0.6B int4 firmware on Qwenputer's software reference machine (ref.Machine) with ENLARGED memories (software model only):
  RAM 4 MB, RAM relocated to 0x1000000 (ROM stays at the real 0x40000 so link values match the circuit's; code is 9 KB), DROM = the 624.6 MB weight/table blob at 0x10000000,
  WROM = 74.5M words (27-bit weight pointer; the WB *token* still carries only the low 15 bits of the base, as in the real vocabulary; the full pointer is machine state).
Writes the exact trace (uint32 tokens, uint8 op-kind per token) in chunks + stats.  Resumable (ckpt.pkl).
usage: ref_big.py ELF BLOB WROM OUTDIR [QP_RISCV_DIR]"""
import sys, os, time, json, pickle, numpy as np
elf_path, blob_path, wrom_path, out = sys.argv[1:5]
sys.path.insert(0, sys.argv[5] if len(sys.argv) > 5 else os.path.expanduser('~/qp/experimental/riscv'))
from chipformer.rv import spec as S, ref, elf as E
os.makedirs(out, exist_ok=True)
blob = open(blob_path, 'rb').read(); wrom = open(wrom_path, 'rb').read()
BLOB_BASE = 0x10000000   # RAM / DROM / ROM keep their real spec values (RAM 0x0 8192 words, DROM 0x20000, ROM 0x40000)
S.WP_BITS = 27; S.WROM_WORDS = (len(wrom) + 7) // 8; S.WROM_BASE = 0x80000000
M27 = (1 << 27) - 1
MEMLOAD = ('LB', 'LBU', 'LH', 'LHU', 'LW')
KIND = dict(other=0, load=1, vdot=2, store=3, wb=4, ecall=5, branch=6, alu=7, div=8, smaqa=9, custom=10)

ST = dict(ram_ld=0, ram_sbsh=0, ram_sw=0, drom_ld=0, blob_ld=0, ram_max_ea=0)
M32 = 0xFFFFFFFF
DROMW = [0] * 32768
class Big(ref.Machine):
    def drom_word(self, ea):
        if ea >= BLOB_BASE:
            o = (ea - BLOB_BASE) & ~3
            return int.from_bytes(self.blob[o:o + 4].ljust(4, b"\0"), "little") if o < len(self.blob) else None
        return super().drom_word(ea)
    def setup(self):
        self.wbset = set(); self.kind = [0] * len(self.rom); self.dec = [None] * len(self.rom); self.inj = []; self.touch = {}; self.nbuf = lambda: 0
        self.dr_p = list(DROMW); self.dr_c = list(DROMW)   # circuit DROM table mirror: persistent (chained) and per-chunk pristine (restarted)
        ST.update(drom_inj=0, drom_need_real=0)
        for i, w in enumerate(self.rom):
            if w == 0: continue
            self.dec[i] = ref.decode(w); op = self.dec[i].op
            if op == 'WB': self.wbset.add(i)
            self.kind[i] = (KIND['load'] if op in ref.LOAD_OPS else KIND['store'] if op in ref.STORE_OPS else KIND['vdot'] if op == 'VDOT' else KIND['smaqa'] if op == 'SMAQA'
                            else KIND['wb'] if op == 'WB' else KIND['ecall'] if op == 'ECALL' else KIND['branch'] if op in ref.BRANCH_OPS or op in ('JAL', 'JALR')
                            else KIND['div'] if op in ref.DIV_OPS else KIND['custom'] if op in ref.CUSTOM else KIND['alu'])
    def record(self):
        i = self.anchor + self.n
        if self.n != S.N_MAX:
            d = self.dec[i]
            if d is not None:
                op = d.op
                if op in MEMLOAD or op in ('SB', 'SH', 'SW'):
                    ea = (self.x[d.rs1] + d.imm) & M32
                    ri = self.ram_index(ea)
                    if ri is not None:      # answered by the circuit's own RAM: never injected
                        self.touch[ri & 0x1FFF] = (ea & ~3, 2 if op == 'SW' else 1 if op in ('SB', 'SH') else 0)
                        ST['ram_sw' if op == 'SW' else 'ram_sbsh' if op in ('SB', 'SH') else 'ram_ld'] += 1; ST['ram_max_ea'] = max(ST['ram_max_ea'], ea)
                    elif op != 'SW':        # DROM-class load: real DROM rodata (0x20000..) or the blob at 0x10000000 -> circuit DROM table slot (word index mod 32768)
                        w = self.drom_word(ea); w = 0 if w is None else w; ds = (ea >> 2) & 0x7FFF
                        blob = ea >= BLOB_BASE; ST['blob_ld' if blob else 'drom_ld'] += 1
                        if self.dr_p[ds] != w or self.dr_c[ds] != w:
                            if not blob: ST['drom_need_real'] += 1
                            self.inj.append((self.nbuf(), 3, ds, w)); ST['drom_inj'] += 1
                        self.dr_p[ds] = w; self.dr_c[ds] = w
                elif op == 'VDOT':
                    self.inj.append((self.nbuf(), 2, (self.wb + i) & M27, 0))
        if self.n != S.N_MAX and i in self.wbset:
            a = self.x[ref.decode(self.rom[i]).rs1]
            self.wb = (a - i - 1) & M27; self.n += 1
            return [S.tok('WB', self.wb & 0x7FFF)]
        return super().record()
    def wrom_cells(self, wp):
        o = 8 * (wp & M27); c = self.wrom[o:o + 8]
        return list(c) if len(c) == 8 else list(c) + [0] * (8 - len(c))

prog = E.program(elf_path)
_d = np.frombuffer(bytes(prog.rodata).ljust(4 * 32768, b'\0'), dtype='<u4'); DROMW[:] = [int(v) for v in _d]
m = Big.from_program(prog); m.blob = blob; m.wrom = wrom; m.setup()
ckp = out + '/ckpt.pkl'; chunk = 0; ntok = 0; t0 = time.time(); fam = {}
if os.path.exists(ckp):
    st = pickle.load(open(ckp, 'rb')); chunk, ntok, fam = st.pop('chunk'), st.pop('ntok'), st.pop('fam')
    for k, v in st.items(): setattr(m, k, v)
    print(f'resumed at chunk {chunk} steps {m.steps}', flush=True)
else:
    first = np.array([S.tok('GO')], dtype=np.uint32); ntok = 1
def dump_state():
    tea = np.full(8192, -1, np.int64); tval = np.zeros(8192, np.int64); top = np.full(8192, -1, np.int8)
    for sl, (ea, o_) in m.touch.items():
        tea[sl] = ea; tval[sl] = m.ram[ea >> 2]; top[sl] = o_
    np.save(f'{out}/ramimg_{chunk:04d}.npy', np.array(m.ram, dtype=np.uint32))
    pickle.dump(dict(x=list(m.x), anchor=m.anchor, n=m.n, wb=m.wb, steps=m.steps, tok_start=ntok, halted=m.halted, bn=[list(r) for r in m.bn], touch_ea=tea, touch_val=tval, touch_op=top), open(f'{out}/state_{chunk:04d}.pkl', 'wb'))
    m.dr_c = list(DROMW)
CH = 1 << 21; buf = []; kbuf = []; nb = 0
def flush():
    global chunk, buf, kbuf, nb
    np.save(f'{out}/tok_{chunk:04d}.npy', np.array(buf, dtype=np.uint32)); np.save(f'{out}/kind_{chunk:04d}.npy', np.array(kbuf, dtype=np.uint8))
    np.save(f'{out}/inj_{chunk:04d}.npy', np.array(m.inj, dtype=np.int64).reshape(-1, 4)); m.inj = []
    chunk += 1; buf = []; kbuf = []; nb = 0
    dump_state()
    st = {k: getattr(m, k) for k in ('x', 'ram', 'bn', 'anchor', 'n', 'halted', 'steps', 'wb', 'out', 'fault', 'exit_code', 'touch', 'dr_p')}
    st.update(chunk=chunk, ntok=ntok, fam=fam); pickle.dump(st, open(ckp + '.tmp', 'wb')); os.replace(ckp + '.tmp', ckp)
if chunk == 0 and not buf: buf.append(S.tok('GO')); kbuf.append(0); nb = 1; ntok = 1
if chunk == 0: dump_state()
m.nbuf = lambda: nb
rom_kind = m.kind; rec = m.record; last = time.time()
while not m.halted:
    i = m.anchor + m.n; k = rom_kind[i] if m.n != S.N_MAX else 0
    r = rec(); L = len(r); buf += r; kbuf += [k] * L; nb += L; ntok += L
    if nb >= CH:
        flush()
        if chunk % 8 == 0: print(f'{ntok:,} tokens {m.steps:,} instr {time.time()-t0:.0f}s ({m.steps/(time.time()-t0):,.0f} instr/s)', flush=True)
if buf or m.inj: flush()
fams = {}
for c in range(chunk):
    t = np.load(f'{out}/tok_{c:04d}.npy'); b = np.zeros(S.VOCAB, dtype=np.int64)
    for f, (base, size) in S.FAMILIES.items(): fams[f] = fams.get(f, 0) + int(((t >= base) & (t < base + size)).sum())
    for nme, v in S.CONTROLS.items(): fams[nme] = fams.get(nme, 0) + int((t == v).sum())
instr = sum(fams.get(f, 0) for f in ('F0', 'F1', 'F2', 'J', 'BYTE', 'WB')) + sum(fams.get(f, 0) for f in ('NT', 'ST0', 'ST1', 'SS0', 'SS1', 'HALT'))
res = dict(ram_stats=ST, trace_tokens=ntok, instructions_machine=m.steps, instructions_by_final_tokens=instr, refreshes=fams.get('R', 0), families=fams, out_hex=bytes(m.out).hex(),
           argmax=int.from_bytes(bytes(m.out[:4]), 'little', signed=True), fault=m.fault, exit_code=m.exit_code, chunks=chunk, wall_s=time.time() - t0)
json.dump(res, open(out + '/summary.json', 'w'), indent=1); print(json.dumps(res), flush=True)
