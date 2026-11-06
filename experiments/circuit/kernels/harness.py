"""Harness: assemble a kernel with the Qwenputer (npu branch) assembler, run it on the Qwenputer reference
Machine (the exact trace the compiled circuit must reproduce), count instructions and TRACE TOKENS, check outputs
against a numpy reference.  Needs a checkout of github.com/divagr18/qwenputer branch `npu`:
    export QWENPUTER_NPU=/path/to/qwenputer/experimental/riscv
"""
import os, sys, collections
import numpy as np
sys.dont_write_bytecode = True
_p = os.environ.get('QWENPUTER_NPU', os.path.expanduser('~/qwenos-work/qwenputer-npu/experimental/riscv'))
sys.path.insert(0, _p)
from chipformer.rv import asm, ref, spec as S      # noqa: E402

M32 = 0xFFFFFFFF
ACT = 0x100          # activations (int8), RAM, low addresses so pointer values stay one-token
OUT = 0x1000         # outputs (int32 words)
ACTW = 0x800         # second area (e.g. biases / packed acts)

def to_sw(a):  # numpy int -> signed 32
    a = int(a) & M32
    return a - (1 << 32) if a >> 31 else a

class Result:
    def __init__(s, m, toks, prog):
        s.m, s.toks, s.prog = m, toks, prog
        s.instr = m.steps
        s.fam = collections.Counter(S.family(t)[0] for t in toks)
        s.ntok = len(toks) - 1                      # minus the GO token
    def word(s, addr): return s.m.ram[addr >> 2]
    def sword(s, addr): return to_sw(s.word(addr))
    def byte(s, addr): return (s.word(addr & ~3) >> (8 * (addr & 3))) & 255

def run(src, ram_bytes=None, rodata=b'', wrom=b'', max_records=50_000_000):
    """ram_bytes: dict addr->bytes preloaded into RAM.  Returns Result."""
    prog = asm.assemble(src)
    prog.rodata, prog.wrom = bytes(rodata), bytes(wrom)
    m = ref.Machine.from_program(prog)
    for a, b in (ram_bytes or {}).items():
        for i, v in enumerate(b):
            w = (a + i) >> 2; sh = 8 * ((a + i) & 3)
            m.ram[w] = (m.ram[w] & ~(255 << sh) & M32) | (v << sh)
    toks = [S.tok('GO')]; by_op = collections.Counter(); cnt_op = collections.Counter()
    while not m.halted and len(toks) < max_records:
        pc = m.pc; w_ = m.rom[(pc - S.ROM_BASE) >> 2]
        op = ('EXT%d' % (w_ >> 12 & 7) if w_ & 0x7F == 0x5B else ref.decode(w_).op) if m.n != S.N_MAX else 'REFRESH'
        t = m.record(); toks += t; by_op[op] += len(t); cnt_op[op] += 1
    assert m.halted and m.fault is None, ('fault', m.fault)
    r = Result(m, toks, prog); r.by_op, r.cnt_op = by_op, cnt_op
    return r

def pack_wrom_nibbles(W):
    """W: int array (n_words, 8) of weights in [-8,7] -> WROM bytes (cell = w+8)."""
    return bytes((np.asarray(W, dtype=np.int64).reshape(-1) + 8).astype(np.uint8).tolist())

def i8bytes(a): return bytes((np.asarray(a, dtype=np.int64).reshape(-1) & 255).astype(np.uint8).tolist())
