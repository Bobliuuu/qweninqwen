"""Multi-lane (SPMD, lockstep) simulation of the Qwenputer RV32IM core on a row-split int8 matvec.

N lanes = N copies of the datapath (registers, ALU, RAM) in different DeltaNet heads/layers, one shared
control path (PC, ROM fetch, decode), all executing the SAME program on different weight rows.  Each lane is a
real `ref.Machine` (Qwenputer reference emulator, bit-exact model of the compiled circuit).  One joint step =
one instruction on every lane; the joint step must be emitted through ONE token stream, so the cost of a joint
step is bounded by the token channel (vocabulary 248,320 rows = 17.92 bits), not by compute.

Outputs per N: exactness vs numpy, instructions per lane, single-lane tokens, and three joint token costs:
  unbounded  : lanes' chunk tokens emitted in parallel (needs an N-way tuple vocabulary; not available)
  packed     : ceil(joint bits / 17.92) per step (bits = shared control + sum of per-lane payload bits)
  serial     : sum of lane tokens (lanes time-multiplexed; no speedup, lower bound on benefit)
Usage: python lanes_sim.py --rows 16 --k 64 --lanes 1 2 4 8 16 --json out.json
Set QWENPUTER=<path to qwenputer/experimental/riscv>.
"""
import argparse, json, math, os, sys
import numpy as np

QP = os.environ.get('QWENPUTER', os.path.expanduser('~/qwenos-work/qwenputer/experimental/riscv'))
sys.path.insert(0, QP)
from chipformer.rv import asm, ref, spec as S  # noqa: E402

BUS_BITS = math.log2(248320)          # stock Qwen3.5 vocabulary (0.8B and 4B): 17.92 bits per trace token
KERNEL = """
_start:
  li s0, {K}
  li s1, {DROM}
  mul t0, a0, s0
  add s1, s1, t0
  la s2, y
  slli t0, a0, 2
  add s2, s2, t0
  mv s3, a1
row:
  la t1, x
  add t2, t1, s0
  li t3, 0
inner:
  lb t4, 0(s1)
  lb t5, 0(t1)
  mul t4, t4, t5
  add t3, t3, t4
  addi s1, s1, 1
  addi t1, t1, 1
  bne t1, t2, inner
  sw t3, 0(s2)
  addi s2, s2, 4
  addi s3, s3, -1
  bnez s3, row
  li a7, 93
  li a0, 0
  ecall
.data
x: .space {K}
.align 2
y: .space {Y}
"""


def build_lane(W, x, row0, nrows):
    R, K = W.shape
    prog = asm.assemble(KERNEL.format(K=K, DROM=S.DROM_BASE, Y=4 * R))
    prog.rodata = W.astype(np.int8).tobytes()
    data = bytearray(prog.data); off = prog.labels['x'] - prog.data_base
    data[off:off + K] = x.astype(np.int8).tobytes(); prog.data = bytes(data)
    m = ref.Machine.from_program(prog)
    m.x[10], m.x[11] = row0, nrows                     # fork: lane id -> row slice
    return m, prog


def bits_of(rec):
    """(kind, payload bits) of one record in the *information* sense."""
    fams = [S.family(t)[0] for t in rec]
    last = fams[-1]
    if last in ('F0', 'F1', 'F2'):                      # rd value: its sign-extended size
        v = S.decode_value(rec); s = v - (1 << 32) if v >> 31 else v
        return 'val', max(1, abs(s).bit_length() + 1), len(rec)
    if last == 'J': return 'ctl', 16, 1
    return 'ctl', 1, 1                                  # NT, ST*, HALT: a symbol, ~1 bit


def run(R, K, N, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.integers(-128, 128, (R, K)).astype(np.int8); x = rng.integers(-128, 128, K).astype(np.int8)
    ref_y = (W.astype(np.int64) @ x.astype(np.int64)).astype(np.int32)
    per = R // N; lanes = [build_lane(W, x, i * per, per)[0] for i in range(N)]
    steps = 0; tok_serial = 0; tok_unb = 0; uni_steps = 0; tok_pack = 0; bits_tot = 0.0; bits_val = 0
    single_tokens = 0
    while not lanes[0].halted:
        recs = [m.record() for m in lanes]; steps += 1
        assert len({(m.anchor + m.n) for m in lanes}) == 1 or all(m.halted for m in lanes), 'lanes diverged'
        info = [bits_of(r) for r in recs]
        ctl = max((b for k, b, _ in info if k == 'ctl'), default=0)   # shared control symbol, emitted once
        uniform = all(r == recs[0] for r in recs)                       # SPMD-uniform value (pointer, counter): emitted once
        val = (max((b for k, b, _ in info if k == 'val'), default=0) if uniform else sum(b for k, b, _ in info if k == 'val'))
        bits = ctl + val + (2 if val else 0)                            # 2 bits: size class of the value chunk set
        uni_steps += uniform
        tok_serial += sum(len(r) for r in recs)
        tok_unb += max(len(r) for r in recs)
        tok_pack += max(1, math.ceil(bits / BUS_BITS))
        bits_tot += bits
    # gather y from lane RAMs
    y = np.zeros(R, np.int32)
    for i, m in enumerate(lanes):
        for r in range(i * per, (i + 1) * per):
            addr = prog_y(R, K) + 4 * r; w = m.ram[(addr - S.RAM_BASE) >> 2]
            y[r] = np.int32(np.uint32(w))
    exact = bool(np.array_equal(y, ref_y))
    return dict(R=R, K=K, lanes=N, exact=exact, steps=steps, instr_per_lane=steps, macs=R * K,
                instr_per_mac_total=steps * N / (R * K), tok_serial=tok_serial, tok_unbounded=tok_unb, tok_packed=tok_pack,
                tok_ideal_bits=math.ceil(bits_tot / BUS_BITS), uniform_frac=uni_steps / steps, bits_per_step=bits_tot / steps)


_YADDR = {}
def prog_y(R, K):
    if (R, K) not in _YADDR:
        p = asm.assemble(KERNEL.format(K=K, DROM=S.DROM_BASE, Y=4 * R)); _YADDR[(R, K)] = p.labels['y']
    return _YADDR[(R, K)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rows', type=int, default=16); ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--lanes', type=int, nargs='+', default=[1, 2, 4, 8, 16]); ap.add_argument('--json')
    a = ap.parse_args(); out = []
    base = None
    for N in a.lanes:
        r = run(a.rows, a.k, N); base = base or r['tok_serial']
        r['speedup_unbounded'] = base / r['tok_unbounded']; r['speedup_packed'] = base / r['tok_packed']
        r['speedup_ideal_bits'] = base / r['tok_ideal_bits']; out.append(r); print(json.dumps(r), flush=True)
    if a.json: json.dump(out, open(a.json, 'w'), indent=1)
