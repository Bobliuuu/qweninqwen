"""WB27 validation (1): tiny firmware, IR evaluate (teacher-forced, check=True with all exactness assertions).
The circuit's 27-bit weight pointer (wbase low 15 + wbase_hi 12, read from the head outputs at every position)
must equal a reference machine with S.WP_BITS = 27 (its WB value before the 15-bit token mask), incl. upper
bits non-zero, the borrow/wrap case x[rs1] < anchor + n + 1, and x[rs1] bits >= 27 (dropped mod 2^27).
Run in Qwenputer-npu experimental/riscv: PYTHONPATH=.:tests:<this dir> python test_wb27.py"""
import random, sys, time, json
import numpy as np
from chipformer.rv import core as core0, ref, spec as S
from chipformer.rv.asm import assemble
from chipformer.q35 import ir
import wb27

EXIT = "\nli a7, 93\nli a0, 0\necall\n"


def sx32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


def program(rng, bases, n_runs, n_words):
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {sx32(rng.getrandbits(32))}" for r in regs]
    for i in range(n_runs):
        b = bases[i % len(bases)]
        lines += [f"li s4, {sx32(b)}", "wb s4"]
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.2:
                lines.append(f"xori {rng.choice(regs)}, {rng.choice(regs)}, {rng.randint(-2048, 2047)}")
            lines.append(f"vdot s2, {rng.choice(regs)}, {rng.choice(regs)}")
    cells = bytes(rng.randrange(16) for _ in range(8 * n_words))
    return "\n".join(lines) + EXIT + "\n.wrom\n" + "\n".join(
        f".byte {', '.join(str(x) for x in cells[i:i + 64])}" for i in range(0, len(cells), 64))


def ref27(prog):
    """Reference with S.WP_BITS = 27: record the full WB value (token payload masked to 15 bits, WROM read mod 2^15)."""
    seen, otok, ob = [], S.tok, S.WP_BITS

    def tok(name, payload=0):
        if name == "WB":
            seen.append(payload); payload &= 0x7FFF
        return otok(name, payload)

    class M27(ref.Machine):
        def wrom_cells(self, wp):
            return ref.Machine.wrom_cells(self, wp & 0x7FFF)
    S.tok, S.WP_BITS = tok, 27
    try:
        m = M27.from_program(prog); toks = m.run(10**6)
    finally:
        S.tok, S.WP_BITS = otok, ob
    return toks, seen, m.wb


def check(c, cw, prog, name, log):
    toks = ref.Machine.from_program(prog).run(10**6)
    toks27, wb27s, wbend = ref27(prog)
    assert toks27 == toks, "27-bit reference trace differs from the 15-bit one"
    cw.snapshot(c, ref.Machine.from_program(prog))
    t0 = time.time(); mat = ir.evaluate(c, toks, check=True, kernel=len(toks) < 1500); te = time.time() - t0
    pred = cw.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:5]
    lo_e, hi_e = c.wb27_reads
    val = lambda es: [np.rint(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items())).astype(np.int64) if e.terms else np.zeros(len(toks), np.int64) for e in es]
    lo = sum(b << i for i, b in enumerate(val(lo_e))); hi = sum(b << i for i, b in enumerate(val(hi_e)))
    circ27 = lo | hi << 15
    # expected at position p: the full value of the last WB token at <= p; position p with toks[p+1] == WB is the
    # exec position (hi already latched there, lo latched one position later from the token): checked separately
    exp = np.zeros(len(toks), np.int64); cur, k, execpos = 0, 0, []
    for p, t in enumerate(toks):
        if S.family(t)[0] == "WB":
            cur = wb27s[k]; k += 1
        exp[p] = cur
        if p + 1 < len(toks) and S.family(toks[p + 1])[0] == "WB":
            execpos.append(p)
    steady = np.ones(len(toks), bool); steady[execpos] = False
    nbad = int((circ27[steady] != exp[steady]).sum())
    # at the exec position the hi latch already holds the NEW value's bits 15..26
    nxt = np.array([wb27s[i] for i in range(len(execpos))], np.int64)
    exec_hi_bad = int((hi[execpos] != nxt >> 15).sum()) if execpos else 0
    nhi = sum(1 for v in wb27s if v >> 15); n_r = sum(S.family(t)[0] == 'R' for t in toks)
    r = dict(name=name, tokens=len(toks), wb_records=len(wb27s), wb_with_upper_bits=nhi, wb_wrapped_borrow=sum(1 for v in wb27s if v >= (1 << 27) - 4096), R_refresh_tokens=n_r, wb27_values=[hex(v) for v in wb27s[:12]],
             positions_checked=int(steady.sum()), wb27_mismatch=nbad, exec_pos_hi_mismatch=exec_hi_bad, ref_final_wb27=hex(wbend),
             circ_final_wb27=hex(int(circ27[-1])), token_mismatch=len(bad), eval_s=round(te, 1))
    print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    assert nbad == 0 and exec_hi_bad == 0 and int(circ27[-1]) == wbend
    return r


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "wb27_test.jsonl"
    log = open(out, "w")
    t0 = time.time(); c0 = core0.build(); s0 = c0.stats(); t_b0 = time.time() - t0
    cw = wb27.load_core(); t0 = time.time(); c = cw.build(); s1 = c.stats(); t_b1 = time.time() - t0
    nodes = lambda cc: len(cc.nodes)
    st = dict(baseline=s0, wb27=s1, nodes=(nodes(c0), nodes(c)), build_s=(round(t_b0, 1), round(t_b1, 1)))
    print("STATS", json.dumps(st, default=str), flush=True); log.write("STATS " + json.dumps(st, default=str) + "\n")
    # bases: x[rs1] values; borrow: x < w + 1 (small x) -> wraps to 2^27 - (w + 1 - x); bits 27..31 set; exact 2^27 - 1 boundary
    B1 = [0x05ABC123, 0x00000000, 0x00000003, 0x07FFFFFF, 0xFFFFFFFF, 0x80001234, 0x12345678, 0x00008000, 0x00007FFF, 0x0400_0000]
    rng = random.Random(1)
    check(c, cw, assemble(program(rng, B1, 10, 256)), "fixed_bases", log)
    for seed in range(3):   # random bases, many runs (n reaches 63 -> R refresh before some WB)
        rng = random.Random(100 + seed)
        bases = [rng.choice([rng.getrandbits(32), rng.getrandbits(27), rng.randrange(64), (1 << 27) - rng.randrange(64)]) for _ in range(40)]
        check(c, cw, assemble(program(rng, bases, 40, 128)), f"random_{seed}", log)
    print("ALL OK", flush=True); log.write("ALL OK\n")
