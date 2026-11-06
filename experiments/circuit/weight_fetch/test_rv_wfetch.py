"""Weight fetch: VDOT weights from rows of the outer model's own table (ROW tokens), teacher-forced.

The table rows (uint8 cells 0..15, ROW_CELLS per row) are the ROW tokens' embeddings; the circuit
selects the slot's cells, unpacks the int4 bits, latches them in a DeltaNet head and feeds the VDOT
multiplier; the reference machine reads the same table. WROM_BIN=<wrom.bin> takes the rows from
the real Qwen3-0.6B int4 stream (one nibble per byte), else random.
"""
import os
import random
import sys
import time

import numpy as np
import pytest

from chipformer.q35 import ir
from chipformer.rv import core, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

sys.path.insert(0, os.path.dirname(__file__))
from test_rv_core import EXIT  # noqa: E402
from test_rv_vdot import sx32, word  # noqa: E402

NROWS = 6


def make_table(seed=0, nrows=NROWS):
    p = os.environ.get("WROM_BIN")
    if p and os.path.exists(p):
        off = int(os.environ.get("WROM_OFF", 7_000_003))
        raw = np.fromfile(p, dtype=np.uint8, count=off + nrows * S.ROW_CELLS)[off:]
        return (raw[: nrows * S.ROW_CELLS] & 15).reshape(nrows, S.ROW_CELLS).copy()
    rng = np.random.default_rng(seed)
    return rng.integers(0, 16, (nrows, S.ROW_CELLS), dtype=np.uint8)


def wf_program(rng, tab, W, n_blocks, per_block):
    """Blocks of `wb; wrow; vdot x per_block` (other instructions between some), W-aligned."""
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {sx32(word(rng))}" for r in regs]
    for _ in range(n_blocks):
        row = rng.randrange(len(tab) + 1)  # one past the table too: reads as zeros
        win = rng.randrange(S.ROW_CELLS // 8 // W)
        x = win * W - 1  # wb: the next word (wrow) is at WP = x, the first vdot at win * W
        acc = rng.choice(["s2", "s3"])
        init = rng.choice([0, rng.randint(-2**14, 2**14), rng.randint(-2**31, 2**31 - 1)])
        lines += [f"li s4, {x & 0x7FFF}", f"li s5, {row}", f"li {acc}, {init}", "wb s4", "wrow s5"]
        for i in range(per_block):
            if i and rng.random() < 0.0:
                lines.append("nop")
            lines.append(f"vdot {acc}, {rng.choice(regs)}, {rng.choice(regs)}")
    return "\n".join(lines) + EXIT


def check(c, prog, tab, W):
    m = ref.Machine.from_program(prog)
    m.rowtab, m.wrow_w = tab, W
    core.snapshot(c, m)
    m2 = ref.Machine.from_program(prog)
    m2.rowtab, m2.wrow_w = tab, W
    toks = m2.run()
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True)
    pred = core.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:5]
    return toks, time.time() - t0


@pytest.fixture(scope="module", params=[1, 8])
def circ_tab(request):
    W = request.param
    tab = make_table()
    S.ROW_TABLE = tab
    return W, tab, core.build(wrow=W)


@pytest.mark.parametrize("seed", range(3))
def test_weight_fetch(circ_tab, seed):
    W, tab, c = circ_tab
    rng = random.Random(seed)
    toks, dt = check(c, assemble(wf_program(rng, tab, W, 4, 1 if W == 1 else min(W, 5))), tab, W)
    fams = [S.family(t)[0] for t in toks]
    assert "ROW" in fams and fams[-1] == "HALT"


def test_negative_control(circ_tab):
    """The reference reads a table with every cell shifted by one: the circuit must disagree."""
    W, tab, c = circ_tab
    rng = random.Random(7)
    prog = assemble(wf_program(rng, tab, W, 4, 1 if W == 1 else min(W, 5)))
    bad_tab = (tab + 1) & 15
    with pytest.raises(AssertionError):
        check(c, prog, bad_tab, W)


def test_stats(circ_tab):
    W, tab, c = circ_tab
    st = c.stats()
    print(f"\nWF_STATS W={W} {st}")
    assert st["max_heads"] <= c.target.key_heads and st["max_units"] <= c.target.intermediate
