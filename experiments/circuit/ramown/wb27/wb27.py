"""WB27: the circuit holds the full 27-bit weight pointer (wbase low 15 bits + wbase_hi 12 bits).

Source patch for Qwenputer-npu experimental/riscv chipformer/rv/core.py (2717232), applied the same way
circuit_run.py applies its bring-up patches (string replace, exec as a new module). Nothing else changes:
  1. the WB uop's adder "wb.d" is widened from 4 to 7 nibbles: wb27 = x[rs1][27:0] + ~w (16-bit ~w extended
     with ones in bits 16..27) mod 2^27 = (x[rs1] - (anchor + n) - 1) mod 2^27, incl. the borrow out of the low
     15 bits and the wrap. Bits 0..14 are bit-identical to the old adder's output (same nibble sums 0..3,
     carry-lookahead is exact), so the WB token (family WB, 15-bit payload) is unchanged.
  2. a new 1-slot latch head "wbase_hi" (cols 2..13 = wb27 bits 15..26), written at the position where the WB
     uop executes (gate = the WB case's own literals nr + [wbop] (+ nb), i.e. exactly where the circuit emits
     the WB token), from the circuit's own adder output. No token carries these bits, no host writes them.
  3. snapshot() also initialises wbase_hi from m.wb27 (or m.wb) >> 15 (only used at the step-0 state restore).
The low 15 bits keep living in the existing "wbase" head (written from the WB token payload), so the WROM
addressing (WP = wbase + w mod 2^15) and the injection path are untouched.
"""
WB_HI_BITS = 12

_OLD_ADD = 'wbd_out = add_nibbles(c, A[:16], [one - b for b in w[:16]], 4, [wbop], "wb.d")[:S.WP_BITS]'
_NEW_ADD = ('wbd27 = add_nibbles(c, A[:28], [one - b for b in w[:16]] + [one] * 12, 7, [wbop], "wb.d")  # WB27: 7-nibble adder\n'
            '    wbd_out, wbd_hi = wbd27[:S.WP_BITS], wbd27[15:27]')
_OLD_CASE = 'case(nr + [wbop], "WB", wbd_out)\n'
_NEW_CASE = ('case(nr + [wbop], "WB", wbd_out)\n'
             '    # WB27: latch bits 15..26 of the 27-bit pointer where the WB uop executes (same literals as the WB case)\n'
             '    (wbhi_rd,) = thead(c, "wbase_hi", {0: one}, {0: one},\n'
             '                       [{"data": {2 + i: wbd_hi[i] for i in range(12)}, "ref": one,\n'
             '                         "beta": c.and_(nr + [wbop] + nb, "wb.hi.we")}])\n'
             '    c.tables["wbase_hi"] = c.nodes[-1]; c.wb27_reads = (wbd, wbhi_rd)\n')
_OLD_SNAP = '    c.tables["wbase"].init = [{0: wb}]\n'
_NEW_SNAP = ('    c.tables["wbase"].init = [{0: wb}]\n'
             '    if "wbase_hi" in c.tables:  # WB27\n'
             '        wbh = np.zeros(HD); wbh[list(REF_COLS)] = A_REF; hi = getattr(m, "wb27", m.wb) >> 15 & 0xFFF\n'
             '        for i in range(12):\n'
             '            wbh[2 + i] = A_BIT * (hi >> i & 1)\n'
             '        c.tables["wbase_hi"].init = [{0: wbh}]\n')


_WP27 = ('    # WB27 option wp27: the VDOT weight address WP27 = wbase27 + w mod 2^27 (same as the reference ref_big VDOT address), not consumed\n'
         '    c.wp27 = add_nibbles(c, wbd + wbhi_rd + [Expr()], w[:16] + [Expr()] * 12, 7, [], "wp27")[:27]\n')


def patch_core_src(src: str, wp27: bool = False) -> str:
    """wp27=True also builds the 27-bit VDOT address adder (for the WROW composition; off in the chain runs)."""
    case = _NEW_CASE + (_WP27 if wp27 else '')
    for old, new in ((_OLD_ADD, _NEW_ADD), (_OLD_CASE, case), (_OLD_SNAP, _NEW_SNAP)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def load_core(extra=(), wp27=False):
    """The patched core as a module (extra: further (old, new) replacements, applied first)."""
    import types
    from chipformer.rv import core as core0
    src = open(core0.__file__).read()
    for old, new in extra:
        assert old in src, old
        src = src.replace(old, new)
    mod = types.ModuleType('chipformer.rv.core_wb27'); mod.__file__ = core0.__file__
    exec(compile(patch_core_src(src, wp27), 'core_wb27.py', 'exec'), mod.__dict__)
    return mod


# ---- circuit_run.py --chain hooks (boundary readback / compare) ----
def extract_hi(o, grab, bit):
    """Read the circuit's wbase_hi head state (its own latch), add o['wb_hi'], o['wb27']."""
    o['wb_hi'] = bit(grab('wbase_hi')[0, 0], [2 + i for i in range(WB_HI_BITS)])
    o['wb27'] = o['wb'] | o['wb_hi'] << 15


def compare_hi(o, rs, d):
    d['wb27'] = (o['wb27'], rs['wb']); d['wb27_ok'] = o['wb27'] == rs['wb']
