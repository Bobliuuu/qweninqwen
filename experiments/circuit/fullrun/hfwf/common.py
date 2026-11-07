"""Shared setup for the HF merge/check of the weight-fetch (WF) circuit: env, the BRING-UP core (same source patch as circuit_run.py:
in_ram := one so the 8192-word aliased RAM absorbs the full model's addresses), the ROW id remap.
Import this FIRST (it sets WF_ROW_CELLS / WF_ROW_SIZE before spec is imported). Needs PYTHONPATH to the Qwenputer riscv tree patched with weight_fetch/wf_npu.patch."""
import os, sys, types
NROWS = int(os.environ.get('WF_NROWS', 155205))   # 155,205 = WF table; 156,396 = CF table (rowtab_cf.npy: weights + blob constants as rows)
REALMAP = bool(os.environ.get('REALMAP'))        # CF+RAM firmware: real 32 KB RAM map, in_ram stays REAL, only in_drom is widened (same patch as circuit_run.py REALMAP=1)
os.environ.setdefault('WF_ROW_CELLS', '3840'); os.environ['WF_ROW_SIZE'] = str(NROWS)
from chipformer.rv import core as _core0
_src = open(_core0.__file__).read(); _old = 'in_ram = c.and_([1 - b for b in EA[15:32]], "in_ram")'
assert _old in _src
if REALMAP:
    _old2 = 'in_drom = c.and_([1 - b for b in EA[18:32]] + [EA[17]], "in_drom")'; assert _old2 in _src; _src = _src.replace(_old2, 'in_drom = 1 - in_ram')
else:
    _src = _src.replace(_old, 'in_ram = one')
core = types.ModuleType('chipformer.rv.core_bringup'); core.__file__ = _core0.__file__
exec(compile(_src, 'core_bringup.py', 'exec'), core.__dict__)
from chipformer.rv import spec as S
ROW_BASE, ROW_SIZE = S.FAMILIES['ROW']; assert ROW_SIZE == NROWS
DEFAULT_ROW_SIZE = 1 << 18      # numbering of the reference trace (ref_big.py ran with the default ROW family size)
def remap_trace(t):
    """reference-trace token ids (ROW family 2^18 wide) -> ids of the circuit/checkpoint built with a 155,205-row ROW family (controls shift down)."""
    import numpy as np
    t = np.asarray(t, np.int64).copy(); hi = t >= ROW_BASE + DEFAULT_ROW_SIZE
    assert not ((t >= ROW_BASE + NROWS) & (t < ROW_BASE + DEFAULT_ROW_SIZE)).any(), 'trace touches a ROW id beyond the table'
    t[hi] -= DEFAULT_ROW_SIZE - NROWS
    return t
