"""Circuit size of the weight-fetch variants (layers, units, heads, dims) vs the 11-layer baseline. Run per config via env:
WF_ENT_BITS=4|6 WF_ROW_CELLS=N python tests/wf_stats.py"""
import json, os, sys
import numpy as np
from chipformer.rv import core, spec as S
out = []
c0 = core.build(); b = c0.stats(); out.append(dict(cfg="baseline (WROM)", **b))
S.ROW_TABLE = np.zeros((2, S.ROW_CELLS), dtype=np.uint8)
for W in (1, 4, 8, 16, 32):
    c = core.build(wrow=W); st = c.stats()
    lv = {n.name: n.level for n in c.nodes}
    sel = max(v for k, v in lv.items() if k.startswith("wf.sel")); unp = max(v for k, v in lv.items() if k.startswith("wf.u"))
    lat = [v for k, v in lv.items() if k.startswith("wlatch")]
    out.append(dict(cfg=f"W={W} ent_bits={S.ROW_ENT_BITS} row_cells={S.ROW_CELLS}", **st, sel_layer=(sel + 1) // 2, unpack_layer=(unp + 1) // 2,
                    latch_layer=[(v + 1) // 2 for v in lat], latch_key_heads=len(lat),
                    d_units=st["ffn_units"] - b["ffn_units"], d_heads=st["heads"] - b["heads"]))
    print(json.dumps(out[-1]), flush=True)
print(json.dumps(out[0]))
