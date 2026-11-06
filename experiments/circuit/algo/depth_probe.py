"""Circuit size / depth of the qwenTPU tile machine for different (rows, cols, act bits, weight bits, g, h).
Uses the *upstream compiler* from reference/nested-qwen-lab (qwentpu.circuit.configure/build/horner_plan): numbers are measured from the real IR,
not modelled.  Output: layers (= outer Qwen3.5 decoder layers = circuit depth), dims (outer hidden size), heads, ffn units, Horner levels."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('QWENTPU_LAB', os.path.join(HERE, '..', '..', '..', 'reference', 'nested-qwen-lab')))
from qwentpu import circuit as C

def probe(kr, nc, ab, wb, g=1, h=1):
    t = time.time(); C.configure(kr, nc, ab, wb, g, h)
    plan = C.horner_plan(); b = C.build(); gr = b[0] if isinstance(b, tuple) else b; st = gr.stats()
    return dict(kr=kr, nc=nc, ab=ab, wb=wb, g=g, h=h, horner_levels=len(plan), clamp_passes=sum(p[5] for p in plan), vocab=len(C.VOCAB), build_s=round(time.time() - t, 2), **st)

def main(configs=None):
    cfgs = configs or [(8, 8, 3, 3, 1, 1), (8, 8, 4, 4, 1, 1), (8, 8, 6, 4, 1, 1), (8, 8, 6, 6, 1, 1), (8, 8, 2, 2, 1, 1), (8, 8, 4, 2, 1, 1),
                       (16, 8, 4, 4, 1, 1), (32, 8, 4, 4, 1, 1), (32, 8, 4, 4, 2, 1), (32, 8, 4, 4, 1, 2), (32, 8, 4, 4, 2, 2),
                       (32, 16, 4, 4, 1, 1), (32, 16, 4, 4, 2, 2), (64, 8, 4, 4, 2, 1), (32, 8, 3, 3, 1, 1), (32, 8, 2, 2, 1, 1)]
    out = []
    for c in cfgs:
        try: r = probe(*c)
        except Exception as e: r = dict(cfg=list(c), error=repr(e)[:200])
        print(json.dumps(r), flush=True); out.append(r)
    return dict(probes=out)

if __name__ == '__main__': main()
