"""CPU diagnostic: why does `circuit_run.py --reset0` assert 'reference state_0000 != program reset state' on the CF+RAM firmware?
Builds the program exactly as circuit_run.py (REALMAP), compares ref.Machine.from_program(prog) with refdir/state_0000.pkl and ramimg_0000.npy, prints every difference.
  modal run diag_reset0.py"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-diag-reset0')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data')

@app.function(image=img, timeout=1800, volumes={'/data': data}, memory=16384)
def diag(elf='/data/cfram/ramcf.elf', refdir='/data/cfram/refRC'):
    import sys, pickle, numpy as np; data.reload()
    sys.path.insert(0, '/data/qpriscv_wf'); sys.path.insert(0, '/data/circuit_wf/engine')
    from chipformer.rv import spec as S, ref, elf as E
    prog = E.program(elf); m = ref.Machine.from_program(prog)
    st = pickle.load(open(f'{refdir}/state_0000.pkl', 'rb')); out = dict(keys=sorted(st.keys()))
    out['x_diff'] = {i: (st['x'][i], m.x[i]) for i in range(len(m.x)) if st['x'][i] != m.x[i]}
    for k in ('anchor', 'n', 'wb'): out[k] = (st.get(k), getattr(m, k, None))
    out['latch'] = (st.get('latch'), getattr(m, 'latch', None)); out['wb27'] = st.get('wb27')
    ri = np.load(f'{refdir}/ramimg_0000.npy'); mr = np.array([int(v) for v in m.ram], dtype=np.int64)
    out['ram_len'] = (len(ri), len(mr)); d = np.nonzero(ri[:len(mr)].astype(np.int64) != mr[:len(ri)])[0]
    out['ram_ndiff'] = int(len(d)); out['ram_first_diffs'] = [(int(i), int(ri[i]), int(mr[i])) for i in d[:20]]
    out['ram_nonzero_ref'] = int((ri != 0).sum()); out['ram_nonzero_prog'] = int((mr != 0).sum())
    out['S_SP_INIT'] = getattr(S, 'SP_INIT', None); out['RAM_BASE'] = getattr(S, 'RAM_BASE', None)
    return out

@app.local_entrypoint()
def main(elf: str = '/data/cfram/ramcf.elf', refdir: str = '/data/cfram/refRC'): print(json.dumps(diag.remote(elf, refdir), indent=1, default=str))
