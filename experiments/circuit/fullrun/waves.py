"""waves.py REFDIR OUTDIR ID NBOX [WAVE=20]: run this box's chunks (ch % NBOX == ID) on the circuit in waves, waiting for the reference run to produce them. Skips finished waves (resumable)."""
import os, sys, time, subprocess, glob
ref, out, bid, nbox = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); W = int(sys.argv[5]) if len(sys.argv) > 5 else 20
os.makedirs(out, exist_ok=True); done = set()
for f in glob.glob(out + '/result_*.json'):
    import json; done |= set(json.load(open(f))['chunks'])
def ready(c): return all(os.path.exists(f'{ref}/{n}_{c:04d}.{e}') for n, e in (('tok', 'npy'), ('inj', 'npy'), ('state', 'pkl'), ('kind', 'npy')))
while True:
    fin = os.path.exists(ref + '/summary.json')
    total = len(glob.glob(ref + '/tok_*.npy'))
    mine = [c for c in range(total) if c % nbox == bid and c not in done and ready(c) and (c + 1 < total or fin)]
    if not mine:
        if fin: break
        time.sleep(20); continue
    if len(mine) < W and not fin: time.sleep(20); continue
    wave = mine[:W]; print('wave', wave, flush=True)
    subprocess.call([os.path.expanduser('~/qp/experimental/riscv/.venv/bin/python'), os.path.dirname(os.path.abspath(__file__)) + '/circuit_run.py', os.path.expanduser('~/fr/w785/ref.elf'), ref, out,
                     '--chunks', ','.join(map(str, wave)), '--block', '100000', '--wrom', os.path.expanduser('~/fr/w785/wrom.bin')], cwd=os.path.expanduser('~/qp/experimental/riscv'), env=dict(os.environ, PYTHONPATH='.'))
    done |= set(wave)
