"""CF + REMAPPED RAM CHAINED run on Modal: the final-token run (no weight, RAM or blob host injection; circuit carries its own state across chunk boundaries).
Firmware /data/cfram/ramcf.elf, reference /data/cfram/refRC (73 chunks), table /data/rowtab_cf.npy. Derived from modal_wf_cfram.py (per-chunk) + modal_wf_chain.py (--chain).
Only the FIRST chunk of each segment is seeded from the reference snapshot (registers, pc, RAM image, latch); every later chunk continues from the circuit's own state.
A single 73-chunk chain is ~12 h on one GPU (> 4 h function timeout), so split into SEGMENTS: each segment is one chained run, and the seams are the only host-seeded chunk starts.

  modal run --detach cf/modal_cfram_chain.py --chunks 0-9            (one segment, smoke / first segment)
  python cf/spawn_cfram_chain.py 8                                    (73 chunks as 8 segments, spaced launches; Modal rate-limits AppCreate)
  python cf/spawn_cfram_chain.py 1                                    (ONE segment of all 73 chunks: needs timeout raise, ~12-13 h)
Outputs: qfr-out /cfram_chain/chain_<first>_<last>(.stdout|/). Pull with `modal volume get qfr-out cfram_chain/<name>.stdout`.
NOT YET RUN. Open: confirm circuit_run.py --chain works with REALMAP + --ramimg (ramimg is applied to the first chunk only); if it asserts, run with --reset0 on chunk 0 only."""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-cfram-chain')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=4 * 3600 - 60, volumes={'/data': data, '/out': out}, retries=0)

def _run(chunks, extra):
    data.reload(); first, last = chunks.split(',')[0], chunks.split(',')[-1]; tag = f'chain_{int(first):04d}_{int(last):04d}' + ''.join(extra).replace('-', '')
    os.makedirs('/out/cfram_chain', exist_ok=True); o = f'/out/cfram_chain/{tag}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', WF_ROW_CELLS='3840')
    cmd = ['python', '-u', '/data/circuit_wfram/fullrun/circuit_run.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', chunks, '--ramimg', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy', '--chain'] + list(extra)
    with open(o + '.stdout', 'w') as f:
        p = subprocess.Popen(cmd, cwd='/data/qpriscv_wf', env=env, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:   # commit the log every 5 min so progress is readable from outside
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired: pass
            out.commit()
    out.commit()
    return dict(tag=tag, rc=p.returncode, tail=open(o + '.stdout').read()[-3000:])

@app.function(gpu='A100-40GB', **KW)
def a100(chunks, extra=()): return _run(chunks, list(extra))

@app.local_entrypoint()
def main(chunks: str = '0-9', reset0: bool = False):
    a, b = chunks.split('-') if '-' in chunks else (chunks, chunks)
    cl = ','.join(str(c) for c in range(int(a), int(b) + 1))
    print(json.dumps(a100.remote(cl, ('--reset0',) if reset0 else ())), flush=True)
