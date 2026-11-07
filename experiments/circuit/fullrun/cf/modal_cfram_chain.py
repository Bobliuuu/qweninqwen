"""CF + REMAPPED RAM CHAINED run on Modal: the final-token run (no weight, RAM or blob host injection; circuit carries its own state across chunk boundaries).
Firmware /data/cfram/ramcf.elf, reference /data/cfram/refRC (75 chunks, 0..74), table /data/rowtab_cf.npy. Derived from modal_wf_cfram.py (per-chunk) + modal_wf_chain.py (--chain).
Only chunk 0 of a fresh start is seeded from the reference; a resumed run starts from the circuit's OWN saved state (circuit_state_NNNN.pkl, --save-state); every later chunk continues from the circuit's own state.
One unbroken 75-chunk chain is ~11-13 h on one GPU; function timeout is the Modal max (24 h) and a crash resumes from the last saved boundary (cf/chain_resume.py). spawn_cfram_chain.py (parallel segments) still seeds each segment from the reference.

  modal run --detach cf/modal_cfram_chain.py --chunks 0-9            (one segment, smoke / first segment)
  python cf/spawn_cfram_chain.py 8                                    (75 chunks as 8 segments, spaced launches; Modal rate-limits AppCreate)
  python cf/chain_resume.py launch 0-74                               (ONE unbroken chain, all 75 chunks, ~11-13 h, timeout 24 h; if it dies: python cf/chain_resume.py launch   -> resumes from the last circuit_state_NNNN.pkl)
Outputs: qfr-out /cfram_chain/chain_<first>_<last>(.stdout|/). Pull with `modal volume get qfr-out cfram_chain/<name>.stdout`.
Validated 2026-11-06: chunks 0-1 then 2 resumed from the circuit's own state = unsplit 0-2 (all boundaries ok, 0 free mismatches). See logs/cfram_chain.
FINAL PATH flags (this file): REALMAP=1 REALDECODE=1 (real in_ram/in_drom decode, no bring-up patch), --fulltable (whole row table), --stop-on-halt (circuit's own HALT), ROWEMB_CACHE (cached circuit embedding of the row table; write it once with --write-cache).
  modal run --detach cf/modal_cfram_chain.py --chunks 0-2 --reset0      (chained 3-chunk smoke test of the final path; chunk 0 boots from the program alone)
  modal run --detach cf/modal_cfram_chain.py --chunks 0-0 --reset0 --steps 2000 --write-cache --outname rowemb_build   (one-off: build /data/cfram/rowemb_cf.npy)
NOTE: the committed version of this file (78232dc) lacked the closing triple quote of this docstring (SyntaxError); fixed here.
"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-cfram-chain')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True); wfhf = modal.Volume.from_name('qfr-wfhf')   # wfhf: the merged checkpoint /ckpt_cf (the row table is read FROM ITS embed_tokens.weight)
KW = dict(image=img, timeout=24 * 3600 - 60,   # Modal max 24 h: one unbroken 75-chunk chain is ~11-13 h at ~3.8k steps/s
           volumes={'/data': data, '/out': out, '/wf': wfhf}, retries=0)

def _run(chunks, extra, outname='', resume=False, write_cache=False):
    """chunks: 'a,b,c' consecutive. outname: shared out dir name (all segments of one unbroken chain share it so circuit_state_NNNN.pkl accumulates). resume: start from the circuit's OWN saved state for chunks[0] (written by --save-state of the previous run), never the reference."""
    data.reload(); out.reload(); wfhf.reload(); first, last = chunks.split(',')[0], chunks.split(',')[-1]
    tag = outname or (f'chain_{int(first):04d}_{int(last):04d}' + ''.join(extra).replace('-', ''))
    os.makedirs('/out/cfram_chain', exist_ok=True); o = f'/out/cfram_chain/{tag}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', REALDECODE='1', WF_ROW_CELLS='3840', ROWEMB_CACHE='/data/cfram/rowemb_cf.npy')   # FINAL PATH: real in_ram/in_drom decode (no bring-up patch), cached circuit embedding of the whole row table
    if write_cache: env['ROWEMB_WRITE'] = '1'
    cmd = ['python', '-u', '/data/circuit_wfram/fullrun/circuit_run.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', chunks, '--ramimg', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy', '--rowtab-ckpt', '/wf/ckpt_cf', '--chain', '--save-state', '--fulltable', '--stop-on-halt', '--no-host-writes'] + list(extra)
    if resume:
        st = f'{o}/circuit_state_{int(first):04d}.pkl'; assert os.path.exists(st), f'no circuit state {st}: cannot resume at chunk {first}'
        cmd += ['--resume-state', st]
    with open(f'{o}/run_{int(first):04d}_{int(last):04d}.stdout', 'w') as f:
        p = subprocess.Popen(cmd, cwd='/data/qpriscv_wf', env=env, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:   # commit the log + circuit states every 5 min so progress is readable and a crash loses at most a few chunks
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired: pass
            out.commit()
    out.commit()
    if write_cache: data.commit()
    return dict(tag=tag, rc=p.returncode, tail=open(f'{o}/run_{int(first):04d}_{int(last):04d}.stdout').read()[-3000:])

@app.function(gpu='A100-40GB', **KW)
def a100(chunks, extra=(), outname='', resume=False, write_cache=False): return _run(chunks, list(extra), outname, resume, write_cache)

@app.local_entrypoint()
def main(chunks: str = '0-9', outname: str = 'unbroken', resume: bool = False, reset0: bool = False, steps: int = 0, write_cache: bool = False):
    """--resume: chunks[0] starts from the circuit's own circuit_state_<chunks[0]>.pkl in the shared outname dir (written by --save-state of the previous run). Use cf/chain_resume.py to find the chunk to resume at."""
    a, b = chunks.split('-') if '-' in chunks else (chunks, chunks)
    cl = ','.join(str(c) for c in range(int(a), int(b) + 1))
    ex = (('--reset0',) if reset0 else ()) + (('--steps', str(steps)) if steps else ())
    print(json.dumps(a100.remote(cl, ex, outname, resume, write_cache)), flush=True)
