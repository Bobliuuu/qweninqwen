"""Run fullrun/circuit_run.py with the REAL circuit address decode (no bring-up patch).

circuit_run.py (REALMAP=1) rewrites the compiled core's source: `in_drom = 1 - in_ram` replaces the real
`in_drom = and(~EA[18:32], EA[17])` so that loads of the 0x10000000 weight/table blob are answered by the DROM table.
On the CF (constants-fetch) firmware there are NO blob loads (per-chunk stats_NNNN.json: blob_ld == 0 for all chunks; the blob constants come
from table rows via WROW+VDOT), so the widening is dead weight and a bring-up host crutch. This launcher removes it at run time:
it reads circuit_run.py ($CIRCUIT_RUN or ../../fullrun/circuit_run.py), turns the single `_src.replace(_old2, 'in_drom = 1 - in_ram')`
into a no-op, and execs the rest unchanged (argv, __file__, __name__ preserved). circuit_run.py itself is not modified.
With this, in_ram = real (EA[15:32]==0, 32 KB), in_drom = real (0x20000..0x3FFFF, 128 KB), anything else (e.g. a blob address) is neither.
usage: REALMAP=1 [CIRCUIT_RUN=path] python run_realdecode.py <circuit_run.py args...>"""
import os, sys
path = os.environ.get('CIRCUIT_RUN') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'fullrun', 'circuit_run.py')
path = os.path.abspath(path); src = open(path).read()
OLD = "_src = _src.replace(_old2, 'in_drom = 1 - in_ram')"
assert src.count(OLD) == 1, f'expected exactly one widening line in {path}, found {src.count(OLD)}'
assert os.environ.get('REALMAP'), 'REALMAP=1 required (real 32 KB RAM map firmware)'
src = src.replace(OLD, "pass   # REAL DECODE: in_drom stays the real 128 KB DROM window (patch removed by ramown/realdecode/run_realdecode.py)")
sys.argv[0] = path; sys.path.insert(0, os.path.dirname(path))
print('[realdecode] in_drom widening removed; real in_ram + in_drom decode active', flush=True)
exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})
