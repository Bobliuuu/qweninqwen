"""Split the 75-chunk CF+RAM reference into N chained segments and launch each as `modal run --detach cf/modal_cfram_chain.py` (25 s apart: Modal rate-limits AppCreate).
usage: python cf/spawn_cfram_chain.py N [--dry]   (N=8 -> segments of 9-10 chunks, ~1.6-1.8 h each at ~10 min/chunk)
Logs: ~/qweninqwen-work/cfram_chain_out/launch_<first>.log. NOT launched automatically; run only after the user says go."""
import subprocess, sys, time, os
N = int(sys.argv[1]); dry = '--dry' in sys.argv; TOTAL = 75
base, rem = divmod(TOTAL, N); segs = []; s = 0
for i in range(N):
    n = base + (1 if i < rem else 0); segs.append((s, s + n - 1)); s += n
od = os.path.expanduser('~/qweninqwen-work/cfram_chain_out'); os.makedirs(od, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
for a, b in segs:
    cmd = ['modal', 'run', '--detach', os.path.join(here, 'modal_cfram_chain.py'), '--chunks', f'{a}-{b}']
    print(segs.index((a, b)), a, b, ' '.join(cmd), flush=True)
    if dry: continue
    subprocess.Popen(cmd, stdout=open(f'{od}/launch_{a:04d}.log', 'w'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(25)
