"""Find where to resume an unbroken CF+RAM chain and (re)launch it from the circuit's OWN saved state (circuit_state_NNNN.pkl written by circuit_run.py --save-state at every chunk boundary).
  python cf/chain_resume.py status                 latest saved state on qfr-out cfram_chain/<outname>
  python cf/chain_resume.py launch [A-B]           start (A=0 fresh) or, with no range, resume at the latest saved state through chunk 72
Nothing here uses the reference snapshot except for chunk 0 of a fresh start and the pass/fail comparison."""
import subprocess, sys, re, os, json
OUT = 'unbroken'; LAST = 72; here = os.path.dirname(os.path.abspath(__file__))
def latest():
    r = subprocess.run(['modal', 'volume', 'ls', 'qfr-out', f'cfram_chain/{OUT}'], capture_output=True, text=True).stdout
    ks = sorted(int(m) for m in re.findall(r'circuit_state_(\d{4})\.pkl', r)); return ks[-1] if ks else None
if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'; k = latest()
    if cmd == 'status': print(json.dumps(dict(outname=OUT, latest_state_chunk=k, next=('fresh start at 0' if k is None else f'resume at chunk {k}'))))
    else:
        if len(sys.argv) > 2: a, b = [int(x) for x in sys.argv[2].split('-')]; resume = a > 0
        else:
            assert k is not None, 'no saved circuit state to resume from; pass a range like 0-72 for a fresh start'; a, b, resume = k, LAST, True
        args = ['modal', 'run', '--detach', os.path.join(here, 'modal_cfram_chain.py'), '--chunks', f'{a}-{b}', '--outname', OUT] + (['--resume'] if resume else [])
        print(' '.join(args), flush=True)
        if '--dry' not in sys.argv: subprocess.Popen(args, stdout=open(os.path.expanduser(f'~/qweninqwen-work/cfram_chain_out/launch_unbroken_{a:04d}.log'), 'w'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
