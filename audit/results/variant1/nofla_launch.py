"""Launcher used for the audit runs on the GPU box: forces the Hugging Face torch GDN path (no fla/triton), as in the original
A100 run (see experiments/vslice/FULL_TOKEN_RESULT.md), then runs audit/run_audit.py unchanged with the given arguments."""
import sys, runpy, os
sys.modules['fla'] = None
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(here, 'run_audit.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
