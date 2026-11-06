"""Run any agent's driver script under HostGuard.
  python run_audit.py [--mode raise|log] [--fast] [--out report.json] [--check-outer] driver.py [driver args...]
The driver runs as __main__ (runpy).  Any exact transformers Qwen3_5ForCausalLM it calls is the 'outer'.
Exit code 0 only if there were no violations (and, with --check-outer, the outer-is-real check passed)."""
import os
import sys, os, json, runpy, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hostguard import HostGuard, HostViolation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="log"); ap.add_argument("--fast", action="store_true",
        help="no per-op python stack inspection (much faster; loses foreign-code-in-forward + stale-depth detection)")
    ap.add_argument("--out", default=None); ap.add_argument("--no-numpy-patch", action="store_true")
    ap.add_argument("--check-outer", action="store_true", help="after the run, run outer_check on the outer model(s) seen")
    ap.add_argument("--scope", action="append", default=[], help="kind:file.py:func  e.g. weight_prep:engine.py:write_frames "
        "(kinds: weight_prep|transport|build|reference|validation). Declares that python function as an exempt scope without editing the agent's code; listed in report.")
    ap.add_argument("--variant", type=int, default=0, help="offset added to every torch/numpy/random seed (+env QIQ_VARIANT) so a "
        "second run draws different host data; use with compare_runs.py to detect host-injected data in the stream")
    ap.add_argument("driver"); ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.variant:
        import torch, random, numpy as np
        os.environ["QIQ_VARIANT"] = str(a.variant)
        _ts, _ns, _rs = torch.manual_seed, np.random.seed, random.seed
        torch.manual_seed = lambda s: _ts(int(s) + 7919 * a.variant)
        np.random.seed = lambda s=None: _ns(None if s is None else (int(s) + 7919 * a.variant) % 2**32)
        random.seed = lambda s=None, *x: _rs(None if s is None else int(s) + 7919 * a.variant)
        torch.manual_seed(0)
    drv = os.path.abspath(a.driver)
    sys.argv = [drv] + a.rest
    sys.path.insert(0, os.path.dirname(drv))
    g = HostGuard(outer=None, exempt_frames=[tuple(x.split(":", 2)) for x in a.scope], mode=a.mode, stack_check=not a.fast, patch_numpy=not a.no_numpy_patch, name=os.path.basename(drv))
    seen = []
    import torch.nn.modules.module as tm
    import gc
    from transformers import Qwen3_5ForCausalLM
    def _see(m, args):
        if g._is_outer(m) and all(m is not s for s in seen):
            seen.append(m)
            if type(m) is not Qwen3_5ForCausalLM:      # keep the wrapping CausalLM alive (driver may drop it) for outer_check
                for p in gc.get_objects():
                    if type(p) is Qwen3_5ForCausalLM and getattr(p, "model", None) is m: seen.append(p)
    h = tm.register_module_forward_pre_hook(_see)
    err = None
    try:
        with g:
            try: runpy.run_path(drv, run_name="__main__")
            except SystemExit as e:
                if e.code not in (0, None): err = f"driver exited {e.code}"
    except HostViolation as e:
        err = str(e)
    except Exception:
        err = traceback.format_exc()
    h.remove()
    print(g.report_text())
    rep = g.report(); rep["driver_error"] = err
    import hashlib
    rep["source_sha256"] = {}
    for mn, mo in list(sys.modules.items()):
        fn = getattr(mo, "__file__", None)
        if fn and fn.endswith(".py") and os.sep + "site-packages" + os.sep not in fn and not fn.startswith(sys.base_prefix) and os.path.exists(fn):
            rep["source_sha256"][fn] = hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16]
    rep["source_sha256"][drv] = hashlib.sha256(open(drv, "rb").read()).hexdigest()[:16]
    print("sources hashed:", len(rep["source_sha256"]))
    if a.check_outer:
        from outer_check import check_outer
        rep["outer_checks"] = []
        import gc
        from transformers import Qwen3_5ForCausalLM
        full = [m for m in seen if type(m) is Qwen3_5ForCausalLM]
        for m in seen:
            if type(m) is not Qwen3_5ForCausalLM:      # driver calls model.model(...): find the wrapping CausalLM
                for o in gc.get_objects():
                    if type(o) is Qwen3_5ForCausalLM and getattr(o, "model", None) is m and all(o is not f for f in full): full.append(o)
        for m in full:
            r = check_outer(m); print(r["text"]); rep["outer_checks"].append({k: v for k, v in r.items() if k != "text"})
    if not seen: print("WARNING: no Qwen3_5ForCausalLM forward observed -> nothing was run on an outer model")
    if err: print("DRIVER ERROR:", err[-800:])
    if a.out: json.dump(rep, open(a.out, "w"), indent=1, default=str)
    ok = rep["PASS"] and err is None and bool(seen) and all(c["PASS"] for c in rep.get("outer_checks", []))
    print("AUDIT RESULT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
main()
