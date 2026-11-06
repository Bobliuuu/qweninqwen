"""'OUTER MODEL IS REAL' checker.
check_outer(model, inner=None, declared_constants=None, distinct_thresh=64) -> dict with PASS, text, details.
Checks
  1 class identity: type(model) is transformers Qwen3_5ForCausalLM; all submodules are transformers qwen3_5 / torch.nn classes
  2 no monkeypatching: no instance 'forward' override, no instance/class hooks, every class method's code lives in
    site-packages/transformers|torch|fla; module-level callables of modeling_qwen3_5 not replaced; installed transformers
    qwen3_5/modeling files hash == pip RECORD hash
  3 'config only, parameters only': params are plain nn.Parameter; buffers listed; config dumped
  4 parameter inventory: count, layer types, per-param nnz / distinct nonzero values; params with many distinct values
    are DATA-BEARING and must be listed in declared_constants (regex -> reason), else FAIL
  5 inner weights are not baked: scale-invariant window-fingerprint search of the inner weights inside every outer parameter
"""
import os
import re, os, sys, hashlib, base64, importlib, importlib.metadata as md, collections
import torch


def _observer_only(hk):
    """True if hook's bytecode never returns anything but None, or only returns the result of a `.__setitem__/.append` call."""
    import dis
    co = getattr(hk, "__code__", None)
    if co is None: return False
    ins = list(dis.get_instructions(co))
    for i, x in enumerate(ins):
        if x.opname == "RETURN_VALUE":
            p = ins[i - 1]
            if p.opname == "LOAD_CONST" and p.argval is None: continue
            # lambda m,i,o: d.__setitem__(...)  /  l.append(...)  : last call is a METHOD call named __setitem__/append
            j = i - 1
            while j >= 0 and ins[j].opname not in ("LOAD_METHOD", "LOAD_ATTR"): j -= 1
            if p.opname in ("CALL", "CALL_METHOD", "CALL_FUNCTION") and j >= 0 and ins[j].argval in ("__setitem__", "append", "update", "setdefault"):
                continue
            return False
        if x.opname == "RETURN_CONST" and x.argval is not None: return False
    return True


def pristine_code_diff(mod):
    """Re-exec the installed modeling source in a private module and compare code objects of all module-level functions and
    class methods against the live module.  Returns list of differences (monkeypatch / rebinding detector)."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(mod.__name__ + "_pristine", mod.__file__)
    pm = importlib.util.module_from_spec(spec); pm.__package__ = mod.__package__
    import sys as _sys; _sys.modules[pm.__name__] = pm
    try: spec.loader.exec_module(pm)
    finally: _sys.modules.pop(pm.__name__, None)
    def key(co): return (co.co_code, co.co_names, tuple(c for c in co.co_consts if not isinstance(c, types.CodeType)))
    def unwrap(f):
        seen = 0
        while hasattr(f, "__wrapped__") and seen < 5: f = f.__wrapped__; seen += 1
        return f
    diffs = []
    for k, v in vars(pm).items():
        lv = getattr(mod, k, None)
        if isinstance(v, type) and v.__module__ == pm.__name__:
            for mk, mv in vars(v).items():
                f = unwrap(mv.__func__ if isinstance(mv, (staticmethod, classmethod)) else mv)
                if hasattr(f, "__code__"):
                    lc = getattr(lv, "__dict__", {}).get(mk)
                    lc = unwrap(lc.__func__ if isinstance(lc, (staticmethod, classmethod)) else lc)
                    if lc is None or not hasattr(lc, "__code__") or key(lc.__code__) != key(f.__code__): diffs.append(f"{k}.{mk}")
        elif callable(v) and hasattr(unwrap(v), "__code__") and getattr(v, "__module__", None) == pm.__name__:
            lf = unwrap(lv)
            if not hasattr(lf, "__code__") or key(lf.__code__) != key(unwrap(v).__code__): diffs.append(k)
    return diffs


def _file_hash_ok():
    bad, n = [], 0
    try:
        files = md.files("transformers")
        root = None
        for f in files:
            p = str(f)
            if re.match(r"transformers/(models/qwen3_5/.*\.py|modeling_utils\.py|modeling_layers\.py|cache_utils\.py|masking_utils\.py|"
                        r"modeling_rope_utils\.py|activations\.py|modeling_flash_attention_utils\.py|integrations/sdpa_attention\.py|"
                        r"models/qwen3_next/modeling_qwen3_next\.py)$", p) and f.hash:
                data = open(f.locate(), "rb").read(); n += 1
                h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                if f.hash.mode != "sha256" or f.hash.value != h: bad.append(p)
    except Exception as e:
        return None, [f"hash check unavailable: {e}"]
    return n, bad


def _windows_fp(x, W=8, q=2000):
    """x: 1D float tensor -> int64 hashes of scale-invariant windows (anchor = max-abs element). Returns (hash, valid)."""
    u = x.unfold(0, W, 1)
    a = u.abs().amax(1, keepdim=True)
    valid = (a[:, 0] > 1e-30)
    r = (u / a.clamp_min(1e-30) * q).round().to(torch.int64)
    nd = (u[:, 1:] != u[:, :1]).sum(1)
    valid &= nd >= 4
    mult = torch.tensor([1000003, 998244353, 1000000007, 19260817, 7000001, 3000017, 5000011, 777777773], dtype=torch.int64)[:W]
    return (r * mult).sum(1), valid


def inner_fingerprints(inner, n_samples=30000, W=8, seed=0):
    """inner: HF model / state dict / safetensors path / list of paths.  Samples windows from rows of 2D+ weights."""
    sd = {}
    if isinstance(inner, torch.nn.Module): sd = inner.state_dict()
    elif isinstance(inner, dict): sd = inner
    else:
        from safetensors import safe_open
        paths = [inner] if isinstance(inner, str) else inner
        for p in paths:
            with safe_open(p, "pt") as f:
                for k in f.keys(): sd[k] = f.get_tensor(k)
    g = torch.Generator().manual_seed(seed)
    mats = [(k, v) for k, v in sd.items() if v.dim() >= 2 and v.numel() >= 4096]
    tot = sum(v.numel() for _, v in mats); hs = set(); per = {}
    for k, v in mats:
        ns = max(50, int(n_samples * v.numel() / tot))
        flat = v.detach().float().reshape(-1, v.shape[-1]) if v.shape[-1] >= W else None
        if flat is None: continue
        rows = torch.randint(0, flat.shape[0], (ns,), generator=g); cols = torch.randint(0, flat.shape[1] - W + 1, (ns,), generator=g)
        idx = cols[:, None] + torch.arange(W)
        win = flat[rows[:, None], idx]
        a = win.abs().amax(1, keepdim=True)
        ok = (a[:, 0] > 1e-30) & ((win[:, 1:] != win[:, :1]).sum(1) >= 4)
        r = (win / a.clamp_min(1e-30) * 2000).round().to(torch.int64)
        mult = torch.tensor([1000003, 998244353, 1000000007, 19260817, 7000001, 3000017, 5000011, 777777773], dtype=torch.int64)[:W]
        h = (r * mult).sum(1)[ok]
        per[k] = set(h.tolist()); hs |= per[k]
    return hs, per


def scan_param_for(p, hs, W=8, chunk=1 << 18):
    """Count windows in param p (rows & columns direction) whose fingerprint is in hs."""
    if not hs: return 0
    hsT = torch.tensor(sorted(hs), dtype=torch.int64)
    hits = 0
    views = [p.detach().float()]
    if p.dim() == 2: views.append(p.detach().float().t())
    for v in views:
        flat = v.reshape(-1, v.shape[-1]) if v.shape[-1] >= W else None
        if flat is None: continue
        for r0 in range(0, flat.shape[0], max(1, chunk // max(1, flat.shape[1]))):
            blk = flat[r0:r0 + max(1, chunk // max(1, flat.shape[1]))]
            u = blk.unfold(1, W, 1).reshape(-1, W)
            a = u.abs().amax(1, keepdim=True)
            ok = (a[:, 0] > 1e-30) & ((u[:, 1:] != u[:, :1]).sum(1) >= 4)
            if not ok.any(): continue
            u, a = u[ok], a[ok]
            r = (u / a * 2000).round().to(torch.int64)
            mult = torch.tensor([1000003, 998244353, 1000000007, 19260817, 7000001, 3000017, 5000011, 777777773], dtype=torch.int64)[:W]
            hits += int(torch.isin((r * mult).sum(1), hsT).sum())
    return hits


def check_outer(model, inner=None, declared_constants=None, distinct_thresh=64, max_hits=3, verbose=True, allow_swaps=None):
    import transformers
    from transformers import Qwen3_5ForCausalLM
    mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    declared_constants = declared_constants or {}
    fails, warns, info, L = [], [], {}, []
    # 1 identity
    if type(model) is not Qwen3_5ForCausalLM: fails.append(f"type(model) is {type(model)}, not transformers Qwen3_5ForCausalLM")
    types = collections.Counter(); foreign_types = []
    for n, m in model.named_modules():
        c = type(m); types[c.__module__ + "." + c.__name__] += 1
        if not (c.__module__.startswith("torch.nn") or c.__module__.startswith("transformers.models.qwen3_5")
                or c.__module__.startswith("transformers.") or c.__module__.startswith("fla.") ):
            foreign_types.append((n, c))
    if foreign_types: fails.append(f"non-library module classes: {foreign_types[:5]}")
    # 2 patching
    SP = os.sep + "site-packages" + os.sep
    def lib_code(f):
        co = getattr(f, "__code__", None)
        return co is None or SP in co.co_filename or co.co_filename.startswith(sys.base_prefix)
    patched = []
    for n, m in model.named_modules():
        if "forward" in m.__dict__: patched.append(f"{n}: instance forward override")
        for hn in ("_forward_hooks", "_forward_pre_hooks"):
            for hk in (getattr(m, hn, None) or {}).values():
                co = getattr(hk, "__code__", None)
                if co is not None and SP in co.co_filename: continue          # library-installed hook (transformers output capture)
                loc = f"{os.path.basename(co.co_filename)}:{co.co_firstlineno}" if co else str(hk)
                (warns if hn == "_forward_hooks" and _observer_only(hk) else patched).append(
                    f"{n}: {hn} hook at {loc}" + (" (observer: no return value; math inside would be caught by HostGuard)" if hn == "_forward_hooks" and _observer_only(hk) else " (may MODIFY activations / inject host data)"))
        for hn in ("_backward_hooks", "_backward_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks", "_state_dict_pre_hooks"):
            if getattr(m, hn, None): patched.append(f"{n}: {hn} non-empty")
        extra = [k for k, v in m.__dict__.items() if callable(v) and not isinstance(v, torch.nn.Module) and k not in ("training",)
                 and k not in ("_compiled_call_impl",)]
        if extra: patched.append(f"{n}: instance callables {extra}")
    for c in {type(m) for m in model.modules()}:
        for k, v in vars(c).items():
            f = v.__func__ if isinstance(v, (staticmethod, classmethod)) else v
            if callable(f) and hasattr(f, "__code__") and not lib_code(f): patched.append(f"{c.__name__}.{k} code from {f.__code__.co_filename}")
    for k, v in vars(mod).items():
        if callable(v) and hasattr(v, "__code__") and not lib_code(v): patched.append(f"modeling_qwen3_5.{k} replaced by {v.__code__.co_filename}")
    try:
        pd = pristine_code_diff(mod)
        allowed = [d for d in pd if any(re.fullmatch(p, d) for p in (allow_swaps or {}))]
        if pd and len(allowed) != len(pd): fails.append(f"live modeling_qwen3_5 code differs from pristine source: {[d for d in pd if d not in allowed]}")
        if allowed: warns.append(f"declared code swaps: {allowed}")
        info["pristine_code_diff"] = pd
    except Exception as e:
        warns.append(f"pristine-code comparison unavailable: {e!r}")
    ghooks = [k for k in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks")
              if getattr(torch.nn.modules.module, k, None)]
    if ghooks: warns.append(f"global torch module hooks registered: {ghooks} (HostGuard installs some; others must be justified)")
    if patched: fails.append(f"monkeypatching: {patched[:6]}")
    n_files, bad = _file_hash_ok()
    if n_files is None: warns.append(bad[0])
    elif bad: fails.append(f"installed transformers files differ from pip RECORD: {bad}")
    info["transformers"] = transformers.__version__; info["files_hash_verified"] = n_files
    # 3 params
    nonparam = [n for n, p in model.named_parameters() if type(p) is not torch.nn.Parameter]
    if nonparam: fails.append(f"non-plain Parameters: {nonparam[:5]}")
    buffers = [(n, tuple(b.shape), float(b.abs().sum() != 0)) for n, b in model.named_buffers()]
    cfg = model.config
    ltypes = collections.Counter(getattr(cfg, "layer_types", []))
    info.update(n_params=sum(p.numel() for p in model.parameters()), layer_types=dict(ltypes), hidden=cfg.hidden_size,
                inter=cfg.intermediate_size, vocab=cfg.vocab_size, layers=cfg.num_hidden_layers, buffers=buffers,
                dtype=str(next(model.parameters()).dtype), tie=cfg.tie_word_embeddings)
    # 4 inventory
    inv = []; databearing = []
    for n, p in model.named_parameters():
        d = p.detach().float().reshape(-1)
        nz = d[d != 0]
        if nz.numel() > 4_000_000: nzs = nz[torch.randperm(nz.numel())[:2_000_000]]
        else: nzs = nz
        nd = int(torch.unique(nzs).numel()) if nzs.numel() else 0
        inv.append({"name": n, "shape": tuple(p.shape), "nnz": int(nz.numel()), "distinct_nz": nd})
        if nd > distinct_thresh:
            reason = next((r for pat, r in declared_constants.items() if re.fullmatch(pat, n)), None)
            databearing.append((n, nd, reason))
            if reason is None: fails.append(f"param {n} has {nd} distinct nonzero values (>{distinct_thresh}): data-bearing, not declared")
    info["declared_data_bearing"] = [(n, nd, r) for n, nd, r in databearing if r]
    # 5 inner not baked
    if inner is not None:
        hs, per = inner_fingerprints(inner)
        total_hits = {}
        for (n, p), row in zip(model.named_parameters(), inv):
            if row["distinct_nz"] <= 16: continue
            h = scan_param_for(p, hs)
            if h: total_hits[n] = h
        info["inner_windows_sampled"] = len(hs); info["inner_hits_in_outer_params"] = total_hits
        bad = {k: v for k, v in total_hits.items() if v > max_hits}
        if bad: fails.append(f"INNER WEIGHT FRAGMENTS FOUND in outer params (baked): {bad}")
    L.append(f"OUTER CHECK: class={type(model).__module__}.{type(model).__name__} transformers={transformers.__version__} "
             f"hash-verified {n_files} source files")
    L.append(f"  params={info['n_params']:,} dtype={info['dtype']} layers={info['layers']} types={info['layer_types']} "
             f"D={info['hidden']} inter={info['inter']} vocab={info['vocab']} tie={info['tie']}")
    L.append(f"  params with nonzero values: {sum(1 for r in inv if r['nnz'])}/{len(inv)}; "
             f"data-bearing (> {distinct_thresh} distinct): {len(databearing)} (declared {len(info['declared_data_bearing'])})")
    for n, nd, r in databearing[:10]: L.append(f"    {n}: {nd} distinct -> {'DECLARED: ' + r if r else 'UNDECLARED'}")
    if inner is not None:
        L.append(f"  inner-fingerprint search: {info['inner_windows_sampled']} windows; hits={info['inner_hits_in_outer_params'] or 'none'}")
    for w in warns: L.append("  WARN " + w)
    for f in fails: L.append("  FAIL " + f)
    L.append("OUTER IS REAL: " + ("PASS" if not fails else "FAIL"))
    return {"PASS": not fails, "fails": fails, "warns": warns, "info": info, "inventory": inv, "text": "\n".join(L)}
