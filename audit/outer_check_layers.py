"""outer-is-real for LAYER-MAJOR executors (no Qwen3_5ForCausalLM is ever instantiated; the outer model is a sequence of stock
Qwen3_5DecoderLayer objects built one at a time).  check_outer() cannot be applied to such a driver (it needs the CausalLM), so this
script applies the SAME checks per layer:  exact stock class / library-only submodules, no instance forward override or hooks,
live modeling_qwen3_5 code == pristine installed source, installed transformers files == pip RECORD, plain Parameters,
parameter inventory (distinct nonzero values; > distinct_thresh = data-bearing), and the scale/transposition-invariant
inner-weight fingerprint search over every parameter with > 16 distinct nonzero values.
It rebuilds the plan exactly as test_vs2.run() does (same Cfg/layout/plan/compile_entry) -- no outer forward is run here.
Usage (from experiments/vslice):  python ../../audit/outer_check_layers.py real151936_L28 --out outer_layers0.json
Exit 0 iff every layer passes."""
import sys, os, json, time, argparse, collections, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.getcwd())
import torch
import outer_check as oc

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scale"); ap.add_argument("--out", default="outer_layers.json")
    ap.add_argument("--distinct_thresh", type=int, default=64); ap.add_argument("--max_hits", type=int, default=3)
    a = ap.parse_args()
    torch.set_grad_enabled(False)   # the driver's Runner.run is @torch.no_grad()
    import vs2, test_vs2 as T
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    import importlib, transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    parts = a.scale[4:].split("_L"); Vt = int(parts[0]) if parts[0] else 4096; nl = int(parts[1]) if len(parts) > 1 else 1
    c = vs2.Cfg(1024, 16, 8, 128, 3072, nl, Vt, theta=1e6, I_MLP=8192)
    toy = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", num_hidden_layers=nl, torch_dtype=torch.float32).eval()
    Wd, P = T.weights(toy, c)
    S = vs2.build_layout(c); plan = vs2.make_plan(S, c)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rn = vs2.Runner(S, c, plan, P, dev=dev)
    # inner weights = the RAW HF checkpoint tensors (what a "baked" copy would contain) AND the gain-folded matrices the host streams
    sd = {k: v for k, v in toy.state_dict().items()}
    hs_raw, _ = oc.inner_fingerprints(sd)
    hs_fold, _ = oc.inner_fingerprints({f"folded{k}": v for k, v in Wd.items() if v.dim() == 2})
    hs = hs_raw | hs_fold
    fails_global, warns_global = [], []
    pd = oc.pristine_code_diff(mod)
    if pd: fails_global.append(f"live modeling_qwen3_5 code differs from pristine source: {pd}")
    n_files, bad = oc._file_hash_ok()
    if n_files is None: warns_global.append(bad[0])
    elif bad: fails_global.append(f"installed transformers files differ from pip RECORD: {bad}")
    SP = os.sep + "site-packages" + os.sep
    per_layer = []; agg = collections.Counter(); db_names = collections.Counter(); t0 = time.time()
    for li, entry in enumerate(plan):
        with torch.device(dev): layer = Qwen3_5DecoderLayer(rn.cfgv[rn.kind_cfg[entry[0]]], li).eval()
        vs2.compile_entry(layer, entry, S.D, c, S, P)
        fails = []
        if type(layer) is not Qwen3_5DecoderLayer: fails.append(f"type {type(layer)}")
        for n, m in layer.named_modules():
            cm = type(m).__module__
            if not (cm.startswith("torch.nn") or cm.startswith("transformers.")): fails.append(f"foreign module class {n}:{cm}")
            if "forward" in m.__dict__: fails.append(f"{n}: instance forward override")
            for hn in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
                for hk in (getattr(m, hn, None) or {}).values():
                    co = getattr(hk, "__code__", None)
                    if not (co is not None and SP in co.co_filename): fails.append(f"{n}: {hn} hook")
            extra = [k for k, v in m.__dict__.items() if callable(v) and not isinstance(v, torch.nn.Module) and k not in ("training", "_compiled_call_impl")]
            if extra: fails.append(f"{n}: instance callables {extra}")
        if [n for n, p in layer.named_parameters() if type(p) is not torch.nn.Parameter]: fails.append("non-plain Parameters")
        inv = []; hits_l = {}
        for n, p in layer.named_parameters():
            d = p.detach().float().reshape(-1); nz = d[d != 0]
            nzs = nz[torch.randperm(nz.numel(), device=nz.device)[:2_000_000]] if nz.numel() > 4_000_000 else nz
            nd = int(torch.unique(nzs).numel()) if nzs.numel() else 0
            inv.append((n, tuple(p.shape), int(nz.numel()), nd))
            if nd > a.distinct_thresh: fails.append(f"param {n} has {nd} distinct nonzero values (>{a.distinct_thresh}): data-bearing"); db_names[n] += 1
            if nd > 16:
                h = oc.scan_param_for(p.detach().cpu(), hs)
                if h: hits_l[n] = h
        bad_h = {k: v for k, v in hits_l.items() if v > a.max_hits}
        if bad_h: fails.append(f"INNER WEIGHT FRAGMENTS FOUND in outer params: {bad_h}")
        agg["layers"] += 1; agg["nz_params"] += sum(1 for r in inv if r[2]); agg["params"] += len(inv)
        agg["max_distinct_nz"] = max(agg["max_distinct_nz"], max([r[3] for r in inv] + [0]))
        per_layer.append(dict(li=li, kind=entry[0], fails=fails, max_distinct=max([r[3] for r in inv] + [0]),
                              hits=hits_l, params=[dict(name=r[0], shape=r[1], nnz=r[2], distinct_nz=r[3]) for r in inv]))
        del layer
        if dev == "cuda": torch.cuda.empty_cache()
        if li % 50 == 0: print(f"  [outer-layers] {li}/{len(plan)} t={time.time()-t0:.0f}s", flush=True)
    nfail = sum(1 for r in per_layer if r["fails"])
    kinds = collections.Counter(e[0] for e in plan)
    res = dict(PASS=(not fails_global and nfail == 0), transformers=transformers.__version__, torch=torch.__version__,
               files_hash_verified=n_files, pristine_code_diff=pd, global_fails=fails_global, global_warns=warns_global,
               n_layers=len(plan), layer_kinds=dict(kinds), layers_failing=nfail, max_distinct_nz_any_param=int(agg["max_distinct_nz"]),
               data_bearing_param_names=dict(db_names), inner_windows=len(hs), inner_windows_raw=len(hs_raw), inner_windows_folded=len(hs_fold),
               layers_with_inner_hits=[r["li"] for r in per_layer if r["hits"]], per_layer=per_layer)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(f"OUTER-LAYERS CHECK: layers={len(plan)} kinds={dict(kinds)} transformers={transformers.__version__} hash-verified files={n_files}")
    print(f"  layers failing={nfail}; max distinct nonzero values in any param={agg['max_distinct_nz']} (thresh {a.distinct_thresh}); "
          f"inner windows searched={len(hs)} (raw {len(hs_raw)} + gain-folded {len(hs_fold)}); layers with fragment hits={res['layers_with_inner_hits'][:10]}")
    for k, v in db_names.most_common(10): print(f"  DATA-BEARING param {k}: in {v} layers")
    for f in fails_global: print("  FAIL", f)
    for r in per_layer:
        for f in r["fails"][:3]: print(f"  FAIL layer {r['li']} ({r['kind']}): {f}")
        if sum(1 for q in per_layer if q["fails"]) and r["li"] > 40 and nfail > 12: break
    print("OUTER IS REAL (layer-major):", "PASS" if res["PASS"] else "FAIL"); sys.exit(0 if res["PASS"] else 1)

main()
