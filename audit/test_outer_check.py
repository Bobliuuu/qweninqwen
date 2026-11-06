import os
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qweninqwen"))
from machine import Layout, make_outer, gdn_const_init
from outer_check import check_outer
def mk():
    L = Layout(); L.alloc("x", 63)
    return make_outer(L, ["linear_attention", "full_attention"], 128, gdn=dict(hk=2, hv=2, dk=16, dv=16), attn=dict(heads=2, kv=1, hd=16), vocab=60), L
ok = True
def expect(name, r, want):
    global ok; good = r["PASS"] == want; ok &= good
    print("PASS" if good else "FAIL!!", name, "-> outer-check PASS=%s" % r["PASS"]); 
m, L = mk(); gdn_const_init(m.model.layers[0], L)
r = check_outer(m); print(r["text"]); expect("clean zero/const outer", r, True)
inner = {"w": torch.randn(256, 256), "w2": torch.randn(64, 512)}
r = check_outer(m, inner=inner); expect("clean vs inner", r, True)
m, L = mk()                                    # bake an inner matrix (scaled by 3.7, transposed) into mlp
with torch.no_grad(): m.model.layers[0].mlp.gate_proj.weight[:64, :63] = 3.7 * inner["w"][:63, :64].t()
r = check_outer(m, inner=inner, declared_constants={r".*gate_proj.*": "test"}); print(r["text"]); expect("baked inner scaled+transposed", r, False)
m, L = mk(); m.model.layers[0].mlp.forward = lambda x: x
expect("instance forward patch", check_outer(m), False)
m, L = mk(); m.model.layers[1].register_forward_hook(lambda a, b, c: None)
expect("hook (None-returning observer => warn)", check_outer(m), True)
m, L = mk()
with torch.no_grad(): m.model.layers[0].mlp.up_proj.weight.normal_()
expect("undeclared data-bearing", check_outer(m), False)
import transformers.models.qwen3_5.modeling_qwen3_5 as mm
orig = mm.Qwen3_5RMSNorm.forward
mm.Qwen3_5RMSNorm.forward = lambda self, x: x
m, L = mk(); expect("class forward monkeypatch", check_outer(m), False); mm.Qwen3_5RMSNorm.forward = orig
class Sub(type(m)): pass
m.__class__ = Sub; expect("subclass", check_outer(m), False)
import sys as _s; sys_ = _s
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'experiments', 'state'))
import gdn_impl; gdn_impl.set_impl("torch", "torch")
m, L = mk(); r = check_outer(m); print("gdn_impl swap ->", r["PASS"], r["fails"]); gdn_impl.set_impl("fla", "fla")
m, L = mk(); cap = {}; m.model.layers[0].register_forward_hook(lambda mo, i, o: cap.__setitem__("o", o))
r = check_outer(m); expect("observer hook is warn only", r, True); print(r["warns"])
m, L = mk(); m.model.layers[0].register_forward_hook(lambda mo, i, o: o * 2)
expect("modifying hook", check_outer(m), False)
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
