"""Positive/negative tests for HostGuard. Run: python test_hostguard.py"""
import os
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qweninqwen"))
from hostguard import HostGuard, HostViolation, transport, reference, validation_read
from machine import Layout, make_outer

X4x4=torch.randn(4,4); X4=torch.randn(4); X10=torch.randn(10); X3=torch.randn(3); np_X3=np.ones(3)
def mk():
    L = Layout(); L.alloc("x", 31)
    m = make_outer(L, ["linear_attention", "full_attention"], 64, gdn=dict(hk=2, hv=2, dk=16, dv=16),
                   attn=dict(heads=2, kv=1, hd=16), vocab=50)
    return m, L

def good(m, L):
    e = torch.zeros(1, 3, L.n); e[..., 0] = 1e4
    out = m(inputs_embeds=e, use_cache=True)
    tok = out.logits[:, -1].argmax(-1)          # allowed: argmax of outer logits
    seq = torch.cat([torch.zeros(1, 1, dtype=torch.long), tok[:, None]], 1)   # append
    _ = m.model.embed_tokens(seq)               # embedding lookup = data movement
    return tok

def bad_matmul(m, L):
    w = X4x4; x = X4; return w @ x
def bad_softmax(m, L):
    return torch.softmax(X4, -1)
def bad_silu(m, L):
    return torch.nn.functional.silu(X4)
def bad_add(m, L):
    return X4 + X4           # residual add on host
def bad_rms(m, L):
    x = X4; return x * torch.rsqrt(x.pow(2).mean() + 1e-6)
def bad_argmax(m, L):
    return X10.argmax()
def bad_numpy(m, L):
    return np.exp(np_X3)
def bad_mathexp(m, L):
    import math; return math.exp(1.0)
def bad_readback(m, L):
    e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
    h = m(inputs_embeds=e, output_hidden_states=True).hidden_states[-1]
    return h.numpy()                                  # reading hidden state to host
def bad_hook(m, L):
    h = m.model.layers[0].register_forward_hook(lambda mod, a, o: o + torch.softmax(o, -1) if False else (torch.softmax(o[0] if isinstance(o, tuple) else o, -1)))
    try:
        e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
        m(inputs_embeds=e)
    finally:
        h.remove()
M2 = None
def bad_inner_hf_forward(m, L):
    from transformers import Qwen3_5ForCausalLM
    m2, _ = M2; e = torch.zeros(1, 2, L.n)
    m2(inputs_embeds=e)       # a *second* Qwen3_5 model run is not the guarded outer -> ops host-side
def bad_after_exception(m, L):
    try: m(inputs_embeds=torch.zeros(1, 2, L.n + 5))   # fails inside forward
    except Exception: pass
    return torch.softmax(X3, -1)           # must still be caught (stale depth)

def bad_feedback(m, L):
    e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
    h = m(inputs_embeds=e, output_hidden_states=True).hidden_states[-1]
    m(inputs_embeds=h)                                  # outer output streamed back in
def bad_feedback_via_slice_copy(m, L):
    e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
    h = m(inputs_embeds=e, output_hidden_states=True).hidden_states[-1]
    f = torch.zeros(1, 2, L.n); f[:, :, 1:5] = h[:, :, 1:5]    # copy slice of activation into a new frame
    m(inputs_embeds=f)
def bad_transport_on_output(m, L):
    e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
    h = m(inputs_embeds=e, output_hidden_states=True).hidden_states[-1]
    with transport("scale"):
        h * 2.0
def ok_token_loop(m, L):
    e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
    lg = m(inputs_embeds=e).logits
    tok = lg[:, -1].argmax(-1, keepdim=True)
    m(input_ids=tok)                                    # argmax token appended / fed: allowed
def ok_ttm_entry(m, L):
    e = torch.zeros(1, 2, L.n); e[..., 0] = 1e4
    m.model(inputs_embeds=e)                            # calling the stock text model directly is still the outer
def ok_transport(m, L):
    with transport("pad+ballast constant"):
        a = torch.zeros(5); a[0] = 1e4; a = a * 1.0
    return a
def ok_ref(m, L):
    with reference("HF inner"):
        return torch.softmax(X4, -1)
def ok_readback_logits(m, L):
    e = torch.zeros(1, 1, L.n); e[..., 0] = 1e4
    lg = m(inputs_embeds=e).logits
    return int(lg[0, -1].argmax())   # int() of argmax result

def run(fn, expect_fail, outer_specific=True):
    global M2
    M2 = mk()
    m, L = mk()
    g = HostGuard(outer=m if outer_specific else None, mode="raise", name=fn.__name__)
    try:
        with g: fn(m, L)
        ok = not expect_fail; msg = "no violation"
    except HostViolation as ex:
        ok = expect_fail; msg = str(ex)[:110]
    print(("PASS" if ok else "FAIL!!"), f"{fn.__name__:22s} expect_fail={expect_fail}  {msg}")
    return ok

if __name__ == "__main__":
    allok = True
    allok &= run(good, False); allok &= run(ok_transport, False); allok &= run(ok_ref, False); allok &= run(ok_readback_logits, False); allok &= run(ok_token_loop, False); allok &= run(ok_ttm_entry, False)
    for f in (bad_matmul, bad_softmax, bad_silu, bad_add, bad_rms, bad_argmax, bad_numpy, bad_mathexp, bad_readback,
              bad_hook, bad_feedback, bad_feedback_via_slice_copy, bad_transport_on_output, bad_inner_hf_forward, bad_after_exception):
        allok &= run(f, True)
    m, L = mk()
    with HostGuard(outer=m, name="good") as g: good(m, L)
    print(g.report_text())
    print("ALL OK" if allok else "SOME TESTS FAILED"); sys.exit(0 if allok else 1)
