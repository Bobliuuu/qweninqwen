"""HostGuard: enforce the true-qweninqwen rule that the HOST does no inner-model math.

Rule (INTERFACE.md s2): the host may only stream weights/tokens/embeddings/addresses/position ids and do the
outer argmax + append.  NO host matmul, attention, norm, activation, softmax, RoPE, residual add, sampling.

Mechanism
  * a TorchDispatchMode sees every ATen op.  Each op is classified FORBIDDEN / GRAY / MOVE / unknown.
  * "inside the outer forward" = an instance of EXACTLY transformers Qwen3_5ForCausalLM is executing forward (global
    module hooks + a python-stack check, so a stale depth counter after an exception cannot hide host ops).
  * outside the outer forward (= host):  FORBIDDEN -> violation;  GRAY (elementwise arithmetic, compare, where..) ->
    violation unless inside `with transport(reason):`;  MOVE -> allowed;  unknown -> violation (strict);
    argmax/max-index only on tensors derived from outer logits.
  * inside the outer forward, a FORBIDDEN op whose python call stack contains foreign (non transformers/torch) frames
    (e.g. a forward hook on an outer submodule that computes something) is a violation ("foreign code in forward").
  * Tensor.numpy/tolist/item/__array__ readback of anything that is not outer logits (or argmax result) is a violation
    unless inside `with validation_read(reason):` (logged, must never be fed back).
  * numpy/math compute functions called from user code (not site-packages) are violations (best effort; `a @ b` on
    ndarrays cannot be intercepted -> see static_scan.py).
  * `with reference(reason):` exempts the HF inner reference model (for T-checks); everything inside is counted
    separately and listed in the report so a reviewer can read it.

Usage:
    from hostguard import HostGuard
    with HostGuard(mode='raise') as g:        # auto-detects Qwen3_5ForCausalLM instances
        run_driver()
    print(g.report_text())
Drivers may `from hostguard import reference, build, weight_prep, transport, validation_read` (no-ops when no guard is active).
"""
import os
import collections, contextlib, math, os, sys, json, traceback, threading
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn.modules.module as _tm

# ---------------------------------------------------------------- op classes
FORBIDDEN = set("""
mm bmm addmm baddbmm matmul linear einsum dot vdot mv addmv outer ger tensordot inner addbmm chain_matmul
_softmax softmax _log_softmax log_softmax _softmax_backward_data softplus silu gelu relu relu6 elu leaky_relu hardtanh
sigmoid tanh exp exp2 expm1 log log2 log10 log1p rsqrt sqrt pow float_power var std var_mean std_mean
native_layer_norm layer_norm rms_norm _fused_rms_norm native_group_norm batch_norm native_batch_norm
mean sum prod cumsum cumprod logsumexp norm linalg_vector_norm linalg_norm renorm nansum
sin cos tan sinh cosh erf erfc erfinv mish hardswish hardsigmoid
convolution _convolution conv1d conv2d conv_depthwise2d slow_conv_dilated2d
topk sort argsort multinomial bernoulli normal uniform rand randn randint dropout native_dropout
scaled_dot_product_attention _scaled_dot_product_flash_attention _scaled_dot_product_flash_attention_for_cpu
_scaled_dot_product_efficient_attention _scaled_dot_product_cudnn_attention _scaled_dot_product_attention_math
_scaled_dot_product_fused_attention_overrideable _flash_attention_forward
max min amax amin aminmax maximum minimum cummax cummin median kthvalue mode argmin
logaddexp atan atan2 asin acos sign_ lerp addcmul addcdiv fma
_fft_c2c fft_fft einops_rearrange
""".split())
GRAY = set("""
add sub mul div true_divide floor_divide neg abs floor ceil round trunc clamp clamp_min clamp_max clip where
masked_fill masked_scatter reciprocal fmod remainder sign eq ne lt le gt ge logical_and logical_or logical_not
logical_xor bitwise_and bitwise_or bitwise_not bitwise_xor bitwise_left_shift bitwise_right_shift isnan isinf isfinite
nan_to_num rsub square cdist all any count_nonzero nonzero bucketize searchsorted unique unique_consecutive
bincount histc rshift lshift __rshift__ __lshift__ __and__ __or__ __xor__ index_add index_reduce
""".split())
MOVE = set("""
view _unsafe_view reshape expand expand_as permute transpose t squeeze unsqueeze select slice narrow index
index_select index_put _index_put_impl take take_along_dim gather scatter scatter_add_ cat concat stack hstack vstack
copy _to_copy to clone contiguous detach alias lift_fresh lift_fresh_copy empty empty_like empty_strided
zeros zeros_like ones ones_like full full_like new_zeros new_full new_empty new_ones new_empty_strided fill zero
arange eye embedding split split_with_sizes chunk unbind as_strided unfold flip roll repeat repeat_interleave tile
constant_pad_nd _local_scalar_dense tril triu select_scatter slice_scatter diagonal_scatter set resize resize_
index_fill index_copy _reshape_alias view_as flatten unflatten movedim swapaxes cpu masked_select item numpy_T _pin_memory is_pinned
_has_compatible_shallow_copy_type is_same_size result_type numel diag diag_embed meshgrid detach_ _assert_async
lift _make_dual _conj conj real imag view_as_real view_as_complex tensor
""".split())
ARGMAX_LIKE = {"argmax"}


def _base(name):
    return name[:-1] if name.endswith("_") and not name.endswith("__") and len(name) > 1 else name


def classify(func):
    n = func.overloadpacket.__name__
    b = _base(n)
    for c in (n, b, n.lstrip("_"), b.lstrip("_")):
        if c in ARGMAX_LIKE:
            return "argmax", b
        if c in FORBIDDEN:
            return "forbidden", b
        if c in MOVE:
            return "move", b
        if c in GRAY:
            return "gray", b
    if any(s in n for s in ("attention", "softmax", "norm", "matmul", "conv", "gemm")):
        return "forbidden", b
    return "unknown", b


_TAINT = "_qiq_taint"


def _flat_tensors(args, kwargs):
    out = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor): out.append(a)
        elif isinstance(a, (list, tuple)):
            out += [x for x in a if isinstance(x, torch.Tensor)]
    return out


def _is_tainted(t):
    if getattr(t, _TAINT, False): return True
    b = getattr(t, "_base", None)
    return b is not None and getattr(b, _TAINT, False)


def _set_taint(t, v=True):
    try:
        setattr(t, _TAINT, v)
        b = getattr(t, "_base", None)
        if b is not None and v: setattr(b, _TAINT, v)
    except Exception:
        pass


class HostViolation(RuntimeError):
    pass


# ---------------------------------------------------------------- active guard + no-op scopes
_ACTIVE = None


@contextlib.contextmanager
def _scope(kind, reason):
    g = _ACTIVE
    if g is None:
        yield
        return
    g._scope_push(kind, reason)
    try:
        yield
    finally:
        g._scope_pop(kind)


def reference(reason="HF inner reference"):
    """Exempt scope: HF inner reference model / comparison math. Counted + listed in report."""
    return _scope("reference", reason)


def build(reason="compile constant outer structure (no inner data)"):
    """Exempt scope for COMPILE-TIME work that writes the outer model's constant parameters (norm gains, structural
    selectors).  All ops allowed, counted and listed; compiling INNER WEIGHT values here defeats the claim
    (outer_check's inner-fingerprint search + AUDIT review catch that)."""
    return _scope("build", reason)


def weight_prep(reason="token-independent weight reformat (e.g. silu-inverse of weights, norm precompensation)"):
    """Exempt scope for host math on INNER WEIGHTS ONLY, independent of any activation/token (payload construction).
    Everything is allowed but listed; tensors derived from outer outputs may not be touched inside (checked)."""
    return _scope("weight_prep", reason)


def transport(reason="weight transport/reformat"):
    """Exempt scope for GRAY host ops (elementwise arith used for pad/tile/(de)quantise/ballast constants).
    FORBIDDEN ops are never exempt here."""
    return _scope("transport", reason)


def validation_read(reason="validation readback (never fed back)"):
    return _scope("validation", reason)


_ORIG_NUMPY = torch.Tensor.numpy
_SITE = os.sep + "site-packages" + os.sep
_LIB_OK = (_SITE + "transformers" + os.sep, _SITE + "torch" + os.sep, _SITE + "fla" + os.sep,
           _SITE + "flash_attn" + os.sep, _SITE + "causal_conv1d" + os.sep, _SITE + "accelerate" + os.sep)


def _is_lib(fn):
    return _SITE in fn or fn.startswith(sys.base_prefix) or fn.startswith("<frozen")


class HostGuard(TorchDispatchMode):
    def __init__(self, outer=None, mode="raise", stack_check=True, patch_numpy=True, name="run", exempt_frames=(), taint=True):
        """outer: a specific outer model instance (else: any exact Qwen3_5ForCausalLM instance).
        mode: 'raise' (fail at first violation) or 'log' (collect all)."""
        super().__init__()
        assert mode in ("raise", "log")
        from transformers import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        self.OuterCls = Qwen3_5ForCausalLM
        self.EntryCls = (Qwen3_5ForCausalLM, Qwen3_5TextModel)   # drivers call m(...) or m.model(...): both are outer forwards
        self.outer = outer
        self.mode, self.stack_check, self.patch_numpy, self.name = mode, stack_check, patch_numpy, name
        self.taint = taint
        self.round_digests = []
        self.exempt_frames = [tuple(x) for x in exempt_frames]   # (kind, file_suffix, funcname)
        self.frame_scope_ops = collections.Counter(); self._fs_hit = None
        self.depth = 0
        self.scopes = []                       # stack of (kind, reason)
        self.scope_log = collections.OrderedDict()   # (kind, reason) -> {'entries':n,'ops':Counter}
        self.outer_ops = collections.Counter()
        self.host_ops = collections.Counter()          # class:name for allowed host ops
        self.exempt_ops = collections.Counter()        # kind:name
        self.violations = []
        self.rounds = 0
        self.steps = 0
        self.bytes_streamed = collections.Counter()
        self.round_log = []
        self.readbacks = []
        self.logits_ptrs = {}                  # storage ptr -> nbytes (derived from outer logits)
        self.outer_calls = 0
        self.other_hf_forward = collections.Counter()  # non-outer transformers PreTrainedModel forwards
        self._fwd_codes = set()
        self._handles = []
        self._patches = []
        self._seen_outer = []

    # ------------------------------------------------ scopes
    def _scope_push(self, kind, reason):
        self.scopes.append((kind, reason))
        d = self.scope_log.setdefault((kind, reason), {"entries": 0, "ops": collections.Counter()})
        d["entries"] += 1

    def _scope_pop(self, kind):
        self.scopes.pop()

    def _in_scope(self, kind):
        return any(k == kind for k, _ in self.scopes)

    # ------------------------------------------------ outer-forward tracking
    def _is_outer(self, m):
        if type(m) not in self.EntryCls:       # exact class identity, no subclass
            return False
        if self.outer is None: return True
        return m is self.outer or m is getattr(self.outer, "model", None)

    def _collect_codes(self):
        for C in self.EntryCls:
            f = C.forward
            seen = 0
            while f is not None and seen < 10:
                if hasattr(f, "__code__"):
                    self._fwd_codes.add(f.__code__)
                f = getattr(f, "__wrapped__", None); seen += 1

    def _pre_hook(self, module, args):
        if self._is_outer(module):
            if self.depth == 0:
                self.rounds += 1
                kw = {}
                try:
                    kw = sys._getframe(1).f_locals.get("kwargs", {}) or {}
                except Exception:
                    pass
                rb = collections.Counter(); n_pos = 0
                if self.taint:
                    for k, v in list(enumerate(args)) + list(kw.items()):
                        if isinstance(v, torch.Tensor) and _is_tainted(v) and not (self._in_scope("validation")):
                            self._violate("outer-output-fed-back", f"forward arg {k}", self._user_caller(3)[0],
                                          "a tensor derived from outer activations is streamed as an outer input")
                self._digest_round(args, kw)
                for k, v in list(enumerate(args)) + list(kw.items()):
                    if isinstance(v, torch.Tensor):
                        rb[str(k)] += v.numel() * v.element_size()
                        if str(k) in ("inputs_embeds", "input_ids", "0") and v.dim() >= 2:
                            n_pos = max(n_pos, v.shape[1])
                self.steps += n_pos
                self.bytes_streamed.update(rb)
                self.round_log.append({"positions": n_pos, "bytes": dict(rb)})
            self.depth += 1
        elif isinstance(module, torch.nn.Module) and type(module).__module__.startswith(
                "transformers.models.") and type(module).__name__.endswith(("ForCausalLM", "Model")) and self.depth == 0:
            self.other_hf_forward[type(module).__name__] += 1

    def _digest_round(self, args, kw):
        """per-position sha256 of everything streamed in (reproducibility/token-independence audit)."""
        import hashlib
        d = {}
        for k, v in list(enumerate(args)) + list(kw.items()):
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.numel() <= 6e7:
                x = v.detach().cpu().contiguous()
                T = x.shape[1] if (x.dim() >= 2 and str(k) in ("inputs_embeds", "input_ids", "position_ids", "0")) else 1
                x = x.reshape(-1) if T == 1 else x.reshape(x.shape[0], T, -1) if x.dim() >= 2 else x
                b = _ORIG_NUMPY(x.reshape(-1)) if T == 1 else None
                if T == 1: d[str(k)] = [hashlib.sha256(b.tobytes()).hexdigest()[:10]]
                else:
                    xx = _ORIG_NUMPY(x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x.reshape(-1))
                    if x.dim() == 3: d[str(k)] = [hashlib.sha256(r.tobytes()).hexdigest()[:10] for r in xx]
                    else: d[str(k)] = [hashlib.sha256(xx.tobytes()).hexdigest()[:10]]
        self.round_digests.append(d)

    def _post_hook(self, module, args, output):
        if self._is_outer(module):
            self.depth = max(0, self.depth - 1)
            lg = getattr(output, "logits", None)
            if lg is None and isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                lg = output[0]
            if isinstance(lg, torch.Tensor):
                self._reg(lg)

    def _reg(self, t):
        try:
            setattr(t, "_qiq_logits", True)
            b = getattr(t, "_base", None)
            if b is not None: setattr(b, "_qiq_logits", True)
        except Exception:
            pass

    def _is_logits_derived(self, t):
        return bool(getattr(t, "_qiq_logits", False)) or bool(getattr(getattr(t, "_base", None), "_qiq_logits", False))

    def _stack_info(self):
        """(inside_outer_forward, foreign_frame_or_None, user_site)"""
        f = sys._getframe(2)
        foreign = None; user = None; inside = False
        while f is not None:
            co = f.f_code
            if co in self._fwd_codes:
                inside = True
                break
            fn = co.co_filename
            if not _is_lib(fn) and user is None:
                user = f"{os.path.basename(fn)}:{f.f_lineno} in {co.co_name}"
            f = f.f_back
        if inside:
            # frames between the op and the outer forward frame must be library code
            f = sys._getframe(2)
            while f is not None and f.f_code not in self._fwd_codes:
                fn = f.f_code.co_filename
                if not (_is_lib(fn)) and "torch" not in fn.split(os.sep)[-3:-1]:
                    foreign = f"{os.path.basename(fn)}:{f.f_lineno} in {f.f_code.co_name}"; break
                f = f.f_back
        else:
            f = sys._getframe(2)
            while f is not None:
                if not _is_lib(f.f_code.co_filename):
                    user = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} in {f.f_code.co_name}"; break
                f = f.f_back
        return inside, foreign, user

    _CONSTR = {"make_outer", "reset_parameters", "_init_weights", "initialize_weights", "_initialize_weights"}

    def _host_exempt(self, kinds=("reference", "build", "weight_prep", "transport", "validation")):
        return any(self._in_scope(k) for k in kinds) or self._construction_frame(3) or (bool(self.exempt_frames) and self._frame_scope() is not None)

    def _frame_scope(self):
        """declared (CLI --scope kind:file.py:func) exemptions keyed on python frames, so agents' code need not be edited."""
        f = sys._getframe(2)
        while f is not None:
            co = f.f_code
            for kind, fsuf, fn in self.exempt_frames:
                if co.co_name == fn and co.co_filename.endswith(fsuf):
                    self._fs_hit = f"{os.path.basename(co.co_filename)}:{fn}"
                    return kind
            f = f.f_back
        return None

    def _construction_frame(self, d=2):
        f = sys._getframe(d)
        while f is not None:
            co = f.f_code
            fn = co.co_filename
            if (co.co_name == "make_outer" and fn.endswith(os.sep + "machine.py")) or \
               (co.co_name in self._CONSTR and co.co_name != "make_outer" and _SITE in fn) or \
               fn.endswith(os.sep + "nn" + os.sep + "init.py"):
                return True
            f = f.f_back
        return False

    # ------------------------------------------------ violations
    def _violate(self, kind, opname, site, detail=""):
        v = {"kind": kind, "op": opname, "site": site, "detail": detail}
        self.violations.append(v)
        if len(self.violations) > 200000: self.violations.pop(0)
        if self.mode == "raise":
            raise HostViolation(f"HOST VIOLATION [{kind}] op={opname} at {site} {detail}")

    # ------------------------------------------------ dispatch
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        cls, name = classify(func)
        key = name
        inside = self.depth > 0
        if inside and (cls != "move") and self.stack_check:
            ins, foreign, user = self._stack_info()
            if not ins:
                self.depth = 0; inside = False       # stale depth (exception) -> treat as host
            elif foreign and cls in ("forbidden", "gray", "unknown", "argmax"):
                self._violate("foreign-code-in-outer-forward", name, foreign,
                              "math op reached from non-library frame inside outer forward (hook/patched forward?)")
        if inside:
            self.outer_ops[key] += 1
            out = func(*args, **kwargs)
            if self.taint:
                for o in (out if isinstance(out, (list, tuple)) else [out]):
                    if isinstance(o, torch.Tensor): _set_taint(o)
            return out
        # ---- host side
        tin = self.taint and any(_is_tainted(t) for t in _flat_tensors(args, kwargs))
        if tin and cls in ("forbidden", "gray", "unknown") and not self._in_scope("reference") \
                and not (self.exempt_frames and self._frame_scope() == "reference"):
            self._violate("host-math-on-outer-output", name, self._stack_info()[2],
                          "host computes on a tensor derived from outer activations (not allowed even in transport/weight_prep)")
        if self._construction_frame():
            self.exempt_ops["auto-construction:" + key] += 1
            return func(*args, **kwargs)
        fscope = self._frame_scope() if (self.exempt_frames and cls != "move") else None
        for kind in ("reference", "build", "weight_prep", "validation", "transport"):
            if self._in_scope(kind) or fscope == kind:
                if kind in ("reference", "build", "weight_prep") or cls in ("move", "gray") or (kind == "validation" and cls == "move"):
                    if kind in ("build", "weight_prep", "transport"):
                        for a in list(args) + list(kwargs.values()):
                            if isinstance(a, torch.Tensor) and _is_tainted(a):
                                self._violate("outer-data-in-exempt-scope", name, self._stack_info()[2],
                                              f"{kind} scope touches a tensor derived from outer output")
                    self.exempt_ops[f"{kind}:{key}"] += 1
                    for k, r in self.scopes:
                        self.scope_log[(k, r)]["ops"][key] += 1
                    if fscope == kind and not self._in_scope(kind):
                        self.frame_scope_ops[(kind, self._fs_hit)] += 1
                    out = func(*args, **kwargs)
                    if tin and cls == "move":
                        for o in (out if isinstance(out, (list, tuple)) else [out]):
                            if isinstance(o, torch.Tensor): _set_taint(o)
                        if func._schema.is_mutable and args and isinstance(args[0], torch.Tensor): _set_taint(args[0])
                    return out
        if cls == "move":
            self.host_ops["move:" + key] += 1
            out = func(*args, **kwargs)
            if tin:
                for o in (out if isinstance(out, (list, tuple)) else [out]):
                    if isinstance(o, torch.Tensor): _set_taint(o)
                if func._schema.is_mutable and args and isinstance(args[0], torch.Tensor): _set_taint(args[0])
            if any(isinstance(a, torch.Tensor) and self._is_logits_derived(a) for a in args[:2]):
                for o in (out if isinstance(out, (list, tuple)) else [out]):
                    if isinstance(o, torch.Tensor):
                        self._reg(o)
            return out
        site = self._stack_info()[2]
        if cls == "argmax":
            t = args[0]
            if isinstance(t, torch.Tensor) and self._is_logits_derived(t):
                self.host_ops["argmax(outer-logits)"] += 1
                out = func(*args, **kwargs)
                self._reg(out); _set_taint(out, False)
                return out
            self._violate("argmax-on-non-logits", name, site, "host argmax of a tensor not derived from outer logits")
        elif cls == "forbidden":
            self._violate("host-math", name, site)
        elif cls == "gray":
            self._violate("host-elementwise", name, site, "outside `transport()` scope")
        else:
            self._violate("unclassified-host-op", func.overloadpacket.__name__, site)
        self.host_ops[f"{cls}!:{key}"] += 1
        return func(*args, **kwargs)

    # ------------------------------------------------ readback / numpy / math patches
    def _user_caller(self, depth=2):
        f = sys._getframe(depth)
        while f is not None:
            if not _is_lib(f.f_code.co_filename):
                return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} in {f.f_code.co_name}", True
            f = f.f_back
        return "lib", False

    def _patch(self, obj, attr, new):
        old = getattr(obj, attr)
        setattr(obj, attr, new)
        self._patches.append((obj, attr, old))

    def _install_patches(self):
        g = self
        for attr in ("numpy", "tolist", "item", "__array__"):
            old = getattr(torch.Tensor, attr)

            def mk(old, attr):
                def w(t, *a, **k):
                    if g.depth == 0 and not g._host_exempt(("reference", "build", "validation")):
                        site, user = g._user_caller()
                        ok = g._is_logits_derived(t) or (attr == "item" and t.numel() == 1 and g._is_logits_derived(t))
                        g.readbacks.append({"fn": attr, "numel": t.numel(), "site": site, "logits": ok})
                        if user and not ok:
                            g._violate("readback-non-logits", attr, site,
                                       "host reads an outer tensor that is not logits (use validation_read())")
                    return old(t, *a, **k)
                return w
            self._patch(torch.Tensor, attr, mk(old, attr))
        if self.patch_numpy:
            names = ["matmul", "dot", "vdot", "inner", "outer", "einsum", "tensordot", "exp", "expm1", "log", "log1p",
                     "tanh", "sqrt", "cumsum", "sin", "cos", "power", "var", "std", "mean", "sum", "prod", "cumprod",
                     "convolve", "correlate", "argmax", "max", "amax", "sort", "argsort", "maximum", "minimum"]
            for n in names:
                if hasattr(np, n):
                    old = getattr(np, n)

                    def mkn(old, n):
                        def w(*a, **k):
                            if not g._host_exempt():
                                site, user = g._user_caller()
                                if user:
                                    g._violate("host-numpy-math", "np." + n, site)
                            return old(*a, **k)
                        w.__name__ = getattr(old, "__name__", n)
                        return w
                    self._patch(np, n, mkn(old, n))
            for n in ("exp", "tanh", "sqrt", "log", "erf", "sin", "cos", "pow", "fsum", "expm1", "log1p"):
                if hasattr(math, n):
                    old = getattr(math, n)

                    def mkm(old, n):
                        def w(*a, **k):
                            if not g._host_exempt():
                                site, user = g._user_caller()
                                if user:
                                    g._violate("host-python-math", "math." + n, site)
                            return old(*a, **k)
                        return w
                    self._patch(math, n, mkm(old, n))

    # ------------------------------------------------ context manager
    def __enter__(self):
        global _ACTIVE
        assert _ACTIVE is None, "nested HostGuard"
        _ACTIVE = self
        self._collect_codes()
        self._handles.append(_tm.register_module_forward_pre_hook(self._pre_hook))
        self._handles.append(_tm.register_module_forward_hook(self._post_hook, always_call=True))
        self._install_patches()
        return super().__enter__()

    def __exit__(self, *exc):
        global _ACTIVE
        r = super().__exit__(*exc)
        for h in self._handles: h.remove()
        for obj, attr, old in reversed(self._patches): setattr(obj, attr, old)
        self._handles, self._patches = [], []
        _ACTIVE = None
        return r

    # ------------------------------------------------ report
    def report(self):
        tot_out = sum(self.outer_ops.values())
        math_in = {k: v for k, v in self.outer_ops.items()}
        viol = collections.Counter(v["kind"] for v in self.violations)
        return {
            "name": self.name, "mode": self.mode, "PASS": len(self.violations) == 0,
            "outer_rounds(forward calls)": self.rounds, "outer_steps(positions)": self.steps,
            "bytes_streamed_total": sum(self.bytes_streamed.values()), "bytes_streamed_by_arg": dict(self.bytes_streamed),
            "ops_inside_outer_forward": tot_out, "outer_op_histogram": dict(self.outer_ops.most_common(25)),
            "host_ops_allowed": dict(self.host_ops), "exempt_ops": dict(self.exempt_ops),
            "exempt_scopes": [{"kind": k, "reason": r, "entries": d["entries"], "ops": sum(d["ops"].values())}
                              for (k, r), d in self.scope_log.items()],
            "host_readbacks": [r for r in self.readbacks][:50], "n_host_readbacks": len(self.readbacks),
            "non_outer_hf_forwards_on_host": dict(self.other_hf_forward),
            "round_digests": self.round_digests,
            "declared_frame_scopes": [{"kind": k, "frame": f, "ops": n} for (k, f), n in self.frame_scope_ops.items()],
            "violation_counts": dict(viol), "violations": self.violations[:100],
            "violation_groups": [{"kind": k[0], "op": k[1], "site": k[2], "count": n} for k, n in
                                 collections.Counter((v["kind"], v["op"], v["site"]) for v in self.violations).most_common(40)],
        }

    def report_text(self):
        r = self.report()
        L = [f"=== HostGuard report [{r['name']}] mode={r['mode']} -> {'PASS' if r['PASS'] else 'FAIL'} ==="]
        L.append(f"outer rounds={r['outer_rounds(forward calls)']} steps={r['outer_steps(positions)']} "
                 f"bytes streamed={r['bytes_streamed_total']:,} {r['bytes_streamed_by_arg']}")
        L.append(f"ATen ops INSIDE outer forward: {r['ops_inside_outer_forward']:,} (top: "
                 + ", ".join(f"{k}:{v}" for k, v in list(r['outer_op_histogram'].items())[:8]) + ")")
        L.append(f"host ops allowed: {r['host_ops_allowed']}")
        L.append(f"host readbacks: {r['n_host_readbacks']} (non-logits: {sum(1 for x in self.readbacks if not x['logits'])})")
        for s in r["exempt_scopes"]:
            L.append(f"EXEMPT scope {s['kind']}: '{s['reason']}' entries={s['entries']} ops={s['ops']}  <-- reviewer must read")
        if r["non_outer_hf_forwards_on_host"]:
            L.append(f"WARNING non-outer HF model forwards executed: {r['non_outer_hf_forwards_on_host']}")
        for d in r["declared_frame_scopes"]:
            L.append(f"EXEMPT (declared via --scope) {d['kind']} frame {d['frame']} ops={d['ops']}  <-- reviewer must read")
        for v in r["violation_groups"][:25]:
            L.append(f"VIOLATION x{v['count']} {v['kind']}: op={v['op']} at {v['site']}")
        if len(r["violation_groups"]) > 25:
            L.append(f"... {len(r['violation_groups'])-25} more groups, {len(self.violations)} violations total")
        return "\n".join(L)
