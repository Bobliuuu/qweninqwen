"""Static (AST) scan of driver/runner code for host-side inner math the dynamic guard can't see (ndarray @, C ext, etc.).
  python static_scan.py [-v] file_or_dir ...     (read-only)
Reports per finding: file:line kind text [EXEMPT scope reason].  HOT = names that are inner math; ARITH (-v) = + - * / ** on
anything (informational).  Also flags imports/uses of HF inner models (AutoModelForCausalLM, Qwen2/3ForCausalLM) and `.forward(`/`generate(`.
Exit 1 if any non-exempt HOT/MATMUL finding."""
import os
import ast, sys, os

HOT = set("""matmul mm bmm einsum softmax log_softmax silu gelu relu sigmoid tanh exp expm1 log log1p rsqrt sqrt pow cumsum dot
inner outer tensordot linear layer_norm rms_norm scaled_dot_product_attention conv1d convolve var std mean norm
sin cos topk sort multinomial sample erf softplus apply_rotary_pos_emb rotate_half rms silu_and_mul addmm baddbmm
""".split())
HF_NAMES = {"AutoModelForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen2Model", "Qwen3Model", "AutoModel"}
EXEMPT = {"reference", "build", "transport", "validation_read"}


class V(ast.NodeVisitor):
    def __init__(self, fn, src, verbose):
        self.fn, self.lines, self.verbose, self.out, self.stack = fn, src.splitlines(), verbose, [], []
    def add(self, node, kind):
        ex = self.stack[-1] if self.stack else None
        self.out.append((self.fn, node.lineno, kind, self.lines[node.lineno - 1].strip()[:110], ex))
    def visit_With(self, node):
        pushed = 0
        for it in node.items:
            c = it.context_expr
            if isinstance(c, ast.Call):
                f = c.func; n = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if n in EXEMPT:
                    r = ast.unparse(c.args[0]) if c.args else ""
                    self.stack.append(f"{n}({r})"); pushed += 1
        self.generic_visit(node)
        for _ in range(pushed): self.stack.pop()
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult): self.add(node, "MATMUL(@)")
        elif self.verbose and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)): self.add(node, "arith")
        self.generic_visit(node)
    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.MatMult): self.add(node, "MATMUL(@=)")
        elif self.verbose: self.add(node, "arith")
        self.generic_visit(node)
    def visit_Call(self, node):
        f = node.func; n = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if n in HOT: self.add(node, f"HOT({n})")
        elif n in HF_NAMES: self.add(node, f"HF-INNER({n})")
        elif n in ("generate",): self.add(node, "HF-generate")
        elif n in ("argmax", "argmin", "max", "min", "item", "tolist", "numpy"): self.add(node, f"READ/ARG({n})")
        self.generic_visit(node)
    def visit_Name(self, node):
        if node.id in HF_NAMES and isinstance(node.ctx, ast.Load): pass
        self.generic_visit(node)


def scan(paths, verbose=False):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for r, d, fs in os.walk(p):
                d[:] = [x for x in d if x not in ("__pycache__", ".git", "ref_copy")]
                files += [os.path.join(r, f) for f in fs if f.endswith(".py")]
        else: files.append(p)
    res = []
    for fn in sorted(files):
        try: src = open(fn).read(); tree = ast.parse(src)
        except Exception as e: res.append((fn, 0, "PARSE-ERR", str(e)[:80], None)); continue
        v = V(fn, src, verbose); v.visit(tree); res += v.out
    return res


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "-v"]; res = scan(a, "-v" in sys.argv)
    bad = 0
    for fn, ln, kind, txt, ex in res:
        flag = f"  [EXEMPT {ex}]" if ex else ""
        print(f"{fn}:{ln}: {kind}: {txt}{flag}")
        if (kind.startswith("HOT") or kind.startswith("MATMUL") or kind.startswith("HF-")) and not ex: bad += 1
    print(f"{len(res)} findings, {bad} non-exempt HOT/MATMUL/HF"); sys.exit(1 if bad else 0)
