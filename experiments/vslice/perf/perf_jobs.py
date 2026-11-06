"""Remote jobs for vs2 perf work (imported inside the Modal container)."""
import time, json, os, sys, math, torch

def _t(fn, n=5):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n

def micro(rows=3000, D=40775, nout=8288, nnz_row=2):
    import torch.nn.functional as F
    dev = 'cuda'; torch.manual_seed(0); out = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name())
    import transformers; out['transformers'] = transformers.__version__
    idx_r = torch.arange(nout, device=dev).repeat_interleave(nnz_row)
    idx_c = torch.randint(0, D, (nout * nnz_row,), device=dev)
    val = torch.randn(nout * nnz_row, device=dev)
    Wd = torch.zeros(nout, D, device=dev); Wd.index_put_((idx_r, idx_c), val, accumulate=True)
    x = torch.randn(rows, D, device=dev)
    ref = F.linear(x, Wd)
    out['dense_fp32_ms'] = _t(lambda: F.linear(x, Wd)) * 1e3
    torch.backends.cuda.matmul.allow_tf32 = True
    out['dense_tf32_ms'] = _t(lambda: F.linear(x, Wd)) * 1e3
    out['tf32_relerr'] = ((F.linear(x, Wd) - ref).abs().max() / ref.abs().max()).item()
    torch.backends.cuda.matmul.allow_tf32 = False
    Wc = Wd.to_sparse_csr(); Wcoo = Wd.to_sparse()
    def tryit(name, fn):
        try:
            y = fn(); out[name + '_relerr'] = ((y - ref).abs().max() / ref.abs().max()).item()
            out[name + '_ms'] = _t(fn) * 1e3
        except Exception as e: out[name + '_err'] = repr(e)[:200]
    tryit('F.linear_csr', lambda: F.linear(x, Wc))
    tryit('F.linear_coo', lambda: F.linear(x, Wcoo))
    tryit('spmm_csr_xT', lambda: torch.sparse.mm(Wc, x.t()).t())
    tryit('spmm_coo_xT', lambda: torch.sparse.mm(Wcoo, x.t()).t())
    xt = x.t().contiguous()
    tryit('spmm_csr_xTcontig', lambda: torch.sparse.mm(Wc, xt).t())
    tryit('matmul_x_csrT', lambda: x @ Wc.t())
    # gather route: unique input cols
    cols = torch.unique(idx_c); Wsub = Wd[:, cols].contiguous()
    tryit('gather_dense', lambda: F.linear(x[:, cols], Wsub))
    out['n_cols'] = int(cols.numel())
    print(json.dumps(out), flush=True)
    return out


def run(variant='base', **kw):
    from test_vs2_fast import run_fast
    t = time.time(); r = run_fast(**kw); r['variant'] = variant; r['wall_s'] = time.time() - t
    return r
