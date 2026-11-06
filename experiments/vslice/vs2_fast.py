"""vs2_fast: faster layer-major executor for vs2 (same outer layers, same frames, same math; vs2.py is untouched).

What it changes versus vs2.Runner.run (each switch is independent and measured in PERF_NOTES.md):
  fastbuild    construct outer layers on the meta device + to_empty (no random init of ~1-4 GB of weights that compile_entry zeroes anyway)
  sparse       after compile_entry, every nn.Linear weight is converted to a COO *sparse* weight (the compiled weights have 1-3 nonzeros per row;
               zeros are skipped, nothing is approximated).  Also removes the dense 1-4 GB layer from memory.
  cache_layers keep the compiled layers on the GPU across positions (needs sparse: a dense layer is up to 4 GB) instead of rebuilding 1006 layers per position
  cache_pay    keep the token-independent weight payload tiles (silu-inverse of the inner weights, computed once) on the GPU across positions
  seg          rows per layer call (vs2 uses 2048)
  tf32         allow TF32 matmuls (changes numerics; error is reported)
  prof         per-phase / per-submodule CUDA-event timing (adds a few % overhead)
Host-side rules are unchanged: nothing here evaluates inner-model math; the host only builds the same flags/addresses/payload frames.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vs2 import *          # noqa: Runner, Prog, compile_entry, tile_payload, transport, build, Qwen3_5DecoderLayer, ...


class Phase:
    """accumulating wall-clock timers; sync=True makes them GPU-accurate (only used when profiling)."""
    def __init__(self, dev, sync): self.t = {}; self.n = {}; self.dev = dev; self.sync = sync and dev == 'cuda'
    def start(self):
        if self.sync: torch.cuda.synchronize()
        return time.time()
    def stop(self, name, t0):
        if self.sync: torch.cuda.synchronize()
        dt = time.time() - t0; self.t[name] = self.t.get(name, 0.0) + dt; self.n[name] = self.n.get(name, 0) + 1


SUBMODS = ['input_layernorm', 'post_attention_layernorm', 'mlp', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'linear_attn', 'linear_attn.in_proj_qkv',
           'linear_attn.in_proj_z', 'linear_attn.in_proj_b', 'linear_attn.in_proj_a', 'linear_attn.conv1d', 'linear_attn.norm', 'linear_attn.out_proj',
           'self_attn']


def sparsify(layer, fmt='coo'):
    """dense nn.Linear weights -> COO sparse weights (exact: only structural zeros are dropped)."""
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                m.weight = nn.Parameter((w.to_sparse() if fmt == 'coo' else w.to_sparse_csr()), requires_grad=False)
                del w
    return layer


class FastRunner(Runner):
    def __init__(self, S, c, plan, P, dev='cuda', eps=1e-30, fastbuild=False, sparse=False, cache_layers=False, cache_pay=False, seg=2048,
                 prof=False, kernprof=False, graph=False, sparse_fmt='coo'):
        super().__init__(S, c, plan, P, dev=dev, eps=eps)
        self.graph = graph; self.sparse_fmt = sparse_fmt; self.plans = {}; self.graphs = {}; self.goff = {}; self.Hs = None; self.pool = torch.cuda.graph_pool_handle() if dev == 'cuda' else None
        self.fastbuild, self.sparse, self.cache_layers, self.cache_pay, self.seg, self.prof, self.kernprof = fastbuild, sparse, cache_layers, cache_pay, seg, prof, kernprof
        self.lcache = {}; self.paycache = {}; self.ph = Phase(dev, prof); self.sub = {}; self.kind_t = {}; self.kp_done = {}; self.kp_text = []

    # ------------------------------------------------------------------ build
    def make_layer(self, li, entry):
        dev = self.dev
        if self.cache_layers and li in self.lcache: return self.lcache[li]
        cf = self.cfgv[self.kind_cfg[entry[0]]]
        with build("construct a stock Qwen3_5DecoderLayer (params all overwritten by compile_entry)"):
            if self.fastbuild:
                with torch.device('meta'): layer = Qwen3_5DecoderLayer(cf, li).eval()
                layer = layer.to_empty(device=dev)
            else:
                with torch.device(dev): layer = Qwen3_5DecoderLayer(cf, li).eval()
        compile_entry(layer, entry, self.D, self.c, self.S, self.P)
        if self.sparse: layer = sparsify(layer, self.sparse_fmt)
        if self.cache_layers: self.lcache[li] = layer
        return layer

    # ------------------------------------------------------------------ payload
    def job_rows(self, pr, job):
        dev = self.dev; S, c = pr.S, pr.c
        if 'dense' in job: return job['dense'].to(dev)
        if 'pay' in job: pay = job['pay'].to(dev)
        else:
            key = (id(job['W']), job['o0'], job['j'])
            pay = self.paycache.get(key) if self.cache_pay else None
            if pay is None:
                pay = tile_payload(job['W'], job['o0'], job['j'], c, dev).to(dev)
                if self.cache_pay: self.paycache[key] = pay
        m = pay.shape[0]
        F = torch.zeros(m, self.D, device=dev); F[:, 0] = B
        ar = torch.arange(m, device=dev); F[ar, S.SEL.start + ar] = 1.0
        F[:, S.WF.start + job['wf']] = 1.0; F[:, S.PAY.start:S.PAY.stop] = pay
        return F

    # ------------------------------------------------------------------ timing hooks
    def _hook(self, layer, evs):
        hs = []
        def mk(name):
            def pre(m, a, k=None): e = torch.cuda.Event(enable_timing=True); e.record(); evs.setdefault(name, []).append([e, None])
            def post(m, a, o, k=None): e = torch.cuda.Event(enable_timing=True); e.record(); evs[name][-1][1] = e
            return pre, post
        for n, m in layer.named_modules():
            if n in SUBMODS:
                pre, post = mk(n); hs.append(m.register_forward_pre_hook(pre)); hs.append(m.register_forward_hook(post))
        return hs

    # ------------------------------------------------------------------ per-layer plan (token independent -> cached)
    def layer_plan(self, li, tl_li, tl_m):
        if li in self.plans: return self.plans[li]
        sel = np.nonzero((tl_li == -1) | (tl_li == li))[0]; segs = []; cur = []; nb = 0
        for k in sel.tolist():
            m = int(tl_m[k])
            if nb + m > self.seg and cur: segs.append(cur); cur = []; nb = 0
            cur.append(k); nb += m
        if cur: segs.append(cur)
        self.plans[li] = segs; return segs

    def _body(self, pr, li, kind, layer, segs, off, is_c, tl_idx, cpos_t):
        """one outer layer over all its rows (control rows from the static Hs buffer + the weight-write jobs addressed to it).  Pure device work for
        non-attention layers, so it can be CUDA-graph captured.  Returns new cache offset."""
        dev = self.dev; Hs = self.Hs; ph = self.ph
        for cur in segs:
            t0 = ph.start()
            rows = []; cidx = []; pids = []; pos_ = 0; i = 0; L = len(cur)
            attn = kind in ('A', 'AM')
            while i < L:
                k = cur[i]
                if is_c[k]:
                    j = i
                    while j < L and is_c[cur[j]]: j += 1
                    a, b = int(tl_idx[cur[i]]), int(tl_idx[cur[j - 1]]) + 1
                    rows.append(Hs[a:b]); cidx.append((pos_, a, b))
                    if attn: pids.append(cpos_t[a:b])
                    pos_ += b - a; i = j
                else:
                    F = self.job_rows(pr, pr.jobs[int(tl_idx[k])]); rows.append(F)
                    if attn: pids.append(torch.full((F.shape[0],), pr.cur_pos, device=dev))
                    pos_ += F.shape[0]; i += 1
            x = torch.cat(rows)[None]; n = x.shape[1]
            ph.stop('assemble', t0); t0 = ph.start()
            mask = None; pe = None; pid = None
            cp = torch.arange(off, off + n, device=dev)
            if attn:
                pid = torch.cat(pids)[None]; pe = self.rot(x, pid[None].expand(3, 1, -1))
                with transport("causal mask from position indices (token-independent addresses)"):
                    mask = (torch.arange(off + n, device=dev)[None, :] <= (torch.arange(n, device=dev)[:, None] + off))[None, None]
            y = layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pid, past_key_values=self.cache, cache_position=cp, use_cache=True)
            if isinstance(y, tuple): y = y[0]
            ph.stop('layer_fwd', t0); t0 = ph.start()
            for (p0, a, b) in cidx: Hs[a:b] = y[0, p0:p0 + (b - a)].float()
            off += n
            ph.stop('writeback', t0)
        return off

    def _snap(self, li):
        cl = self.cache.layers[li]; out = []
        for d, f in ((cl.conv_states, cl.is_conv_states_initialized), (cl.recurrent_states, cl.is_recurrent_states_initialized)):
            out.append((d[0], d[0].clone()) if f[0] else None)
        return out

    @staticmethod
    def _restore(snap):
        for e in snap:
            if e is not None: e[0].copy_(e[1])

    def _capture(self, li, body):
        """warm up eagerly (state restored afterwards), capture, then replay once for the real result."""
        snap = self._snap(li); self.Hs_bak.copy_(self.Hs)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st): body()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        self._restore(snap); self.Hs.copy_(self.Hs_bak)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool): body()
        self.graphs[li] = g; g.replay()

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def run(self, pr, trace_rows=None, trace_layers=None):
        dev = self.dev; ph = self.ph; trace = {}
        t = ph.start(); cf = pr.ctrl_frames()
        if self.Hs is None or self.Hs.shape != cf.shape:
            assert not self.graphs, 'control-frame count changed after graphs were captured'
            self.Hs = torch.empty(cf.shape, device=dev); self.Hs_bak = torch.empty_like(self.Hs)
        self.Hs.copy_(cf); ph.stop('h2d_ctrl', t)
        cpos_t = torch.tensor(pr.cpos, device=dev)
        for li, e in enumerate(self.plan):
            if e[0] == 'AM' and self.offs.get(li, 0) > 0: self.cache.layers[li].reset(); self.offs[li] = 0
        t = ph.start()
        n_tl = len(pr.timeline); is_c = np.fromiter((x[0] == 'c' for x in pr.timeline), bool, n_tl)
        tl_idx = np.fromiter((x[1] for x in pr.timeline), np.int64, n_tl)
        jli = np.array([j['li'] for j in pr.jobs], np.int64); jm = np.array([j.get('m', 1) for j in pr.jobs], np.int64)
        z = np.where(is_c, 0, tl_idx)
        tl_li = np.where(is_c, -1, jli[z] if len(jli) else -1); tl_m = np.where(is_c, 1, jm[z] if len(jli) else 1)
        ph.stop('timeline_np', t)
        for li, entry in enumerate(self.plan):
            kind = entry[0]; tk = ph.start()
            t0 = ph.start(); layer = self.make_layer(li, entry); ph.stop('build', t0)
            off = self.offs.get(li, 0)
            use_graph = self.graph and kind not in ('A', 'AM')
            if use_graph and li in self.graphs:
                t0 = ph.start(); self.graphs[li].replay(); ph.stop('graph_replay', t0)
                off += self.goff[li]
            else:
                t0 = ph.start(); segs = self.layer_plan(li, tl_li, tl_m); ph.stop('plan_segs', t0)
                evs = {}; hs = self._hook(layer, evs) if self.prof else []
                prof_ctx = None
                if self.kernprof and kind not in self.kp_done and li > 0:
                    from torch.profiler import profile, ProfilerActivity
                    prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); prof_ctx.__enter__(); self.kp_done[kind] = li
                body = lambda: self._body(pr, li, kind, layer, segs, off, is_c, tl_idx, cpos_t)
                if use_graph:
                    t0 = ph.start(); self._capture(li, body); ph.stop('graph_capture', t0)
                    self.goff[li] = sum(int(tl_m[k]) if not is_c[k] else 1 for sg in segs for k in sg)
                    off += self.goff[li]
                else:
                    off = body()
                if prof_ctx is not None:
                    torch.cuda.synchronize(); prof_ctx.__exit__(None, None, None)
                    self.kp_text.append(f'=== kernel profile of one {kind} layer (li={li}) ===\n' + prof_ctx.key_averages().table(sort_by='cuda_time_total', row_limit=14, max_name_column_width=70))
                if self.prof:
                    torch.cuda.synchronize()
                    for nme, lst in evs.items():
                        s_ = sum(a.elapsed_time(b) for a, b in lst if b is not None) / 1e3
                        d = self.sub.setdefault(nme, {}); d[kind] = d.get(kind, 0.0) + s_
                    for h in hs: h.remove()
            self.offs[li] = off
            if trace_rows is not None and (trace_layers is None or li in trace_layers):
                for nme, r in trace_rows.items(): trace[(nme, li)] = self.Hs[r.to(dev) if torch.is_tensor(r) else r].cpu().clone()
            if not self.cache_layers: del layer
            if self.prof: self.kind_t[kind] = self.kind_t.get(kind, 0.0) + (ph.start() - tk)
            if os.environ.get('VS2_PROGRESS') and li % 100 == 0: print(f'  [progress] layer {li}/{len(self.plan)} {entry[0]} t={dict((k, round(v, 1)) for k, v in ph.t.items())}', flush=True)
        t = ph.start(); out = self.Hs.cpu(); ph.stop('d2h_final', t)
        return out, trace
