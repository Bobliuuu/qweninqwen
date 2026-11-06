"""Driver for vs2_fast.FastRunner: same token program / error metric as test_vs2.run for the real 28-layer Qwen3-0.6B, with a per-phase time breakdown.
run_fast(cfg) -> dict.   Does not touch test_vs2.py."""
import os, sys, time, json, math, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vs2_fast import *            # FastRunner, vs2 names
from test_vs2 import weights, logits_from
from transformers import Qwen3ForCausalLM, AutoTokenizer
try:
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'audit')); from hostguard import reference
except Exception:
    import contextlib; reference = lambda *a, **k: contextlib.nullcontext()


def run_fast(positions=2, nlayers=28, vocab=151936, tf32=False, **rk):
    dev = 'cuda'; torch.backends.cuda.matmul.allow_tf32 = bool(tf32); torch.backends.cudnn.allow_tf32 = bool(tf32)
    c = Cfg(1024, 16, 8, 128, 3072, nlayers, vocab, theta=1e6, I_MLP=8192)
    with reference("load the real HF Qwen3-0.6B inner weights (weights + reference model)"):
        toy = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', num_hidden_layers=nlayers, torch_dtype=torch.float32).eval()
    prompt = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')('The capital of France is')['input_ids']
    Wd, P = weights(toy, c); S = build_layout(c); plan = make_plan(S, c)
    emb = lambda t: toy.model.embed_tokens.weight[t].float()
    rn = FastRunner(S, c, plan, P, dev=dev, **rk)
    out = dict(rk=rk, tf32=tf32, layers=len(plan), D=S.D, steps=[], prompt=prompt)
    t0 = time.time(); pr = Prog(S, c, plan, dev); init_steps(pr); rn.run(pr); torch.cuda.synchronize(); out['init_s'] = time.time() - t0
    print('init_s', out['init_s'], flush=True)
    seq = list(prompt)
    for step in range(positions):
        for d in (rn.ph.t, rn.ph.n): d.clear()
        rn.sub.clear(); rn.kind_t.clear(); rn.stats.update(rows=0)
        torch.cuda.synchronize(); t00 = time.time()
        pr = Prog(S, c, plan, dev); token_program(pr, seq[step], step, Wd, emb); t_prog = time.time() - t00
        t1 = time.time()
        with reference("HF inner Qwen3 forward on host: reference logits for the error metric only; never fed back"):
            with torch.no_grad(): rr = toy(torch.tensor([seq[:step + 1]])).logits[0, -1][:c.V]
        t_ref = time.time() - t1
        lmrows = torch.tensor([pr.marks[f'lm{b}'] for b in range(c.nblk)])
        tm7 = [i for i, e in enumerate(plan) if e == ('TM', 7, 0)][0]
        t2 = time.time()
        Hs, tr = rn.run(pr, trace_rows={'lm': lmrows}, trace_layers={tm7})
        torch.cuda.synchronize(); t_run = time.time() - t2
        bestblk = tr[('lm', tm7)][:, S.VL[8].start]
        with reference("error metric: in-model block maxima vs HF block maxima (reporting only, never fed back)"):
            ref_blk = torch.stack([rr[b * c.O:(b + 1) * c.O].max() for b in range(c.nblk)])
            err = ((bestblk - ref_blk).abs().max() / rr.abs().max()).item()
        amo = Hs[pr.marks['emit']][S.AMO.start:S.AMO.stop]
        with transport("host decode codebook: binary rail code of every vocab id (index arithmetic only, no model data)"):
            tt = torch.arange(c.V); codes = torch.zeros(c.V, 36)
            for j in range(8):
                bit = ((tt % c.O) >> j) & 1; codes[:, 2 * j] = bit; codes[:, 2 * j + 1] = 1 - bit
            for j in range(10):
                bit = ((tt // c.O) >> j) & 1; codes[:, 16 + 2 * j] = bit; codes[:, 16 + 2 * j + 1] = 1 - bit
        lg = codes @ amo
        tot = time.time() - t00
        rec = dict(step=step, frames=pr.total, total_s=tot, prog_s=t_prog, ref_s=t_ref, run_s=t_run, maxrel=err, outer_argmax=int(lg.argmax()), hf_argmax=int(rr.argmax()),
                   top_code_score=float(lg.max()), phases={k: round(v, 3) for k, v in rn.ph.t.items()}, rows=rn.stats['rows'],
                   sub={k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rn.sub.items()}, kind_s={k: round(v, 3) for k, v in rn.kind_t.items()},
                   bestblk=bestblk.tolist())
        out['steps'].append(rec); print({k: v for k, v in rec.items() if k not in ('bestblk', 'sub')}, flush=True)
        if rn.prof and step == 0: print('SUB', json.dumps(rec['sub']), flush=True)
        if len(seq) == step + 1: seq.append(int(lg.argmax()))
    if rn.kp_text: out['kernprof'] = rn.kp_text; print('\n'.join(rn.kp_text), flush=True)
    out['peak_gpu_gb'] = torch.cuda.max_memory_allocated() / 1e9
    return out
