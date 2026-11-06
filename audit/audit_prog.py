"""Frame-program audit for trueqiq vs2 (host stream).  No outer model is executed; only the host Prog is built and inspected.
usage: python audit_prog.py <real151936_L28|real4096_L1|d256> <out.json> [prompt text] [nsteps]
Checks, for every host-written frame of every token program:
  S1 slot whitelist: nonzero only in BALLAST, EMBX, PAY, SEL, INIT, RINIT, WF, RF, ACTF, MASK*, MASKM, PAD, BLKCODE, register RA/WA
  S2 flag/address fields hold exactly 0/1 (no analog values); PAY empty in control frames; EMBX nonzero only in the 'emb' frame and equal to the embedding row of the token
  S3 write payloads decode (silu(v)-silu(ref)) to +-Wd slices; Wd == inner weight * folded gain (nothing else)
  S4 token independence digests: program with EMBX zeroed hashed per step (compare across prompts with compare_prog.py)
"""
import sys, os, json, hashlib, math, time, torch
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_vs2 as T
from vs2 import *
from transformers import Qwen3ForCausalLM, AutoTokenizer

scale, out_path = sys.argv[1], sys.argv[2]
prompt_txt = sys.argv[3] if len(sys.argv) > 3 else 'The capital of France is'
nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
parts = scale[4:].split('_L'); Vt = int(parts[0]); nl = int(parts[1])
c = Cfg(1024, 16, 8, 128, 3072, nl, Vt, theta=1e6, I_MLP=8192)
toy = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', num_hidden_layers=nl, torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); prompt = tok(prompt_txt)['input_ids']
Wd, P = T.weights(toy, c); S = build_layout(c); plan = make_plan(S, c)
rep = dict(scale=scale, D=S.D, outer_layers=len(plan), prompt=prompt, V=Vt, L=nl, violations=[], steps=[])
# ---- allowed host slots
allowed = torch.zeros(S.D, dtype=torch.bool); names = {}
def al(sl_, nm):
    allowed[sl_] = True; names[nm] = (sl_.start, sl_.stop)
allowed[0] = True
for nm in ('EMBX', 'PAY', 'SEL', 'INIT', 'RINIT', 'WF', 'RF', 'ACTF'): al(getattr(S, nm), nm)
for l in range(nl): al(S.MASK[l], f'MASK{l}')
if c.tourn:
    for nm in ('PAD', 'BLKCODE', 'MASKM'): al(getattr(S, nm), nm)
for k, r in S.regs.items(): al(r['RA'], k + '_RA'); al(r['WA'], k + '_WA')
def slot_name(i):
    for nm, (a, b) in dict(EMBX=(S.EMBX.start, S.EMBX.stop), X=(S.X.start, S.X.stop), Y=(S.Y.start, S.Y.stop), AO=(S.AO.start, S.AO.stop), QKVR=(S.QKVR.start, S.QKVR.stop), GU=(S.GU.start, S.GU.stop), ACT=(S.ACT.start, S.ACT.stop), RESX=(S.RESX.start, S.RESX.stop), RESH=(S.RESH.start, S.RESH.stop), NIN=(S.NINS.start, S.NINS.stop)).items():
        if a <= i < b: return nm
    return f'slot{i}'
flagish = allowed.clone(); flagish[S.EMBX] = False; flagish[S.PAY] = False; flagish[0] = False
# ---- S3a: Wd == weight*gain
def check_wd():
    out = {}
    for l, ly in enumerate(toy.model.layers):
        a = ly.self_attn; ml = ly.mlp; gi = ly.input_layernorm.weight; gp = ly.post_attention_layernorm.weight
        for op, ref in ((4*l, torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]) * gi[None]), (4*l+1, a.o_proj.weight),
                        (4*l+2, torch.cat([ml.gate_proj.weight, ml.up_proj.weight]) * gp[None]), (4*l+3, ml.down_proj.weight)):
            out[op] = bool(torch.equal(Wd[op], ref))
    out[4*nl] = bool(torch.equal(Wd[4*nl], toy.lm_head.weight[:Vt] * toy.model.norm.weight[None]))
    return out
wd_ok = check_wd(); rep['Wd_equals_weight_times_gain'] = all(wd_ok.values()); rep['Wd_ops'] = len(wd_ok)
# rawness: how many distinct gain-folded vs raw
rep['gain_folded_ops'] = [op for op in wd_ok if op % 4 in (0, 2) and op < 4*nl] + [4*nl]
emb = lambda t: toy.model.embed_tokens.weight[t].float()
seq = list(prompt); fake_next = [1000, 2000, 3000, 4000]       # tokens "emitted" for steps beyond the prompt (any value; program must not care)
silu64 = lambda u: u / (1 + torch.exp(-u))
def decode_job(job):
    pay = job['pay'] if 'pay' in job else tile_payload(job['W'], job['o0'], job['j'], c, 'cpu')
    return pay
import random; random.seed(0)
for step in range(nsteps):
    t0 = time.time(); pr = Prog(S, c, plan, 'cpu'); tk = seq[step] if step < len(seq) else fake_next[step - len(seq)]
    token_program(pr, tk, step, Wd, emb)
    Fr = pr.ctrl_frames(); nr = Fr.shape[0]; emb_row = pr.marks['emb']
    viol = {}
    nz = (Fr != 0)
    bad = nz & ~allowed[None]; bad[:, 0] = False
    if bad.any():
        cols = bad.any(0).nonzero()[:, 0].tolist(); viol['S1_slot_not_whitelisted'] = sorted({slot_name(i) for i in cols})
    # S2
    fv = Fr[:, flagish]
    if (fv.abs() * ((fv != 0) & (fv != 1)).float()).sum() > 0: viol['S2_flag_not_0_1'] = int(((fv != 0) & (fv != 1)).sum())
    if nz[:, S.PAY.start:S.PAY.stop].any(): viol['S2_PAY_in_control_frame'] = True
    e_nz = nz[:, S.EMBX.start:S.EMBX.stop].any(1).nonzero()[:, 0].tolist()
    if e_nz != [emb_row] and e_nz != []: viol['S2_EMBX_in_other_frames'] = e_nz[:5]
    if not torch.equal(Fr[emb_row, S.EMBX.start:S.EMBX.stop], emb(tk)): viol['S2_EMBX_not_embedding_row'] = True
    if Fr[:, 0].ne(B).any(): viol['S2_ballast_not_const'] = True
    # S3b: dense frames of jobs: only structure (dense_job code); check dense inits like control frames; decode sample payloads
    njobs = len(pr.jobs); dec = []
    pick = set(range(0, njobs, max(1, njobs // 60))) | {njobs - 1}
    maxerr = 0.0
    for ji in sorted(pick):
        job = pr.jobs[ji]
        if 'dense' in job:
            d = job['dense']; b2 = (d != 0) & ~allowed[None]; b2[:, 0] = False
            if b2.any(): viol.setdefault('S1_dense_job_slot', []).append(ji)
            continue
        pay = decode_job(job).double(); O, H, P = c.O, c.H, c.P
        v = pay.reshape(2 * P, H, O + 1)
        rows = silu64(v[:, :, :O]) - silu64(v[:, :, O:O + 1])
        W = job['W']; o0, j = job['o0'], job['j']
        Wb = torch.zeros(O, H * P, dtype=torch.float64); sl_ = W[o0:o0 + O, j * H * P:(j + 1) * H * P].double(); Wb[:sl_.shape[0], :sl_.shape[1]] = sl_
        Wt = Wb.reshape(O, H, P).permute(2, 1, 0); exp = torch.cat([Wt, -Wt], 0)
        e = ((rows - exp).abs().max() / (exp.abs().max() + 1e-30)).item(); maxerr = max(maxerr, e)
    rep.setdefault('payload_decode_maxrel', 0.0); rep['payload_decode_maxrel'] = max(rep['payload_decode_maxrel'], maxerr)
    if maxerr > 1e-4: viol['S3_payload_not_weights'] = maxerr
    # S4 digests
    F0 = Fr.clone(); F0[:, S.EMBX.start:S.EMBX.stop] = 0
    dig = hashlib.sha256(F0.numpy().tobytes()).hexdigest()[:16]
    jd = hashlib.sha256(repr([(j['li'], j.get('o0'), j.get('j'), j.get('wf'), j.get('m')) for j in pr.jobs]).encode()).hexdigest()[:16]
    pd = hashlib.sha256(repr(pr.cpos).encode()).hexdigest()[:16]
    rep['steps'].append(dict(step=step, token=tk, ctrl_rows=nr, jobs=njobs, write_rows=pr.total - nr, ctrl_digest_noEMB=dig, jobs_digest=jd, cpos_digest=pd,
                             payload_sampled=len(pick), viol=viol, sec=time.time() - t0))
    if viol: rep['violations'].append((step, viol))
    print(rep['steps'][-1], flush=True)
rep['PASS'] = not rep['violations'] and rep['Wd_equals_weight_times_gain']
json.dump(rep, open(out_path, 'w'), indent=1)
print('PROG AUDIT', 'PASS' if rep['PASS'] else 'FAIL', 'D', S.D, 'layers', len(plan))
