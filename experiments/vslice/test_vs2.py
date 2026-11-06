import os
import sys, time, json, torch, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs2 import *
from transformers import Qwen3Config, Qwen3ForCausalLM

def make_toy(c, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(hidden_size=c.d, intermediate_size=c.inter, num_hidden_layers=c.L, num_attention_heads=c.nq,
                      num_key_value_heads=c.nkv, head_dim=c.hd, vocab_size=c.V, rms_norm_eps=c.eps, rope_theta=c.theta,
                      max_position_embeddings=256, tie_word_embeddings=False, attention_bias=False)
    m = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2: p.normal_(0, 0.9 / math.sqrt(p.shape[1]) if 'embed' not in n else 1.0)
            else: p.copy_(1 + 0.2 * torch.randn_like(p))
    return m

@torch.no_grad()
def weights(m, c):
    Wd = {}; P = dict(gq=[], gk=[])
    for l, ly in enumerate(m.model.layers):
        a = ly.self_attn; ml = ly.mlp
        gi = ly.input_layernorm.weight; gp = ly.post_attention_layernorm.weight
        Wd[4*l] = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]) * gi[None]
        Wd[4*l+1] = a.o_proj.weight.clone()
        Wd[4*l+2] = torch.cat([ml.gate_proj.weight, ml.up_proj.weight]) * gp[None]
        Wd[4*l+3] = ml.down_proj.weight.clone()
        P['gq'].append(a.q_norm.weight.detach().clone()); P['gk'].append(a.k_norm.weight.detach().clone())
    Wd[4*c.L] = m.lm_head.weight[:c.V] * m.model.norm.weight[None]
    return Wd, P

def logits_from(Hs_row, S, c, dev):
    x = Hs_row.to(dev)[None]; D = x.shape[-1]
    nrm = Qwen3_5RMSNorm(D, eps=1e-30).to(dev)
    with torch.no_grad(): nrm.weight.fill_(B / math.sqrt(D) - 1.0)
    return nrm(x)[0, S.LOG.start:S.LOG.stop].cpu()

def run(c, prompt, ngen, dev, tag, toy=None):
    toy = toy or make_toy(c); Wd, P = weights(toy, c)
    S = build_layout(c); plan = make_plan(S, c)
    print(f'[{tag}] d={c.d} D={S.D} outer_layers={len(plan)} I_MLP={c.I_MLP}', flush=True)
    emb = lambda t: toy.model.embed_tokens.weight[t].float()
    rn = Runner(S, c, plan, P, dev=dev)
    pr = Prog(S, c, plan, dev); init_steps(pr); rn.run(pr)
    seq = list(prompt); out = dict(tag=tag, d=c.d, D=S.D, layers=len(plan), steps=[])
    t00 = time.time()
    for step in range(int(os.environ.get('NSTEPS', len(prompt) + ngen))):
        pr = Prog(S, c, plan, dev); token_program(pr, seq[step], step, Wd, emb)
        t0 = time.time()
        rn.stats.update(rows=0, build_s=0.0, run_s=0.0)
        with torch.no_grad(): rr = toy(torch.tensor([seq[:step + 1]])).logits[0, -1][:c.V]
        if not c.tourn:
            Hs, _ = rn.run(pr)
            lg = logits_from(Hs[pr.marks['emit']], S, c, dev)
            err = ((lg - rr).abs().max() / rr.abs().max()).item()
        else:
            lmrows = torch.tensor([pr.marks[f'lm{b}'] for b in range(c.nblk)])
            tm7 = [i for i, e in enumerate(plan) if e == ('TM', 7, 0)][0]
            Hs, tr = rn.run(pr, trace_rows={'lm': lmrows}, trace_layers={tm7})
            bestblk = tr[('lm', tm7)][:, S.VL[8].start]                       # in-model block maxima
            ref_blk = torch.stack([rr[b * c.O:(b + 1) * c.O].max() for b in range(c.nblk)])
            err = ((bestblk - ref_blk).abs().max() / rr.abs().max()).item()
            dd = (bestblk - ref_blk).abs(); wi = dd.argsort(descending=True)[:4]
            print('  worst blocks', [(int(b), round(float(bestblk[b]), 4), round(float(ref_blk[b]), 4)) for b in wi], 'nblk', c.nblk)
            amo = Hs[pr.marks['emit']][S.AMO.start:S.AMO.stop]
            tt = torch.arange(c.V); codes = torch.zeros(c.V, 36)
            for j in range(8):
                bit = ((tt % c.O) >> j) & 1; codes[:, 2 * j] = bit; codes[:, 2 * j + 1] = 1 - bit
            for j in range(10):
                bit = ((tt // c.O) >> j) & 1; codes[:, 16 + 2 * j] = bit; codes[:, 16 + 2 * j + 1] = 1 - bit
            lg = codes @ amo
            print('  rails', [round(v, 3) for v in amo.tolist()[:8]], 'top code score', lg.max().item())
        if step >= len(prompt) - 1 and len(seq) == step + 1: seq.append(int(lg.argmax()))
        rec = dict(step=step, frames=pr.total, sec=time.time() - t0, build_s=rn.stats['build_s'], run_s=rn.stats['run_s'],
                   maxrel=err, outer_argmax=int(lg.argmax()), hf_argmax=int(rr.argmax()))
        out['steps'].append(rec); print(rec, flush=True)
    out['tokens'] = seq; out['maxrel_worst'] = max(s['maxrel'] for s in out['steps'])
    out['params_per_layer_M'] = None
    print(json.dumps({k: v for k, v in out.items() if k != 'steps'}), flush=True)
    json.dump(out, open(f'res_vs2_{tag}.json', 'w'), indent=1)

if __name__ == '__main__':
    scale = sys.argv[1]; ngen = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if scale == 'd256': c = Cfg(256, 4, 2, 64, 512, 2, 512, I_MLP=4096)
    elif scale == 'd512': c = Cfg(512, 8, 4, 64, 1024, 2, 1024, I_MLP=4096)
    elif scale.startswith('real'): pass
    elif scale == 'd1024': c = Cfg(1024, 16, 8, 128, 3072, 1, 4096, theta=1e6, I_MLP=8192)
    toy = None; prompt = [5, 17, 301, 42]
    if scale.startswith('real'):
        from transformers import AutoTokenizer
        parts = scale[4:].split('_L'); Vt = int(parts[0]) if parts[0] else 4096
        nl = int(parts[1]) if len(parts) > 1 else 1
        c = Cfg(1024, 16, 8, 128, 3072, nl, Vt, theta=1e6, I_MLP=8192)
        toy = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', num_hidden_layers=nl, torch_dtype=torch.float32).eval()
        prompt = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')('The capital of France is')['input_ids']
        print('prompt', prompt)
    run(c, prompt, ngen, dev, scale, toy)
