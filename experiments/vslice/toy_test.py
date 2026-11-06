import os
import sys, time, torch, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vslice import *
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

def weights(m, c):
    Wd = {}; gains = []
    for l, ly in enumerate(m.model.layers):
        a = ly.self_attn; ml = ly.mlp
        gi = ly.input_layernorm.weight; gp = ly.post_attention_layernorm.weight
        Wd[4*l] = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]) * gi[None]
        Wd[4*l+1] = a.o_proj.weight.clone()
        Wd[4*l+2] = torch.cat([ml.gate_proj.weight, ml.up_proj.weight]) * gp[None]
        Wd[4*l+3] = ml.down_proj.weight.clone()
        gains.append((a.q_norm.weight.repeat(c.nq), a.k_norm.weight.repeat(c.nkv)))
    Wd[4*c.L] = m.lm_head.weight * m.model.norm.weight[None]
    return Wd, gains

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = Cfg(); toy = make_toy(c)
    prompt = [5, 17, 301, 42]; ngen = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with torch.no_grad():
        ids = torch.tensor([prompt]); ref = toy(ids).logits[0]
    gen_ref = list(prompt); 
    with torch.no_grad():
        for _ in range(ngen): gen_ref.append(int(toy(torch.tensor([gen_ref])).logits[0, -1].argmax()))
    print('HF toy tokens', gen_ref)
    m, S = build_outer(c); m = m.to(dev).eval()
    print('outer D', S.D, 'layers', len(m.model.layers), 'params', sum(p.numel() for p in m.parameters())/1e6, 'M', dev)
    Wd, gains = weights(toy, c)
    emb = lambda t: toy.model.embed_tokens.weight[t].double()
    cache = DynamicCache(config=m.config); off = 0
    pr = Prog(S, c, m); init_steps(pr)
    fr = torch.stack(pr.frames); _, off = run_frames(m, fr, pr.pos, cache, off, dev=dev)
    toks = list(prompt); pos = 0; seq = list(prompt); t0 = time.time()
    for step in range(len(prompt) + ngen):
        tok = seq[step]
        pr = Prog(S, c, m); token_program(pr, tok, step, Wd, gains, emb)
        fr = torch.stack(pr.frames)
        logits, off = run_frames(m, fr, pr.pos, cache, off, dev=dev)
        lg = logits[-1].cpu(); r = ref[step] if step < len(prompt) else None
        if step >= len(prompt) - 1 and len(seq) == step + 1: seq.append(int(lg.argmax()))
        rr = toy(torch.tensor([seq[:step+1]])).logits[0, -1]
        print(f'step {step} frames {fr.shape[0]} outer_argmax {int(lg.argmax())} hf_argmax {int(rr.argmax())} '
              f'maxerr {(lg-rr).abs().max():.3e} /max|ref| {(lg-rr).abs().max()/rr.abs().max():.3e} t={time.time()-t0:.1f}s', flush=True)
    print('outer tokens', seq, 'HF', gen_ref)
