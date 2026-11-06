"""Real Qwen3-0.6B -> int4 EXT-free firmware inputs (cfg.h, blob.bin, wrom.bin) for a given prompt; also HF argmax per position.
usage: gen_full.py OUTDIR TOKEN[,TOKEN...] [wbits=4] [ext=0]"""
import sys, os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../compiler')
import gen_fx
from transformers import AutoModelForCausalLM
out = sys.argv[1]; prompt = [int(t) for t in sys.argv[2].split(',')]; wbits = int(sys.argv[3]) if len(sys.argv) > 3 else 4; ext = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).eval()
with torch.no_grad(): lg = m(torch.tensor([prompt])).logits[0].numpy()
c = m.config; cd = dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
    head_dim=c.head_dim, intermediate_size=c.intermediate_size, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps, rope_theta=1000000.0)
sd = {k: v.numpy() for k, v in m.state_dict().items()}
st = gen_fx.gen(sd, cd, prompt, out, wbits, ext)
top2 = np.sort(lg, 1)[:, -2:]
json.dump(dict(prompt=prompt, wbits=wbits, ext=ext, hf_argmax=lg.argmax(1).tolist(), hf_gap=(top2[:, 1] - top2[:, 0]).tolist(), **st), open(out + '/gen.json', 'w'))
print(open(out + '/gen.json').read())
