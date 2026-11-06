"""Firmware top-1 vs HF on diverse teacher-forced prompts (algo's 12 + a ~128-token held-out natural passage).
usage: fidelity_prompts.py WBITS EXT OUTDIR   (gen once with MAXPOS=160, swap PROMPT in cfg.h per sequence)"""
import sys, os, re, json, subprocess, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import gen_fx
wbits, ext, out = int(sys.argv[1]), bool(int(sys.argv[2])), sys.argv[3]; apl = int(sys.argv[4]) if len(sys.argv) > 4 else 1
PROMPTS = ["The capital of France is", "In 1969, humans first landed on the Moon, and", "def fibonacci(n):\n    if n < 2:\n        return n",
           "Water boils at 100 degrees Celsius at sea level because", "The quick brown fox jumps over the lazy dog. The dog", "Photosynthesis is the process by which plants",
           "Once upon a time, in a small village by the sea, there lived", "The theory of general relativity describes gravity as", "Q: What is 12 times 12?\nA:",
           "import numpy as np\nx = np.arange(10)\nprint(x.sum())", "Le chat est sur la table et le chien", "The largest planet in the solar system is"]
HELD = ("The old lighthouse keeper climbed the spiral stairs every evening, carrying a small lamp and a thermos of tea. From the top he could see the fishing boats "
        "returning through the fog, their engines muttering like distant thunder. Years ago the harbor had been busy with merchants and sailors from a dozen countries, "
        "but now only a few families remained, mending nets and telling stories about storms that no one else remembered. When the wind rose, the glass trembled, and he "
        "would sit by the window and listen, wondering whether the sea was warning him or simply keeping him company.")
name = 'Qwen/Qwen3-0.6B'; tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
seqs = [tok(p).input_ids for p in PROMPTS] + [tok(HELD).input_ids[:128]]
refs = []
with torch.no_grad():
    for s in seqs: refs.append(m(torch.tensor([s])).logits[0].argmax(-1).numpy())
c = m.config; cd = dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
    head_dim=c.head_dim, intermediate_size=c.intermediate_size, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps, rope_theta=1000000.0)
sd = {k: v.numpy() for k, v in m.state_dict().items()}; del m
gen_fx.gen(sd, cd, seqs[0], out, wbits, ext, maxpos=160, dump=False, apl=apl)
cfg = open(f'{out}/cfg.h').read(); emu = f'{HERE}/fw/rvemu'; res = []
for i, s in enumerate(seqs):
    c2 = re.sub(r'#define PROMPT .*', '#define PROMPT {' + ','.join(map(str, s)) + '}', cfg); c2 = re.sub(r'#define NTOK \d+', f'#define NTOK {len(s)}', c2)
    open(f'{out}/cfg.h', 'w').write(c2)
    subprocess.check_call([f'{HERE}/fw/build.sh', f'{out}/fw.elf', '-I', out, f'{HERE}/fw/qwen3_fx.c', f'{HERE}/fw/vdk.S'])
    log = subprocess.check_output([emu, f'{out}/fw.elf', '-b', '0x10000000', f'{out}/blob.bin', '-w', f'{out}/wrom.bin', '-o', f'{out}/out.bin'], text=True)
    am = np.fromfile(f'{out}/out.bin', dtype=np.int32); ok = int((am == refs[i]).sum())
    ptok = int(np.sum(am[0:1] == refs[i][0:1])); res.append(dict(prompt=(PROMPTS + ['<held-out 128-token passage>'])[i][:40], n=len(s), match=ok, pos0_ok=bool(am[0] == refs[i][0]), pos1plus_match=int((am[1:] == refs[i][1:]).sum()),
                    instr=int(re.search(r'instr=(\d+)', log).group(1)))); print(json.dumps(res[-1]), flush=True)
tot = sum(r['n'] for r in res[:12]); mt = sum(r['match'] for r in res[:12])
summary = dict(wbits=wbits, apl=apl, ext=ext, algo12_match=mt, algo12_total=tot, held_match=res[12]['match'], held_total=res[12]['n'], per_prompt=res)
json.dump(summary, open(f'{HERE}/res_fidelity_prompts_w{wbits}a{apl}.json', 'w'), indent=1); print(json.dumps({k: v for k, v in summary.items() if k != 'per_prompt'}))
