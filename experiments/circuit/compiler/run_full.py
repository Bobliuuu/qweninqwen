"""Full Qwen3-0.6B (real HF weights) -> RV32IM firmware -> emulator; instruction counts per token/op and check vs HF.
usage: run_full.py MODE(f32|q4v) NTOK OUTDIR.  Heavy (f32: ~1e11 instr/token in the emulator): run on the GPU box, CPU only."""
import sys, os, json, subprocess, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import gen_fw
mode, ntok, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
name = 'Qwen/Qwen3-0.6B'
tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
prompt = tok("The capital of France is Paris. The capital of Germany is", return_tensors='pt').input_ids[0][:ntok].tolist()
with torch.no_grad(): ref = m(torch.tensor([prompt])).logits[0].numpy()
c = m.config; cd = dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
    head_dim=c.head_dim, intermediate_size=c.intermediate_size, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps, rope_theta=1000000.0)
sd = {k: v.numpy() for k, v in m.state_dict().items()}; del m
st = gen_fw.gen(sd, cd, prompt, out, mode, dump=False)
elf = f'{out}/fw.elf'; subprocess.check_call([f'{HERE}/fw/build.sh', elf, '-I', out, f'{HERE}/fw/qwen3.c'] + ([f'{HERE}/fw/vdk.S'] if mode == 'q4v' else []))
emu = f'{HERE}/fw/rvemu'
if not os.path.exists(emu): subprocess.check_call(['cc', '-O2', '-o', emu, emu + '.c'])
args = [emu, elf, '-b', '0x10000000', f'{out}/blob.bin', '-o', f'{out}/out.bin', '-p'] + (['-w', f'{out}/wrom.bin'] if mode == 'q4v' else [])
log = subprocess.check_output(args, text=True)
am = np.fromfile(f'{out}/out.bin', dtype=np.int32)
marks = [int(l.split()[2]) for l in log.splitlines() if l.startswith('mark')]
syms = {l.split()[1]: int(l.split()[2]) for l in log.splitlines() if l.startswith('sym')}
res = dict(model=name, mode=mode, prompt=prompt, argmax_fw=am.tolist(), argmax_hf=ref.argmax(1).tolist(), instr_per_pos=[b - a for a, b in zip(marks, marks[1:])],
           instr_total=int(log.split('instr=')[1].split()[0]), classes=log.splitlines()[0], by_symbol=syms, blob=st)
json.dump(res, open(f'{HERE}/res_full_{mode}.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'by_symbol'}))
