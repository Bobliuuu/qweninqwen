"""Step 1 gate: real Qwen3-0.6B -> integer firmware (per-layer weight bits + per-layer activation precision) -> RV32 emulator, prompt [785], position 0.
usage: run_q.py NAME OUTJSON 'SPEC'   SPEC = json {"wbits":4,"ext":1,"boost":{"2":8,"26":8,"27":8},"apl":{"2":2,"26":2,"27":2},"lm_apl":1}
boost: layer -> weight bits for ALL 7 matrices of that layer; apl: layer -> activation planes (1=int8, 2=16-bit). CPU only."""
import sys, os, json, subprocess, numpy as np, torch
from transformers import AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__)); COMP = os.environ.get('COMP', HERE + '/../../compiler'); sys.path.insert(0, HERE)
import gen_fx_q
name, outjson, spec = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
wbits = spec.get('wbits', 4); ext = bool(spec.get('ext', 1))
boost = {(int(l), 'proj'): int(b) for l, b in spec.get('boost', {}).items()}
aplmap = {int(l): int(a) for l, a in spec.get('apl', {}).items()}
if spec.get('lm_apl'): aplmap[28] = int(spec['lm_apl'])
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).eval(); prompt = [785]
with torch.no_grad(): ref = m(torch.tensor([prompt])).logits[0, -1].numpy()
c = m.config; cd = dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
    head_dim=c.head_dim, intermediate_size=c.intermediate_size, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps, rope_theta=1000000.0)
sd = {k: v.numpy() for k, v in m.state_dict().items()}; del m
out = f'/tmp/q_{name}'; os.makedirs(out, exist_ok=True)
st = gen_fx_q.gen(sd, cd, prompt, out, wbits, ext, dump=True, boost=boost, aplmap=aplmap)
elf = f'{out}/fw.elf'
subprocess.check_call([f'{COMP}/fw/build.sh', elf, '-I', out, f'{HERE}/qwen3_fx_q.c', f'{COMP}/fw/vdk.S'] + (['-DEXT=1'] if False else []))
emu = f'{COMP}/fw/rvemu'
if not os.path.exists(emu): subprocess.check_call(['cc', '-O2', '-o', emu, emu + '.c', '-lm'])
log = subprocess.check_output([emu, elf, '-b', '0x10000000', f'{out}/blob.bin', '-w', f'{out}/wrom.bin', '-o', f'{out}/out.bin', '-p'], text=True)
raw = np.fromfile(f'{out}/out.bin', dtype=np.uint8); V = cd['vocab_size']; rec = 4 + 4 * V
o = raw.reshape(1, rec); am = int(o[0, :4].copy().view(np.int32)[0]); lg = o[0, 4:].copy().view(np.int32) / 65536.0
rank = lambda l, t: int((l > l[t]).sum())
top5 = np.argsort(-lg)[:5].tolist()
marks = [int(l.split()[2]) for l in log.splitlines() if l.startswith('mark')]
res = dict(name=name, spec=spec, argmax_fw=am, argmax_hf=int(ref.argmax()), rank_15846_in_fw=rank(lg, 15846), rank_fw_argmax_in_hf=rank(ref, am), top5_fw=top5,
           top5_fw_tokens_logit=[float(lg[t]) for t in top5], gap_fw_top1_top2=float(np.sort(lg)[-1] - np.sort(lg)[-2]),
           cos_logits_fw_vs_hf=float(lg @ ref / np.linalg.norm(lg) / np.linalg.norm(ref)), instr=[b - a for a, b in zip(marks, marks[1:])], blob=st)
json.dump(res, open(outjson, 'w'), indent=1); print(json.dumps(res))
