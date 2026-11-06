"""Tiny Qwen3 (random weights) -> integer firmware qwen3_fx.c -> emulator vs HF logits. Variants: wbits 4/8, base RV32IM nonlinears vs custom-1 EXT.
Run on the toolchain host:  python test_tiny_fx.py"""
import os, subprocess, sys, json, numpy as np, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import gen_fx

def build_run(sd, cd, prompt, out, wbits, ext, dump=True, extra=()):
    stats = gen_fx.gen(sd, cd, prompt, out, wbits, ext, dump=dump)
    elf = f'{out}/fw.elf'
    subprocess.check_call([f'{HERE}/fw/build.sh', elf, '-I', out, f'{HERE}/fw/qwen3_fx.c', f'{HERE}/fw/vdk.S', *extra])
    emu = f'{HERE}/fw/rvemu'
    if not os.path.exists(emu): subprocess.check_call(['cc', '-O2', '-o', emu, emu + '.c', '-lm'])
    log = subprocess.check_output([emu, elf, '-b', '0x10000000', f'{out}/blob.bin', '-w', f'{out}/wrom.bin', '-o', f'{out}/out.bin', '-p'], text=True)
    return stats, log

def parse(log, out, n, V, dump=True):
    raw = np.fromfile(f'{out}/out.bin', dtype=np.uint8); rec = 4 + (4 * V if dump else 0)
    o = raw.reshape(n, rec); am = o[:, :4].copy().view(np.int32).ravel()
    lg = o[:, 4:].copy().view(np.int32).reshape(n, V) / 65536.0 if dump else None
    marks = [int(l.split()[2]) for l in log.splitlines() if l.startswith('mark')]
    syms = {l.split()[1]: int(l.split()[2]) for l in log.splitlines() if l.startswith('sym')}
    return am, lg, marks, syms, log.splitlines()[0]

def make_model(cfgd, seed=0, std=0.3):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(Qwen3Config(**cfgd, tie_word_embeddings=True, attention_bias=False)).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2: p.normal_(0, std)
            elif 'norm' in n: p.uniform_(0.5, 1.5)
    return m

if __name__ == '__main__':
    cfgd = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128,
                vocab_size=128, max_position_embeddings=64, rope_theta=1000000.0, rms_norm_eps=1e-6)
    prompt = [5, 17, 99, 3, 64, 120, 8, 42]
    m = make_model(cfgd)
    with torch.no_grad(): ref = m(torch.tensor([prompt])).logits[0].numpy()
    sd = {k: v.numpy() for k, v in m.state_dict().items()}
    res = []
    for wbits in (4, 8):
        for ext in (False, True):
            out = f'/tmp/qq_fx_{wbits}_{int(ext)}'
            st, log = build_run(sd, cfgd, prompt, out, wbits, ext)
            am, lg, marks, syms, first = parse(log, out, len(prompt), cfgd['vocab_size'])
            rel = float(np.abs(lg - ref).max() / np.abs(ref).max())
            res.append(dict(wbits=wbits, ext=ext, max_rel_logit_err=rel, rms_rel_err=float(np.sqrt(((lg - ref) ** 2).mean()) / ref.std()),
                            argmax_fw=am.tolist(), argmax_hf=ref.argmax(1).tolist(), instr_per_pos=[b - a for a, b in zip(marks, marks[1:])], top_syms=dict(sorted(syms.items(), key=lambda kv: -kv[1])[:8])))
            print(json.dumps(res[-1]))
    json.dump(res, open(f'{HERE}/res_tiny_fx.json', 'w'), indent=1)
