"""Tiny-config end-to-end: random Qwen3 (HF) -> RV32IM firmware (gen_fw.py + fw/qwen3.c) -> emulator (fw/rvemu.c) vs HF logits.
Needs: torch, transformers, riscv64-unknown-elf-gcc + picolibc, a C compiler.  Run on the GPU box (it has the toolchain)."""
import os, subprocess, sys, json, numpy as np, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen_fw

def run(cfgd, prompt, mode, out, seed=0, std=0.3):
    torch.manual_seed(seed)
    cfg = Qwen3Config(**cfgd, tie_word_embeddings=True, attention_bias=False)
    m = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2: p.normal_(0, std)
            elif 'norm' in n: p.uniform_(0.5, 1.5)
        ref = m(torch.tensor([prompt])).logits[0].numpy()
    sd = {k: v.numpy() for k, v in m.state_dict().items()}
    cd = dict(cfgd); cd.setdefault('rms_norm_eps', 1e-6); cd.setdefault('rope_theta', cfg.rope_parameters['rope_theta'] if getattr(cfg, 'rope_parameters', None) else cfg.rope_theta)
    stats = gen_fw.gen(sd, cd, prompt, out, mode)
    elf = f'{out}/fw.elf'
    srcs = [f'{HERE}/fw/qwen3.c'] + ([f'{HERE}/fw/vdk.S'] if mode == 'q4v' else [])
    subprocess.check_call([f'{HERE}/fw/build.sh', elf, '-I', out, *srcs])
    emu = f'{HERE}/fw/rvemu'
    if not os.path.exists(emu): subprocess.check_call(['cc', '-O2', '-o', emu, emu + '.c'])
    args = [emu, elf, '-b', '0x10000000', f'{out}/blob.bin', '-o', f'{out}/out.bin', '-p']
    if mode == 'q4v': args += ['-w', f'{out}/wrom.bin']
    log = subprocess.check_output(args, text=True)
    raw = np.fromfile(f'{out}/out.bin', dtype=np.uint8); V = cfgd['vocab_size']; rec = 4 + 4 * V
    outp = raw.reshape(len(prompt), rec); am = outp[:, :4].copy().view(np.int32).ravel(); lg = outp[:, 4:].copy().view(np.float32).reshape(len(prompt), V)
    marks = [int(l.split()[2]) for l in log.splitlines() if l.startswith('mark')]
    first = log.splitlines()[0]
    rel = float(np.abs(lg - ref).max() / np.abs(ref).max())
    # argmax agreement counted only where HF top1-top2 gap is above the error scale
    srt = np.sort(ref, 1); gap = srt[:, -1] - srt[:, -2]; ok = [(a == r) or g < 3 * rel * np.abs(ref).max() for a, r, g in zip(am, ref.argmax(1), gap)]
    return dict(mode=mode, max_rel_logit_err=rel, argmax_fw=am.tolist(), argmax_hf=ref.argmax(1).tolist(), argmax_ok_given_gap=bool(all(ok)),
                instr_total=int(first.split('instr=')[1].split()[0]), instr_per_pos=[b - a for a, b in zip(marks, marks[1:])], blob=stats, emu_line=first)

if __name__ == '__main__':
    cfgd = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128,
                vocab_size=128, max_position_embeddings=64, rope_theta=1000000.0, rms_norm_eps=1e-6)
    prompt = [5, 17, 99, 3, 64, 120, 8, 42]
    res = [run(cfgd, prompt, md, f'/tmp/qq_tiny_{md}') for md in ('f32', 'q4v')]
    for r in res: print(json.dumps(r))
    json.dump(res, open(f'{HERE}/res_tiny.json', 'w'), indent=1)
    assert res[0]['max_rel_logit_err'] < 1e-3 and res[0]['argmax_fw'] == res[0]['argmax_hf']
    print('PASS')
