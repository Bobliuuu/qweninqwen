import sys, numpy as np, torch, json
import test_tiny_fx as T, gen_fx
cfgd = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=128, max_position_embeddings=64, rope_theta=1000000.0, rms_norm_eps=1e-6)
prompt = [5, 17, 99, 3, 64, 120, 8, 42]
m = T.make_model(cfgd)
with torch.no_grad(): ref = m(torch.tensor([prompt])).logits[0].numpy()
sd = {k: v.numpy() for k, v in m.state_dict().items()}
import subprocess
for wb, apl in ((8, 1), (8, 2), (12, 1), (12, 2)):
    out = f'/tmp/qq_dbg_{wb}_{apl}'
    gen_fx.gen(sd, cfgd, prompt, out, wb, True, dump=True, apl=apl)
    subprocess.check_call([f'fw/build.sh', f'{out}/fw.elf', '-I', out, 'fw/qwen3_fx.c', 'fw/vdk.S'])
    log = subprocess.check_output(['fw/rvemu', f'{out}/fw.elf', '-b', '0x10000000', f'{out}/blob.bin', '-w', f'{out}/wrom.bin', '-o', f'{out}/out.bin', '-p'], text=True)
    am, lg, marks, syms, first = T.parse(log, out, len(prompt), 128)
    print(wb, apl, 'rel', float(np.abs(lg - ref).max() / np.abs(ref).max()), am.tolist(), ref.argmax(1).tolist())
