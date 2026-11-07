"""Step 2 of the quantfix: generate the FINAL-contract CF+RAM firmware stack (layers 2/26/27: 8-bit weights + 16-bit activations; EXT off = plain RV32IM nonlinears, as the circuit's ramcf.elf) on Modal CPU,
all under NEW paths on volume qfr-data (/data/cfram_q/...). Never touches /data/cfram, /data/w785, /data/rowtab_cf.npy.
  modal run --detach quantfix/modal_cfq.py::main --what ctrl    # backward-compat control: cf_gen_q.py on the OLD /data/w785 must reproduce /data/rowtab_cf.npy bit for bit
  modal run --detach quantfix/modal_cfq.py::main --what build   # gen_fx_q (boost+apl) -> cfg.h blob.bin wrom.bin -> cf_gen_q.py -> build ramcf_q.elf (+ size / symbol report)
Outputs (build): /data/cfram_q/{cfg/{cfg.h,blob.bin,wrom.bin}, cf/{cf_cfg.h,cfetch.S,rowtab_cf.npy,cf_layout.json}, ramcf_q.elf, build_report.txt}."""
import modal, subprocess, os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); CIRC = os.path.abspath(HERE + '/../..')
app = modal.App('qwen-in-qwen-cfq')
img = (modal.Image.debian_slim(python_version='3.11').apt_install('gcc-riscv64-unknown-elf', 'picolibc-riscv64-unknown-elf', 'build-essential')
       .pip_install('torch', 'transformers', 'numpy', 'accelerate').add_local_dir(CIRC, '/root/circuit', ignore=['__pycache__', '*.elf', 'logs', '*.npy', '*.pkl', '*.bin']))
data = modal.Volume.from_name('qfr-data')
SPEC = dict(wbits=4, boost={2: 8, 26: 8, 27: 8}, aplmap={2: 2, 26: 2, 27: 2})
def md5(p, n=1 << 24):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()

@app.function(image=img, cpu=8, memory=28672, timeout=7200, volumes={'/data': data})
def ctrl():
    """cf_gen_q.py on the old W4 cfg (/data/w785) must give the SAME rowtab as the original cf_gen.py and as /data/rowtab_cf.npy; macros must resolve to the same row ids."""
    data.reload(); o = {}
    for nm, script in (('orig', 'cf/cf_gen.py'), ('q', 'quantfix/cf_gen_q.py')):
        d = f'/tmp/ctrl_{nm}'; subprocess.run(['python', f'/root/circuit/fullrun/{script}', '/data/w785', d], check=True, capture_output=True, text=True)
        o[nm] = dict(rowtab=md5(d + '/rowtab_cf.npy'), layout=json.load(open(d + '/cf_layout.json')))
    o['existing_rowtab_cf'] = md5('/data/rowtab_cf.npy'); o['layout_same'] = {k: o['orig']['layout'][k] == o['q']['layout'][k] for k in ('rows', 'KH', 'KL', 'KF', 'KS', 'LM_ROW0', 'LS_ROW')}
    # macro semantics: CF_LBASE(l), CFW_SHIFT(l), CF_FN_ROW, CFW_SHIFT_LM identical?
    import re
    def macros(d):
        t = open(d + '/cf_cfg.h').read(); L = o['orig']['layout']; KH, KL = L['KH'], L['KL']
        if 'CF_LBASE_T' in t:
            lb = [int(v) for v in re.search(r'CF_LBASE_T\[\d+\] = \{(.*?)\}', t).group(1).split(',')]; sh = [int(v) for v in re.search(r'CFW_SHIFT_T\[\d+\] = \{(.*?)\}', t).group(1).split(',')]
        else:
            m = re.search(r'CF_LBASE\(l\) \((\d+) \+ \(l\) \* (\d+)\)', t); a, b = int(m.group(1)), int(m.group(2)); lb = [a + l * b for l in range(28)]
            m = re.search(r'CFW_SHIFT\(l\) \((\d+) \+ \(l\) \* (\d+)\)', t); a, b = int(m.group(1)), int(m.group(2)); sh = [a + l * b for l in range(28)]
        return lb, sh, int(re.search(r'CF_FN_ROW \(?(\d+)', t).group(1)), int(re.search(r'CFW_SHIFT_LM \(?(-?\d+)', t).group(1))
    o['macros_same'] = macros('/tmp/ctrl_orig') == macros('/tmp/ctrl_q')
    for nm in ('orig', 'q'): o[nm].pop('layout')
    print(json.dumps(o), flush=True); return o

@app.function(image=img, cpu=8, memory=28672, timeout=10800, volumes={'/data': data})
def build(spec_json='', outroot='/data/cfram_q', ext=False):
    import sys, torch, numpy as np
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, '/root/circuit/fullrun/quantfix'); import gen_fx_q
    spec = json.loads(spec_json) if spec_json else {}
    boost = {(int(l), 'proj'): int(b) for l, b in spec.get('boost', {'2': 8, '26': 8, '27': 8}).items()}
    aplmap = {int(l): int(a) for l, a in spec.get('apl', {'2': 2, '26': 2, '27': 2}).items()}
    m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).eval(); c = m.config
    cd = dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
              head_dim=c.head_dim, intermediate_size=c.intermediate_size, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps, rope_theta=1000000.0)
    sd = {k: v.numpy() for k, v in m.state_dict().items()}; del m
    cfg = '/tmp/cfg'; st = gen_fx_q.gen(sd, cd, [785], cfg, 4, ext, dump=False, boost=boost, aplmap=aplmap); del sd
    rep = [f'gen: {st}']
    for n in ('cfg.h', 'blob.bin', 'wrom.bin'): rep.append(f'{n} md5 {md5(cfg + "/" + n)} size {os.path.getsize(cfg + "/" + n)}')
    cfd = '/tmp/cf'; r = subprocess.run(['python', '/root/circuit/fullrun/quantfix/cf_gen_q.py', cfg, cfd], capture_output=True, text=True); rep.append('cf_gen_q: ' + (r.stdout[-1500:] + r.stderr[-1500:]))
    assert r.returncode == 0, r.stderr[-2000:]
    # build the ELF exactly like cf/build_fw_ram_cf.sh but with the quantfix firmware (PMAX=2 planes)
    C = '/root/circuit/compiler/fw'; H = '/root/circuit/ramown'; PICO = '/usr/lib/picolibc/riscv64-unknown-elf'; GCC = 'riscv64-unknown-elf-gcc'
    libgcc = subprocess.check_output([GCC, '-march=rv32im', '-mabi=ilp32', '-print-libgcc-file-name'], text=True).strip(); elf = '/tmp/ramcf_q.elf'
    cmd = [GCC, '-march=rv32im', '-mabi=ilp32', '-O2', '-DNOMARK', '-DPUTCHAR', '-DWF', '-DPMAX=2', '-DTILE=128', '-isystem', PICO + '/include', '-I' + cfg, '-I' + cfd, '-I' + C, '-nostdlib', '-nostartfiles', '-static', '-T' + H + '/fw_rom.ld',
           '-o', elf, C + '/env/crt0.S', '/root/circuit/fullrun/quantfix/fw_qwen3_ram_cf_q.c', C + '/vdk_wf.S', cfd + '/cfetch.S', '-Wl,--start-group', PICO + '/lib/rv32im/ilp32/libc.a', libgcc, '-Wl,--end-group']
    r = subprocess.run(cmd, capture_output=True, text=True); rep.append('gcc rc %d\n%s' % (r.returncode, (r.stdout + r.stderr)[-3000:])); assert r.returncode == 0, rep[-1]
    rep.append(subprocess.check_output(['riscv64-unknown-elf-size', '-A', elf], text=True)); rep.append(subprocess.check_output('riscv64-unknown-elf-nm -S --size-sort ' + elf + ' | tail -14', shell=True, text=True))
    rep.append(subprocess.check_output('riscv64-unknown-elf-nm ' + elf + ' | grep -E " (_end|__heap_start|__heap_end)$"', shell=True, text=True))
    rep.append('elf md5 ' + md5(elf))
    os.makedirs(outroot + '/cfg', exist_ok=True); os.makedirs(outroot + '/cf', exist_ok=True)
    for n in ('cfg.h', 'blob.bin', 'wrom.bin'): subprocess.run(['cp', cfg + '/' + n, outroot + '/cfg/' + n], check=True)
    for n in os.listdir(cfd): subprocess.run(['cp', cfd + '/' + n, outroot + '/cf/' + n], check=True)
    subprocess.run(['cp', elf, outroot + '/ramcf_q.elf'], check=True); open(outroot + '/build_report.txt', 'w').write('\n'.join(rep)); data.commit()
    print('\n'.join(rep)[-6000:], flush=True); return rep

@app.local_entrypoint()
def main(what: str = 'build'):
    if what == 'ctrl': print(json.dumps(ctrl.remote()))
    else: print(json.dumps(build.remote(''), default=str)[-6000:])
