"""Run run_q.py variants on Modal CPU containers in parallel (no GPU): modal run --detach quantfix/modal_q.py --names a,b,c
Variants are defined in VARIANTS below. Results: printed to app logs AND volume qfr-out /quantfix/<name>.json."""
import modal, subprocess, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App('qwen-in-qwen-quantfix')
img = (modal.Image.debian_slim(python_version='3.11').apt_install('gcc-riscv64-unknown-elf', 'picolibc-riscv64-unknown-elf', 'build-essential')
       .pip_install('torch', 'transformers', 'numpy', 'accelerate').add_local_dir(HERE, '/root/quantfix', ignore=['*.json', '__pycache__']).add_local_dir(HERE + '/../../compiler', '/root/compiler', ignore=['*.json', '__pycache__', '*.elf']))
out = modal.Volume.from_name('qfr-out', create_if_missing=True)
L3 = {'2': 1, '26': 1, '27': 1}
VARIANTS = {
 'ctrl_w4':        dict(wbits=4),                                                              # control: must give 13 (reproduces the firmware answer)
 'b8_apl2':        dict(boost={'2': 8, '26': 8, '27': 8}, apl={'2': 2, '26': 2, '27': 2}),
 'b8_apl2_l3':     dict(boost={'2': 8, '3': 8, '26': 8, '27': 8}, apl={'2': 2, '3': 2, '26': 2, '27': 2}),
 'b12_apl2':       dict(boost={'2': 12, '26': 12, '27': 12}, apl={'2': 2, '26': 2, '27': 2}),
 'w4_apl2only':    dict(apl={'2': 2, '26': 2, '27': 2}),
 'b8_apl1':        dict(boost={'2': 8, '26': 8, '27': 8}),
 # EXT OFF = the final CF+RAM circuit firmware (cfg.h without #define EXT: plain RV32IM nonlinears, as ramcf.elf / refRC were built); the first gate ran EXT=1
 'ctrl_w4_noext':  dict(wbits=4, ext=0),
 'b8_apl2_noext':  dict(boost={'2': 8, '26': 8, '27': 8}, apl={'2': 2, '26': 2, '27': 2}, ext=0),
}
@app.function(image=img, cpu=8, memory=28672, timeout=7200, volumes={'/out': out})
def go(name, spec):
    os.makedirs('/out/quantfix', exist_ok=True)
    r = subprocess.run(['python', '-u', '/root/quantfix/run_q.py', name, f'/out/quantfix/{name}.json', json.dumps(spec)], capture_output=True, text=True, env=dict(os.environ, COMP='/root/compiler'))
    out.commit(); print(name, r.stdout[-3000:], r.stderr[-2000:])
    return dict(name=name, rc=r.returncode, tail=r.stdout[-1500:] + r.stderr[-1500:])
@app.local_entrypoint()
def main(names: str = ','.join(VARIANTS)):
    for r in go.starmap([(n, VARIANTS[n]) for n in names.split(',')]): print(json.dumps(r)[:1800], flush=True)
