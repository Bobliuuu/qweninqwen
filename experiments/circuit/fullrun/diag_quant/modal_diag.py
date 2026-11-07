"""Run diag_quant.py on a Modal CPU container (no GPU): modal run --detach modal_diag.py. Result JSON printed to the app logs AND saved on volume qfr-out /diag_quant/diag_quant.json."""
import modal, subprocess, os
app = modal.App('qwen-in-qwen-diagquant')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'transformers', 'numpy', 'accelerate').add_local_dir(os.path.dirname(os.path.abspath(__file__)), '/root', ignore=['*.json', '__pycache__'])
out = modal.Volume.from_name('qfr-out', create_if_missing=True)
@app.function(image=img, cpu=8, memory=24576, timeout=3600, volumes={'/out': out})
def go(SCRIPT='diag_quant.py'):
    os.makedirs('/out/diag_quant', exist_ok=True)
    r = subprocess.run(['python', '-u', '/root/' + SCRIPT, '/out/diag_quant/' + SCRIPT.replace('.py', '.json')], capture_output=True, text=True)
    out.commit(); print(r.stdout[-6000:]); print(r.stderr[-3000:])
    return dict(rc=r.returncode, tail=r.stdout[-4000:] + r.stderr[-2000:])
@app.local_entrypoint()
def main(script: str = 'diag_quant.py'): print(go.remote(script), flush=True)
