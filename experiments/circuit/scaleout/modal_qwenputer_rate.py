"""Measure the real Qwenputer (RV32IM circuit compiled into Qwen3.5 weights) trace-token rate per GPU type on Modal.

Runs upstream `chipformer.rv.timing --executor-only` (Triton executor v2 + sparse torch executor, CUDA graphs,
crc32 trace) at several batch sizes (= independent Qwenputer instances stepped in lockstep on one GPU).
Launch ONLY with:  modal run --detach experiments/circuit/scaleout/modal_qwenputer_rate.py --gpus L4,A100,H100
Results: Volume `qiq-scaleout` /rate_<GPU>.json and `QPRATE_JSON` lines in the app logs.
"""
import json, os, re, subprocess, sys, time
import modal

QP_COMMIT = "bec1a0c"
app = modal.App("qiq-scaleout-rate")
vol = modal.Volume.from_name("qiq-scaleout", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("numpy", "safetensors", "transformers>=5.19.0", "torch", "triton")
         .run_commands("git clone https://github.com/divagr18/qwenputer /opt/qwenputer && cd /opt/qwenputer && git checkout " + QP_COMMIT))


def _run(gpu, batches, steps):
    env = dict(os.environ, PYTHONPATH="/opt/qwenputer/experimental/riscv")
    import torch
    info = dict(gpu_request=gpu, device=torch.cuda.get_device_name(0), torch=torch.__version__)
    rows = []
    for sparse in ("2", "1"):
        p = subprocess.run([sys.executable, "-u", "-m", "chipformer.rv.timing", "--executor-only", "--sparse", sparse,
                            "--batches", batches, "--steps", str(steps)], cwd="/opt/qwenputer/experimental/riscv",
                           env=env, capture_output=True, text=True)
        print(p.stdout[-3000:], p.stderr[-1500:], flush=True)
        for m in re.finditer(r"executor(?: v2 \(triton\))? batch\s+(\d+).*?([\d.]+) ms/step\s+(\d+) tok/s\s+(\d+) instr/s", p.stdout):
            rows.append(dict(executor="triton_v2" if sparse == "2" else "sparse_torch", batch=int(m[1]), ms_step=float(m[2]),
                             tok_s=int(m[3]), instr_s=int(m[4])))
        if not rows or p.returncode:
            rows.append(dict(executor=sparse, error=p.stderr[-500:]))
    info["rows"] = rows
    print("QPRATE_JSON " + json.dumps(info), flush=True)
    with open(f"/vol/rate_{gpu}.json", "w") as f: json.dump(info, f, indent=1)
    vol.commit()
    return info


fns = {}
for g in ["L4", "A10G", "L40S", "A100-40GB", "H100"]:
    fns[g] = app.function(image=image, gpu=g, timeout=3000, volumes={"/vol": vol}, name="rate_" + g.replace("-", "_"))(
        lambda batches="1,8,32,128", steps=1000, _g=g: _run(_g, batches, steps))


@app.local_entrypoint()
def main(gpus: str = "L4,A100-40GB,H100", batches: str = "1,8,32,128", steps: int = 1000):
    for g in gpus.split(","):
        c = fns[g].spawn(batches, steps)
        print("spawned", g, c.object_id)
