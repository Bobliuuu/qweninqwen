"""Shard the circuit MAC workload of ONE real Qwen3-0.6B layer (layer 0) across N Modal containers.

Kernel = the circuit baseline (reference/nested-qwen-lab/parallel_complete.py Engine): a 32x31 INT8xINT4 tile
executed by a Qwen GDN query/state read + compiled Qwen reduction layers in BF16 (992 MACs per call, bit exact).
Workload = a deterministic stratified sample of T tiles of the seven real layer-0 matrices (q,k,v,o,gate,up,down),
INT4 per-row quantised exactly like parallel_complete.py case 5.  Strong scaling: same T tiles split over N workers.

Launch ONLY with:  modal run --detach experiments/circuit/scaleout/modal_scaleout.py --tiles 256 --ns 1,2,4,8,16
The orchestration runs inside Modal (survives laptop loss); results go to Volume `qiq-scaleout` and app logs (JSON lines).
"""
import hashlib, json, os, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reference", "nested-qwen-lab"))
app = modal.App("qiq-scaleout")
vol = modal.Volume.from_name("qiq-scaleout", create_if_missing=True)
LAYER_NPZ = "/opt/layer0.npz"


def _prep():
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    p = hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors")
    names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
             "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
    out = {}
    with safe_open(p, framework="pt") as sf:
        for k, n in names.items():
            out[k] = sf.get_tensor(f"model.layers.0.{n}.weight").float().numpy()
    np.savez(LAYER_NPZ, **out)


image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "safetensors", "huggingface_hub", "transformers==5.20.2")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
         .run_function(_prep)
         .add_local_dir(KERNEL_DIR, "/app", ignore=["*.log", "*.json", "__pycache__", "cpu_reference", "gpu_reference"]))

TR, TC = 32, 31   # tile: 32 inputs (rows of w) x 31 outputs


def make_tiles(T, seed=7):
    """Deterministic stratified sample of T tiles from the seven real layer-0 matrices -> list of (name, o0, i0)."""
    import numpy as np
    d = np.load(LAYER_NPZ)
    pool = []
    for k in ["q", "k", "v", "o", "gate", "up", "down"]:
        out, inn = d[k].shape
        for o0 in range(0, out - TC + 1, TC):
            for i0 in range(0, inn - TR + 1, TR):
                pool.append((k, o0, i0))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=T, replace=False)
    return [pool[i] for i in sorted(idx)], len(pool), d


def tile_case(d, tile, idx):
    """INT4 weights w (32,31) and INT8 activations x (32,) for a tile (per-output-row scale from the full real row)."""
    import numpy as np
    k, o0, i0 = tile
    W = d[k]
    scales = np.max(np.abs(W[o0:o0 + TC, :]), axis=1) / 7
    w = np.clip(np.rint(W[o0:o0 + TC, i0:i0 + TR] / scales[:, None]), -7, 7).astype(np.int64).T
    x = np.random.default_rng(1000 + idx).integers(-128, 128, TR)
    return w, x


@app.function(image=image, cpu=2.0, memory=4096, timeout=3600, volumes={"/vol": vol})
def worker(wid, tiles_idx, T, n_workers, run_id, barrier):
    import sys, platform
    sys.path.insert(0, "/app")
    import numpy as np, torch
    torch.set_num_threads(2)
    tiles, _, d = make_tiles(T)
    import parallel_complete as pc
    t0 = time.perf_counter(); eng = pc.Engine(); compile_s = time.perf_counter() - t0
    cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    # barrier so all N workers start the timed section together (the Dict is only a rendezvous counter)
    barrier[f"{run_id}:{wid}"] = 1
    t_wait = time.time()
    while sum(1 for k in range(n_workers) if f"{run_id}:{k}" in barrier) < n_workers and time.time() - t_wait < 600:
        time.sleep(0.2)
    start = time.time(); c0 = time.perf_counter()
    h = {}; ok = True
    for i in tiles_idx:
        w, x = tile_case(d, tiles[i], i)
        eng.load(w); got = pc.decode(eng(eng.encode(x)))
        ref = x @ w
        ok &= bool(np.array_equal(got, ref))
        h[i] = hashlib.sha256(np.asarray(got, dtype=np.int64).tobytes()).hexdigest()
    compute_s = time.perf_counter() - c0
    return dict(wid=wid, n_tiles=len(tiles_idx), macs=len(tiles_idx) * TR * TC, compile_s=compile_s, compute_s=compute_s,
                t_start=start, t_end=time.time(), kernel_exact=ok, cpu=cpu, hashes=h)


@app.function(image=image, timeout=6 * 3600, volumes={"/vol": vol})
def orchestrate(T=256, ns=(1, 2, 4, 8, 16)):
    import numpy as np
    tiles, pool_size, d = make_tiles(T)
    ref = {i: hashlib.sha256(np.asarray(tile_case(d, tiles[i], i)[1] @ tile_case(d, tiles[i], i)[0], dtype=np.int64).tobytes()).hexdigest()
           for i in range(T)}
    results = []
    for N in ns:
        run_id = f"{int(time.time())}-N{N}"
        barrier = modal.Dict.from_name("qiq-scaleout-barrier", create_if_missing=True)
        shards = [list(range(w, T, N)) for w in range(N)]   # round-robin so every shard mixes matrices
        tw = time.time()
        outs = list(worker.starmap([(w, shards[w], T, N, run_id, barrier) for w in range(N)]))
        wall = time.time() - tw
        macs = sum(o["macs"] for o in outs)
        window = max(o["t_end"] for o in outs) - min(o["t_start"] for o in outs)
        hashes = {}
        for o in outs: hashes.update(o["hashes"])
        verified = all(o["kernel_exact"] for o in outs) and all(hashes[i] == ref[i] for i in range(T)) and len(hashes) == T
        row = dict(N=N, tiles=T, macs=macs, pool_tiles_in_layer=pool_size,
                   makespan_compute_s=window, wall_incl_coldstart_s=wall,
                   agg_macs_per_s=macs / window,
                   sum_of_worker_rates=sum(o["macs"] / o["compute_s"] for o in outs),
                   per_worker_macs_per_s=[o["macs"] / o["compute_s"] for o in outs],
                   mean_compile_s=sum(o["compile_s"] for o in outs) / N,
                   verified_vs_reference=verified, cpus=sorted({o["cpu"] for o in outs}),
                   container_seconds=sum(o["compute_s"] + o["compile_s"] for o in outs))
        results.append(row)
        print("SCALEOUT_ROW " + json.dumps(row), flush=True)
    base = results[0]["agg_macs_per_s"]
    for r in results:
        r["speedup"] = r["agg_macs_per_s"] / base; r["efficiency"] = r["speedup"] / r["N"]
    final = dict(workload="Qwen3-0.6B layer 0 real weights, 7 matrices, INT4/INT8 32x31 tiles", kernel="circuit baseline (nested-qwen-lab parallel_complete Engine, CPU BF16)",
                 tiles=T, rows=results, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open("/vol/scaleout_results.json", "w") as f: json.dump(final, f, indent=1)
    vol.commit()
    print("SCALEOUT_FINAL " + json.dumps(final), flush=True)
    return final


@app.local_entrypoint()
def main(tiles: int = 256, ns: str = "1,2,4,8,16"):
    call = orchestrate.spawn(tiles, tuple(int(x) for x in ns.split(",")))
    print("spawned orchestrate call", call.object_id, "(results: volume qiq-scaleout /scaleout_results.json, JSON in app logs)")
