"""Qwenputer fused-executor benchmark/optimiser (A10G). Uses the chipformer package of divagr18/qwenputer branch `npu`
(PYTHONPATH=<qwenputer>/experimental/riscv).  Measures trace tokens/s of FusedExec FreeRun3 for batch B and K steps per CUDA graph,
checks every row of every run against the reference machine's trace prefix (exact token equality).
usage: python bench_exec.py ELF --steps N --batch 1,8,64 --k 1,4,16 [--profile] [--out res.json]"""
import argparse, json, os, time
import numpy as np, torch
from chipformer.rv import core, elf, fastexec, fusedexec, ref
from chipformer.rv import spec as S

class FreeRunK(fusedexec.FreeRun3):
    """FreeRun3 with K executor steps captured into ONE CUDA graph (token of each step copied into an on-GPU buffer inside the graph)."""
    def capture_k(self, K):
        if not hasattr(self, "use4"): self.tune()
        heads = [st for st in self.ex.steps if st["kind"] == "head"]
        bufs = self.X + self.Y + [st["S"] for st in heads] + [st["hist"] for st in heads]
        saved = [b.clone() for b in bufs]
        self.K = K; self.outbuf = torch.zeros(self.B, K, dtype=torch.long, device=self.dev)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for j in range(K):
                self._step(); self.outbuf[:, j].copy_(self.tok)
        for b, s0 in zip(bufs, saved): b.copy_(s0)
        self.gk = g
    def run_k(self, n):
        outs = []
        for _ in range(-(-n // self.K)):
            self.gk.replay(); outs.append(self.outbuf.clone())
        torch.cuda.synchronize()
        return torch.cat(outs, 1)[:, :n].cpu().numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("elf"); ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", default="1"); ap.add_argument("--k", default="1"); ap.add_argument("--out", default=None)
    ap.add_argument("--profile", action="store_true"); ap.add_argument("--want", default="want.npy")
    a = ap.parse_args()
    prog = elf.program(a.elf)
    if os.path.exists(a.want) and len(np.load(a.want)) > a.steps: want = np.load(a.want)
    else:
        m = ref.Machine.from_program(prog); t = time.time(); tr = m.run(int(a.steps * 1.1)); want = np.asarray(tr); np.save(a.want, want)
        print(f"reference prefix {len(want)} tokens in {time.time()-t:.0f}s", flush=True)
    res = dict(gpu=torch.cuda.get_device_name(), steps=a.steps, runs=[])
    c = core.build()
    for B in [int(x) for x in a.batch.split(",")]:
        for K in [int(x) for x in a.k.split(",")]:
            core.snapshot(c, ref.Machine.from_program(prog))
            fr = FreeRunK(c, B, "cuda")
            for r in range(B): fr.load(r, S.tok("GO"))
            if K == 1: fr.capture()
            else: fr.capture_k(K)
            torch.cuda.synchronize(); t = time.time()
            if K == 1:
                out = torch.zeros(B, a.steps, dtype=torch.long, device="cuda")
                for s in range(a.steps): fr.graph.replay(); out[:, s].copy_(fr.tok)
                torch.cuda.synchronize(); got = out.cpu().numpy()
            else: got = fr.run_k(a.steps)
            dt = time.time() - t
            ok = bool((got == want[1:a.steps + 1][None]).all())
            r = dict(B=B, K=K, seconds=dt, ms_per_step=dt / a.steps * 1e3, agg_tok_per_s=B * a.steps / dt, per_row_tok_per_s=a.steps / dt, exact_all_rows=ok)
            print(json.dumps(r), flush=True); res["runs"].append(r)
            if a.profile and B == 1 and K == 1:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
                    for _ in range(50): fr.graph.replay()
                    torch.cuda.synchronize()
                print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60), flush=True)
            del fr; torch.cuda.empty_cache()
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
main()
