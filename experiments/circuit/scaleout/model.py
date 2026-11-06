"""Scale-out model: seconds/token of a sequential RV32IM instruction trace on Qwenputer (Qwen3.5 with a CPU in its weights).

s/token = trace_tokens_per_token / rate, with
  trace_tokens_per_token = instr_per_token * tok_per_instr
  rate (independent-sessions)  = trace tok/s of ONE session (batch-1 CUDA-graph step latency), batch only adds throughput
  rate (split-matvec, K sessions) : each session executes 1/K of every matvec; per stage all K exchange activations.
     t = (I/K) * tpi * step_ms(batch_K) + S * (sync_tokens * step_ms)   [SPLIT MODE needs ruling R3, not measured]
Inputs measured: rates.json (from modal_qwenputer_rate.py); upstream L4 figures cited in docs/TIERS.md.
Usage: python model.py [rates.json]   -> writes model_out.json and prints markdown tables.
"""
import json, sys
MAC_TOK = 5.96e8          # Qwen3-0.6B: 28 layers x 15.73M + 155.6M tied lm_head (repo cites 4.94e8 w/o head-ish; +-20%)
TIERS = {  # name: (instr/token, tok per instr, source)   -- TIERS.md (commit e713778), all [P]rojected
    "T1 plain C": (4.0e9, 1.4), "T2 VDOT/SMAQA": (3.7e8, 1.4), "T3 QwenPU+algo": (8.6e7, 1.4)}
STAGES = 28 * 4 + 1       # serial matvec stages per token (qkv | o | gate+up | down) + head
SYNC_TOK = 2000           # [E] trace tokens per stage to publish/gather an activation slice (assumption)
PRICE = {"L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583, "H100": 0.001097}  # modal.com/pricing $/s

def load(path):
    r = {}
    try:
        for g, d in json.load(open(path)).items():
            r[g] = {(x["executor"], x["batch"]): x for x in d["rows"] if "batch" in x}
    except FileNotFoundError:
        pass
    return r

def main():
    rates = load(sys.argv[1] if len(sys.argv) > 1 else "rates.json")
    # fall back to upstream L4 figure (docs/rv32.md): triton v2 batch1 0.177 ms, b8 0.441, b32 1.502 [M upstream]
    rates.setdefault("L4", {k: dict(ms_step=v) for k, v in {("triton_v2", 1): 0.177, ("triton_v2", 8): 0.441, ("triton_v2", 32): 1.502}.items()})
    out = {"independent": [], "split": []}
    print("## independent sessions (latency of ONE token = batch-1 step; extra sessions add throughput only)")
    print("| tier | GPU | step ms (b=1) | s/token (1 session) | agg tok/s @b=32 | s/token amortised, 32 prompts in flight | $/token (b=1) | $/token (b=32) |\n|---|---|---|---|---|---|---|---|")
    for tn, (ipt, tpi) in TIERS.items():
        T = ipt * tpi
        for g, d in rates.items():
            b1 = d.get(("triton_v2", 1)); b32 = d.get(("triton_v2", 32))
            if not b1: continue
            s1 = T * b1["ms_step"] / 1e3
            s32 = T * b32["ms_step"] / 1e3 / 32 if b32 else float("nan")
            p = PRICE.get(g, float("nan"))
            row = dict(tier=tn, gpu=g, tok_per_token=T, step_ms_b1=b1["ms_step"], s_per_token=s1, s_per_token_amortised_b32=s32,
                       usd_per_token_b1=s1 * p, usd_per_token_b32=s32 * p)
            out["independent"].append(row)
            print(f"| {tn} | {g} | {b1['ms_step']:.3f} | {s1:.3g} ({s1/3600:.3g} h) | {32/(b32['ms_step']/1e3) if b32 else float('nan'):.0f} | {s32:.3g} ({s32/3600:.3g} h) | ${s1*p:.3g} | ${s32*p:.3g} |")
    print("\n## split-matvec across K sessions (PROJECTED, needs cross-session transport ruling R3), best GPU in table")
    g = min((x for x in rates if ("triton_v2", 1) in rates[x]), key=lambda x: rates[x][("triton_v2", 1)]["ms_step"])
    print(f"GPU={g}; sync cost {SYNC_TOK} trace tokens/stage x {STAGES} stages [E]; K sessions on K/ngpu per GPU (batched step time from table; interpolated linearly in ms vs batch)")
    print("| tier | K | s/token | $/token |\n|---|---|---|---|")
    pts = sorted((b, rates[g][("triton_v2", b)]["ms_step"]) for (e, b) in rates[g] if e == "triton_v2")
    def step_ms(b):  # per-GPU step time with b sessions resident, linear interp/extrap
        for (b0, m0), (b1, m1) in zip(pts, pts[1:]):
            if b <= b1: return m0 + (m1 - m0) * (b - b0) / (b1 - b0)
        (b0, m0), (b1, m1) = pts[-2], pts[-1]; return m1 + (m1 - m0) * (b - b1) / (b1 - b0)
    for tn, (ipt, tpi) in TIERS.items():
        T = ipt * tpi
        for K in (1, 8, 32, 128, 512, 2048):
            per_gpu = min(K, 32); ngpu = -(-K // 32)
            ms = step_ms(per_gpu) + (0.05 * (ngpu > 1))      # +0.05 ms/step cross-GPU hop [E]
            s = (T / K + STAGES * SYNC_TOK * (K > 1)) * ms / 1e3
            usd = s * ngpu * PRICE[g]
            out["split"].append(dict(tier=tn, K=K, gpus=ngpu, s_per_token=s, usd_per_token=usd, gpu=g))
            print(f"| {tn} | {K} | {s:.3g} ({s/3600:.3g} h) | ${usd:.3g} |")
    json.dump(out, open("model_out.json", "w"), indent=1)

if __name__ == "__main__":
    main()
