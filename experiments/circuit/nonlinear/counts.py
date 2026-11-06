"""Per-token elements of every non-matmul op of Qwen3-0.6B and the RV32 firmware cost (instructions / Qwenputer trace tokens) of each,
for plain RV32IM firmware vs RV32IM + the proposed extensions, from the measured slopes in res_kernels.json.  Also matvec cost for context."""
import json, sys
R = json.load(open("res_kernels.json"))
L, d, nq, nkv, hd, I, V = 28, 1024, 16, 8, 128, 3072, 151936
def cost(k, n, fixed_calls=0):
    r = R[k]; return dict(instr=n * r["instr_per_elem"] + fixed_calls * r.get("fixed_instr", 0), tokens=n * r["tokens_per_elem"] + fixed_calls * r.get("fixed_instr", 0) * (r["tokens_per_elem"] / r["instr_per_elem"]))
def table(T):
    rows = []
    def add(name, elems, base, ext, note=""):
        rows.append(dict(op=name, elements=elems, base_instr=base["instr"], base_tokens=base["tokens"], ext_instr=ext["instr"], ext_tokens=ext["tokens"], note=note))
    nrm_el = L * (2 * d + nq * hd + nkv * hd) + d                       # in/post/q/k norms per layer + final
    nrm_scalar = L * (2 + nq + nkv) + 1
    sq = cost("k_sumsq", nrm_el); sc = cost("k_rms_scale", nrm_el)
    rb = R["k_rsqrt_base"]; re_ = R["k_rsqrt_ext"]
    add("RMSNorm (sumsq + scale*gain)", nrm_el, dict(instr=sq["instr"] + sc["instr"] + nrm_scalar * rb["instr"], tokens=sq["tokens"] + sc["tokens"] + nrm_scalar * rb["tokens"]),
        dict(instr=sq["instr"] + sc["instr"] + nrm_scalar * re_["instr"], tokens=sq["tokens"] + sc["tokens"] + nrm_scalar * re_["tokens"]), f"{nrm_scalar} rsqrt scalars/token (RSQ replaces {rb['instr']:.0f} instr by {re_['instr']:.0f})")
    n_sw = L * I
    add("SwiGLU silu(g)*u", n_sw, cost("k_silu_base", n_sw), cost("k_silu_ext", n_sw), "SIGM instruction")
    rows_sm = L * nq
    n_sm = rows_sm * T
    add(f"softmax over T={T} ({rows_sm} rows)", n_sm, cost("k_softmax_base", n_sm, rows_sm), cost("k_softmax_ext", n_sm, rows_sm), "EXP2 instruction; log2e/sqrt(hd) folded into q weights")
    n_rp = L * (nq + nkv) * (hd // 2); n_sincos = hd // 2
    sb = R["k_sincos_base"]
    add("RoPE rotate (pairs)", n_rp, dict(instr=n_rp * R["k_rope_base"]["instr_per_elem"] + n_sincos * sb["instr"], tokens=n_rp * R["k_rope_base"]["tokens_per_elem"] + n_sincos * sb["tokens"]),
        dict(instr=n_rp * R["k_rope_ext"]["instr_per_elem"] + n_sincos * sb["instr"], tokens=n_rp * R["k_rope_ext"]["tokens_per_elem"] + n_sincos * sb["tokens"]), f"ROT16; includes {n_sincos} sincos table lookups/token (software in both)")
    n_add = L * 2 * d
    add("residual add", n_add, cost("k_add", n_add), cost("k_add", n_add), "no extension")
    n_rq = L * (d + nq * hd + d + I) + d
    add("requant int32->int8 (matvec inputs)", n_rq, cost("k_requant", n_rq), cost("k_requant", n_rq), "no extension")
    n_dq = L * (nq * hd + 2 * nkv * hd + d + 2 * I + d)
    add("dequant matvec outputs (scale to Q12/Q16)", n_dq, dict(instr=n_dq * 3.0, tokens=n_dq * 4.2), dict(instr=n_dq * 3.0, tokens=n_dq * 4.2), "ESTIMATE 3 instr/elt (mulh, shift, store); not measured")
    add("argmax over vocab (fused with lm_head epilogue)", V, dict(instr=V * 2.5, tokens=V * 2.6), dict(instr=V * 1.0, tokens=V * 1.0), "ESTIMATE: base cmp+branch+idx ~2.5; AMX 1 (measured unfused: 4.1 vs 3.25 incl. lw)")
    add("argmax over vocab (logits in RAM, unfused)", V, cost("k_argmax_base", V), cost("k_argmax_ext", V), "measured; needs 600 KB of logits storage: not realistic with 32 KB RAM")
    return rows
def matvec():
    macs_layer = d * (nq * hd + 2 * nkv * hd) + nq * hd * d + 3 * d * I
    macs = L * macs_layer + V * d
    out = {}
    for k, nm in (("k_dot_i8_base", "plain RV32IM int8"), ("k_dot_i4_base", "plain RV32IM int4"), ("k_dot_smaqa", "SMAQA int8"), ("k_dot_vdot1", "VDOT int4, 1 row"), ("k_dot_vdot8", "VDOT int4, 8 rows share activations")):
        out[nm] = dict(instr=macs * R[k]["instr_per_mac"], tokens=macs * R[k]["tokens_per_mac"], instr_per_mac=R[k]["instr_per_mac"])
    return macs, out
if __name__ == "__main__":
    full = {}
    macs, mv = matvec(); full["matvec_macs"] = macs; full["matvec"] = mv
    for T in (1, 64, 512, 2048):
        rows = table(T); full[f"T={T}"] = rows
        tb = sum(r["base_instr"] for r in rows if "unfused" not in r["op"]); te = sum(r["ext_instr"] for r in rows if "unfused" not in r["op"])
        tbt = sum(r["base_tokens"] for r in rows if "unfused" not in r["op"]); tet = sum(r["ext_tokens"] for r in rows if "unfused" not in r["op"])
        full[f"T={T}_total_nonlinear"] = dict(base_instr=tb, ext_instr=te, base_tokens=tbt, ext_tokens=tet)
    json.dump(full, open("res_counts.json", "w"), indent=1)
    print(f"matvec MACs/token = {macs:,}")
    for nm, v in mv.items(): print(f"  {nm:40s} {v['instr']:.3e} instr ({v['instr_per_mac']:.3f}/MAC)  {v['tokens']:.3e} tokens")
    for T in (1, 512):
        print(f"\nT={T}")
        for r in full[f"T={T}"]:
            print(f"  {r['op']:55s} n={r['elements']:>9,}  base {r['base_instr']:.2e} instr {r['base_tokens']:.2e} tok | ext {r['ext_instr']:.2e} instr {r['ext_tokens']:.2e} tok")
        print("  TOTAL (excl. unfused argmax)", {k: f"{v:.3e}" for k, v in full[f'T={T}_total_nonlinear'].items()})
