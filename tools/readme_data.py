#!/usr/bin/env python3
"""Derive docs/derived/readme_data.json (used by README.spec.yml) from the committed run files under
experiments/circuit/fullrun: seams_0-95 json, fleet_result.json, FINAL_VERDICT.json, quantfix results (hfcheck pos/neg, b8_apl2)."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FR = os.path.join(ROOT, "experiments/circuit/fullrun")
J = lambda p: json.load(open(os.path.join(FR, p)))


def main():
    seams = J("logs/cfram_q_vast/seams_0-95_20261107_004845.json")
    fleet = J("logs/cfram_q_vast/fleet_result.json")["done"]
    verdict = J("logs/cfram_q_vast/FINAL_VERDICT.json")
    pos, neg = J("quantfix/results/hfcheck_cfq_pos_130k.json"), J("quantfix/results/hfcheck_cfq_neg.json")
    prec = J("quantfix/results/b8_apl2.json")
    chunks, hw = [], {}
    for r in seams["rows"]:
        f = fleet[str(r["chunk"])]
        gpu = "A100-40GB (Modal)" if r["chunk"] in verdict["modal_chunks"] else (f["gpu"] or "?").replace("NVIDIA ", "").replace("GeForce ", "") + " (Vast)"
        chunks.append(dict(chunk=r["chunk"], minutes=round(f["seconds"] / 60, 2), seconds=f["seconds"], steps=r["steps_run"], free_tokens=r["free_tokens"],
                           mismatches=r["free_mismatches"], forced=r["forced_tokens"], injections=r["injections"], clean=r["clean"], gpu=gpu,
                           seam_exact=bool(r.get("seam", {}).get("exact", True))))
        h = hw.setdefault(gpu, dict(gpu=gpu, chunks=0, seconds=0))
        h["chunks"] += 1; h["seconds"] += f["seconds"]
    for h in hw.values():
        h["mean_min"] = round(h["seconds"] / h["chunks"] / 60, 2)
    last = seams["rows"][-1]
    out = dict(
        chunks=chunks, hardware=sorted(hw.values(), key=lambda h: -h["chunks"]),
        totals=dict(chunks=len(chunks), clean=sum(c["clean"] for c in chunks), seams_exact=sum(1 for r in seams["rows"] if r.get("seam", {}).get("exact")),
                    mismatches=sum(c["mismatches"] for c in chunks), forced=sum(c["forced"] for c in chunks), injections=sum(c["injections"] for c in chunks),
                    free_tokens=sum(c["free_tokens"] for c in chunks), job_hours=round(sum(c["seconds"] for c in chunks) / 3600, 2),
                    halt_step=last["halt_step"], final_word=last["readout"]["circuit_word"], modal_chunks=len(verdict["modal_chunks"]),
                    verdict_passed=bool(verdict["passed"])),
        stock=[dict(check="positive (real table)", steps=pos["steps_checked"], mismatches=pos["n_mismatch"], row_tokens=pos["n_row_tokens_checked"], vdot_tokens=pos["n_vdot_tokens_checked"], injections=pos["injections_applied"]),
               dict(check="negative control (table +1)", steps=neg["steps_checked"], mismatches=neg["n_mismatch"], row_tokens=neg["n_row_tokens_checked"], vdot_tokens=neg["n_vdot_tokens_checked"], injections=neg["injections_applied"])],
        stock_meta=dict(vocab=pos["full_vocab"], rows=pos["table_rows"], hidden=pos["hidden"]),
        precision=dict(argmax_circuit_fw=prec["argmax_fw"], argmax_hf=prec["argmax_hf"], cos=round(prec["cos_logits_fw_vs_hf"], 3), gap=round(prec["gap_fw_top1_top2"], 3),
                       instr=json.loads(prec["instr"])[0] if isinstance(prec["instr"], str) else prec["instr"][0]),
    )
    dst = os.path.join(ROOT, "docs/derived/readme_data.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst, out["totals"], out["hardware"])


if __name__ == "__main__":
    sys.exit(main())
