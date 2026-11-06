"""Instructions / trace tokens / seconds per inner token of Qwen3-0.6B firmware on the Qwenputer-style RV32 CPU inside Qwen3.5.
Matvec instr/MAC values are MEASURED on the ISS (results/iss.json); attention/nonlinear per-element costs are ESTIMATES (flagged) and are <10% of the total.
Trace tokens per instruction 1.2-1.55 (Qwenputer docs/rv32.md); executor speed 5,500 trace tokens/s (L4, fused executor, NPU README)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D, I, NL, NQ, NKV, HD, V = 1024, 3072, 28, 16, 8, 128, 151936
PER_LAYER = dict(q=D * NQ * HD, k=D * NKV * HD, v=D * NKV * HD, o=NQ * HD * D, gate=D * I, up=D * I, down=I * D)
LAYER_MACS = sum(PER_LAYER.values()); HEAD_MACS = D * V; MACS = NL * LAYER_MACS + HEAD_MACS
assert MACS == 595_984_384
TOK_PER_INSTR, TOKS = 1.4, 5500.0

def nonlinear_instr(ctx=128):
    """ESTIMATED instructions (RV32IM, table-based) for everything that is not a weight matvec; per-element costs stated."""
    e = dict(rmsnorm=(2 * NL + 1) * D * 9 + (2 * NL * (NQ + NKV) * HD) * 9,    # input/post/final norms + q/k norm: ~9 instr/elt (square-acc, isqrt via recip table, scale)
             rope=NL * (NQ + NKV) * HD * 8,                                    # ~8 instr/elt
             silu_mul=NL * I * 8,                                              # table lookup silu(gate)*up, requant
             residual=NL * 2 * D * 4,
             attention=NL * NQ * ctx * (2 * 4 + 12) + NL * NQ * HD * ctx * 0,  # score loop overhead + exp table + normalisation (MACs counted separately)
             argmax=V * 4)                                                     # ~4 instr/logit compare-update loop
    return e

def attn_macs(ctx=128): return NL * NQ * HD * ctx * 2

def table(iss):
    nl = sum(nonlinear_instr().values())
    rows = []
    for name, ipm in iss.items():
        instr = (MACS + attn_macs()) * ipm + nl
        rows.append(dict(kernel=name, instr_per_mac=ipm, instr_per_token=instr, trace_tokens=instr * TOK_PER_INSTR, seconds=instr * TOK_PER_INSTR / TOKS))
    return rows

if __name__ == '__main__':
    p = os.path.join(HERE, 'results', 'iss.json')
    iss = {'(placeholder) 1.0': 1.0}
    if os.path.exists(p):
        r = json.load(open(p)); iss = {f"v{k['variant']} rb{k['rb']} u{k['unr']}": k['instr_per_mac'] for k in r['kernels']}
    print(json.dumps(dict(macs=MACS, nonlinear=nonlinear_instr(), rows=table(iss)), indent=1))
