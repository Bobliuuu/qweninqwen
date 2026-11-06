"""Memory footprint of Qwen3-0.6B as an RV32IM firmware vs Qwenputer DeltaNet-RAM capacity (analytic; prints JSON)."""
import json
q = dict(D=1024, L=28, NH=16, NKV=8, HD=128, HID=3072, V=151936)
proj = q['L'] * (q['D'] * q['NH'] * q['HD'] + 2 * q['D'] * q['NKV'] * q['HD'] + q['NH'] * q['HD'] * q['D'] + 3 * q['D'] * q['HID'])
emb = q['V'] * q['D']; norms = q['L'] * (2 * q['D'] + 2 * q['HD']) + q['D']
kv_per_pos = q['L'] * 2 * q['NKV'] * q['HD']
act = q['D'] * 4 + q['NH'] * q['HD'] * 2 + q['HID'] * 2     # x, xb, q, ao, hg, hu floats (ctx-independent scratch, elements)
need = dict(projection_weights=proj, embedding_lm_head_tied=emb, total_weights=proj + emb, norm_floats=norms,
            kv_values_per_position=kv_per_pos, scratch_floats=act)
have_wrom = 261_120
def gdn(layers, vheads, dk=128, dv=128): return layers * vheads * dk * dv
cap = {}
for name, L, vh in (('Qwen3.5-0.8B', 18, 16), ('Qwen3.5-4B', 24, 32)):
    cells = gdn(L, vh)
    cap[name] = dict(gdn_layers=L, state_cells=cells,
        int4_weights_at_1bit_per_cell=cells // 4, int4_weights_at_8bit_per_cell=cells * 8 // 4, int4_weights_at_24bit_per_cell=cells * 24 // 4,
        shortfall_vs_total_weights_best_case=(proj + emb) / (cells * 24 // 4),
        shortfall_vs_total_weights_8bit=(proj + emb) / (cells * 8 // 4),
        shortfall_vs_total_weights_1bit=(proj + emb) / (cells // 4))
out = dict(need=need, qwenputer_wrom_int4_weights=have_wrom, ratio_need_over_have=(proj + emb) / have_wrom,
           int4_bytes=(proj + emb) // 2, kv_int8_bytes_per_position=kv_per_pos, qwenputer_ram_bytes=32 * 1024, capacity=cap,
           note="state_cells = value_heads*128*128 per GDN layer; best case stores 24 exact bits per fp32 state cell (delta-rule overwrite); Qwenputer today uses 1 bit per cell, 12.5% of a chip layer's rows")
print(json.dumps(out, indent=1))
json.dump(out, open('res_footprint.json', 'w'), indent=1)
