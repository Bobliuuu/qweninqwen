"""Exact byte capacities of Qwen3.5-0.8B / 4B (and Qwen3-0.6B needs) from the HF configs.
Pure arithmetic, no GPU.  python geometry.py"""
import json

QW35 = {
 "0.8B": dict(H=1024, I=3584, L=24, n_full=6, n_dn=18, kvh=2, hd=256, kh=16, vh=16, dk=128, dv=128, vocab=248320),
 "4B":   dict(H=2560, I=9216, L=32, n_full=8, n_dn=24, kvh=4, hd=256, kh=16, vh=32, dk=128, dv=128, vocab=248320),
}
Q3 = dict(H=1024, I=3072, L=28, nq=16, nkv=8, hd=128, vocab=151936)

def q3_need():
    q = Q3
    attn = q["H"] * q["nq"] * q["hd"] * 2 + q["H"] * q["nkv"] * q["hd"] * 2   # q,o + k,v
    mlp = 3 * q["H"] * q["I"]
    layer = attn + mlp
    body = layer * q["L"]
    head = q["vocab"] * q["H"]          # tied embedding / lm head
    kv_per_tok = q["L"] * q["nkv"] * q["hd"] * 2   # values (K and V)
    return dict(per_layer=layer, body=body, head=head, total=body + head,
                int4_bytes=(body + head) / 2, int8_bytes=body + head, kv_values_per_tok=kv_per_tok)

def outer(g):
    dn_entries = g["n_dn"] * g["vh"] * g["dk"] * g["dv"]            # fp32 state matrix entries
    dn_keyrows = g["n_dn"] * g["kh"] * g["dk"]                      # one-hot addressable (key head,row) pairs
    kv_vals_tok = g["n_full"] * g["kvh"] * g["hd"] * 2              # bf16 K and V values per context position
    mlp_rom = g["L"] * g["I"] * g["H"]                              # down_proj entries (all layers)
    emb = g["vocab"] * g["H"]
    return dict(dn_entries=dn_entries, dn_keyrows=dn_keyrows, kv_vals_per_pos=kv_vals_tok,
                kv_bytes_per_pos_bf16=kv_vals_tok * 2, mlp_rom_entries_per_layer=g["I"] * g["H"],
                mlp_rom_entries_all=mlp_rom, emb_entries=emb, dn_state_bytes_fp32=dn_entries * 4)

if __name__ == "__main__":
    need = q3_need(); print("Qwen3-0.6B", json.dumps(need, indent=1))
    for k, g in QW35.items():
        o = outer(g); print(k, json.dumps(o, indent=1))
        n = need["total"]
        for bits in (1, 4, 8):   # bits stored per DeltaNet state entry
            cap_bytes = o["dn_entries"] * bits / 8
            print(f"  {k} DN state @{bits} bit/entry: {cap_bytes/1e6:.2f} MB; short of int4 weights by {need['int4_bytes']/cap_bytes:.0f}x")
        print(f"  {k} inner KV (int8, {need['kv_values_per_tok']} B/tok) tokens fitting in DN@8bit: {o['dn_entries']/need['kv_values_per_tok']:.0f}")
        print(f"  {k} outer KV positions per inner token (8 bits/value, bf16 store holds 1 value/slot): {need['kv_values_per_tok']/o['kv_vals_per_pos']:.2f}")
        print(f"  {k} ROM layers needed in down_proj for int4 @2 nibbles/entry: {need['total']/2/o['mlp_rom_entries_per_layer']:.1f} ; @1 nibble/entry: {need['total']/o['mlp_rom_entries_per_layer']:.1f}")
        print(f"  {k} embedding entries {o['emb_entries']/1e6:.0f}M vs nibbles needed {need['total']/1e6:.0f}M -> {need['total']/o['emb_entries']:.2f} nibbles/entry")
