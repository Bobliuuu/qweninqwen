"""Weight-memory problem: can the DeltaNet state / FFN weights of the real Qwen3.5-0.8B or -4B hold Qwen3-0.6B (596M weights)?  Arithmetic from the HF configs
(hidden, layers, linear heads, FFN) -- configs fetched from huggingface.co/Qwen/Qwen3.5-{0.8B,4B}/config.json."""
import json
CFG = {'Qwen3.5-0.8B': dict(h=1024, L=24, lin=18, inter=3584, hv=16, dk=128, dv=128),
       'Qwen3.5-4B': dict(h=2560, L=32, lin=24, inter=9216, hv=32, dk=128, dv=128)}
INNER_W = 595_984_384; INNER_INT4_BYTES = INNER_W // 2
QWENPUTER_INT4 = 261_120

def run():
    out = {}
    for n, c in CFG.items():
        st_layer = c['hv'] * c['dk'] * c['dv']; st_all = st_layer * c['lin']
        ffn_layer = 3 * c['h'] * c['inter']; ffn_all = ffn_layer * c['L']
        r = dict(state_floats_per_gdn_layer=st_layer, gdn_layers=c['lin'], state_floats_all=st_all,
                 # a) Qwenputer layout: one bit per state float
                 int4_weights_bit_layout=st_all // 4, bit_layout_bytes=st_all // 8,
                 # b) exact integers in fp32 state: 24 bits per float (6 int4)
                 int4_weights_24bit_layout=st_all * 6, int8_weights_24bit_layout=st_all * 3, bytes_24bit=st_all * 3,
                 # c) frames: inputs_embeds of hidden floats, 24 bits each, written into state/consumed per token
                 frame_bytes=c['h'] * 3, frames_to_stream_int4_model=INNER_INT4_BYTES / (c['h'] * 3),
                 # d) ROM in checkpoint: FFN matrices (gate/up/down) hold 6 int4 per fp32 weight
                 ffn_params_per_layer=ffn_layer, ffn_params_all=ffn_all, rom_int4_capacity=ffn_all * 6,
                 ffn_layers_needed_for_inner_int4=INNER_W / 6 / ffn_layer)
        r['fraction_of_inner_int4_in_state_24bit'] = r['int4_weights_24bit_layout'] / INNER_W
        r['fraction_in_state_bit_layout'] = r['int4_weights_bit_layout'] / INNER_W
        out[n] = r
    out['qwenputer_int4_weights'] = QWENPUTER_INT4; out['inner_int4_bytes'] = INNER_INT4_BYTES
    out['qwen_inner_vs_qwenputer'] = INNER_W / QWENPUTER_INT4
    return out

if __name__ == '__main__': print(json.dumps(run(), indent=1))
