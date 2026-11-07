"""Decode the WF/CF row table FROM THE MERGED CHECKPOINT's embed_tokens.weight (the model's own parameters), instead of loading rowtab_cf.npy.
Layout (hfwf/merge_wf.py): row r of the ROW family is checkpoint embedding row  chip_token_offset + row_family_base + r  ; its 3,840 4-bit cells are bf16 integers 0..15 in the columns chipformer.json['cell_columns'].
Returns uint8 [row_rows, 3840], bit-identical to the table the checkpoint was built from (asserted integer-valued, 0..15). Reads only the shard that holds embed_tokens.weight, in 8192-row blocks."""
import json, numpy as np

def load_rowtab(ckdir):
    import torch
    from safetensors import safe_open
    meta = json.load(open(f'{ckdir}/chipformer.json')); wm = json.load(open(f'{ckdir}/model.safetensors.index.json'))['weight_map']
    key = next(k for k in wm if k.endswith('embed_tokens.weight')); cols = torch.tensor(meta['cell_columns'])
    r0 = meta['chip_token_offset'] + meta['row_family_base']; n = meta['row_rows']; out = np.empty((n, len(cols)), np.uint8)
    with safe_open(f'{ckdir}/{wm[key]}', 'pt') as fh:
        sl = fh.get_slice(key)
        for a in range(0, n, 8192):
            b = min(a + 8192, n); v = sl[r0 + a:r0 + b][:, cols].float(); u = v.to(torch.uint8)
            assert bool((u.float() == v).all()) and bool((v >= 0).all()) and bool((v < 16).all()), f'non-4-bit cell in checkpoint rows {a}..{b}'
            out[a:b] = u.numpy()
    return out
