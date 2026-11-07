"""Merge Qwen3.5-4B + the WF (weight-fetch) circuit INCLUDING the 155,205-row weight table into ONE stock-architecture checkpoint (text-only Qwen3_5ForCausalLM).
Same recipe as chipformer.rv.merge.build_merged (chip layers first, Qwen's 32 layers zero-widened, tied head reads only the output code), with:
  * circuit = core.build(Target, T, wrow=8) (12 chip layers; bring-up core, see common.py)
  * the ROW family has exactly NROWS=155,205 tokens; the embedding row of ROW(r) = its code/payload features (as before) PLUS table row r:
    3840 4-bit entries (wf_table.build_table of wrom.bin) written as bf16 integers into the 3840 'row.c{j}' feature columns. The table is a PARAMETER (embed_tokens.weight).
  * hidden size picked by pick_hidden for the real slot count (the 3840 cells + the ~1.4 K other dims cannot share a 4096 residual: see the printed slots).
usage: python merge_wf.py QWEN35_4B_DIR WROM_BIN_or_TABLE_NPY OUT_DIR   (CF+RAM: env WF_NROWS=156396 REALMAP=1, table /data/rowtab_cf.npy)"""
import sys, json, re, time
from pathlib import Path
import numpy as np, torch
from common import *
from safetensors import safe_open
from safetensors.torch import save_file
from chipformer.q35.merge import IM_START, IN_COLS, OUT_ROWS, TAU_GAIN, WIDE_NORMS, pick_hidden
from chipformer.q35.emit import Layout, allocate, check_bf16, emit_layers
from chipformer.q35.ir import Target
from chipformer.q35.tiny import code_gain
from chipformer.rv import bridge as BR
from chipformer.rv import merge as M
from chipformer.rv.tiny import embedding_matrix
from wf_table import build_table
SRC_TEXT, DST = M.SRC_TEXT, M.DST

def main(src, wrom_path, dst, hidden=None, mm=None):
    t00 = time.time(); src, dst = Path(src), Path(dst)
    cfg = json.loads((src / 'config.json').read_text()); tc0 = cfg['text_config']
    d0, v0, inter = tc0['hidden_size'], tc0['vocab_size'], tc0['intermediate_size']
    T = BR.Text(src / 'tokenizer.json')
    c = core.build(Target(tc0['linear_num_key_heads'], tc0['linear_num_value_heads'], inter), T, wrow=8)
    keep = set(c.keep); alloc = allocate(c, None, keep, inter)
    if hidden is None: hidden, mm = pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=mm, slots=alloc.slots)
    assert d0 + mm + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden); lay.scale
    n_chip = c.n_layers(); n0 = tc0['num_hidden_layers']
    print(f'circuit {c.stats()}, token features {len(c.token_fns)}, slots {alloc.n_slots}, hidden {hidden} (m={mm}), {n_chip} chip layers, S.VOCAB {S.VOCAB}', flush=True)
    tc = {k: v for k, v in tc0.items() if k != 'full_attention_interval'}
    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + n0, mtp_num_hidden_layers=0, layer_types=['linear_attention'] * n_chip + tc0['layer_types'],
              vocab_size=-(-(v0 + S.VOCAB) // 64) * 64, tie_word_embeddings=True, architectures=['Qwen3_5ForCausalLM'], model_type='qwen3_5_text')
    vocab = tc['vocab_size']
    from chipformer.q35.mute import fit_anti_sink, mute
    sinks = fit_anti_sink(src, 'cuda'); torch.cuda.empty_cache()
    scale = (d0 / hidden) ** 0.5; sd = {}; base_emb = None
    for f in sorted(src.glob('*.safetensors')):
        with safe_open(f, 'pt') as fh:
            for k in fh.keys():
                if not k.startswith(SRC_TEXT): continue
                w = fh.get_tensor(k)
                if k == SRC_TEXT + 'embed_tokens.weight': base_emb = w; continue
                if IN_COLS.search(k): w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
                elif OUT_ROWS.search(k): w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
                elif WIDE_NORMS.search(k):
                    g = (1.0 + w.float()) * scale - 1.0; w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
                k2 = DST + k[len(SRC_TEXT):]
                sd[re.sub(r'^(model\.layers\.)(\d+)\.', lambda mt: f'{mt[1]}{int(mt[2]) + n_chip}.', k2)] = w
    mute(sd, tc, n0, lay.dim(next(iter(c.mu.terms))), hidden ** 0.5, prefix=DST, sinks=sinks, first=n_chip)
    layers = emit_layers(c, lay, tc, 0, DST, alloc, keep)
    for k in list(layers): sd[k] = layers.pop(k).to(torch.bfloat16)
    print(f'chip layers emitted {time.time()-t00:.0f}s', flush=True)
    emb = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    emb[:v0] = M.text_rows(c, lay, T, base_emb, v0); del base_emb
    chip = embedding_matrix(c, lay, S.VOCAB)       # ROW table is None here: cells are 0; every other feature (payload, code) is exact
    emb[v0:v0 + S.VOCAB] = chip[:S.VOCAB].to(torch.bfloat16); del chip
    print(f'embedding rows built {time.time()-t00:.0f}s', flush=True)
    # ---- the weight table: rows of the ROW family ----
    tab = np.load(wrom_path) if str(wrom_path).endswith('.npy') else build_table(open(wrom_path, 'rb').read())   # .npy = prebuilt table (CF: rowtab_cf.npy)
    assert tab.shape == (NROWS, S.ROW_CELLS), (tab.shape, NROWS)
    cell_dims = [next(iter(e.terms)) for e in c.rowcell]; cols = torch.tensor([lay.dim(d) for d in cell_dims])
    assert len(set(cols.tolist())) == S.ROW_CELLS and cols.min() >= d0
    r0 = v0 + ROW_BASE; assert (emb[r0:r0 + NROWS][:, cols] == 0).all()
    for a in range(0, NROWS, 8192):
        emb[r0 + a:r0 + min(a + 8192, NROWS)][:, cols] = torch.from_numpy(tab[a:a + 8192].astype(np.float32)).to(torch.bfloat16)
    check_bf16('embedding', emb[:, d0:].float()[:300000]); sd[DST + 'embed_tokens.weight'] = emb
    code = [next(iter(e.terms)) for e in c.code]; tau = next(iter(c.tau.terms))
    norm = sd[DST + 'norm.weight'].float(); norm[d0:] = -1.0
    for d in code: norm[lay.dim(d)] = code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN; check_bf16('final norm', norm[d0:]); sd[DST + 'norm.weight'] = norm.to(torch.bfloat16)
    dst.mkdir(parents=True, exist_ok=True); shard, size, shards = {}, 0, []
    for k in sorted(sd):
        shard[k] = sd[k].contiguous(); size += sd[k].numel() * 2
        if size > 2 ** 31: shards.append(shard); shard, size = {}, 0
    shards.append(shard)
    index = {'metadata': {'total_size': sum(t.numel() * 2 for t in sd.values())}, 'weight_map': {}}
    for i, sh in enumerate(shards):
        name = f'model-{i + 1:05d}-of-{len(shards):05d}.safetensors'; save_file(sh, str(dst / name), metadata={'format': 'pt'}); index['weight_map'].update({k: name for k in sh})
    (dst / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2)); (dst / 'config.json').write_text(json.dumps(tc, indent=2))
    M.extend_tokenizer(src, dst, v0)
    gen = {'eos_token_id': [248046, tc0.get('eos_token_id', 248044)], 'pad_token_id': tc0.get('eos_token_id', 248044), 'bad_words_ids': [[IM_START]], 'do_sample': False}
    (dst / 'generation_config.json').write_text(json.dumps(gen, indent=2))
    meta = {'layout': 'rv1+wf', 'chip_token_offset': v0, 'chip_vocab': S.VOCAB, 'chip_layers': n_chip, 'qwen_layers_from': n_chip, 'hidden': hidden, 'chip_dims_offset': d0,
            'ballast_dims': mm, 'circuit': c.stats(), 'slots': alloc.n_slots, 'muted': True, 'code_dims': [lay.dim(d) for d in code], 'tau_dim': lay.dim(tau),
            'mu_dim': lay.dim(next(iter(c.mu.terms))), 'nfam': core.NFAM, 'npay': core.NPAY, 'tool_call': [T.TCO, T.TCC], 'printable': T.printable,
            'wrow': 8, 'row_cells': S.ROW_CELLS, 'row_ent_bits': S.ROW_ENT_BITS, 'row_family_base': ROW_BASE, 'row_rows': NROWS, 'cell_columns': cols.tolist(),
            'table_param_bytes_bf16': NROWS * hidden * 2, 'table_payload_bits': NROWS * S.ROW_CELLS * 4, 'bringup_core': ('REALMAP: real in_ram, in_drom := 1 - in_ram' if REALMAP else 'in_ram := one'), 'realmap': REALMAP}
    (dst / 'chipformer.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('hidden', 'chip_layers', 'slots', 'circuit', 'chip_vocab', 'table_param_bytes_bf16')}), f'total {time.time()-t00:.0f}s', flush=True)
if __name__ == '__main__': main(*sys.argv[1:4])
