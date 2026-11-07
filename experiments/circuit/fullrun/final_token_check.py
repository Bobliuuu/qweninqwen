"""FIX G: what does "the token is correct" mean, and is the circuit's final token the real Qwen3-0.6B's token?

Two independent questions, answered separately and never mixed:
 (1) PURE READ-OUT: the final answer token is decoded from the CIRCUIT'S OWN emitted tokens (the 4 BYTE records the firmware putchar's before its exit ecall = HALT), not from the
     reference trace. The reference is used only afterwards as a post-hoc comparison. `pure` = Modal A100: runs circuit_run_pure.py (zero host writes, zero token overrides) on the
     last chunk of the CF+RAM reference and dumps the circuit's own token stream; `decode` = decode any token npy (circuit or reference).
 (2) FIDELITY vs REAL Qwen3-0.6B: `hf` = HF Qwen3-0.6B on the same prompt in fp32 and bf16, plus the quantised numerics the firmware actually uses (int4 per-row RTN weights, optional
     int8 per-token activations) as torch fake-quant. Top-k logits, gap, rank of the firmware's token.
The firmware's inner contract is int4 weights + fixed-point nonlinears, so its token is only expected to equal HF fp32 where the quantisation error is smaller than the logit gap.

  python final_token_check.py decode TOK.npy SPEC.py            decode the final putchar word from a token stream (reference or circuit)
  modal run final_token_check.py::hf --prompt 785 --scan 3000      (Modal, 1 GPU, ~5 min)   -> /out/finaltoken/hf_<prompt>.json
  modal run --detach final_token_check.py::pure --chunk 74         (Modal, 1 A100, ~10 min) -> /out/finaltoken/pure_chunk0074/{circuit_tail.npy,result.json}
  python final_token_check.py report HF.json PURE.json OUT.md
"""
import sys, os, json

# ----------------------------------------------------------------------------------------------------------------- decode (pure python/numpy)
def load_spec(path):
    import importlib.util
    s = importlib.util.spec_from_file_location('rv_spec', path); S = importlib.util.module_from_spec(s); s.loader.exec_module(S); return S

def decode_final(tokens, S):
    """Final putchar word from a token stream: the BYTE(b) records in order (little-endian) until the HALT control. Returns dict(word, bytes, halt_index, n_byte)."""
    import numpy as np
    t = np.asarray(tokens); bytes_ = []; idx = []; halt = None
    for i, x in enumerate(t.tolist()):
        fam = S.family(int(x))[0]
        if fam == 'BYTE': bytes_.append(int(S.family(int(x))[1])); idx.append(i)
        elif fam == 'HALT': halt = i; break
    word = sum(b << (8 * k) for k, b in enumerate(bytes_[:4])) if len(bytes_) >= 4 else None
    return dict(word=word, bytes=bytes_, byte_positions=idx, halt_index=halt, n_tokens=len(t), complete=(word is not None and halt is not None and len(bytes_) == 4))

# ----------------------------------------------------------------------------------------------------------------- HF fidelity
def hf_check(prompt, scan_n=0, topk=10, dev='cuda'):
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = 'Qwen/Qwen3-0.6B'; tok = AutoTokenizer.from_pretrained(name)
    def load(dtype): return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    ids = torch.tensor([prompt], device=dev)
    def fq_w(W, bits=4):                       # SAME per-row symmetric RTN as compiler/gen_fx.quant and run_full_fx.fq
        qmax = 7.0 if bits == 4 else 119.0; lo, hi = (-8, 7) if bits == 4 else (-120, 119)
        s = (W.abs().max(1, keepdim=True).values / qmax).clamp(min=1e-6); return torch.clamp(torch.round(W / s), lo, hi) * s
    def fq_a(x, bits=8):
        qmax = 2 ** (bits - 1) - 1; s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax; return torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
    def summarize(logits, label):
        lg = logits[0, -1].float().cpu(); top = torch.topk(lg, topk); order = torch.argsort(lg, descending=True)
        rank = {int(t): int((order == t).nonzero()[0, 0]) for t in (13, 15846, 454)}
        return dict(label=label, argmax=int(top.indices[0]), argmax_text=tok.decode([int(top.indices[0])]), gap_top1_top2=float(top.values[0] - top.values[1]),
                    topk=[(int(i), tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)], rank_of={str(k): v for k, v in rank.items()},
                    logit_of={str(k): float(lg[k]) for k in (13, 15846, 454)})
    out = dict(prompt=prompt, prompt_text=tok.decode(prompt), runs=[])
    m = load(torch.float32)
    with torch.no_grad(): out['runs'].append(summarize(m(ids).logits, 'HF fp32'))
    base = {n: p.detach().clone() for n, p in m.named_parameters()}
    def restore():
        with torch.no_grad():
            for n, p in m.named_parameters(): p.copy_(base[n])
    def quantise_weights():
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.ndim == 2 and ('proj' in n or 'embed' in n): p.copy_(fq_w(base[n]))      # embed is tied to lm_head in Qwen3-0.6B: quantises both, exactly like run_full_fx
    quantise_weights()
    with torch.no_grad(): out['runs'].append(summarize(m(ids).logits, 'HF fp32 + int4 per-row weights only (= firmware weight contract)'))
    hs = []
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear): hs.append(mod.register_forward_pre_hook(lambda mo, a: (fq_a(a[0]),)))
    with torch.no_grad(): out['runs'].append(summarize(m(ids).logits, 'HF fp32 + int4 weights + int8 per-token activations at every linear (closest to firmware W4A8, exact nonlinears)'))
    for h in hs: h.remove()
    scan = None
    if scan_n:                                  # single-token-prompt agreement of the quantised contract vs fp32: tells how often "firmware == HF" can hold at all
        toks = torch.arange(scan_n, device=dev)[:, None]
        def am(model, hooks=False):
            hh = [mod.register_forward_pre_hook(lambda mo, a: (fq_a(a[0]),)) for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)] if hooks else []
            r = []
            with torch.no_grad():
                for i in range(0, scan_n, 256):
                    lg = model(toks[i:i + 256]).logits[:, -1].float(); t2 = torch.topk(lg, 2); r.append(torch.stack([t2.indices[:, 0].float(), (t2.values[:, 0] - t2.values[:, 1])], 1).cpu())
            for h in hh: h.remove()
            return torch.cat(r).numpy()
        restore(); a32 = am(m); quantise_weights(); aw4 = am(m); aw4a8 = am(m, True)
        agree4 = (a32[:, 0] == aw4[:, 0]); agree48 = (a32[:, 0] == aw4a8[:, 0])
        scan = dict(n=scan_n, w4_agree_frac=float(agree4.mean()), w4a8_agree_frac=float(agree48.mean()),
                    candidates_both_agree_gap_ge_1=[int(i) for i in np.nonzero(agree4 & agree48 & (a32[:, 1] >= 1.0))[0][:40]], prompt_in_agree_set=bool(agree4[prompt[0]] and agree48[prompt[0]]) if len(prompt) == 1 else None)
    out['scan_single_token_prompts'] = scan
    restore(); del m
    m = load(torch.bfloat16)
    with torch.no_grad(): out['runs'].insert(1, summarize(m(ids).logits, 'HF bf16'))
    return out

# ----------------------------------------------------------------------------------------------------------------- report
def report(hf_json, pure_json, out_md):
    hf = json.load(open(hf_json)); pr = json.load(open(pure_json)) if pure_json and os.path.exists(pure_json) else None
    L = [f"# Final-token check, prompt {hf['prompt']} ({hf['prompt_text']!r})", '', '## HF Qwen3-0.6B (real model) vs the firmware numerics', '',
         '| run | argmax | text | gap top1-top2 | rank of token 13 | rank of 15846 |', '|---|---|---|---|---|---|']
    for r in hf['runs']: L.append(f"| {r['label']} | {r['argmax']} | {r['argmax_text']!r} | {r['gap_top1_top2']:.3f} | {r['rank_of']['13']} | {r['rank_of']['15846']} |")
    if hf.get('scan_single_token_prompts'): s = hf['scan_single_token_prompts']; L += ['', f"Single-token-prompt scan (first {s['n']} ids): int4 weights agree with fp32 argmax on {100*s['w4_agree_frac']:.1f}%, int4+int8 activations on {100*s['w4a8_agree_frac']:.1f}%."]
    if pr: L += ['', '## Circuit read-out', '', f"Circuit's own tail decoded: word={pr['circuit']['word']} bytes={pr['circuit']['bytes']} halt_at={pr['circuit']['halt_index']}; reference tail word={pr['reference']['word']}; circuit == reference: {pr['circuit']['word'] == pr['reference']['word']}; free mismatches in this chunk: {pr.get('free_mismatches')}."]
    open(out_md, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

# ----------------------------------------------------------------------------------------------------------------- Modal entrypoints
try:
    import modal
    app = modal.App('qwen-in-qwen-finaltoken')
    data = modal.Volume.from_name('qfr-data'); outv = modal.Volume.from_name('qfr-out', create_if_missing=True)
    img_hf = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'transformers', 'accelerate')
    img_c = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')

    @app.function(image=img_hf, gpu='A10G', timeout=3600, volumes={'/out': outv})
    def hf_remote(prompt, scan):
        r = hf_check(prompt, scan); os.makedirs('/out/finaltoken', exist_ok=True)
        json.dump(r, open(f'/out/finaltoken/hf_{"_".join(map(str, prompt))}.json', 'w'), indent=1); outv.commit(); print(json.dumps(r)); return r

    @app.local_entrypoint()
    def hf(prompt: str = '785', scan: int = 3000):
        r = hf_remote.remote([int(x) for x in prompt.split(',')], scan); print(json.dumps(r, indent=1)[:6000])

    @app.function(image=img_c, gpu='A100-40GB', timeout=4 * 3600, memory=65536, cpu=8, volumes={'/data': data, '/out': outv})
    def pure_remote(chunk, steps):
        """PURE free-run of ONE chunk of the CF+RAM reference through circuit_run_pure.py (zero host writes/overrides/injections); then dump the CIRCUIT's own tokens and decode its final word."""
        import runpy, numpy as np, torch
        data.reload(); o = f'/out/finaltoken/pure_chunk{chunk:04d}'; os.makedirs(o, exist_ok=True)
        os.environ.update(PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', WF_ROW_CELLS='3840'); os.chdir('/data/qpriscv_wf'); sys.path[:0] = ['.', '/data/circuit_wf/engine']
        sys.argv = ['circuit_run_pure.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', str(chunk), '--ramimg', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy'] + (['--steps', str(steps)] if steps else [])
        g = runpy.run_path('/data/circuit_wfram/fullrun/circuit_run_pure.py', run_name='__main__')
        S = g['S']; n = int(g['lens'][0]); raw = g['RAW'][0, :n].cpu().numpy(); ref = np.load(f'/data/cfram/refRC/tok_{chunk:04d}.npy')
        np.save(f'{o}/circuit_tail.npy', raw); res = json.load(open([f'{o}/{f}' for f in os.listdir(o) if f.startswith('result_')][0]))
        spec_path = '/data/qpriscv_wf/chipformer/rv/spec.py'; Sx = load_spec(spec_path)
        out = dict(chunk=chunk, circuit=decode_final(raw, Sx), reference=decode_final(ref, Sx), free_mismatches=res['free_mismatches'], host_writes=res['host_writes'], token_overrides=res['token_overrides'], injections_applied=res['injections_applied'], steps_run=res['steps_run'], gpu=res['gpu'])
        json.dump(out, open(f'{o}/result.json', 'w'), indent=1); outv.commit(); print(json.dumps(out)); return out

    @app.local_entrypoint()
    def pure(chunk: int = 74, steps: int = 0):
        print(json.dumps(pure_remote.remote(chunk, steps), indent=1))
except ImportError:
    pass

if __name__ == '__main__' and len(sys.argv) > 1:
    if sys.argv[1] == 'decode':
        import numpy as np; print(json.dumps(decode_final(np.load(sys.argv[2]), load_spec(sys.argv[3]))))
    elif sys.argv[1] == 'report': report(sys.argv[2], sys.argv[3] if len(sys.argv) > 4 else None, sys.argv[-1])
