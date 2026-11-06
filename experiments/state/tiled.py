"""Tiled projection driver: full W[Oall,Kall] @ x via paged GDN state (host sends only flags/addresses/weight payload)."""
import torch, time
from engine import *
from transformers import DynamicCache

def run_stream(model, frames, seg=512, cache=None, hook_layer=-1):
    """push frames [n,D] through the outer model in segments (state carried in a DynamicCache). returns last-layer output [n,D]."""
    dev = next(model.parameters()).device
    cap = {}
    h = model.model.layers[hook_layer].register_forward_hook(lambda m, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
    cache = cache if cache is not None else DynamicCache(config=model.config)
    outs = []
    with torch.no_grad():
        for a in range(0, frames.shape[0], seg):
            fb = frames[a:a + seg].to(dev); cp = torch.arange(a, a + fb.shape[0], device=dev)
            model.model(inputs_embeds=fb.float()[None], past_key_values=cache, use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
            outs.append(cap['o'][0])
    h.remove()
    return torch.cat(outs), cache

def project(spec, model, W, X, seg=2048, return_stats=False, max_blocks_per_call=8, fast_prep=True):
    """y[m, Oall] = X[m,Kall] @ W^T.  X is supplied to the X slot of READ frames (test harness; in the full model it is produced
    in-model).  K-tile accumulation over heads x layers happens inside the model (out_proj + residual).
    Frames are built on the model's device, a group of output blocks at a time; state persists in a DynamicCache."""
    dev = next(model.parameters()).device; spec.dev = dev
    spec.ft = torch.float32 if (dev.type == 'cuda' and fast_prep) else torch.float64
    Oall, Kall = W.shape; assert Kall <= spec.K
    Xp = torch.zeros(X.shape[0], spec.K, device=dev); Xp[:, :Kall] = X.to(dev)
    cache = DynamicCache(config=model.config); pos = 0
    t_build = t_run = 0.0; nfr = 0; ys = []
    def sync():
        if dev.type == 'cuda': torch.cuda.synchronize()
    def push(fr):
        nonlocal pos, cache, t_run
        cap = {}
        hk = model.model.layers[-1].register_forward_hook(lambda m, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
        outs = []
        sync(); t0 = time.time()
        with torch.no_grad():
            for a in range(0, fr.shape[0], seg):
                fb = fr[a:a + seg].float(); cp = torch.arange(pos, pos + fb.shape[0], device=dev); pos += fb.shape[0]
                model.model(inputs_embeds=fb[None], past_key_values=cache, use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
                outs.append(cap['o'][0][:, spec.Y.start:spec.Y.stop] if True else None)
        hk.remove(); sync(); t_run += time.time() - t0
        return torch.cat(outs)
    push(spec.init_frames()); nfr += spec.nL
    blocks = list(range(0, Oall, spec.O * spec.G_))
    for g in range(0, len(blocks), max_blocks_per_call):
        sync(); t0 = time.time(); chunks = []; rp = []; n = 0
        for o0 in blocks[g:g + max_blocks_per_call]:
            rows, pl = spec.tile_weights(W, o0)
            chunks.append(spec.write_frames(rows, pl)); n += len(rows)
            chunks.append(spec.read_frames(Xp)); rp.append((n, n + X.shape[0])); n += X.shape[0]
        fr = torch.cat(chunks); sync(); t_build += time.time() - t0
        out = push(fr); nfr += fr.shape[0]
        ys += [out[a:b].double().cpu() for a, b in rp]
    y = torch.cat(ys, 1)[:, :Oall]
    if return_stats:
        return y, dict(frames=nfr, run_s=t_run, build_s=t_build, frames_per_s=nfr / t_run, weights=W.numel(), weights_per_s=W.numel() / t_run, D=spec.layout.n)
    return y
