# trueqiq_state (agent B): GDN recurrent state as float weight memory  -- PRELIMINARY (API first, numbers below are measured)

## API (engine.py, tiled.py, run_block.py has `load_w`)
```python
from machine import Layout, make_outer        # ../trueqiq/machine.py
from engine import build_state_matvec         # StateMV
lay = Layout(); spec = build_state_matvec(lay, nL=1, H=16, K=896, B=1e6)   # allocs B_WF,B_RF,B_INIT,B_SEL,B_PAY{l},B_X,B_Y
# ... allocate C's / other slots in the same layout, THEN:
model = make_outer(lay, ['linear_attention']*nL, intermediate, gdn=dict(hk=16,hv=16,dk=128,dv=128), vocab=...)
spec.compile(model, l0=0)    # layers l0..l0+nL-1 become the paged matvec bank
frames = spec.init_frames()                       # once per layer: writes ballast constant (row dk-1)   [nL frames]
rows, pl = spec.tile_weights(W, o0)               # real W[Oall,Kall] block of 126 output rows -> 126 write rows (63 pairs, +c/-c)
frames = spec.write_frames(rows, pl)              # [126, D] host transport frames (flags, one-hot address, silu-pre-inverted payload)
frames = spec.read_frames(X)                      # READ frame(s): X slot (or: the in-model slot spec.X filled by other layers), RF=1
# after the READ frame, slot spec.Y of the layer-stack output = (W[o0:o0+126, :K] @ x)   (float, ~1e-6 rel)
```
`tiled.project(spec, model, W, X)` runs a whole W[Oall,Kall] (any Oall; Kall <= spec.K) through the stream with state carried in a
DynamicCache (pass `attention_mask={'full_attention':None,'linear_attention':None}`, `cache_position`, `position_ids`; a pure-GDN
stack otherwise crashes in the stock mask builder).

## Frame protocol (all host-written fields are transport only)
* slot 0 BALLAST = spec.B (**B >= 1e6 required**: with 1e4 the transparent outer norm adds x-dependent scale error |x|^2/(2B^2) ~1e-3 for |x|~1e3;
  measured below).  RF=1 read frame; WF[l]=1 write for layer l; INIT=1 + SEL[dk-1]=1 init frame; SEL one-hot = key row; PAY[l] = payload.
* WRITE frame: SEL[r]=1, WF[l]=1, PAY[l][h*127+c] = silu^-1(w_c + s) (c<126), PAY[..+126]=silu^-1(s)  (s = per-row/head shift, weights only).
  The host also pre-compensates the transparent-norm scale of the *write* frame (fixed point on weights only).  Writes of all nL layers x 16 heads
  happen in the SAME frame (each (layer,head) has its own payload channels) => nL*H*126 weights per frame.
* READ frame: x in slot X (K floats).  Layer l head h contracts x tile t=l*H+h (63 elems).  out_proj sums heads (and the residual sums layers):
  K-tile accumulation is inside the model.   Y[c] = sum_t W_t x_t.
* Non-written/overwritten semantics: each write frame overwrites row r completely (beta=1,g=0): no clear needed between tiles; reads are non-destructive (beta=0).
