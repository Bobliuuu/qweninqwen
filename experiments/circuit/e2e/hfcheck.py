"""Teacher-forced check of a firmware trace through the REAL merged Qwen3.5-4B checkpoint (stock Qwen3_5 chip layers via HF modules).

python hfcheck.py MODEL_DIR ELF [--steps N]    (run from <qwenputer npu>/experimental/riscv with its .venv)
Same as chipformer.rv.npucheck.check (which is wrong for programs that putchar: it omits the echo token) but handles BYTE echoes and reports the first mismatching step with context instead of crashing.
"""
import argparse, json, time, sys
import torch
from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S
from chipformer.rv.chat import RvChat

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("elf"); ap.add_argument("--steps", type=int, default=45000)
a = ap.parse_args()
chat = RvChat(a.model)
prog = elf.program(a.elf)
want = ref.Machine.from_program(prog).run(a.steps + 1)[: a.steps + 1]
ids = chat.prompt_ids([{"role": "user", "content": "Hi"}])
ccache = chat._chip_cache()
h = chat.cm.embed_tokens(torch.tensor([ids], device=chat.dev))
with torch.no_grad():
    for L in range(chat.n_chip):
        h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=ccache)
core.snapshot(chat.c, ref.Machine.from_program(prog))
hf.inject(chat.c, ccache)
if chat.graph is None:
    chat._capture()
g, bad, t0 = len(ids), None, time.time()
nm = lambda t: S.name(t) if 0 <= t < S.VOCAB else f"<non-rv {t}>"
echoes = []
with torch.no_grad():
    for i in range(len(want) - 1):
        hc = chat._chip_step(chat.v0 + want[i], g, ccache); g += 1
        got = chat.circuit_token(hc)
        if S.family(want[i])[0] == "BYTE":
            # The stream is BYTE, then the circuit's echo (the Qwen text token of the byte), then the next chip token
            # (bridge.Session.run). The raw reference trace omits the echo, so check it and feed it before going on.
            p = S.family(want[i])[1]
            ok = got is not None and got < chat.v0 and (p >= 256 or chat.tok.decode([got]) == chr(p))
            if not ok:
                bad = {"at": i, "want": f"echo of BYTE({p})", "got_raw": got, "context_in": [nm(w) for w in want[max(0, i - 6): i + 1]]}
                break
            echoes.append(chat.tok.decode([got]))
            hc = chat._chip_step(got, g, ccache); g += 1
            got = chat.circuit_token(hc)
        if got != chat.v0 + want[i + 1]:
            gg = None if got is None else got - chat.v0
            bad = {"at": i, "want": nm(want[i + 1]), "got": None if gg is None else nm(gg), "got_raw": got,
                   "context_in": [nm(w) for w in want[max(0, i - 6): i + 1]], "context_next": [nm(w) for w in want[i + 1: i + 4]]}
            break
r = {"elf": a.elf, "steps_checked": (bad["at"] if bad else len(want) - 1), "steps_requested": a.steps, "mismatch": bad, "echoes": echoes,
     "seconds": round(time.time() - t0, 1)}
print("RESULT " + json.dumps(r)); sys.exit(0 if bad is None else 1)
