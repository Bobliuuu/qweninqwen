"""spawn_chain.py: spawn chain jobs on the DEPLOYED app (modal deploy modal_chain.py first). usage: spawn_chain.py NAME CHUNKS(a-b) GPU 'extra flags' [NAME ...]  (groups of 4)"""
import sys, modal
args = sys.argv[1:]
for i in range(0, len(args), 4):
    name, ch, gpu, extra = args[i:i + 4]; a, b = (ch.split('-') + [ch])[:2] if '-' in ch else (ch, ch)
    cl = ','.join(str(x) for x in range(int(a), int(b) + 1))
    f = modal.Function.from_name('qwen-in-qwen-chain', gpu.lower()); c = f.spawn(name, cl, extra.split()); print(name, c.object_id, flush=True)
