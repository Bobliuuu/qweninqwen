cd /tmp/romx && flock -w 7200 ~/.gpu.lock python3 -c "
import json, mac_layer as M
for (K,H,I) in [(230,2560,9216),(86,1024,3584)]: print(json.dumps(M.run(K,H=H,I=I)),flush=True)
"
