"""Vast fleet manager (runs on the Mac). Manages every running instance whose label starts with 'qiq-': prep (env setup + data relay from the hub box), then a shared work queue of single-chunk jobs
(exactly the Modal chunk_job flags, see run_chunks.sh). Results are rsynced back to ~/qweninqwen-work/cfram_q_vast_pull/cfram_q_par_full/chunk_NNNN (layout of cf/collect_seams.py --no-pull).
The first chunk on each box is its own verification (clean = rc 0, 0 free mismatches, 0 forced, not diverged); a box that fails is destroyed, slower-than-2x-per-dollar boxes are destroyed after >=1 chunk.
Status line every 2 min -> ~/qweninqwen-work/cfram_q_vast_out/status.txt. When the queue is empty each idle box is destroyed; at the end every qiq- instance is destroyed and verified.
usage: python fleet.py [--chunks 0-91] [--hub 59513405]. The Vast key is read by vast_api.py only (never sent to boxes)."""
import argparse, collections, json, os, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import vast_api as v
ap = argparse.ArgumentParser(); ap.add_argument('--chunks', default='0-91'); ap.add_argument('--hub', type=int, default=59513405); ap.add_argument('--relay-par', type=int, default=5); a = ap.parse_args()
lo, hi = (int(x) for x in a.chunks.split('-')); HOME = os.path.expanduser('~'); W = f'{HOME}/qweninqwen-work'; PULL = f'{W}/cfram_q_vast_pull/cfram_q_par_full'; OUT = f'{W}/cfram_q_vast_out'; os.makedirs(PULL, exist_ok=True)
KEY = f'{HOME}/.ssh/vast_qiq'; SSHO = ['-i', KEY, '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', '-o', 'BatchMode=yes']
RELAY_PUB = open(f'{OUT}/relay_pub.txt').read().strip(); VD = os.path.dirname(os.path.abspath(__file__))
PRIO = {hi: None}   # last chunk (HALT / final word) is claimed only by the hub (reliable A100 PCIE), first
lock = threading.Lock(); queue = collections.deque([hi] + list(range(lo, hi))); attempts = collections.Counter(); done = {}; failed = {}; boxes = {}; stop = threading.Event(); relay_sem = threading.Semaphore(a.relay_par); best = [None]; T0 = time.time()
def log(m):
    with open(f'{OUT}/fleet.log', 'a') as f: f.write(f'{time.strftime("%F %T")} {m}\n')
def ssh(b, cmd, timeout=120):
    try:
        r = subprocess.run(['ssh', *SSHO, '-p', str(b['port']), f'root@{b["host"]}', cmd], capture_output=True, text=True, timeout=timeout); return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired: return 124, 'timeout'
def scp(b, files, dest):
    return subprocess.run(['scp', '-q', *SSHO[:6], '-P', str(b['port']), *files, f'root@{b["host"]}:{dest}'], capture_output=True, text=True, timeout=120).returncode
def destroy(b, why):
    log(f'DESTROY {b["id"]} {b["label"]}: {why}')
    for _ in range(3):
        try: v.call('DELETE', f'/instances/{b["id"]}/'); break
        except Exception as e: log(f'destroy retry {b["id"]}: {str(e)[:80]}'); time.sleep(3)
    b['dead'] = True
def prep(b):
    t = time.time()
    while time.time() - t < 900:
        rc, o = ssh(b, 'echo up', 30)
        if rc == 0 and 'up' in o: break
        time.sleep(10)
    else: return False
    ssh(b, f'mkdir -p /data/cfram_q/cf /data/cfram_q/refQ /out/cfram_q_par_full /root/.ssh; grep -q "{RELAY_PUB.split()[1]}" /root/.ssh/authorized_keys || echo "{RELAY_PUB}" >> /root/.ssh/authorized_keys')
    scp(b, [f'{VD}/box_setup.sh', f'{VD}/run_chunks.sh'], '/root/'); ssh(b, 'chmod +x /root/*.sh; grep -q SETUP_OK /root/setup.log 2>/dev/null || (nohup /root/box_setup.sh > /root/setup.log 2>&1 < /dev/null &)')
    while 'HUBSYNC_DONE' not in open(f'{OUT}/hub_sync.log').read(): time.sleep(10)   # Mac -> hub copy (incl. the 4.8 GB cache) must be complete before anyone relays from the hub
    if b['id'] != a.hub:
        with relay_sem:
            for k in range(2):
                hub = boxes[a.hub]; rc, o = ssh(hub, f'/root/relay.sh {b["host"]} {b["port"]} rowemb 2>&1 | tail -n 3', 3000)
                if 'RELAY_DONE' in o: break
                log(f'relay {b["id"]} attempt {k} failed: {o[-200:]}')
            else: return False
    t = time.time()
    while time.time() - t < 1200:
        rc, o = ssh(b, 'tail -n 3 /root/setup.log')
        if 'SETUP_OK' in o:
            b['env'] = o.strip().splitlines()[-2][:120] if len(o.strip().splitlines()) > 1 else ''
            return 'cuda True' in o
        time.sleep(10)
    return False
def verify(res):
    return res.get('free_mismatches') == 0 and res.get('forced_tokens') == 0 and not res.get('diverged')
def run_chunk(b, k, launch=True):
    d = f'/out/cfram_q_par_full/chunk_{k:04d}'; t0 = time.time()
    if launch: ssh(b, f'rm -rf {d}; nohup /root/run_chunks.sh {k} > /root/run_{k}.log 2>&1 < /dev/null &')
    while time.time() - t0 < 4200 and not stop.is_set():
        time.sleep(20); rc, o = ssh(b, f'cat {d}/job_done.json 2>/dev/null', 60)
        if rc == 0 and '"rc"' in o:
            jd = json.loads(o); loc = f'{PULL}/chunk_{k:04d}'; os.makedirs(loc, exist_ok=True)
            subprocess.run(['rsync', '-a', '-e', 'ssh ' + ' '.join(SSHO) + f' -p {b["port"]}', f'root@{b["host"]}:{d}/', loc + '/'], capture_output=True, timeout=600)
            try: res = json.load(open(f'{loc}/chain_{k}_to_{k}.json'))
            except Exception: res = {}
            return jd, res
    return None, None
def worker(b):
    b['state'] = 'prep'
    if not prep(b): log(f'prep failed {b["id"]}'); destroy(b, 'prep failed'); return
    b['state'] = 'ready'; log(f'READY {b["id"]} {b["label"]} {b.get("env","")}'); n = 0; adopt = None
    rc, o = ssh(b, "pgrep -fa 'run_chunks.sh [0-9]+' | grep -v pgrep | grep -o 'run_chunks.sh [0-9]*' | head -1")
    if rc == 0 and o.strip().startswith('run_chunks.sh'):
        try: adopt = int(o.split()[1]); log(f'ADOPT chunk {adopt} running on {b["id"]}')
        except Exception: adopt = None
    while not stop.is_set():
        with lock:
            cand = [c for c in queue if not (c in PRIO and b['id'] != a.hub)]
            if not cand:
                if any(c in PRIO for c in queue): time.sleep(5); continue   # only the hub-reserved chunk is left
                break
            k = adopt if adopt is not None else cand[0]
            if k in queue: queue.remove(k)
            attempts[k] += 1
        b['state'] = f'chunk {k}'; jd, res = run_chunk(b, k, launch=(adopt is None)); adopt = None
        ok = jd is not None and jd.get('rc') == 0 and res and verify(res)
        with lock:
            if ok:
                sec = jd['seconds']; sr = res.get('steps_run', 0) or 0; full = sr >= 1_500_000; cpc = b['dph'] * sec / 3600 * (2097152 / sr if sr else 1)   # $ per FULL chunk (the last chunk is short: do not let it distort the comparison)
                done[k] = dict(box=b['id'], gpu=b['gpu'], seconds=sec, cost=round(cpc, 3), steps_s=round(sr / sec) if sec else None); b['n'] = b.get('n', 0) + 1
                if full:
                    b['cpc'] = cpc
                    if best[0] is None or cpc < best[0]: best[0] = cpc
            else:
                failed.setdefault(k, []).append(dict(box=b['id'], gpu=b['gpu'], jd=jd, res={x: res.get(x) for x in ('free_mismatches', 'forced_tokens', 'diverged')} if res else None))
                if attempts[k] < 3: queue.appendleft(k)
        log(f'chunk {k} on {b["id"]} {b["gpu"]}: {"OK" if ok else "FAIL"} {jd and jd.get("seconds")}s')
        if not ok: destroy(b, f'chunk {k} failed'); return
        if best[0] and b.get('cpc') and b['cpc'] > 2 * best[0] and len(done) >= 6: destroy(b, f'$/chunk {b["cpc"]:.3f} > 2x best {best[0]:.3f}'); return
    destroy(b, 'queue empty'); b['state'] = 'destroyed'
def status():
    while not stop.is_set():
        try:
            ins = v.callv1('/instances/').get('instances', []); spend = sum(i.get('dph_total', 0) * (time.time() - i.get('start_date', time.time())) / 3600 for i in ins)
            up = [b for b in boxes.values() if not b.get('dead')]; comp = sum(1 for b in up if str(b.get('state', '')).startswith('chunk'))
            by = collections.defaultdict(list)
            for k, d in done.items(): by[d['gpu']].append(d)
            per = '; '.join(f'{g}: {len(l)} chunks, {sum(x["seconds"] for x in l) / len(l) / 60:.1f} min, ${sum(x["cost"] for x in l) / len(l):.2f}/chunk' for g, l in by.items())
            line = f'{time.strftime("%F %T")} boxes up={len(up)} computing={comp} instances_rented={len(ins)} chunks done={len(done)}/{hi - lo + 1} failed_attempts={sum(len(x) for x in failed.values())} queue={len(queue)} est_spend_current_fleet=${spend:.2f} rate=${sum(i.get("dph_total", 0) for i in ins):.2f}/h | {per}'
            open(f'{OUT}/status.txt', 'a').write(line + '\n')
        except Exception as e: log(f'status err {e}')
        for _ in range(24):
            if stop.is_set(): break
            time.sleep(5)
for k in range(lo, hi + 1):
    try:
        rj = json.load(open(f'{PULL}/chunk_{k:04d}/chain_{k}_to_{k}.json')); jd = json.load(open(f'{PULL}/chunk_{k:04d}/job_done.json'))
        if jd.get('rc') == 0 and verify(rj):
            done[k] = dict(box=jd.get('host'), gpu=jd.get('gpu'), seconds=jd['seconds'], cost=0.0, steps_s=None, resumed=True); queue = collections.deque(c for c in queue if c != k)
    except Exception: pass
def harvest():
    """pull every finished chunk dir (job_done.json present) that is not yet in PULL from all running boxes (jobs finish while the manager is restarted)"""
    try: ins = [i for i in v.callv1('/instances/').get('instances', []) if str(i.get('label', '')).startswith('qiq-') and i.get('actual_status') == 'running' and i.get('ssh_host')]
    except Exception: return
    def one(i):
        b = dict(host=i['ssh_host'], port=i['ssh_port']); rc, o = ssh(b, 'for d in /out/cfram_q_par_full/chunk_*; do [ -f $d/job_done.json ] && basename $d; done', 60)
        if rc != 0: return
        for name in o.split():
            if not name.startswith('chunk_'): continue
            k = int(name[6:]); loc = f'{PULL}/{name}'
            if k in done or os.path.exists(f'{loc}/chain_{k}_to_{k}.json'): continue
            os.makedirs(loc, exist_ok=True); subprocess.run(['rsync', '-a', '-e', 'ssh ' + ' '.join(SSHO) + f' -p {b["port"]}', f'root@{b["host"]}:/out/cfram_q_par_full/{name}/', loc + '/'], capture_output=True, timeout=600); log(f'harvested {name} from {i["id"]}')
    ts = [threading.Thread(target=one, args=(i,)) for i in ins]; [t.start() for t in ts]; [t.join() for t in ts]
harvest()
for k in range(lo, hi + 1):
    if k in done: continue
    try:
        rj = json.load(open(f'{PULL}/chunk_{k:04d}/chain_{k}_to_{k}.json')); jd = json.load(open(f'{PULL}/chunk_{k:04d}/job_done.json'))
        if jd.get('rc') == 0 and verify(rj):
            done[k] = dict(box=jd.get('host'), gpu=jd.get('gpu'), seconds=jd['seconds'], cost=0.0, steps_s=None, resumed=True); queue = collections.deque(c for c in queue if c != k)
    except Exception: pass
log(f'startup: {len(done)} chunks already done on disk, queue={list(queue)[:6]}...')
threading.Thread(target=status, daemon=True).start()
workers = {}
while True:
    try: ins = v.callv1('/instances/').get('instances', [])
    except Exception as e: log(f'list err {e}'); time.sleep(10); continue
    for i in ins:
        if str(i.get('label', '')).startswith('qiq-') and i.get('actual_status') == 'running' and i.get('ssh_host') and i['id'] not in boxes:
            b = dict(id=i['id'], label=i['label'], host=i['ssh_host'], port=i['ssh_port'], dph=i['dph_total'], gpu=i.get('gpu_name', '?')); boxes[i['id']] = b
            if i['id'] != a.hub and a.hub not in boxes: pass
            t = threading.Thread(target=worker, args=(b,), daemon=True); workers[i['id']] = t; t.start(); log(f'manage {b["id"]} {b["label"]} {b["gpu"]} ${b["dph"]:.3f}/h')
    alive = [t for t in workers.values() if t.is_alive()]
    if len(done) >= hi - lo + 1 or (not queue and not alive and workers): break
    if time.time() - T0 > 6 * 3600: log('6h cap'); break
    time.sleep(15)
stop.set(); time.sleep(2)
for i in v.callv1('/instances/').get('instances', []):
    if str(i.get('label', '')).startswith('qiq-'): destroy(dict(id=i['id'], label=i['label']), 'fleet finished')
time.sleep(5); left = [i['id'] for i in v.callv1('/instances/').get('instances', []) if str(i.get('label', '')).startswith('qiq-')]
open(f'{OUT}/status.txt', 'a').write(f'{time.strftime("%F %T")} FINISHED chunks done={len(done)} failed={sorted(k for k in failed if k not in done)} instances_left={left}\n'); json.dump(dict(done=done, failed=failed), open(f'{OUT}/fleet_result.json', 'w'), indent=1)
