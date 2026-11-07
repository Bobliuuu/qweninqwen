"""Tiny Vast.ai REST helper (key read from ~/.config/vastai/vast_api_key; never printed, never sent to rented machines).
usage: python vast_api.py offers GPUNAME [N]   | balance | list | create OFFER_ID LABEL | destroy ID | show ID"""
import json, os, sys, urllib.request
KEY = open(os.path.expanduser('~/.config/vastai/vast_api_key')).read().strip(); BASE = 'https://console.vast.ai/api/v0'
def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read() or b'{}')
def offers(gpu, n=8, min_disk=90):
    q = {'limit': 300, 'type': 'on-demand', 'verified': {'eq': True}, 'rentable': {'eq': True}, 'num_gpus': {'eq': 1}, 'gpu_name': {'eq': gpu}, 'disk_space': {'gte': min_disk}, 'inet_down': {'gte': 300}, 'reliability2': {'gte': 0.95}, 'cuda_max_good': {'gte': 12.4}, 'order': [['dph_total', 'asc']]}
    return call('POST', '/bundles/', q).get('offers', [])[:n]
if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'balance': u = call('GET', '/users/current/'); print({k: u.get(k) for k in ('balance', 'credit')})
    elif c == 'offers':
        for o in offers(sys.argv[2].replace('_', ' '), int(sys.argv[3]) if len(sys.argv) > 3 else 8): print(o['id'], o['gpu_name'], round(o['dph_total'], 3), 'disk', int(o['disk_space']), 'down', int(o['inet_down']), 'up', int(o['inet_up']), 'rel', round(o['reliability2'], 3), 'cuda', o.get('cuda_max_good'), o.get('geolocation'))
    elif c == 'list':
        for i in call('GET', '/instances/').get('instances', []): print(i['id'], i.get('label'), i.get('actual_status'), i.get('gpu_name'), i.get('dph_total'), i.get('ssh_host'), i.get('ssh_port'))
    elif c == 'destroy': print(call('DELETE', f'/instances/{sys.argv[2]}/'))

def create(offer_id, label, pub, disk=100, image='nvidia/cuda:12.4.1-devel-ubuntu22.04'):
    onstart = f'mkdir -p /root/.ssh && echo "{pub}" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys'
    return call('PUT', f'/asks/{offer_id}/', {'client_id': 'me', 'image': image, 'disk': disk, 'runtype': 'ssh', 'label': label, 'onstart': onstart})
def instances(): return call('GET', '/instances/').get('instances', [])
