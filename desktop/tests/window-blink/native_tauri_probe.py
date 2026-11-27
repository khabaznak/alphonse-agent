"""Exercise a compiled Alphonse Tauri shell with a disposable IPC server on macOS."""
import argparse
from collections import Counter
import copy
import datetime as dt
import json
import os
from pathlib import Path
import plistlib
import re
import socketserver
import subprocess
from tempfile import TemporaryDirectory
import threading
import time

APP_ID = 'local.alphonse.desktop'
FIXTURE = json.loads(Path(__file__).with_name('private-gmail.json').read_text())
USER = 'memory-fixture'
GIB = 1024**3


class Scenario:
    def __init__(self, mode):
        self.mode = mode
        self.started = time.monotonic()
        self.calls = Counter()
        self.history_sizes = Counter()
        self.acknowledged = 0
        self.lock = threading.Lock()
        self.rows = self.make_history(120)

    def make_history(self, count):
        start = dt.datetime.now(dt.timezone.utc)
        rows = []
        for index in range(count):
            request = f'{index + 1:032x}'
            stamp = (start - dt.timedelta(seconds=count-index)).isoformat()
            card = copy.deepcopy(FIXTURE)
            card[0]['createSurface']['surfaceId'] = 'private-gmail:' + request
            card[1]['updateComponents']['surfaceId'] = 'private-gmail:' + request
            for component in card[1]['updateComponents']['components']:
                if component['id'] == 'title':
                    component['text'] = f'Synthetic Inbox summary {index+1}'
            rows.extend([
                {'id':request, 'role':'user', 'content':'What Inbox items need attention?',
                 'source':'desktop', 'created_at':stamp, 'sequence':index*2+1},
                {'id':'owner:'+request, 'role':'assistant',
                 'content':'Gmail Inbox matches were displayed privately in Desktop.',
                 'source':'desktop', 'created_at':stamp, 'sequence':index*2+2,
                 'private_card':{'provider':'gmail','envelopes':card}},
            ])
        return rows

    def visible_replies(self):
        if self.mode == 'fixed':
            return 24
        if self.mode == 'replies':
            return min(120, 4 + int((time.monotonic()-self.started) / 2))
        return 0

    def respond(self, method, params):
        with self.lock:
            self.calls[method] += 1
            if method == 'current_user':
                return {'onboarded':True, 'workspace_configured':True,
                        'user':{'user_id':USER,'display_name':'Synthetic memory test','role':'admin','is_active':True}}
            if method == 'timezone_settings':
                return {'timezone': 'UTC'}
            if method == 'manageable_projects':
                return {'projects': []}
            if method == 'desktop_poll':
                if params.get('user') != USER:
                    raise ValueError('unexpected fixture user')
                return {
                    'daemon_id': 'synthetic-memory-probe', 'next_sequence': 0, 'next_ui_sequence': 0,
                    'questions': [], 'events': [], 'deliveries': [], 'ui_events': [],
                    'status': {'queue': {'ready': 0, 'processing': 0}, 'activity': {'state': 'idle'}, 'active_work': {}},
                    'conversation_has_unseen_messages': self.visible_replies() > self.acknowledged,
                }
            if method == 'desktop_conversation_history':
                if params.get('user') != USER:
                    raise ValueError('unexpected history user')
                replies = self.visible_replies()
                self.history_sizes[replies] += 1
                if params.get('include_private_cards') is True:
                    return {'messages': self.rows[:replies*2], 'private_cards_checked': True}
                return {'messages': [
                    {k:v for k,v in row.items() if k != 'private_card'} for row in self.rows[:replies*2]
                ]}
            if method == 'desktop_mark_conversation_seen':
                self.acknowledged = max(self.history_sizes, default=0)
            if method == 'experimental_owner_status':
                return {'owner_only':True, 'processing_ready':True, 'paused_legacy_requests':0}
            if method == 'owner_scheduled_results_list':
                return {'results':[]}
            return {'items':[]}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        request = json.loads(self.rfile.readline(65536))
        response = self.server.scenario.respond(request['method'],request['params'])
        self.wfile.write(json.dumps({'ok': True,'result':response}, ensure_ascii=False).encode())


def get_renderer(host_pid):
    query = f'processID == {host_pid} AND eventMessage CONTAINS "Current memory footprint:"'
    raw = subprocess.check_output(['/usr/bin/log','show','--last','130s','--style','compact',
                                   '--predicate',query],text=True)
    pids = re.findall(r'WebContent\[(\d+)\] Current memory footprint:',raw)
    return int(pids[-1]) if pids else None


def bytes_count(raw):
    value, unit = re.fullmatch(r'([\d.]+)([KMGT]?)',raw).groups()
    return round(float(value) * 1024**' KMGT'.index(unit or ' '))


def vmmap(pid, output):
    name = subprocess.check_output(['/bin/ps','-p',str(pid),'-o','comm='],text=True)
    if not name.rstrip().endswith('/com.apple.WebKit.WebContent'):
        raise RuntimeError('renderer identity changed')
    raw = subprocess.check_output(['/usr/bin/vmmap','-summary',str(pid)],text=True)
    output.write_text(raw)
    def number(pattern):
        return bytes_count(re.search(pattern,raw,re.M).group(1))
    row = re.search(r'^WebKit Malloc\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)',raw,re.M)
    zone = re.search(r'^WebKit Malloc_\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)',raw,re.M)
    return {
        'footprint':number(r'^Physical footprint:\s+(\S+)'),
        'peak':number(r'^Physical footprint \(peak\):\s+(\S+)'),
        'webkit_malloc_resident':bytes_count(row.group(2)) if row else None,
        'webkit_malloc_swapped':bytes_count(row.group(4)) if row else None,
        'webkit_malloc_live_allocations':int(zone.group(5)) if zone else None,
        'webkit_malloc_allocated':bytes_count(zone.group(6)) if zone else None,
    }


def run(app, mode, duration, result_dir):
    scenario = Scenario(mode)
    samples = []
    with TemporaryDirectory(prefix='alphonse-memory-',dir='/private/tmp') as name:
        fixture_root = Path(name)
        bundle = fixture_root / 'Alphonse Memory Fixture.app'
        subprocess.run(['/usr/bin/ditto','--norsrc','--noextattr',str(app),str(bundle)], check=True)
        info = bundle / 'Contents/Info.plist'
        config = plistlib.loads(info.read_bytes())
        if config['CFBundleIdentifier'] != APP_ID:
            raise RuntimeError('expected Alphonse Desktop bundle')
        config.update(CFBundleIdentifier=APP_ID + '.fixture.' + mode,
                      CFBundleName='Alphonse Memory Fixture',
                      CFBundleDisplayName='Alphonse Memory Fixture')
        info.write_bytes(plistlib.dumps(config))
        subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(bundle)],
                       check=True)
        binary = bundle / 'Contents/MacOS' / config['CFBundleExecutable']
        socket_path = fixture_root / 'fixture.sock'
        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
        with Server(str(socket_path),Handler) as server:
            server.scenario = scenario
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            process = subprocess.Popen([str(binary)],env={
                'PATH':'/usr/bin:/bin','ALPHONSE_V2_SOCKET_PATH':str(socket_path),
                'ALPHONSE_V2_DB_PATH':str(fixture_root / 'fixture.sqlite3'),
                'ALPHONSE_DESKTOP_DIAGNOSTIC_MODE': 'ping-only' if mode == 'ipc' else 'normal',
            },cwd=fixture_root,stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            started = time.monotonic()
            scenario.started = started
            try:
                renderer = None
                while renderer is None and time.monotonic()-started < 75:
                    if process.poll() is not None:
                        raise RuntimeError(f'fixture Desktop exited with {process.returncode}')
                    time.sleep(5)
                    renderer = get_renderer(process.pid)
                if renderer is None:
                    raise RuntimeError(f'WebKit PID unavailable for fixture host {process.pid}')
                deadline = started+duration
                print(f'{mode}: host {process.pid}, renderer {renderer}',flush=True)
                while True:
                    elapsed = round(time.monotonic()-started,1)
                    try:
                        sample = vmmap(renderer,result_dir / f'{mode}-{int(elapsed):03d}.vmmap.txt')
                    except (subprocess.CalledProcessError,RuntimeError):
                        samples.append({'elapsed': elapsed, 'renderer_disappeared': True,
                                        'replacement':get_renderer(process.pid)})
                        break
                    with scenario.lock:
                        sample.update(elapsed=elapsed, calls=dict(scenario.calls),
                                      history_requests=dict(scenario.history_sizes),
                                      acknowledged_replies=scenario.acknowledged)
                    samples.append(sample)
                    print(f'{mode}: {elapsed:5.1f}s {sample["footprint"]/1024**2:6.1f} MiB, '
                          f'{sample["acknowledged_replies"]} acknowledged replies',flush=True)
                    if sample['footprint'] > 1.25*GIB or time.monotonic() >= deadline:
                        break
                    time.sleep(min(15,max(0,deadline-time.monotonic())))
                trace = subprocess.check_output(['/usr/bin/log','show','--last',f'{duration+30}s','--info',
                    '--style','compact','--predicate',f'processID == {process.pid} AND '
                    '(eventMessage CONTAINS "main frame load started" OR eventMessage CONTAINS "didCommitLoadForFrame" '
                    'OR eventMessage CONTAINS "processDidTerminate:")'],text=True)
                (result_dir / f'{mode}-lifecycle.txt').write_text(trace)
                (result_dir / f'{mode}.json').write_text(json.dumps({
                    'mode':mode,'host':process.pid,'renderer':renderer,'samples':samples},indent=2)+'\n')
            finally:
                process.terminate()
                try:
                    process.wait(8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                server.shutdown()
                thread.join(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app',type=Path)
    parser.add_argument('mode',choices=['idle','fixed','replies','ipc'])
    parser.add_argument('result_dir',type=Path)
    parser.add_argument('--seconds',type=int,default=300,choices=range(60,901))
    options = parser.parse_args()
    options.result_dir.mkdir(parents=True,exist_ok=True)
    run(options.app,options.mode,options.seconds,options.result_dir)
