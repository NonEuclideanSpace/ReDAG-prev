from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.reproduce_paper import completed, write_json

RESULTS = ROOT / 'results'
ORDER = [f'{i:02d}' for i in [*range(7, 17), *range(1, 7)]]
STOP = RESULTS / 'STOP_CONTINUOUS'
STATE = RESULTS / 'continuous_status.json'


def prepare_round(base_manifest, round_number):
    if round_number == 1:
        return base_manifest
    base = base_manifest.parent
    data = json.loads(base_manifest.read_text())
    folder = base / 'rounds' / f'round-{round_number:03d}'
    manifest = folder / 'paper_cases.json'
    if manifest.exists():
        return manifest
    data.update(round_number=round_number, quality_retry_policy='record_only',
                seed_base=260500 + (round_number - 1) * 1000)
    if data.get('source_pdf'):
        data['source_pdf'] = os.path.relpath((base / data['source_pdf']).resolve(), folder)
    for case in data['cases']:
        if case.get('reference_image'):
            case['reference_image'] = os.path.relpath((base / case['reference_image']).resolve(), folder)
        if case.get('output'):
            target = base / 'images' / f'round-{round_number:03d}' / Path(case['output']).name
            case['output'] = os.path.relpath(target, folder)
    write_json(manifest, data)
    return manifest


def batch_active(folder):
    with (folder / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / '.continuous.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A continuous supervisor is already running.', flush=True)
            return 1
        previous = json.loads(STATE.read_text()) if STATE.exists() else {}
        round_number = int(previous.get('round', 1))
        finished = list(previous.get('completed_rounds', []))
        state = {'pid': os.getpid(), 'status': 'running', 'round': round_number,
                 'completed_rounds': finished, 'order': ORDER,
                 'policy': 'real_genclaw_draft_then_flux_once_keep_failed_images',
                 'stop_file': str(STOP)}

        def update(**values):
            state.update(values, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            write_json(STATE, state)

        def request_stop(*_):
            STOP.touch()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not STOP.exists():
            manifest = prepare_round(RESULTS / 'paper_cases.json', round_number)
            folder = manifest.parent
            cases = [c for c in json.loads(manifest.read_text())['cases'] if c['source_status'] == 'complete']
            if batch_active(folder):
                update(status='running', phase='following_existing_batch', round=round_number, manifest=str(manifest))
                time.sleep(15)
                continue
            if all(completed(c, folder) for c in cases):
                if round_number not in finished:
                    finished.append(round_number)
                    print(f'Round {round_number:03d} completed; beginning next round.', flush=True)
                round_number += 1
                update(round=round_number, completed_rounds=finished)
                continue
            status_path = folder / 'reproduction_status.json'
            if status_path.exists():
                last = json.loads(status_path.read_text())
                if last.get('status') in {'stopped_on_error', 'blocked'}:
                    update(status='blocked', phase='needs_infrastructure_check', round=round_number,
                           manifest=str(manifest), current=last.get('current'))
                    return 1
            seed = 260500 + (round_number - 1) * 1000
            with (folder / 'batch.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT/'scripts/reproduce_paper.py'),
                    '--manifest', str(manifest), '--ids', ','.join(ORDER), '--seed', str(seed)],
                    cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
                update(status='running', phase='batch', round=round_number, manifest=str(manifest), child_pid=child.pid, seed_base=seed)
                while child.poll() is None:
                    if STOP.exists():
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        break
                    time.sleep(5)
                if child.returncode and not STOP.exists():
                    update(status='blocked', phase='batch_failed', exit_code=child.returncode)
                    return 1
        update(status='stopped', phase='user_stop')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())

