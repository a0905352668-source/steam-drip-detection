#!/usr/bin/env python3
"""Synthetic video and inspection-signal integration test; never opens a camera."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    (ROOT / 'outputs').mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='realtime-check-', dir=ROOT / 'outputs'))
    video, events, control = output / 'synthetic.mp4', output / 'events.jsonl', output / 'cycle.json'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 240))
    if not writer.isOpened():
        raise RuntimeError('Cannot create synthetic replay')
    for index in range(500):
        image = np.full((240, 320, 3), 70, np.uint8)
        if index >= 70:
            overlay = image.copy()
            cv2.circle(overlay, (190, 110), min(45, 3+(index-70)//5), (150, 150, 150), -1)
            image = cv2.addWeighted(image, .8, overlay, .2, 0)
        writer.write(image)
    writer.release()

    def phase(cycle_id, value):
        temporary = control.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(cycle_id=cycle_id, phase=value)))
        os.replace(temporary, control)

    def records():
        if not events.exists():
            return []
        lines = events.read_text(encoding='utf-8').splitlines(keepends=True)
        return [json.loads(line) for line in lines if line.endswith('\n')]

    def wait_for(predicate):
        started = time.monotonic()
        while time.monotonic() - started < 30:
            values = records()
            if predicate(values):
                return values
            if process.poll() is not None:
                raise RuntimeError(f'Runner exited early; inspect {output / "runner.log"}')
            time.sleep(.05)
        raise TimeoutError(f'Expected inspection state not reached; inspect {output / "runner.log"}')

    phase('bag-1', 'idle')
    with (output / 'runner.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([
            sys.executable, '-m', 'realtime.run', '--source', str(video), '--device', args.device,
            '--control-file', str(control), '--reference-seconds', '.5', '--events', str(events),
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(lambda rows: any(row['type'] == 'status' and row['status'].startswith('idle/') for row in rows))
            phase('bag-1', 'reference')
            wait_for(lambda rows: any(row['type'] == 'status' and 'reference_ready' in row['status'] for row in rows))
            phase('bag-1', 'inspect')
            wait_for(lambda rows: any(row['type'] == 'prediction' and row['cycle_id'] == 'bag-1' for row in rows))
            phase('bag-2', 'inspect')
            wait_for(lambda rows: any(row['type'] == 'status' and row.get('cycle_id') == 'bag-2'
                                      and 'reference_required' in row['status'] for row in rows))
            time.sleep(.5)
            rows = records()
            pivot = next(i for i, row in enumerate(rows) if row['type'] == 'cycle' and row['cycle_id'] == 'bag-2')
            assert not any(row['type'] == 'prediction' and row['cycle_id'] == 'bag-1' for row in rows[pivot:])
            assert not any(row['type'] == 'prediction' and row['cycle_id'] == 'bag-2' for row in rows)
            predictions = [row for row in rows if row['type'] == 'prediction']
            assert all(row['has_reference'] and row['target'] == 'steam_proxy' for row in predictions)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=25)
    if process.returncode:
        raise RuntimeError(f'Runner exit {process.returncode}; inspect {output / "runner.log"}')
    report = dict(reference_and_inspection=True, changed_cycle_invalidates_old_results=True,
                  new_cycle_requires_new_reference=True, camera_hardware_tested=False,
                  smoke_accuracy_tested=False, fresh_predictions=len(predictions))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Realtime integration test passed:', output / 'report.json')


if __name__ == '__main__':
    main()
