"""Run timestamped, cycle-aware region classification from camera frames."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time

import cv2

from .core import AlarmState, FrameRing, LatestMailbox
from .cycle import InspectionCycle
from .source import CaptureReader


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    token: int
    cycle_id: str
    window: tuple
    reference: object


class InferenceWorker(threading.Thread):
    def __init__(self, predictor, mailbox, alarm):
        super().__init__(daemon=True, name='region-inference')
        self.predictor, self.mailbox, self.alarm = predictor, mailbox, alarm
        self.result = None
        self.error = None
        self.lock = threading.Lock()
        self.serial = 0

    def run(self):
        token = None
        try:
            while True:
                job = self.mailbox.take()
                if job is None:
                    break
                if job.token != token:
                    self.alarm.states.clear()
                    token = job.token
                result = self.predictor(job.window, job.reference)
                for region in result['regions']:
                    region['state'] = self.alarm.update(result['generation'], region['region'], region['probability'])
                self.predictor.localize(job.window, result, self.alarm.threshold, job.reference)
                result.update(token=job.token, cycle_id=job.cycle_id,
                              has_reference=job.reference is not None)
                result['processing_seconds'] = time.monotonic() - job.window[-1].received_at
                with self.lock:
                    self.serial += 1
                    self.result = result
        except Exception as error:
            self.error = error

    def latest(self):
        with self.lock:
            return self.serial, self.result


def triple(value):
    try:
        values = tuple(float(part) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected 3 comma-separated numbers') from error
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError('Expected 3 finite numbers')
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Video file, RTSP URL, USB index, or adapter config path')
    parser.add_argument('--adapter', help='Python module:factory; e.g. realtime.hikrobot:create_source')
    parser.add_argument('--model', type=Path, default=ROOT / 'steam/checkpoints/best.pt')
    parser.add_argument('--target', choices=('steam', 'smoke'), default='steam',
                        help='steam emits steam_proxy; smoke requires a smoke-specific checkpoint')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--control-file', type=Path, help='PLC bridge state: cycle_id and phase idle/reference/inspect')
    parser.add_argument('--control-timeout', type=float, default=30.0, help='Control-file heartbeat timeout in seconds')
    parser.add_argument('--roi-config', type=Path, help='JSON rois/exclude rectangles in processed-image pixels, xyxy')
    parser.add_argument('--window-seconds', type=float, default=5.0)
    parser.add_argument('--sample-fps', type=float, default=12.0)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--max-gap', type=float, default=0.75)
    parser.add_argument('--reference-seconds', type=float, default=1.0)
    parser.add_argument('--max-result-age', type=float, default=5.0)
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--mean', type=triple, default=(0.5, 0.5, 0.5))
    parser.add_argument('--std', type=triple, default=(0.25, 0.25, 0.25))
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--high-threshold', type=float, default=0.65)
    parser.add_argument('--clear-threshold', type=float, default=0.35)
    parser.add_argument('--confirm', type=int, default=2)
    parser.add_argument('--clear', type=int, default=3)
    parser.add_argument('--motion-threshold', type=float, default=5.0)
    parser.add_argument('--min-motion-area', type=int, default=20)
    parser.add_argument('--max-width', type=int, default=0, help='0 preserves resolution; resize requires ROI coordinate updates')
    parser.add_argument('--buffer-mb', type=int, default=512, help='Ring-only budget; snapshots/decoder/model need extra memory')
    parser.add_argument('--reconnects', type=int, default=10)
    parser.add_argument('--events', type=Path, default=Path('outputs/realtime/events.jsonl'))
    parser.add_argument('--show', action='store_true', help='Open preview window; press q to stop')
    parser.add_argument('--max-seconds', type=float, help='Stop after this many wall-clock seconds')
    args = parser.parse_args(argv)
    for key in ('window_seconds', 'sample_fps', 'interval', 'max_gap', 'reference_seconds', 'control_timeout',
                'max_result_age', 'frames', 'size', 'tile_size', 'batch', 'motion_threshold',
                'min_motion_area', 'buffer_mb'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be positive and finite')
    if args.max_width < 0 or args.reconnects < 0 or not 0 <= args.overlap < 1 or min(args.std) <= 0:
        parser.error('Invalid geometry, reconnect count, or normalization')
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('max-seconds must be positive and finite')
    if args.adapter and ':' not in args.adapter:
        parser.error('adapter must have module:factory format')
    return args


def load_regions(path):
    value = json.loads(path.read_text(encoding='utf-8')) if path else {}
    if not isinstance(value, dict):
        raise ValueError('ROI configuration must be an object')
    result = []
    for name in ('rois', 'exclude'):
        boxes = value.get(name, [])
        if not isinstance(boxes, list) or any(
            not isinstance(box, list) or len(box) != 4
            or any(type(v) is not int for v in box)
            or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]) for box in boxes
        ):
            raise ValueError(f'{name} must contain positive-area integer xyxy rectangles')
        result.append(boxes)
    return result


def usable_result(result, latest, cycle, now, max_age):
    return (result is not None and latest is not None and result['token'] == cycle.token
            and result['generation'] == latest.generation
            and now - result['received_at'] <= max_age
            and now - latest.received_at <= max_age
            and cycle.phase in ('inspect', 'continuous'))


def preview(image, result, status, rois=(), exclusions=()):
    output = image.copy()
    for box in rois:
        cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), (255, 200, 0), 1)
    for box in exclusions:
        cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), (128, 128, 128), 2)
    if result:
        for region in result['regions']:
            if region['state'] == 'clear':
                continue
            color = (0, 0, 255) if region['state'] == 'confirmed' else (0, 180, 255)
            boxes = region['motion_boxes'] or [region['region']]
            for box in boxes:
                cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), color, 2)
                label = f"{result['target']} {region['state']} {region['probability']:.2f}"
                if not region['motion_boxes']:
                    label += ' [ROI]'
                cv2.putText(output, label, (box[0], max(18, box[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    cv2.putText(output, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 220), 1)
    return output


def main(argv=None):
    args = parse_args(argv)
    from .predictor import RegionPredictor
    rois, exclusions = load_regions(args.roi_config)
    if exclusions and not args.control_file:
        raise ValueError('Exclusions require a cycle control file and a reference phase')
    ring = FrameRing(args.window_seconds, args.sample_fps, args.frames, args.max_gap, args.buffer_mb)
    alarm = AlarmState(args.threshold, args.high_threshold, args.clear_threshold, args.confirm, args.clear)
    predictor = RegionPredictor(args.model, args.device, args.frames, args.size, args.mean, args.std,
                                args.tile_size, args.overlap, args.batch, not args.no_tta,
                                args.target, args.motion_threshold, args.min_motion_area, rois, exclusions)
    mailbox = LatestMailbox()
    cycle = InspectionCycle(args.control_file, args.reference_seconds, args.control_timeout)
    reader = CaptureReader(args.source, ring, args.adapter, args.max_width, args.reconnects)
    worker = InferenceWorker(predictor, mailbox, alarm)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_submit = last_sequence = None
    last_token = cycle.token
    seen_result, last_status, evaluated = 0, None, 0
    closed = False
    with args.events.open('a', encoding='utf-8', buffering=1) as log:
        def emit(kind, **fields):
            record = dict(type=kind, time_utc=datetime.now(timezone.utc).isoformat(), **fields)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

        emit('session_start', target='steam_proxy' if predictor.target == 'steam' else 'smoke', model=str(args.model),
             model_domain_validated_for_packing_smoke=False,
             window_seconds=args.window_seconds, sample_fps=args.sample_fps,
             device=predictor.device, cycle_control=bool(args.control_file))
        reader.start()
        worker.start()
        try:
            while True:
                now = time.monotonic()
                if reader.error:
                    raise RuntimeError(f'Capture failed: {reader.error}') from reader.error
                if worker.error:
                    raise RuntimeError(f'Inference failed: {worker.error}') from worker.error
                cycle.poll()
                latest, packets = ring.raw_view()
                cycle.observe(packets)
                if cycle.token != last_token:
                    # Pending jobs are invalidated by token even if already in flight.
                    last_submit = last_sequence = None
                    last_token = cycle.token
                    emit('cycle', cycle_id=cycle.cycle_id, phase=cycle.phase, token=cycle.token)
                full_window = (len(packets) >= args.frames and
                               packets[-1].timestamp - packets[0].timestamp >= args.window_seconds - ring.period - 1e-5)
                window = cycle.window(packets, args.window_seconds, args.frames, ring.period) if full_window else None
                eof = reader.finished.is_set() and reader.status == 'eof'
                if window and not closed and window[-1].sequence != last_sequence and (
                    last_submit is None or window[-1].timestamp - last_submit >= args.interval or eof
                ):
                    mailbox.put(Job(cycle.token, cycle.cycle_id, window, cycle.reference))
                    last_submit, last_sequence = window[-1].timestamp, window[-1].sequence
                if eof and not closed:
                    mailbox.close()
                    closed = True
                serial, result = worker.latest()
                fresh = usable_result(result, latest, cycle, now, args.max_result_age)
                if serial != seen_result:
                    seen_result = serial
                    evaluated += 1
                    public = {k: v for k, v in result.items() if k not in {'received_at', 'completed_at'}}
                    emit('prediction' if fresh else 'discarded_prediction', **public,
                         result_age_seconds=now-result['received_at'], pending_jobs_replaced=mailbox.replaced)
                if cycle.phase == 'reference':
                    status = 'reference_ready' if (cycle.reference is not None and latest is not None
                                                   and cycle.reference_generation == latest.generation) else 'collecting_reference'
                elif cycle.phase not in ('inspect', 'continuous'):
                    status = cycle.phase
                elif args.control_file and (cycle.reference is None or
                                           latest is not None and cycle.reference_generation != latest.generation):
                    status = 'reference_required'
                elif latest is None or now - latest.received_at > args.max_result_age:
                    status = 'no_recent_frame'
                elif fresh:
                    status = 'confirmed' if any(r['state'] == 'confirmed' for r in result['regions']) else 'monitoring'
                else:
                    status = 'warming_up' if not window else 'waiting_for_fresh_inference'
                status = f'{cycle.phase}/{status} camera={reader.status}'
                if status != last_status:
                    print(status, flush=True)
                    emit('status', status=status, cycle_id=cycle.cycle_id)
                    last_status = status
                if args.show and latest is not None:
                    cv2.imshow('Streaming region inspection', preview(latest.image, result if fresh else None,
                                                                     status, rois, exclusions))
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                if eof and not worker.is_alive():
                    if not evaluated and not args.control_file:
                        raise RuntimeError('Video ended before a full inference window; use a longer replay')
                    break
                if args.max_seconds is not None and now - started >= args.max_seconds:
                    break
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        except Exception as error:
            emit('error', message=str(error))
            raise
        finally:
            reader.stop()
            mailbox.close()
            worker.join(timeout=15)
            if args.show:
                cv2.destroyAllWindows()
            emit('session_end', evaluated_windows=evaluated, pending_jobs_replaced=mailbox.replaced)


if __name__ == '__main__':
    main()
