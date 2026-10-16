"""Inspection cycle contract, independent of PLC transport (file/API supported)."""
from pathlib import Path
import json
import time

import numpy as np


class InspectionCycle:
    def __init__(self, control_file=None, reference_seconds=1.0, control_timeout=30.0):
        self.path = Path(control_file) if control_file else None
        self.reference_seconds = reference_seconds
        self.control_timeout = control_timeout
        self.key = None
        self.token = 0
        self.started = time.monotonic()
        self.phase = 'continuous' if self.path is None else 'unknown'
        self.cycle_id = 'continuous' if self.path is None else None
        self.reference = None
        self.reference_generation = None
        self.detail = ''

    def update(self, cycle_id, phase):
        """Call from an existing acquisition/PLC application on phase changes."""
        if phase not in {'idle', 'reference', 'inspect', 'unknown'}:
            raise ValueError("Phase must be idle/reference/inspect")
        key = (str(cycle_id), phase)
        if key == self.key:
            return
        same_cycle = str(cycle_id) == self.cycle_id
        if not same_cycle or phase in {'idle', 'reference', 'unknown'}:
            self.reference = self.reference_generation = None
        self.key = key
        self.cycle_id, self.phase = str(cycle_id), phase
        self.token += 1
        self.started = time.monotonic()

    def poll(self):
        if self.path is None:
            return
        try:
            if time.time() - self.path.stat().st_mtime > self.control_timeout:
                raise ValueError('Control bridge heartbeat expired')
            value = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise ValueError('Control state must be an object')
            if not isinstance(value.get('cycle_id'), str) or not value['cycle_id']:
                raise ValueError("cycle_id must be a non-empty string")
            self.update(value['cycle_id'], value['phase'])
            self.detail = ''
        except (OSError, ValueError, KeyError, TypeError):
            self.update('', 'unknown')
            self.detail = 'control_file_missing_or_invalid'

    def observe(self, packets):
        if self.phase != 'reference' or not packets:
            return
        recent = [packet for packet in packets if packet.received_at >= self.started]
        if len(recent) < 3 or recent[-1].timestamp - recent[0].timestamp < self.reference_seconds:
            return
        indices = np.rint(np.linspace(0, len(recent) - 1, min(8, len(recent)))).astype(int)
        self.reference = np.median(np.stack([recent[i].image for i in indices]), axis=0).astype(np.uint8)
        self.reference_generation = recent[-1].generation

    def window(self, packets, seconds, min_frames, period):
        if self.phase == 'continuous':
            return packets
        if self.phase != 'inspect' or self.reference is None:
            return None
        if not packets or packets[-1].generation != self.reference_generation:
            return None
        window = tuple(packet for packet in packets if packet.received_at >= self.started)
        if len(window) < min_frames or window[-1].timestamp - window[0].timestamp < seconds - period - 1e-5:
            return None
        return window
