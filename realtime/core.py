"""Bounded temporal buffers, spatial coverage, localization, and alarm state."""
from collections import deque
from dataclasses import dataclass
import math
import threading

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float
    received_at: float
    image: np.ndarray
    generation: int


class FrameRing:
    """Timestamp sampled frames; gaps/size changes start a new temporal sequence."""

    def __init__(self, seconds=5.0, sample_fps=12.0, min_frames=48,
                 max_gap=0.75, memory_mb=512):
        if min(seconds, sample_fps, min_frames, max_gap, memory_mb) <= 0:
            raise ValueError("Buffer settings must be positive")
        if seconds * sample_fps + 1 < min_frames:
            raise ValueError("window * sample_fps must accommodate the model frame count")
        self.seconds, self.period = seconds, 1.0 / sample_fps
        self.min_frames, self.max_gap = min_frames, max_gap
        self.limit = int(memory_mb * 1024 ** 2)
        self.frames = deque()
        self.latest = None
        self.generation = 0
        self.next_sample = None
        self.last_timestamp = None
        self.bytes = 0
        self.sequence = 0
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self._reset()

    def _reset(self):
        self.frames.clear()
        self.latest = None
        self.bytes = 0
        self.generation += 1
        self.next_sample = self.last_timestamp = None

    def push(self, image, timestamp, received_at):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Source must return uint8 HxWx3 BGR images")
        if not math.isfinite(timestamp):
            raise ValueError("Invalid frame timestamp")
        with self.lock:
            if self.last_timestamp is not None and (
                timestamp <= self.last_timestamp or timestamp - self.last_timestamp > self.max_gap
                or image.shape != self.latest.image.shape
            ):
                self._reset()
            self.last_timestamp = timestamp
            self.sequence += 1
            # Own the memory: SDKs may immediately reuse their capture buffer.
            packet = Frame(self.sequence, timestamp, received_at, image.copy(), self.generation)
            self.latest = packet
            if self.next_sample is None:
                self.next_sample = timestamp
            if timestamp + 1e-7 < self.next_sample:
                return
            steps = max(1, int((timestamp - self.next_sample + 1e-7) / self.period) + 1)
            self.next_sample += steps * self.period
            self.frames.append(packet)
            self.bytes += packet.image.nbytes
            while len(self.frames) > 1 and self.frames[1].timestamp <= timestamp - self.seconds:
                self.bytes -= self.frames.popleft().image.nbytes
            if self.bytes > self.limit:
                raise MemoryError("Frame ring exceeded --buffer-mb; reduce --max-width or window duration")

    def view(self):
        with self.lock:
            frames = tuple(self.frames)
            ready = (len(frames) >= self.min_frames
                     and frames[-1].timestamp - frames[0].timestamp >= self.seconds - self.period - 1e-5)
            return self.latest, frames if ready else None

    def raw_view(self):
        with self.lock:
            return self.latest, tuple(self.frames)


class LatestMailbox:
    """At most one pending job. Replacing a job never changes an in-flight job."""

    def __init__(self):
        self.condition = threading.Condition()
        self.item = None
        self.closed = False
        self.replaced = 0

    def put(self, item):
        with self.condition:
            if self.closed:
                raise RuntimeError("Mailbox is closed")
            if self.item is not None:
                self.replaced += 1
            self.item = item
            self.condition.notify()

    def take(self):
        with self.condition:
            self.condition.wait_for(lambda: self.item is not None or self.closed)
            item, self.item = self.item, None
            return item

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def tile_grid(height, width, size=512, overlap=0.2):
    """Fixed overlapping regions in xyxy coordinates, including right/bottom edges."""
    if min(height, width, size) <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid grid geometry")
    stride = max(1, round(size * (1 - overlap)))

    def starts(length):
        last = max(0, length - size)
        return sorted(set(list(range(0, last + 1, stride)) + [last]))

    return [(x, y, min(x + size, width), min(y + size, height))
            for y in starts(height) for x in starts(width)]


def sample_frames(frames, count):
    """Sample uniformly in time; do not silently duplicate undersized windows."""
    if len(frames) < count:
        raise ValueError("Insufficient distinct frames")
    timestamps = np.array([frame.timestamp for frame in frames])
    targets = np.linspace(timestamps[0], timestamps[-1], count)
    right = np.searchsorted(timestamps, targets).clip(0, len(frames) - 1)
    left = np.maximum(0, right - 1)
    indices = np.where(abs(timestamps[left] - targets) < abs(timestamps[right] - targets), left, right)
    if len(set(indices.tolist())) < count:
        # Distinct sampling takes precedence if irregular capture gaps cause collisions.
        indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return [frames[i] for i in indices]


def motion_boxes(frames, region, threshold=5.0, min_area=20, max_width=256,
                 reference=None, exclusions=()):
    """Approximate moving subregions, not a learned smoke segmentation mask.

    Absolute intensity threshold avoids min/max normalization amplifying quiet noise.
    Motion is used only for localization, never to exclude a classification region.
    """
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    scale = min(1.0, max_width / width)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    gray = []
    for packet in frames:
        crop = packet.image[y0:y1, x0:x1]
        small = cv2.resize(crop, shape, interpolation=cv2.INTER_AREA)
        gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if len(gray) < 2:
        return []
    differences = [cv2.absdiff(a, b) for a, b in zip(gray[:-1], gray[1:])]
    heat = np.mean(differences, axis=0)
    if reference is not None:
        ref = cv2.resize(reference[y0:y1, x0:x1], shape, interpolation=cv2.INTER_AREA)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Preserve slowly expanding changes that adjacent-frame differencing misses.
        heat = np.maximum(heat, np.abs(np.mean(gray[-min(6, len(gray)):], axis=0) - ref))
    mask = (heat >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for ex0, ey0, ex1, ey1 in exclusions:
        left, top = max(x0, ex0), max(y0, ey0)
        right, bottom = min(x1, ex1), min(y1, ey1)
        if left < right and top < bottom:
            mask[max(0, math.floor((top-y0)*shape[1]/height)):math.ceil((bottom-y0)*shape[1]/height),
                 max(0, math.floor((left-x0)*shape[0]/width)):math.ceil((right-x0)*shape[0]/width)] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area * scale ** 2:
            continue
        # Scale each axis independently to account for integer resize rounding.
        sx, sy = width / shape[0], height / shape[1]
        boxes.append([x0 + int(x * sx), y0 + int(y * sy),
                      min(x1, x0 + math.ceil((x + w) * sx)),
                      min(y1, y0 + math.ceil((y + h) * sy))])
    return boxes


class AlarmState:
    """Candidate immediately; confirmation and clearing use consecutive full windows."""

    def __init__(self, threshold=0.5, high_threshold=0.65,
                 clear_threshold=0.35, confirm=2, clear=3):
        if not 0 <= clear_threshold < threshold <= high_threshold <= 1:
            raise ValueError("Require 0 <= clear < threshold <= high <= 1")
        if min(confirm, clear) < 1:
            raise ValueError("Confirmation and clearing counts must be positive")
        self.threshold, self.high = threshold, high_threshold
        self.clear_threshold, self.confirm, self.clear_count = clear_threshold, confirm, clear
        self.states = {}
        self.generation = None

    def update(self, generation, region, probability):
        if generation != self.generation:
            self.states.clear()
            self.generation = generation
        state = self.states.setdefault(tuple(region), dict(hits=0, misses=0, active=False))
        if probability >= self.threshold:
            state['hits'] += 1
            state['misses'] = 0
            if probability >= self.high or state['hits'] >= self.confirm:
                state['active'] = True
        elif probability < self.clear_threshold:
            state['hits'] = 0
            state['misses'] += 1
            if state['misses'] >= self.clear_count:
                state['active'] = False
        else:
            state['hits'] = state['misses'] = 0
        return 'confirmed' if state['active'] else ('candidate' if probability >= self.threshold else 'clear')
