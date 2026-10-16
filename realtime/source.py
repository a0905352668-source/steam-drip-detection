"""Camera capture is independent of inference and owns the temporal frame ring."""
import importlib
from pathlib import Path
import threading
import time

import cv2


class CaptureReader(threading.Thread):
    """OpenCV source or vendor adapter with isOpened/read/release/get methods.

    SDK adapters must return owned-or-borrowed BGR uint8 images; FrameRing copies
    them before the next read. read() must have a finite timeout in the adapter.
    """

    def __init__(self, source, ring, adapter=None, max_width=0, reconnects=10):
        super().__init__(daemon=True, name="camera-capture")
        self.source, self.ring, self.adapter = source, ring, adapter
        self.max_width, self.reconnects = max_width, reconnects
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.error = None
        self.status = "opening"
        self.is_file = not adapter and Path(source).is_file()

    def open(self):
        if self.adapter:
            module, factory = self.adapter.split(":", 1)
            return getattr(importlib.import_module(module), factory)(self.source)
        if self.source.isdigit() and not self.is_file:
            return cv2.VideoCapture(int(self.source))
        if self.is_file:
            return cv2.VideoCapture(self.source)
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ])

    def run(self):
        capture = None
        try:
            failures = 0
            while not self.stop_event.is_set():
                capture = self.open()
                if not capture.isOpened():
                    failures += 1
                    capture.release()
                    capture = None
                    if self.is_file or failures > self.reconnects:
                        raise RuntimeError("Cannot open camera/source; check connection or adapter")
                    self.status = "reconnecting"
                    self.ring.reset()
                    self.stop_event.wait(0.5)
                    continue
                self.status = "capturing"
                fps = capture.get(cv2.CAP_PROP_FPS) if self.is_file else 0
                if self.is_file and fps <= 0:
                    raise RuntimeError("File has no valid FPS; cannot run timestamped replay")
                index, start = 0, time.monotonic()
                while not self.stop_event.is_set():
                    try:
                        ok, image = capture.read()
                    except TimeoutError:
                        self.status = "waiting_for_frame"
                        continue
                    if not ok:
                        break
                    self.status = "capturing"
                    timestamp = index / fps if self.is_file else time.monotonic()
                    if self.is_file and self.stop_event.wait(max(0, start + timestamp - time.monotonic())):
                        break
                    if self.max_width and image.shape[1] > self.max_width:
                        height = max(1, round(image.shape[0] * self.max_width / image.shape[1]))
                        image = cv2.resize(image, (self.max_width, height), interpolation=cv2.INTER_AREA)
                    self.ring.push(image, timestamp, time.monotonic())
                    failures = 0
                    index += 1
                capture.release()
                capture = None
                if self.is_file:
                    self.status = "eof"
                    break
                if self.stop_event.is_set():
                    break
                failures += 1
                if failures > self.reconnects:
                    raise RuntimeError("Camera disconnected repeatedly")
                self.status = "reconnecting"
                self.ring.reset()
                self.stop_event.wait(0.5)
        except Exception as error:
            self.error = error
            self.status = "error"
        finally:
            if capture is not None:
                capture.release()
            self.finished.set()

    def stop(self):
        self.stop_event.set()
        self.join(timeout=12)
