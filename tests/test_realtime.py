import ctypes
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from realtime.core import AlarmState, Frame, FrameRing, LatestMailbox, motion_boxes, sample_frames, tile_grid
from realtime.cycle import InspectionCycle
from realtime.hikrobot import MVSCamera
from realtime.run import usable_result


def packet(index, image=None, generation=0):
    return Frame(index, float(index), float(index),
                 np.zeros((32, 32, 3), np.uint8) if image is None else image, generation)


class BufferTests(unittest.TestCase):
    def test_fractional_sampling_is_ready_and_bounded(self):
        ring = FrameRing(seconds=5, sample_fps=12, min_frames=48)
        image = np.zeros((10, 10, 3), np.uint8)
        for index in range(301):
            ring.push(image, index / 30, index / 30)
        latest, window = ring.view()
        self.assertIsNotNone(window)
        self.assertLessEqual(len(window), 62)
        self.assertLessEqual(latest.timestamp - window[0].timestamp, 5.1)
        selected = sample_frames(window, 48)
        self.assertEqual(len({item.sequence for item in selected}), 48)

    def test_gap_and_resolution_change_invalidate_history(self):
        ring = FrameRing(seconds=1, sample_fps=10, min_frames=5)
        image = np.zeros((10, 10, 3), np.uint8)
        for index in range(12):
            ring.push(image, index / 10, index / 10)
        self.assertIsNotNone(ring.view()[1])
        ring.push(image, 3, 3)
        self.assertEqual(ring.generation, 1)
        self.assertIsNone(ring.view()[1])
        ring.push(np.zeros((11, 10, 3), np.uint8), 3.1, 3.1)
        self.assertEqual(ring.generation, 2)

    def test_borrowed_sdk_memory_is_copied(self):
        image = np.zeros((10, 10, 3), np.uint8)
        ring = FrameRing(seconds=1, sample_fps=10, min_frames=5)
        ring.push(image, 0, 0)
        image.fill(255)
        self.assertEqual(int(ring.raw_view()[0].image.max()), 0)

    def test_memory_budget_fails_instead_of_silently_shortening_window(self):
        ring = FrameRing(seconds=1, sample_fps=10, min_frames=5, memory_mb=0.0001)
        with self.assertRaises(MemoryError):
            ring.push(np.zeros((10, 10, 3), np.uint8), 0, 0)

    def test_mailbox_keeps_latest_pending_and_drains_on_close(self):
        mailbox = LatestMailbox()
        mailbox.put('old')
        mailbox.put('new')
        mailbox.close()
        self.assertEqual(mailbox.take(), 'new')
        self.assertIsNone(mailbox.take())
        self.assertEqual(mailbox.replaced, 1)


class RegionTests(unittest.TestCase):
    def test_full_grid_has_no_gaps(self):
        covered = np.zeros((1080, 1920), bool)
        for x0, y0, x1, y1 in tile_grid(1080, 1920):
            covered[y0:y1, x0:x1] = True
        self.assertTrue(covered.all())

    def test_quiet_noise_not_normalized_into_motion(self):
        images = [np.full((32, 32, 3), index % 3, np.uint8) for index in range(12)]
        self.assertEqual(motion_boxes([packet(i, image) for i, image in enumerate(images)], (0, 0, 32, 32)), [])

    def test_slow_weak_change_localized_against_reference(self):
        baseline = np.full((32, 32, 3), 40, np.uint8)
        images = []
        for index in range(12):
            image = baseline.copy()
            image[8:20, 10:22] += index
            images.append(packet(index, image))
        self.assertEqual(motion_boxes(images, (0, 0, 32, 32), threshold=5), [])
        boxes = motion_boxes(images, (0, 0, 32, 32), threshold=5, reference=baseline)
        self.assertTrue(any(x0 <= 10 and y0 <= 8 and x1 >= 22 and y1 >= 20 for x0, y0, x1, y1 in boxes))
        self.assertEqual(motion_boxes(images, (0, 0, 32, 32), threshold=5,
                                      reference=baseline, exclusions=[(0, 0, 32, 32)]), [])

    def test_alarm_hysteresis_and_generation_reset(self):
        alarm = AlarmState()
        region = [0, 0, 32, 32]
        self.assertEqual(alarm.update(0, region, .55), 'candidate')
        self.assertEqual(alarm.update(0, region, .55), 'confirmed')
        self.assertEqual(alarm.update(0, region, .1), 'confirmed')
        self.assertEqual(alarm.update(0, region, .1), 'confirmed')
        self.assertEqual(alarm.update(0, region, .1), 'clear')
        self.assertEqual(alarm.update(1, region, .55), 'candidate')


class CycleTests(unittest.TestCase):
    def test_reference_must_precede_inspection_and_is_cleared_per_bag(self):
        cycle = InspectionCycle()
        with patch('realtime.cycle.time.monotonic', return_value=0):
            cycle.update('bag1', 'reference')
        packets = tuple(packet(index) for index in range(4))
        cycle.observe(packets)
        self.assertIsNotNone(cycle.reference)
        with patch('realtime.cycle.time.monotonic', return_value=4):
            cycle.update('bag1', 'inspect')
        packets = tuple(packet(index) for index in range(4, 10))
        self.assertIsNotNone(cycle.window(packets, 5, 5, .1))
        cycle.update('bag2', 'inspect')
        self.assertIsNone(cycle.reference)
        self.assertIsNone(cycle.window(packets, 5, 5, .1))

    def test_reconnect_requires_new_reference(self):
        cycle = InspectionCycle()
        with patch('realtime.cycle.time.monotonic', return_value=0):
            cycle.update('bag1', 'reference')
        cycle.observe(tuple(packet(index) for index in range(4)))
        with patch('realtime.cycle.time.monotonic', return_value=4):
            cycle.update('bag1', 'inspect')
        self.assertIsNone(cycle.window(tuple(packet(i, generation=1) for i in range(4, 10)), 5, 5, .1))

    def test_missing_control_file_cannot_enable_inspection(self):
        with tempfile.TemporaryDirectory() as folder:
            cycle = InspectionCycle(Path(folder) / 'missing.json')
            cycle.poll()
            self.assertEqual(cycle.phase, 'unknown')

    def test_stale_control_heartbeat_disables_inspection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'control.json'
            path.write_text('{"cycle_id":"bag1","phase":"inspect"}')
            old = time.time() - 60
            os.utime(path, (old, old))
            cycle = InspectionCycle(path, control_timeout=30)
            cycle.poll()
            self.assertEqual(cycle.phase, 'unknown')

    def test_malformed_non_object_control_is_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'control.json'
            path.write_text('[]')
            cycle = InspectionCycle(path)
            cycle.poll()
            self.assertEqual(cycle.phase, 'unknown')

    def test_stale_or_previous_cycle_result_is_not_usable(self):
        latest = Frame(1, 20, 20, np.zeros((2, 2, 3), np.uint8), 2)
        cycle = SimpleNamespace(token=3, phase='inspect')
        result = dict(token=3, generation=2, received_at=20)
        self.assertTrue(usable_result(result, latest, cycle, 21, 5))
        self.assertFalse(usable_result(result, latest, cycle, 30, 5))
        self.assertFalse(usable_result(dict(result, token=2), latest, cycle, 21, 5))
        self.assertFalse(usable_result(dict(result, generation=1), latest, cycle, 21, 5))
        cycle.phase = 'idle'
        self.assertFalse(usable_result(result, latest, cycle, 21, 5))


class MVSMemoryTests(unittest.TestCase):
    def make_camera(self, conversion_code=0):
        camera = MVSCamera.__new__(MVSCamera)
        raw = (ctypes.c_ubyte * 12)(*range(12))
        released = []

        def acquire(frame, timeout):
            frame.stFrameInfo = SimpleNamespace(nWidth=2, nHeight=2, nFrameLen=12, enPixelType=0)
            frame.pBufAddr = ctypes.cast(raw, ctypes.POINTER(ctypes.c_ubyte))
            return 0

        def convert(params):
            ctypes.memmove(params.pDstBuffer, raw, 12)
            return conversion_code

        def release(frame):
            released.append(True)
            ctypes.memset(raw, 0, 12)
            return 0

        camera.sdk = SimpleNamespace(MV_FRAME_OUT=SimpleNamespace,
                                     MV_CC_PIXEL_CONVERT_PARAM=SimpleNamespace,
                                     PixelType_Gvsp_BGR8_Packed=1)
        camera.cam = SimpleNamespace(MV_CC_GetImageBuffer=acquire,
                                     MV_CC_ConvertPixelType=convert, MV_CC_FreeImageBuffer=release)
        return camera, released

    def test_image_is_copied_before_sdk_buffer_is_freed(self):
        camera, released = self.make_camera()
        ok, image = camera.read()
        self.assertTrue(ok)
        self.assertEqual(image.ravel().tolist(), list(range(12)))
        self.assertEqual(released, [True])

    def test_sdk_buffer_is_freed_on_conversion_failure(self):
        camera, released = self.make_camera(77)
        with self.assertRaises(RuntimeError):
            camera.read()
        self.assertEqual(released, [True])


if __name__ == '__main__':
    unittest.main()
