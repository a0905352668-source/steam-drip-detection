from pathlib import Path
import unittest

import numpy as np

from realtime.core import Frame
from realtime.predictor import RegionPredictor


class FakeBackend:
    def __init__(self):
        self.inputs = []

    def forward_logits(self, batch, tta):
        self.inputs.append(batch.copy())
        return np.tile(np.array([[0, 1]], np.float32), (len(batch), 1))


class PredictorTests(unittest.TestCase):
    def make_predictor(self, rois=(), exclusions=()):
        predictor = RegionPredictor.__new__(RegionPredictor)
        import torch
        predictor.frames, predictor.size = 4, 16
        predictor.tile_size, predictor.overlap = 16, 0
        predictor.mean, predictor.std = torch.zeros((3, 1, 1, 1)), torch.ones((3, 1, 1, 1))
        predictor.batch_size, predictor.tta, predictor.target = 2, True, 'steam'
        predictor.is_onnx, predictor.device = True, 'cpu'
        predictor.rois, predictor.exclusions = rois, exclusions
        predictor.motion_threshold, predictor.min_motion_area = 5, 4
        predictor.model = FakeBackend()
        return predictor

    def test_no_motion_does_not_skip_any_classification_region(self):
        predictor = self.make_predictor()
        frames = tuple(Frame(i, float(i), float(i), np.full((32, 32, 3), 40, np.uint8), 0) for i in range(4))
        result = predictor(frames)
        self.assertEqual(len(result['regions']), 4)
        self.assertEqual(sum(len(batch) for batch in predictor.model.inputs), 4)
        predictor.localize(frames, result, .5)
        self.assertTrue(all(not region['motion_boxes'] for region in result['regions']))

    def test_excluded_pixels_use_frozen_same_cycle_reference(self):
        predictor = self.make_predictor(rois=[[0, 0, 16, 16]], exclusions=[[0, 0, 8, 16]])
        reference = np.full((32, 32, 3), 40, np.uint8)
        frames = tuple(Frame(i, float(i), float(i), np.full((32, 32, 3), 200, np.uint8), 0) for i in range(4))
        predictor(frames, reference)
        batch = predictor.model.inputs[0]
        np.testing.assert_allclose(batch[0, :, :, :, :8], 40/255)
        np.testing.assert_allclose(batch[0, :, :, :, 8:], 200/255)

    def test_steam_checkpoint_is_rejected_for_smoke_target(self):
        path = Path('steam/checkpoints/best.pt')
        if not path.is_file():
            self.skipTest('Download supplied steam model to exercise the semantic guard')
        with self.assertRaisesRegex(ValueError, 'steam weights cannot certify smoke'):
            RegionPredictor(path, device='cpu', target='smoke')


if __name__ == '__main__':
    unittest.main()
