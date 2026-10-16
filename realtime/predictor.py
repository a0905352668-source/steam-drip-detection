"""Reuse original X3D input normalization; no temporary video encoding."""
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from steam import inference as offline
from .core import motion_boxes, sample_frames, tile_grid


class RegionPredictor:
    def __init__(self, model, device="auto", frames=48, size=256,
                 mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25),
                 tile_size=512, overlap=0.2, batch_size=2, tta=True,
                 target="steam", motion_threshold=5, min_motion_area=20,
                 rois=(), exclusions=()):
        self.path = Path(model)
        if not self.path.is_file():
            raise FileNotFoundError("Model missing; run python scripts/download_models.py")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.frames, self.size = frames, size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
        self.tile_size, self.overlap = tile_size, overlap
        self.batch_size, self.tta, self.target = batch_size, tta, target
        self.motion_threshold, self.min_motion_area = motion_threshold, min_motion_area
        self.rois, self.exclusions = rois, exclusions
        self.is_onnx = self.path.suffix.lower() == ".onnx"
        if self.is_onnx:
            if target == "smoke":
                # ONNX does not carry a reliable smoke/no_smoke semantic contract here.
                raise ValueError("For smoke target use a checkpoint declaring classes ('no_smoke','smoke')")
            offline.REQUIRE_CUDA = device == "cuda"
            self.model = offline.OnnxModel(str(self.path), self.device == "cuda")
            self.device = self.model.device
            shape = self.model.session.get_inputs()[0].shape
            if len(shape) != 5 or any(isinstance(shape[i], int) and shape[i] != v
                                     for i, v in [(1, 3), (2, frames), (3, size), (4, size)]):
                raise ValueError("ONNX input shape does not match configured C/T/H/W")
            if isinstance(shape[0], int) and shape[0] != 1:
                raise ValueError("Only dynamic-batch or batch=1 ONNX models are supported")
            if isinstance(shape[0], int):
                self.batch_size = 1
        else:
            if self.path.suffix.lower() not in {".pt", ".pth"}:
                raise ValueError("Use a .pt/.pth checkpoint or .onnx model")
            checkpoint = torch.load(str(self.path), map_location="cpu", weights_only=False)
            expected = ("no_steam", "steam") if target == "steam" else ("no_smoke", "smoke")
            if tuple(checkpoint.get("classes", ())) != expected:
                raise ValueError(f"Model classes must be {expected}; steam weights cannot certify smoke")
            args = checkpoint.get("args", {}) or {}
            for key, value in [("frames", frames), ("size_h", size), ("size_w", size)]:
                if args.get(key) is not None and args[key] != value:
                    raise ValueError(f"Checkpoint {key}={args[key]} differs from requested {value}")
            for key, value in [("mean", mean), ("std", std)]:
                if args.get(key) is not None and not np.allclose(args[key], value):
                    raise ValueError(f"Checkpoint {key} differs from configured normalization")
            del checkpoint
            self.model = offline.TorchModel(str(self.path), self.device, 2)

    def __call__(self, window, reference=None):
        started = time.monotonic()
        sampled = sample_frames(window, self.frames)
        height, width = sampled[0].image.shape[:2]
        rois = self.rois or [(0, 0, width, height)]
        for x0, y0, x1, y1 in [*rois, *self.exclusions]:
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("ROI/exclusion is outside the current image; update region configuration")
        if self.exclusions and (reference is None or reference.shape != sampled[0].image.shape):
            raise ValueError("Exclusions require a same-cycle reference image")
        regions = sorted(set((x+rx, y+ry, x+rx1, y+ry1)
                             for x, y, right, bottom in rois
                             for rx, ry, rx1, ry1 in tile_grid(bottom-y, right-x, self.tile_size, self.overlap)))
        regions = [box for box in regions if not any(
            ex0 <= box[0] and ey0 <= box[1] and ex1 >= box[2] and ey1 >= box[3]
            for ex0, ey0, ex1, ey1 in self.exclusions)]
        if not regions:
            raise ValueError("No inspection region remains outside exclusions")
        results = []
        for offset in range(0, len(regions), self.batch_size):
            boxes = regions[offset:offset + self.batch_size]
            clips = []
            for x0, y0, x1, y1 in boxes:
                # Original offline preprocessing first exported 256x256 RGB tiles.
                rgb = []
                for frame in sampled:
                    crop = frame.image[y0:y1, x0:x1].copy()
                    for ex0, ey0, ex1, ey1 in self.exclusions:
                        left, top, right, bottom = max(x0, ex0), max(y0, ey0), min(x1, ex1), min(y1, ey1)
                        if left < right and top < bottom:
                            crop[top-y0:bottom-y0, left-x0:right-x0] = reference[top:bottom, left:right]
                    rgb.append(cv2.cvtColor(cv2.resize(crop, (self.size, self.size),
                                                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                clip = torch.from_numpy(np.stack(rgb)).permute(3, 0, 1, 2).float() / 255.0
                clips.append((clip - self.mean) / self.std)
            batch = torch.stack(clips)
            if self.is_onnx:
                logits = self.model.forward_logits(batch.numpy(), self.tta)
                probabilities = offline.softmax_np(logits, axis=1)
            else:
                with torch.inference_mode():
                    # Keep FP32 by default: accuracy is prioritized over throughput.
                    logits = self.model.forward_logits(batch.to(self.device), self.tta)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            if probabilities.shape != (len(boxes), 2) or not np.isfinite(probabilities).all():
                raise RuntimeError("Model returned invalid binary-class probabilities")
            for region, probability in zip(boxes, probabilities[:, 1]):
                results.append({"region": list(region), "probability": float(probability)})
        return dict(generation=window[-1].generation,
                    window_start=window[0].timestamp, window_end=window[-1].timestamp,
                    received_at=window[-1].received_at, completed_at=time.monotonic(),
                    inference_seconds=time.monotonic() - started,
                    frame_size=[width, height], target='steam_proxy' if self.target == 'steam' else 'smoke',
                    regions=results)

    def localize(self, window, result, threshold, reference=None):
        for region in result['regions']:
            if region['probability'] >= threshold:
                region['motion_boxes'] = motion_boxes(
                    window, region['region'], self.motion_threshold, self.min_motion_area,
                    reference=reference, exclusions=self.exclusions)
            else:
                region['motion_boxes'] = []
