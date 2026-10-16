#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/dataset.py - 统一的视频数据集类
支持 steam/freeze/drip 三种任务
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import functional as TF


def list_videos(root: Path, exts=(".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    """列出目录下所有视频文件"""
    files = []
    for ext in exts:
        files += list(root.rglob(f"*{ext}"))
    return sorted(files)


def count_labels(samples: List[Tuple[Path, int]], num_classes: int = 2):
    """统计各类别样本数量"""
    counts = [0] * num_classes
    for _, y in samples:
        counts[y] += 1
    return counts


class VideoFolderDataset(Dataset):
    """
    通用视频分类数据集

    目录结构要求:
        root/
        ├── train/
        │   ├── class1/*.mp4
        │   └── class2/*.mp4
        └── val/
            ├── class1/*.mp4
            └── class2/*.mp4

    功能:
        - 均匀采样 T 帧（不足补尾帧）
        - 支持时间抖动 (temporal jitter)
        - 支持非方形输入 (size_h, size_w)
        - 正类增强：灰度化、亮度/对比度扰动、模糊
        - 负类增强：轻微亮度/对比度扰动
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: Tuple[str, str] = ("negative", "positive"),
        frames: int = 16,
        short_side: int = 256,
        crop_size: int = 224,
        size_h: Optional[int] = None,
        size_w: Optional[int] = None,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.25, 0.25, 0.25),
        allow_corrupt_skip: bool = True,
        train_scale_jitter: Tuple[float, float] = (0.8, 1.2),
        hflip_prob: float = 0.5,
        # 正类增强参数
        pos_gray_p: float = 0.35,
        pos_jitter_p: float = 0.30,
        pos_jitter_brightness: float = 0.10,
        pos_jitter_contrast: float = 0.10,
        pos_blur_p: float = 0.20,
        pos_blur_ksize: int = 3,
        # 负类增强参数
        neg_jitter_p: float = 0.10,
        neg_jitter_brightness: float = 0.05,
        neg_jitter_contrast: float = 0.05,
        # 时间采样 jitter
        temporal_jitter: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.frames = frames
        self.short_side = short_side
        self.crop_size = crop_size
        self.size_h = size_h
        self.size_w = size_w
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

        # 固定类别顺序，正类索引为1
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.pos_idx = 1  # 正类索引

        self.allow_corrupt_skip = allow_corrupt_skip
        self.train_scale_jitter = train_scale_jitter
        self.hflip_prob = hflip_prob if split == "train" else 0.0

        # 正类增强参数
        self.pos_gray_p = float(max(0.0, min(1.0, pos_gray_p)))
        self.pos_jitter_p = float(max(0.0, min(1.0, pos_jitter_p)))
        self.pos_jitter_brightness = float(max(0.0, pos_jitter_brightness))
        self.pos_jitter_contrast = float(max(0.0, pos_jitter_contrast))
        self.pos_blur_p = float(max(0.0, min(1.0, pos_blur_p)))
        self.pos_blur_ksize = int(pos_blur_ksize)

        # 负类增强参数
        self.neg_jitter_p = float(max(0.0, min(1.0, neg_jitter_p)))
        self.neg_jitter_brightness = float(max(0.0, neg_jitter_brightness))
        self.neg_jitter_contrast = float(max(0.0, neg_jitter_contrast))

        # 时间 jitter（仅 train 生效）
        self.temporal_jitter = bool(temporal_jitter) and (split == "train")

        # 收集样本
        self.samples: List[Tuple[Path, int]] = []
        for c in self.classes:
            cdir = self.root / split / c
            vids = list_videos(cdir)
            for v in vids:
                self.samples.append((v, self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No videos found in {self.root}/{split}/({self.classes}).")

        # 过滤损坏视频
        if self.allow_corrupt_skip:
            keep = []
            for p, y in self.samples:
                try:
                    vframes, _, _ = read_video(str(p), pts_unit="sec", output_format="TCHW", start_pts=0, end_pts=0.1)
                    if vframes.numel() > 0:
                        keep.append((p, y))
                except Exception:
                    print(f"⚠️  跳过无法读取的视频: {p}")
            if keep:
                self.samples = keep

        self.label_counts = count_labels(self.samples, num_classes=len(self.classes))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _uniform_indices(total: int, num: int, jitter: bool = False) -> np.ndarray:
        """均匀采样 num 个索引，支持时间抖动"""
        if total <= 0:
            return np.zeros((num,), dtype=np.int64)

        if total >= num:
            idx = np.linspace(0, total - 1, num=num, dtype=np.float32)
            if jitter:
                step = (total - 1) / max(1, num - 1)
                max_jitter = max(1.0, step * 0.3)
                noise = np.random.uniform(-max_jitter, max_jitter, size=num)
                idx = np.clip(idx + noise, 0, total - 1)
            return np.round(idx).astype(np.int64)
        else:
            base = list(range(total))
            base += [total - 1] * (num - total)
            return np.array(base, dtype=np.int64)

    def _load_video_tensor(self, path: Path) -> torch.Tensor:
        vframes, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
        if vframes.numel() == 0:
            raise RuntimeError("Empty video tensor.")
        if vframes.shape[1] == 1:
            vframes = vframes.repeat(1, 3, 1, 1)
        return vframes  # (T, C, H, W)

    def _apply_brightness_contrast(self, v: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
        """应用亮度和对比度扰动"""
        c_delta = (random.random() * 2 - 1) * contrast
        b_delta = (random.random() * 2 - 1) * brightness
        return torch.clamp(v * (1.0 + c_delta) + b_delta, 0.0, 1.0)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            v = self._load_video_tensor(path)
        except Exception:
            if self.allow_corrupt_skip:
                new_idx = random.randint(0, len(self.samples) - 1)
                path, label = self.samples[new_idx]
                v = self._load_video_tensor(path)
            else:
                raise

        T, C, H, W = v.shape

        # 均匀采样 frames 帧（支持时间抖动）
        idxs = self._uniform_indices(T, self.frames, jitter=self.temporal_jitter)
        v = v[idxs]  # (frames, C, H, W)

        # 空间变换
        if self.size_h is not None and self.size_w is not None:
            # 非方形输入：直接 resize
            v = TF.resize(v, [self.size_h, self.size_w], antialias=True)
            if self.split == "train" and random.random() < self.hflip_prob:
                v = torch.flip(v, dims=[-1])
        else:
            # 方形输入：短边缩放 + 裁剪
            if self.split == "train":
                scale = random.uniform(self.train_scale_jitter[0], self.train_scale_jitter[1])
                target_ss = max(64, int(self.short_side * scale))
                v = TF.resize(v, target_ss, antialias=True)
                _, _, H2, W2 = v.shape
                if H2 < self.crop_size or W2 < self.crop_size:
                    min_ss = max(self.crop_size, min(H2, W2))
                    v = TF.resize(v, min_ss, antialias=True)
                    _, _, H2, W2 = v.shape
                top = random.randint(0, max(0, H2 - self.crop_size))
                left = random.randint(0, max(0, W2 - self.crop_size))
                v = TF.crop(v, top, left, self.crop_size, self.crop_size)
                if random.random() < self.hflip_prob:
                    v = torch.flip(v, dims=[-1])
            else:
                v = TF.resize(v, self.short_side, antialias=True)
                v = TF.center_crop(v, [self.crop_size, self.crop_size])

        # (frames, C, H, W) -> (C, T, H, W)
        v = v.permute(1, 0, 2, 3).contiguous()
        v = v.float() / 255.0

        # 类别特定增强（仅训练时）
        if self.split == "train":
            if int(label) == self.pos_idx:
                # 正类增强
                if self.pos_jitter_p > 0 and random.random() < self.pos_jitter_p:
                    v = self._apply_brightness_contrast(v, self.pos_jitter_brightness, self.pos_jitter_contrast)
                if self.pos_gray_p > 0 and random.random() < self.pos_gray_p:
                    gray = 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]
                    v = torch.stack([gray, gray, gray], dim=0)
                # 可选模糊增强
                if self.pos_blur_p > 0 and random.random() < self.pos_blur_p:
                    import torch.nn.functional as F
                    k = self.pos_blur_ksize
                    if k % 2 == 1:
                        sigma = k / 6.0
                        v = F.avg_pool3d(v.unsqueeze(0), kernel_size=(1, k, k), stride=1, padding=(0, k//2, k//2)).squeeze(0)
            else:
                # 负类增强
                if self.neg_jitter_p > 0 and random.random() < self.neg_jitter_p:
                    v = self._apply_brightness_contrast(v, self.neg_jitter_brightness, self.neg_jitter_contrast)

        # 标准化
        v = (v - self.mean) / self.std
        return v, torch.tensor(label, dtype=torch.long)
