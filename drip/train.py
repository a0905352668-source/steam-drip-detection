# -*- coding: utf-8 -*-
"""
train_drip_split.py — 二分类视频低落检测（drip vs nodrip）

相对 v1 的主要改动：
- 增加 train 阶段的时间采样 jitter（同一段视频每次取的帧略有不同，提高时序鲁棒性）
- 为正类(drip)增加可选的轻微高斯模糊增强（模拟边缘虚化/对焦差）
- 新增 --pos_loss_weight 参数，在 balance=class_weight 时可放大 drip 类在 loss 中的权重
  （用于在控制好误报的前提下，进一步提升检出能力）

相对原版的新增改动：
- 为负类(nodrip)增加轻微亮度/对比度增强（幅度较小），提高对异常光照/曝光的鲁棒性
"""

import os
import sys
import time
import copy
import argparse
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision.io import read_video
from torchvision.transforms import functional as TF


# --------------------------- 工具 ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_videos(root: Path, exts=(".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    files = []
    for ext in exts:
        files += list(root.rglob(f"*{ext}"))
    return sorted(files)


def count_labels(samples: List[Tuple[Path, int]], num_classes: int = 2):
    counts = [0] * num_classes
    for _, y in samples:
        counts[y] += 1
    return counts


def parse_3floats(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("需要三个逗号分隔的数值，例如: '0.5,0.5,0.5'")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("mean/std 必须是数值，例如: '0.5,0.5,0.5'")
    return vals  # type: ignore


# --------------------------- 数据集 ---------------------------

class VideoFolderDataset(Dataset):
    """
    读取 root/{split}/{class}/*.mp4
    - 均匀采样 T 帧（不足补尾帧），train 时可加入时间抖动（temporal jitter）
    - 若指定 size_h/size_w：直接 resize 到该非方形尺寸，仅可选水平翻转；
      否则走：短边缩放 + 随机/中心裁剪为 crop_size×crop_size。
    - 输出 (C,T,H,W) float32，[0,1] → 标准化
    - train split：
        * 正类(drip)：灰度化、亮度/对比度扰动、轻微模糊增强（可选）
        * 负类(nodrip)：轻微亮度/对比度扰动（幅度更小）
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        classes: Tuple[str, str] = ("nodrip", "drip"),
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
        # —— 正类(steam)增强参数 ——
        pos_gray_p: float = 0.35,
        pos_jitter_p: float = 0.30,
        pos_jitter_brightness: float = 0.10,
        pos_jitter_contrast: float = 0.10,
        # —— 时间采样 jitter（仅 train 有效）——
        temporal_jitter: bool = True,
        # —— 正类轻微模糊增强（仅 train 有效）——
        pos_blur_p: float = 0.20,
        pos_blur_ksize: int = 3,
        # —— 负类(no_steam) 轻微亮度/对比度增强 ——
        neg_jitter_p: float = 0.10,
        neg_jitter_brightness: float = 0.05,
        neg_jitter_contrast: float = 0.05,
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
        # 固定类别顺序，确保 drip=1
        self.classes = ("nodrip", "drip") if set(classes) == {"nodrip", "drip"} else tuple(sorted(classes))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.pos_idx = self.class_to_idx.get("drip", 1)
        self.allow_corrupt_skip = allow_corrupt_skip
        self.train_scale_jitter = train_scale_jitter
        self.hflip_prob = hflip_prob if split == "train" else 0.0

        # 正类(drip)增强参数
        self.pos_gray_p = float(max(0.0, min(1.0, pos_gray_p)))
        self.pos_jitter_p = float(max(0.0, min(1.0, pos_jitter_p)))
        self.pos_jitter_brightness = float(max(0.0, pos_jitter_brightness))
        self.pos_jitter_contrast = float(max(0.0, pos_jitter_contrast))

        # 负类(nodrip)增强参数（幅度较小）
        self.neg_jitter_p = float(max(0.0, min(1.0, neg_jitter_p)))
        self.neg_jitter_brightness = float(max(0.0, neg_jitter_brightness))
        self.neg_jitter_contrast = float(max(0.0, neg_jitter_contrast))

        # 时间 jitter/模糊只在 train 生效
        self.temporal_jitter = bool(temporal_jitter) and (split == "train")
        self.pos_blur_p = float(max(0.0, min(1.0, pos_blur_p)))
        self.pos_blur_ksize = int(pos_blur_ksize)

        self.samples: List[Tuple[Path, int]] = []
        for c in self.classes:
            cdir = self.root / split / c
            vids = list_videos(cdir)
            for v in vids:
                self.samples.append((v, self.class_to_idx[c]))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No videos found in {self.root}/{split}/({self.classes}).")

        if self.allow_corrupt_skip:
            keep = []
            for p, y in self.samples:
                try:
                    vframes, _, _ = read_video(str(p), pts_unit="sec", output_format="TCHW", start_pts=0, end_pts=0.1)
                    if vframes.numel() == 0:
                        continue
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
        """
        基础：均匀采样 num 个索引。
        若 jitter=True，则在均匀位置附近加一点随机抖动（仅 train 使用），
        让同一段视频每次看到的帧序列略有不同。
        """
        if total <= 0:
            return np.zeros((num,), dtype=np.int64)

        if total >= num:
            idx = np.linspace(0, total - 1, num=num, dtype=np.float32)
            if jitter:
                # 根据步长确定抖动范围（最多约 30% 步长）
                step = (total - 1) / max(1, num - 1)
                max_jitter = max(1.0, step * 0.3)
                noise = np.random.uniform(-max_jitter, max_jitter, size=num)
                idx = np.clip(idx + noise, 0, total - 1)
            return np.round(idx).astype(np.int64)
        else:
            # 帧数不够则重复最后一帧
            base = list(range(total))
            base += [total - 1] * (num - total)
            return np.array(base, dtype=np.int64)

    def _load_video_tensor(self, path: Path) -> torch.Tensor:
        vframes, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
        if vframes.numel() == 0:
            raise RuntimeError("Empty video tensor.")
        if vframes.shape[1] == 1:
            vframes = vframes.repeat(1, 3, 1, 1)
        return vframes  # (T,C,H,W)

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
        # 均匀采样 frames 帧（train 可加时间抖动）
        idxs = self._uniform_indices(T, self.frames, jitter=self.temporal_jitter)
        v = v[idxs]  # (frames, C, H, W)

        # 非方形/方形路径
        if self.size_h is not None and self.size_w is not None:
            # 注意：如果你显式传入 size_h/size_w，将不会启用随机缩放+裁剪
            v = TF.resize(v, [self.size_h, self.size_w], antialias=True)
            if self.split == "train" and random.random() < self.hflip_prob:
                v = torch.flip(v, dims=[-1])
        else:
            if self.split == "train":
                # 随机缩放 + 随机裁剪（推荐启用，提升空间泛化）
                scale = random.uniform(self.train_scale_jitter[0], self.train_scale_jitter[1])
                target_ss = max(64, int(self.short_side * scale))
                v = TF.resize(v, target_ss, antialias=True)
                _, _, H2, W2 = v.shape
                if H2 < self.crop_size or W2 < self.crop_size:
                    min_ss = max(self.crop_size, min(H2, W2))
                    v = TF.resize(v, min_ss, antialias=True)
                    _, _, H2, W2 = v.shape
                top = random.randint(0, H2 - self.crop_size)
                left = random.randint(0, W2 - self.crop_size)
                v = TF.crop(v, top, left, self.crop_size, self.crop_size)
                if random.random() < self.hflip_prob:
                    v = torch.flip(v, dims=[-1])
            else:
                v = TF.resize(v, self.short_side, antialias=True)
                v = TF.center_crop(v, [self.crop_size, self.crop_size])

        # (frames,C,H,W) -> (C,T,H,W)
        v = v.permute(1, 0, 2, 3).contiguous()
        v = v.float() / 255.0

        # ===== 负类(nodrip) 轻微亮度/对比度增强 =====
        if self.split == "train" and int(label) != self.pos_idx and self.neg_jitter_p > 0:
            if random.random() < self.neg_jitter_p:
                c_delta = (random.random() * 2 - 1) * self.neg_jitter_contrast  # 幅度更小
                b_delta = (random.random() * 2 - 1) * self.neg_jitter_brightness
                v = torch.clamp(v * (1.0 + c_delta) + b_delta, 0.0, 1.0)

        # ===== 正类(drip)专用增强：亮度/对比度、灰度化、轻微模糊 =====
        if self.split == "train" and int(label) == self.pos_idx:
            # 亮度/对比度扰动
            if self.pos_jitter_p > 0 and random.random() < self.pos_jitter_p:
                c_delta = (random.random() * 2 - 1) * self.pos_jitter_contrast  # [-a,+a]
                b_delta = (random.random() * 2 - 1) * self.pos_jitter_brightness
                v = torch.clamp(v * (1.0 + c_delta) + b_delta, 0.0, 1.0)

            # 概率性灰度化（保持3通道，通道值相同）
            if self.pos_gray_p > 0 and random.random() < self.pos_gray_p:
                gray = (0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2])  # (T,H,W)
                v = torch.stack([gray, gray, gray], dim=0)

            # 轻微高斯模糊（模拟低落边缘模糊/对焦差）
            if self.pos_blur_p > 0 and random.random() < self.pos_blur_p:
                k = max(1, int(self.pos_blur_ksize))
                if k % 2 == 0:
                    k += 1
                for t in range(v.shape[1]):
                    v[:, t] = TF.gaussian_blur(v[:, t], kernel_size=k)

        # 标准化
        v = (v - self.mean) / self.std
        return v, torch.tensor(label, dtype=torch.long)


# --------------------------- 模型构建（含预训练） ---------------------------

def build_model(arch: str, frames: int, spatial_size: int, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    构建视频分类模型

    Args:
        arch: 模型架构名称
        frames: 时间帧数
        spatial_size: 空间分辨率（用于模型构建，非矩形输入时应传入较小的维度）
        num_classes: 类别数
        pretrained: 是否使用预训练权重
    """
    arch = arch.lower()
    if arch in {"x3d_s", "x3d_m"}:
        from pytorchvideo.models import hub as pv_hub
        model = getattr(pv_hub, arch)(pretrained=pretrained)
        if hasattr(model.blocks[-1], "proj") and isinstance(model.blocks[-1].proj, nn.Linear):
            in_feats = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
        else:
            head = model.blocks[-1]
            proj = None
            for _, m in head.named_modules():
                if isinstance(m, nn.Linear):
                    proj = m
                    break
            if proj is None:
                raise RuntimeError("未找到X3D分类头线性层，请升级 pytorchvideo 或改用 torchvision 模型。")
            in_feats = proj.in_features
            model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
        return model
    elif arch in {"r2plus1d_18", "r3d_18"}:
        from torchvision.models.video import r2plus1d_18, r3d_18
        from torchvision.models.video import R2Plus1D_18_Weights, R3D_18_Weights
        if arch == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            model = r2plus1d_18(weights=weights)
        else:
            weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
            model = r3d_18(weights=weights)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    else:
        raise ValueError(f"未知 arch: {arch}. 可选: x3d_s, x3d_m, r2plus1d_18, r3d_18")


def set_backbone_trainable(model: nn.Module, trainable: bool, arch: str):
    for p in model.parameters():
        p.requires_grad = trainable
    if arch.startswith("x3d"):
        for p in model.blocks[-1].parameters():
            p.requires_grad = True
    else:
        for p in model.fc.parameters():
            p.requires_grad = True


def get_head_parameters(model: nn.Module, arch: str):
    return list(model.blocks[-1].parameters()) if arch.startswith("x3d") else list(model.fc.parameters())


# --------------------------- EMA / TTA / Metrics ---------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            v = esd[k]
            mv = msd[k]
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                esd[k].mul_(d).add_(mv.detach(), alpha=1 - d)
            else:
                esd[k].copy_(mv)


@torch.no_grad()
def _forward_with_tta(model: nn.Module, x: torch.Tensor, tta_flip: bool):
    logits = model(x)
    if tta_flip:
        x_flip = torch.flip(x, dims=[-1])
        logits = logits + model(x_flip)
        logits = logits / 2.0
    return logits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda", tta_flip: bool = False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == "cuda"))
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = _forward_with_tta(model, x, tta_flip)
            loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


@torch.no_grad()
def evaluate_detailed(model: nn.Module, loader: DataLoader, device: str = "cuda", tta_flip: bool = False):
    """返回详细指标并打印：混淆矩阵/各类P/R/F1；扫描阈值优化 steam 的 F1 与 Balanced Acc。"""
    model.eval()
    all_probs1, all_labels = [], []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == "cuda"))
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        with amp_ctx:
            logits = _forward_with_tta(model, x, tta_flip)
            probs = torch.softmax(logits.float(), dim=1)
        all_probs1.append(probs[:, 1].cpu())
        all_labels.append(y)
    p1 = torch.cat(all_probs1).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    def metrics_at(th):
        y_pred = (p1 >= th).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        acc = (tp + tn) / max(1, len(y_true))
        prec1 = tp / max(1, tp + fp)
        rec1 = tp / max(1, tp + fn)
        f1_1 = 2 * prec1 * rec1 / max(1e-12, (prec1 + rec1))
        prec0 = tn / max(1, tn + fn)
        rec0 = tn / max(1, tn + fp)
        f1_0 = 2 * prec0 * rec0 / max(1e-12, (prec0 + rec0))
        bal_acc = 0.5 * (rec0 + rec1)
        cm = np.array([[tn, fp],
                       [fn, tp]], dtype=int)
        return acc, bal_acc, (prec0, rec0, f1_0), (prec1, rec1, f1_1), cm

    # 0.5 默认与最佳阈值
    acc50, bal50, cls0_50, cls1_50, cm50 = metrics_at(0.5)
    best_f1_th, best_f1 = 0.5, -1
    best_bal_th, best_bal = 0.5, -1
    for th in np.linspace(0.05, 0.95, 91):
        acc, bal, _, cls1, _ = metrics_at(th)
        f1 = cls1[2]
        if f1 > best_f1:
            best_f1, best_f1_th = f1, th
        if bal > best_bal:
            best_bal, best_bal_th = bal, th

    print("== Detailed Validation Metrics ==")
    print(f"Default th=0.50 | Acc={acc50:.4f} | BalancedAcc={bal50:.4f} | "
          f"Class0(P/R/F1)={cls0_50[0]:.3f}/{cls0_50[1]:.3f}/{cls0_50[2]:.3f} | "
          f"Class1(P/R/F1)={cls1_50[0]:.3f}/{cls1_50[1]:.3f}/{cls1_50[2]:.3f}")
    print(f"Confusion Matrix @0.50 (rows=true [no_steam,steam]; cols=pred [no_steam,steam]):\n{cm50}")
    print(f"Best F1(steam=1) th={best_f1_th:.2f} | F1={best_f1:.4f}")
    print(f"Best Balanced Acc th={best_bal_th:.2f} | BalancedAcc={best_bal:.4f}")

    return {
        "acc@0.5": acc50,
        "balanced@0.5": bal50,
        "cm@0.5": cm50,
        "best_f1_th": best_f1_th,
        "best_bal_th": best_bal_th,
    }


# --------------------------- 训练主函数 ---------------------------

def main():
    warnings.filterwarnings("once", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="数据根目录，包含 train/ val/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze_epochs", type=int, default=3, help="线性探测epoch数，仅训分类头")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--frames", type=int, default=36)

    # 方形路径（兼容）
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--short_side", type=int, default=256)

    # 非方形尺寸（若提供则覆盖方形流程）
    # 注意：宽度至少需要 112+ 以避免池化层报错
    parser.add_argument("--size_h", type=int, default=480)
    parser.add_argument("--size_w", type=int, default=224)

    parser.add_argument("--arch", type=str, default="x3d_m", choices=["x3d_s", "x3d_m", "r2plus1d_18", "r3d_18"])
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_head_mul", type=float, default=10.0)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--ls", type=float, default=0.05)
    parser.add_argument("--balance", type=str, default="auto", choices=["off", "sampler", "class_weight", "auto"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="outputs/drip/best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # mean/std（逗号分隔）
    parser.add_argument("--mean", type=parse_3floats, default=(0.05, 0.05, 0.05))
    parser.add_argument("--std", type=parse_3floats, default=(0.15, 0.15, 0.15))

    # TTA / EMA
    parser.add_argument("--tta_flip", type=int, default=0)
    parser.add_argument("--ema", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # 正类(drip)增强参数
    parser.add_argument("--pos_gray_p", type=float, default=0.35)
    parser.add_argument("--pos_jitter_p", type=float, default=0.30)
    parser.add_argument("--pos_jitter_brightness", type=float, default=0.10)
    parser.add_argument("--pos_jitter_contrast", type=float, default=0.10)
    parser.add_argument("--temporal_jitter", type=int, default=1, help="train 时帧采样是否加入时间抖动(0/1)")
    parser.add_argument("--pos_blur_p", type=float, default=0.20, help="正类模糊增强概率")
    parser.add_argument("--pos_blur_ksize", type=int, default=3, help="正类模糊核大小(奇数)")

    # 负类(nodrip)轻微增强参数
    parser.add_argument("--neg_jitter_p", type=float, default=0.10,
                        help="负类(nodrip)亮度/对比度扰动概率")
    parser.add_argument("--neg_jitter_brightness", type=float, default=0.05,
                        help="负类亮度扰动幅度(±)")
    parser.add_argument("--neg_jitter_contrast", type=float, default=0.05,
                        help="负类对比度扰动幅度(±)")

    # loss 中正类额外权重（用于在 balance=class_weight 时增强 drip 类的影响）
    parser.add_argument("--pos_loss_weight", type=float, default=1.0,
                        help="drip 类在 loss 中的额外权重系数，>1 会更偏向提高检出(仅 balance=class_weight 生效)")

    args = parser.parse_args()

    # 输入尺寸处理
    if args.size_h is None and args.size_w is None:
        args.size_h, args.size_w = args.size, args.size
    elif (args.size_h is None) != (args.size_w is None):
        raise ValueError("size_h 和 size_w 要么都给，要么都不给。")

    # 校验最小分辨率（模型池化层需要）
    min_dim = min(args.size_h, args.size_w)
    if min_dim < 112:
        raise ValueError(
            f"非矩形输入的最小维度必须 ≥112！\n"
            f"当前设置: {args.size_h}×{args.size_w}，最小维度={min_dim}\n"
            f"X3D模型的池化层需要至少 7×7 的特征图，输入太小会导致池化错误。\n"
            f"建议: --size_h 224 --size_w 224 或更大（如 --size_h 480 --size_w 480）"
        )

    set_seed(args.seed)

    device = args.device
    print(f"Device: {device}")
    print("Enabling TF32 for speed (if Ampere+ GPU).")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 数据集（类别固定为 nodrip/drip）
    classes = ("nodrip", "drip")
    print(f"[Input Size] H×W = {args.size_h}×{args.size_w}")
    print(f"[Norm] mean={args.mean} std={args.std}")
    print(f"[PosAug] gray_p={args.pos_gray_p} jitter_p={args.pos_jitter_p} "
          f"Δbright=±{args.pos_jitter_brightness} Δcontrast=±{args.pos_jitter_contrast} "
          f"blur_p={args.pos_blur_p} blur_ks={args.pos_blur_ksize}")
    print(f"[NegAug] jitter_p={args.neg_jitter_p} "
          f"Δbright=±{args.neg_jitter_brightness} Δcontrast=±{args.neg_jitter_contrast}")
    print(f"[Temporal Jitter] enabled={bool(args.temporal_jitter)}")

    train_set = VideoFolderDataset(
        root=args.root, split="train", classes=classes,
        frames=args.frames,
        short_side=args.short_side, crop_size=args.size,
        size_h=args.size_h, size_w=args.size_w,
        mean=args.mean, std=args.std,
        pos_gray_p=args.pos_gray_p,
        pos_jitter_p=args.pos_jitter_p,
        pos_jitter_brightness=args.pos_jitter_brightness,
        pos_jitter_contrast=args.pos_jitter_contrast,
        temporal_jitter=bool(args.temporal_jitter),
        pos_blur_p=args.pos_blur_p,
        pos_blur_ksize=args.pos_blur_ksize,
        neg_jitter_p=args.neg_jitter_p,
        neg_jitter_brightness=args.neg_jitter_brightness,
        neg_jitter_contrast=args.neg_jitter_contrast,
    )
    val_set = VideoFolderDataset(
        root=args.root, split="val", classes=classes,
        frames=args.frames,
        short_side=args.short_side, crop_size=args.size,
        size_h=args.size_h, size_w=args.size_w,
        mean=args.mean, std=args.std,
        # 验证集不做增强/时间抖动（仍需传参但不会触发，因为 split != 'train'）
        pos_gray_p=args.pos_gray_p,
        pos_jitter_p=args.pos_jitter_p,
        pos_jitter_brightness=args.pos_jitter_brightness,
        pos_jitter_contrast=args.pos_jitter_contrast,
        temporal_jitter=False,
        pos_blur_p=args.pos_blur_p,
        pos_blur_ksize=args.pos_blur_ksize,
        neg_jitter_p=args.neg_jitter_p,
        neg_jitter_brightness=args.neg_jitter_brightness,
        neg_jitter_contrast=args.neg_jitter_contrast,
    )

    print(f"[Data] train={len(train_set)}  val={len(val_set)}  label_counts(train)={train_set.label_counts}")

    # 不均衡 & loss 权重
    sampler = None
    class_weight_tensor = None

    counts = np.array(train_set.label_counts, dtype=np.float64) + 1e-6

    if args.balance in ("sampler", "auto"):
        inv_freq = 1.0 / counts
        sample_weights = [inv_freq[y] for _, y in train_set.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    if args.balance == "class_weight":
        base_weights = counts.sum() / counts
        class_weight_tensor = torch.tensor(base_weights, dtype=torch.float32)
        if args.pos_loss_weight != 1.0 and len(base_weights) > 1:
            print(f"[Loss] 额外放大 drip 类权重: ×{args.pos_loss_weight}")
            class_weight_tensor[1] = class_weight_tensor[1] * args.pos_loss_weight

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.workers > 0), prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, args.batch // 2), shuffle=False,
        num_workers=max(0, args.workers // 2), pin_memory=True, drop_last=False,
        persistent_workers=False,
    )

    # 模型
    # 关键：当使用非方形输入时，模型初始化应使用实际输入分辨率的最小维度
    model_spatial_size = min(args.size_h, args.size_w) if (args.size_h and args.size_w) else args.size_h
    model = build_model(args.arch, args.frames, model_spatial_size, num_classes=2, pretrained=bool(args.pretrained)).to(device)

    # 线性探测：先冻结 backbone
    set_backbone_trainable(model, trainable=False, arch=args.arch)
    head_params = get_head_parameters(model, args.arch)
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    param_groups = [{"params": head_params, "lr": args.lr * args.lr_head_mul}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)

    # Scheduler
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_epochs = max(0, min(args.warmup, args.epochs - 1))
    sched_main = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                                         sched_main], milestones=[warmup_epochs]) if warmup_epochs > 0 else sched_main

    # Loss
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.ls,
        weight=class_weight_tensor.to(device) if class_weight_tensor is not None else None
    )

    # AMP & EMA
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == "cuda"))
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.ckpt) if os.path.dirname(args.ckpt) else ".", exist_ok=True)

    ckpt_dir = os.path.join(os.path.dirname(args.ckpt), "checkpoints_v2")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"检查点将保存在: {ckpt_dir}")

    # 训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = running_acc = seen = 0

        # 解冻 backbone，开始 finetune
        if epoch == args.freeze_epochs + 1:
            print(f"===> Unfreezing backbone for finetuning from epoch {epoch}.")
            set_backbone_trainable(model, trainable=True, arch=args.arch)
            head_params = get_head_parameters(model, args.arch)
            head_ids = {id(p) for p in head_params}
            backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
            optimizer = torch.optim.AdamW(
                [{"params": head_params, "lr": args.lr * args.lr_head_mul},
                 {"params": backbone_params, "lr": args.lr}],
                lr=args.lr, weight_decay=args.wd
            )
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1))

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            if step == 1 and epoch == 1:
                print(f"[Sanity] x.dtype={x.dtype}, param.dtype={next(model.parameters()).dtype}, x.shape={x.shape}")

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            if ema:
                ema.update(model)

            bs = y.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(dim=1) == y).sum().item()
            seen += bs

            if step % 10 == 0 or step == len(train_loader):
                lr0 = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | "
                      f"LR {lr0:.2e} | Loss {(running_loss / seen):.4f} | Acc {(running_acc / seen):.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        # 验证（优先用EMA模型）
        eval_model = ema.ema if ema else model
        val_acc, val_loss = evaluate(eval_model, val_loader, device=device, tta_flip=bool(args.tta_flip))
        dt = time.time() - t0
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | time={dt:.1f}s {'(EMA+TTA)' if ema or args.tta_flip else ''}")

        # 每轮保存检查点
        epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        epoch_ckpt = {
            "epoch": epoch,
            "state_dict": eval_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "args": vars(args),
            "classes": classes,
            "arch": args.arch,
            "is_ema": bool(ema)
        }
        torch.save(epoch_ckpt, epoch_ckpt_path)
        print(f"✅ 保存第 {epoch} 轮检查点到 {epoch_ckpt_path}")

        # 保存最佳（这里仍用 val_acc；你也可以改成 balanced acc）
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt = {
                "epoch": epoch,
                "state_dict": eval_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": best_acc,
                "val_loss": val_loss,
                "args": vars(args),
                "classes": classes,
                "arch": args.arch,
                "is_ema": bool(ema)
            }
            torch.save(best_ckpt, args.ckpt)
            print(f"✅ 更新最佳检查点: {args.ckpt} (val_acc={best_acc:.4f})")

    print(f"训练完成。最佳验证准确率 = {best_acc:.4f}")

    # 结束时输出详细指标 + 推荐阈值（可以用于推理时调高检出率）
    eval_model = ema.ema if ema else model
    metrics = evaluate_detailed(eval_model, val_loader, device=device, tta_flip=bool(args.tta_flip))
    print(f"👉 建议根据业务场景，在 best_f1_th={metrics['best_f1_th']:.2f} "
          f"或 best_bal_th={metrics['best_bal_th']:.2f} 附近调节阈值以平衡误报与低落检出。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
