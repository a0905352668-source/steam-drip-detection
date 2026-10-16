#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/model.py - X3D 视频分类模型构建
"""

import torch.nn as nn


def build_model(arch: str, frames: int, crop_size: int, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    构建视频分类模型

    Args:
        arch: 模型架构，可选 x3d_s, x3d_m, r2plus1d_18, r3d_18
        frames: 输入帧数
        crop_size: 裁剪尺寸
        num_classes: 分类数
        pretrained: 是否使用预训练权重

    Returns:
        nn.Module: 构建好的模型
    """
    arch = arch.lower()

    if arch in {"x3d_s", "x3d_m"}:
        from pytorchvideo.models import hub as pv_hub
        model = getattr(pv_hub, arch)(pretrained=pretrained)
        # 替换分类头
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
                raise RuntimeError("未找到 X3D 分类头线性层，请升级 pytorchvideo 或改用 torchvision 模型。")
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


def load_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = False):
    """
    加载检查点到模型

    Args:
        model: 模型实例
        ckpt_path: 检查点路径
        strict: 是否严格匹配

    Returns:
        dict: 检查点信息
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[load_checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")

    return ckpt
