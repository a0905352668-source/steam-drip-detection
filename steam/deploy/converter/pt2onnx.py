#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_x3d_to_onnx_fixed.py
- 加载你训练得到的 X3D 视频分类模型
- 导出为 ONNX（支持动态 batch / 时间维T，可选）
"""

import os
from pathlib import Path
import torch
import torch.nn as nn

# ==================== 可修改的固定参数 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = str(PROJECT_ROOT / "checkpoints/best.pt")
ONNX_PATH = "outputs/steam/exported.onnx"
ARCH        = None                         # 留空=从ckpt里读；也可手动写 "x3d_m" / "x3d_s"
NUM_CLASSES = 2                            # 你的任务是二分类
FRAMES      = 48                           # 训练用帧数（T）；若导出动态T则仅作为样例输入
SIZE_H      = 256                          # 输入高（与你训练/预处理一致）
SIZE_W      = 256                          # 输入宽（与你训练/预处理一致）
OPSET       = 17                           # 建议 13~17
USE_DYNAMIC = True                         # 是否导出动态维度（batch、T）
INPUT_NAME  = "video"
OUTPUT_NAME = "logits"
# ========================================================


def build_x3d(variant: str, num_classes: int) -> nn.Module:
    """用 pytorchvideo 构建 X3D，并替换分类头到 num_classes。"""
    variant = variant.lower()
    assert variant in {"x3d_s", "x3d_m"}
    from pytorchvideo.models import hub as pv_hub  # 需要 pip install pytorchvideo
    model = getattr(pv_hub, variant)(pretrained=False)
    # 替换分类头
    if hasattr(model.blocks[-1], "proj") and isinstance(model.blocks[-1].proj, nn.Linear):
        in_feats = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
    else:
        head = model.blocks[-1]
        proj = None
        for _, m in head.named_modules():
            if isinstance(m, nn.Linear):
                proj = m; break
        if proj is None:
            raise RuntimeError("未找到 X3D 分类头线性层。")
        in_feats = proj.in_features
        model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
    return model


def main():
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    # 读 ckpt
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) or {}
    arch = (ARCH or args.get("arch", "x3d_m")).lower()
    classes = ckpt.get("classes", ("negative", "positive"))
    print(f"[ckpt] arch={arch} classes={classes}  best_acc={ckpt.get('best_acc', None)}")

    # 构建模型并 load 权重（ckpt 保存的是 state_dict）
    model = build_x3d(arch, num_classes=NUM_CLASSES)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    if missing or unexpected:
        print(f"[load_state_dict] missing={missing} unexpected={unexpected}")
    model.eval()

    # 样例输入 (B, C, T, H, W)
    B = 1
    x = torch.randn(B, 3, FRAMES, SIZE_H, SIZE_W, dtype=torch.float32)

    # 动态轴：batch、T（时间维），H/W 固定（与你预处理/训练保持一致）
    dynamic_axes = None
    if USE_DYNAMIC:
        dynamic_axes = {
            INPUT_NAME:  {0: "batch", 2: "time"},
            OUTPUT_NAME: {0: "batch"}
        }

    torch.onnx.export(
        model, x, ONNX_PATH,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        export_params=True, do_constant_folding=True,
        opset_version=OPSET, dynamic_axes=dynamic_axes
    )
    print(f"✅ Exported ONNX to: {ONNX_PATH}")
    # 简单检查
    try:
        import onnx
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX check passed.")
    except ImportError:
        print("未安装 onnx，跳过结构检查")


def parse_cli():
    import argparse
    global CKPT_PATH, ONNX_PATH, FRAMES, SIZE_H, SIZE_W, USE_DYNAMIC
    parser = argparse.ArgumentParser(description="导出 X3D checkpoint 为 ONNX")
    parser.add_argument("--checkpoint", default=CKPT_PATH)
    parser.add_argument("--output", default=ONNX_PATH)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--size-h", type=int, default=SIZE_H)
    parser.add_argument("--size-w", type=int, default=SIZE_W)
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.size_h, args.size_w) <= 0:
        parser.error("输入规格必须大于 0")
    CKPT_PATH, ONNX_PATH = args.checkpoint, args.output
    FRAMES, SIZE_H, SIZE_W = args.frames, args.size_h, args.size_w
    USE_DYNAMIC = not args.static


if __name__ == "__main__":
    parse_cli()
    main()
