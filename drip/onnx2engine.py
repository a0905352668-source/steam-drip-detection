#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_trt_engine_to_engine_fixed.py
- 从 ONNX 构建 TensorRT 引擎并保存为 .engine（与 .plan 等价，仅扩展名不同）
- 显式批模式；为 (B,3,T,H,W) 配置动态 profile；可开启 FP16
"""

import os
from pathlib import Path

# ==================== 可修改的固定参数 ====================
PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_PATH = str(PROJECT_ROOT / "checkpoints_v4.28.1/OKepoch_40.onnx")
ENGINE_PATH = "outputs/drip/exported.engine"
INPUT_NAME   = "video"                        # ONNX 输入名（确保与导出一致）
B_MIN, B_OPT, B_MAX = 1, 1, 4
T_MIN, T_OPT, T_MAX = 8, 36, 64
H_FIXED = 480
W_FIXED = 224
USE_FP16   = True       # 显卡支持就开（3090/3080等）
WORKSPACE_MB = 4096     # 工作空间（可按需调大）
# ========================================================

def build_engine(onnx_path: str, engine_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flag)
    config  = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1<<20))

    if USE_FP16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ Enable FP16")

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX.")

    # 打印 IO
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"[Input{i}] name={t.name} shape={t.shape} dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"[Output{i}] name={t.name} shape={t.shape} dtype={t.dtype}")

    # 动态 profile
    input_tensor = network.get_input(0)
    assert input_tensor.name == INPUT_NAME, f"输入名不一致：{input_tensor.name} vs {INPUT_NAME}"
    profile = builder.create_optimization_profile()
    min_shape = (B_MIN, 3, T_MIN, H_FIXED, W_FIXED)
    opt_shape = (B_OPT, 3, T_OPT, H_FIXED, W_FIXED)
    max_shape = (B_MAX, 3, T_MAX, H_FIXED, W_FIXED)
    profile.set_shape(INPUT_NAME, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print(f"[Profile] min={min_shape} opt={opt_shape} max={max_shape}")

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(bytes(engine))
    print(f"✅ Saved TensorRT engine to: {engine_path}")

def main():
    Path(ENGINE_PATH).parent.mkdir(parents=True, exist_ok=True)
    assert os.path.exists(ONNX_PATH), f"ONNX not found: {ONNX_PATH}"
    build_engine(ONNX_PATH, ENGINE_PATH)

def parse_cli():
    import argparse
    global ONNX_PATH, ENGINE_PATH, H_FIXED, W_FIXED, T_MIN, T_OPT, T_MAX, USE_FP16
    parser = argparse.ArgumentParser(description="ONNX 转 TensorRT engine；需 NVIDIA CUDA 环境")
    parser.add_argument("--input", default=ONNX_PATH)
    parser.add_argument("--output", default=ENGINE_PATH)
    parser.add_argument("--size-h", type=int, default=H_FIXED)
    parser.add_argument("--size-w", type=int, default=W_FIXED)
    parser.add_argument("--frames", type=int, default=T_OPT)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.size_h, args.size_w) <= 0:
        parser.error("输入规格必须大于 0")
    ONNX_PATH, ENGINE_PATH = args.input, args.output
    H_FIXED, W_FIXED = args.size_h, args.size_w
    T_MIN = T_OPT = T_MAX = args.frames
    USE_FP16 = not args.fp32


if __name__ == "__main__":
    parse_cli()
    main()
