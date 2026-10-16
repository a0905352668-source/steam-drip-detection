# -*- coding: utf-8 -*-
"""
快速版：将原视频按竖直方向切成若干 strip，
对整帧做“给分类模型用”的预处理（灰度 + 帧差 + 短/长 EMA 时间带通），
然后在带通后的 band 图上按竖直切条，为每个 strip 导出一个 3 通道 BGR 视频。

【本版优化点】
- 先把灰度图 resize 到 target_height，再在小分辨率上做 diff / EMA / 带通
- 支持 blur_ksize <=1 时完全跳过模糊
- 抽帧（frame_stride）默认用 2，进一步减小计算量

- 给模型看的，不追求好看，只追求“有滴水 vs 无滴滴水”在时空上更可分。
"""

import os
import cv2
import time
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path


def split_for_cls_fast(
    video_path: str,
    out_dir: str,
    num_strips: int = 10,
    max_frames: Optional[int] = None,

    # 🌟【核心参数1：短窗】突出近期运动，越小越“灵敏”，但噪声也越大
    short_window: int = 5,

    # 🌟【核心参数2：长窗】估计长期活动背景，用来压制“一直在抖的纹理”
    long_window: int = 30,

    # 模糊核大小；0 或 None 表示完全不模糊
    blur_ksize: int = 0,

    # 🌟【控制速度&分辨率】处理/输出的高度（宽度按比例自动计算）
    target_height: int = 360,

    # 🌟【控制速度】抽帧间隔，2 表示每 2 帧处理 1 帧
    frame_stride: int = 2,

    # 🌟【新参数】相邻 strip 之间的重叠比例，0~1，如 0.25 表示重叠 25%
    overlap: float = 0.25,
):
    """
    对单个视频做预处理并按竖直方向切成多个 strip，写出为多路 mp4。
    所有运算都在 resize 后的分辨率上进行，以减少计算量。
    """
    os.makedirs(out_dir, exist_ok=True)

    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap 必须在 [0,1) 之间，当前为 {overlap}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 兜底

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("读取第一帧失败")

    # 原始分辨率
    h, w = frame.shape[:2]

    # ===== 关键：确定“处理分辨率”，后续所有 diff/EMA 都在这里做 =====
    proc_h = target_height
    scale = proc_h / float(h)
    proc_w = int(w * scale + 0.5)

    out_h = proc_h
    out_w = proc_w

    # 计算每个 strip 的宽度（在处理分辨率空间）
    effective = (num_strips - 1) * (1.0 - overlap) + 1.0
    strip_w_out = out_w / effective
    step_out = strip_w_out * (1.0 - overlap)

    strip_w_out_int = int(round(strip_w_out))
    step_out_int = max(1, int(round(step_out)))  # 避免 0

    # 为每个 strip 建立 VideoWriter（注意尺寸是处理分辨率空间）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: List[Tuple[cv2.VideoWriter, int, int]] = []

    out_fps = fps / frame_stride if frame_stride > 0 else fps

    for i in range(num_strips):
        # 在处理分辨率空间计算切分位置
        x0_out = int(round(i * step_out_int))
        x1_out = x0_out + strip_w_out_int
        if x0_out >= out_w:
            # 理论上不会发生；如果发生说明参数比较极端，直接跳过
            break
        if x1_out > out_w:
            x1_out = out_w
        strip_w_actual = x1_out - x0_out
        if strip_w_actual <= 0:
            continue

        out_path = os.path.join(out_dir, f"strip_{i:02d}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                                 (strip_w_actual, out_h))
        writers.append((writer, x0_out, x1_out))

    if not writers:
        raise RuntimeError("未能创建任何 strip writer，请检查 num_strips / overlap 参数")

    # 整帧的 prev_gray / 短期&长期 EMA 累积（都在处理分辨率空间）
    prev_gray = None
    acc_short = None
    acc_long = None

    # 🌟 EMA 系数：近似窗口长度 N 的 EMA：alpha ≈ 1/N
    alpha_s = 1.0 / max(short_window, 1)  # 短期
    alpha_l = 1.0 / max(long_window, 1)   # 长期

    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    print(f"开始处理视频: {video_path}")

    while True:
        if frame_idx == 0:
            cur_frame = frame
        else:
            ret, cur_frame = cap.read()
            if not ret:
                break

        if max_frames is not None and frame_idx >= max_frames:
            break

        # 抽帧：只处理每 frame_stride 帧中的一帧
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # ---------- 整帧预处理：灰度 + 下采样 + 可选模糊 + 帧差 ----------
        gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        # 先缩放到处理分辨率，再做 diff/EMA
        gray_small = cv2.resize(
            gray, (proc_w, proc_h),
            interpolation=cv2.INTER_AREA,
        )

        # 如果 blur_ksize <= 1，就不做模糊，直接用原灰度
        if blur_ksize is None or blur_ksize <= 1:
            gray_proc = gray_small
        else:
            # 保证核大小为奇数
            k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            gray_proc = cv2.GaussianBlur(gray_small, (k, k), 0)

        if prev_gray is None:
            diff = np.zeros_like(gray_proc, dtype=np.float32)
        else:
            diff = cv2.absdiff(gray_proc, prev_gray).astype(np.float32)

        prev_gray = gray_proc

        # ---------- 短期 / 长期 EMA 累积 ----------
        if acc_short is None:
            # 第一帧直接初始化
            acc_short = diff.copy()
            acc_long = diff.copy()
        else:
            # 🌟 短期 EMA：强化“最近 short_window 帧”的变化
            acc_short = (1.0 - alpha_s) * acc_short + alpha_s * diff
            # 🌟 长期 EMA：估计“长期活动背景”（纹理抖动等）
            acc_long = (1.0 - alpha_l) * acc_long + alpha_l * diff

        # ---------- 🌟 时间带通：短期 - 长期 ----------
        band = acc_short - acc_long
        band[band < 0] = 0.0  # 负值截断

        # 归一化到 [0, 255]，得到“运动强度图”
        band_norm = cv2.normalize(
            band, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # band_norm 的尺寸已经是 out_h × out_w（处理分辨率），不再需要 resize
        # ---------- 在 band_norm 上按竖直切条 ----------
        for writer, x0_out, x1_out in writers:
            strip_band = band_norm[:, x0_out:x1_out]  # 单通道
            out_frame = cv2.cvtColor(strip_band, cv2.COLOR_GRAY2BGR)
            writer.write(out_frame)

        frame_idx += 1
        processed_frames += 1

    for writer, _, _ in writers:
        writer.release()
    cap.release()

    # 计算耗时
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"处理完成，总帧数 {frame_idx}，实际处理帧数 {processed_frames}")
    if minutes > 0:
        print(f"⏱️  耗时: {minutes}分{seconds:.2f}秒")
    else:
        print(f"⏱️  耗时: {seconds:.2f}秒")
    print(f"分类用 strip 视频输出目录: {out_dir}")

    return elapsed_time


def process_video_folder(
    input_path: str,
    out_base_dir: str,
    video_extensions: Optional[List[str]] = None,
    **kwargs  # split_for_cls_fast 的其他参数
):
    """
    处理文件夹或单个视频文件

    参数：
        input_path: 视频文件路径或包含视频的文件夹路径
        out_base_dir: 输出根目录
        video_extensions: 视频文件扩展名列表
        **kwargs: 传递给 split_for_cls_fast 的其他参数
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # 规范化扩展名
    video_extensions = [
        ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
        for ext in video_extensions
    ]

    input_path = Path(input_path)

    # 判断是文件还是文件夹
    if input_path.is_file():
        # 单个视频文件
        video_files = [input_path]
    elif input_path.is_dir():
        # 文件夹，查找所有视频文件
        video_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        video_files = sorted(video_files)
    else:
        raise ValueError(f"输入路径不存在或无效: {input_path}")

    if not video_files:
        raise FileNotFoundError(f"未找到视频文件: {input_path}")

    print("=" * 70)
    print(f"输入路径: {input_path}")
    print(f"输出根目录: {out_base_dir}")
    print(f"找到 {len(video_files)} 个视频文件")
    print("=" * 70 + "\n")

    success_count = 0
    failed_count = 0
    total_time = 0.0
    video_times = []  # 记录每个视频的耗时

    for idx, video_file in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] 处理视频: {video_file.name}")
        print("-" * 70)

        # 为每个视频创建单独的输出目录
        video_out_dir = os.path.join(out_base_dir, video_file.stem)

        try:
            video_start = time.time()
            elapsed = split_for_cls_fast(
                str(video_file),
                video_out_dir,
                **kwargs
            )
            video_elapsed = time.time() - video_start
            success_count += 1
            total_time += video_elapsed
            video_times.append((video_file.name, video_elapsed))
            print(f"✅ 成功处理: {video_file.name}")
        except Exception as e:
            failed_count += 1
            print(f"❌ 处理失败: {video_file.name}")
            print(f"   错误信息: {str(e)}")

    print("\n" + "=" * 70)
    print(f"处理完成！成功: {success_count} 个，失败: {failed_count} 个")

    # 显示时间统计
    if success_count > 0:
        print("\n⏱️  时间统计:")
        print("-" * 70)

        # 总耗时
        total_minutes = int(total_time // 60)
        total_seconds = total_time % 60
        if total_minutes > 0:
            print(f"总耗时: {total_minutes}分{total_seconds:.2f}秒")
        else:
            print(f"总耗时: {total_seconds:.2f}秒")

        # 平均耗时
        avg_time = total_time / success_count
        if avg_time >= 60:
            avg_minutes = int(avg_time // 60)
            avg_seconds = avg_time % 60
            print(f"平均耗时: {avg_minutes}分{avg_seconds:.2f}秒/视频")
        else:
            print(f"平均耗时: {avg_time:.2f}秒/视频")

        # 每个视频的详细耗时
        if len(video_times) > 0:
            print("\n各视频耗时详情:")
            for i, (name, t) in enumerate(video_times, 1):
                if t >= 60:
                    m = int(t // 60)
                    s = t % 60
                    print(f"  {i}. {name}: {m}分{s:.2f}秒")
                else:
                    print(f"  {i}. {name}: {t:.2f}秒")

    print("=" * 70)
    if failed_count:
        raise RuntimeError(f"{failed_count} 个视频预处理失败")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="滴水运动带通与竖条视频预处理")
    parser.add_argument("--input", required=True, help="单个原始视频或目录（不递归）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-strips", type=int, default=3)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument("--long-window", type=int, default=30)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if min(args.num_strips, args.short_window, args.long_window, args.target_height, args.frame_stride) <= 0:
        parser.error("数量、窗口、尺寸、步长必须大于 0")
    if not 0 <= args.overlap < 1 or args.long_window <= args.short_window:
        parser.error("overlap 必须在 [0,1)，long-window 必须大于 short-window")
    process_video_folder(str(Path(args.input).expanduser().resolve()),
                         str(Path(args.output).expanduser().resolve()),
                         num_strips=args.num_strips, max_frames=args.max_frames,
                         short_window=args.short_window, long_window=args.long_window,
                         blur_ksize=0, target_height=args.target_height,
                         frame_stride=args.frame_stride, overlap=args.overlap)


if __name__ == "__main__":
    main()
