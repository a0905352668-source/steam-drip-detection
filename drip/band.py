import cv2
import numpy as np

import argparse


def main():
    parser = argparse.ArgumentParser(description="检查滴水运动带通强度")
    parser.add_argument("video")
    args = parser.parse_args()
    video_path = args.video

    cap = cv2.VideoCapture(video_path)

    prev_gray = None
    acc_short = None
    acc_long = None
    alpha_s = 1.0 / 10
    alpha_l = 1.0 / 30

    frame_idx = 0
    max_check_frames = 3000  # 多看一些帧

    # 记录历史最大值，用于判断"突跳"
    history_max = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_check_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            diff = np.zeros_like(gray, dtype=np.float32)
        else:
            diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
        prev_gray = gray

        if acc_short is None:
            acc_short = diff.copy()
            acc_long = diff.copy()
        else:
            acc_short = (1.0 - alpha_s) * acc_short + alpha_s * diff
            acc_long  = (1.0 - alpha_l) * acc_long  + alpha_l * diff

        band = acc_short - acc_long
        band[band < 0] = 0.0

        max_val = band.max()
        mean_val = band.mean()
        history_max.append(max_val)

        # 只打印"显著高于近期平均水平"的帧（可能是水滴）
        if len(history_max) > 30:
            recent_avg = np.mean(history_max[-30:])
            if max_val > recent_avg * 2.0 and max_val > 15:  # 突然翻倍且绝对值>15
                print(f"⚡ 帧 {frame_idx:5d} | band最大={max_val:8.2f} | "
                      f"均值={mean_val:.4f} | 近30帧avg={recent_avg:.2f} | "
                      f"突跳倍数={max_val/recent_avg:.1f}x")

        frame_idx += 1

    cap.release()

    # 最后打印整体统计
    if not history_max:
        raise RuntimeError(f"未读取到视频帧: {video_path}")
    arr = np.array(history_max)
    print(f"\n===== 整体统计 =====")
    print(f"band最大值 - 最小: {arr.min():.2f}, 最大: {arr.max():.2f}, "
          f"中位数: {np.median(arr):.2f}, 95分位: {np.percentile(arr, 95):.2f}")


if __name__ == "__main__":
    main()
