#!/usr/bin/env python3
"""Draw positive strip regions on the corresponding source video."""

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def strip_geometry(
    width: int,
    height: int,
    strip_index: int,
    num_strips: int = 3,
    overlap: float = 0.2,
    target_height: int = 480,
) -> tuple[int, int, int]:
    proc_width = int(width * target_height / float(height) + 0.5)
    effective = (num_strips - 1) * (1.0 - overlap) + 1.0
    strip_width = int(round(proc_width / effective))
    step = max(1, int(round((proc_width / effective) * (1.0 - overlap))))
    x0_proc = int(round(strip_index * step))
    x1_proc = min(proc_width, x0_proc + strip_width)
    return proc_width, x0_proc, x1_proc


def strip_bounds(
    width: int,
    height: int,
    strip_index: int,
    num_strips: int = 3,
) -> tuple[int, int]:
    proc_width, x0_proc, x1_proc = strip_geometry(
        width, height, strip_index, num_strips=num_strips
    )
    x0 = int(round(x0_proc * width / proc_width))
    x1 = int(round(x1_proc * width / proc_width))
    return max(0, x0), min(width - 1, x1)


def motion_bbox(
    strip_video: Path,
    frame_width: int,
    frame_height: int,
    strip_index: int,
    num_strips: int = 3,
) -> Optional[tuple[int, int, int, int]]:
    capture = cv2.VideoCapture(str(strip_video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    if not frames:
        return None

    heat = np.percentile(np.stack(frames).astype(np.float32), 90, axis=0)
    threshold = max(10.0, float(np.percentile(heat, 99)))
    mask = (heat >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[component, cv2.CC_STAT_AREA] < 20:
        return None

    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    w = int(stats[component, cv2.CC_STAT_WIDTH])
    h = int(stats[component, cv2.CC_STAT_HEIGHT])
    strip_height, strip_width = frames[0].shape

    # Edge responses are often clipped by the strip crop. Add enough context
    # to include the physical drip immediately outside an overlapping boundary.
    pad_x = max(30, int(round(w * 1.5)))
    pad_y = max(24, int(round(h * 0.2)))
    local_x0 = x - pad_x
    local_x1 = x + w + pad_x
    local_y0 = max(0, y - pad_y)
    local_y1 = min(strip_height, y + h + pad_y)

    proc_width, strip_x0, _ = strip_geometry(
        frame_width, frame_height, strip_index, num_strips=num_strips
    )
    scale_x = frame_width / float(proc_width)
    scale_y = frame_height / float(strip_height)
    x0 = int(round((strip_x0 + local_x0) * scale_x))
    x1 = int(round((strip_x0 + local_x1) * scale_x))
    y0 = int(round(local_y0 * scale_y))
    y1 = int(round(local_y1 * scale_y))
    return (
        max(0, x0),
        max(0, y0),
        min(frame_width - 1, x1),
        min(frame_height - 1, y1),
    )


def result_strips(
    csv_path: Path, segment_name: str, threshold: float, draw_all: bool
) -> list[tuple[int, float, bool]]:
    matches = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            if path.parent.name != segment_name:
                continue
            probability = float(row["prob_drip"])
            is_drip = probability >= threshold
            if not draw_all and not is_drip:
                continue
            strip_index = int(path.stem.rsplit("_", 1)[-1])
            matches.append((strip_index, probability, is_drip))
    return sorted(matches)


def merged_regions(
    detections: list[tuple[int, float, bool]],
    width: int,
    height: int,
    num_strips: int,
    boxes: dict[int, tuple[int, int, int, int]],
    motion_localize: bool,
) -> list[tuple[list[int], float, bool, tuple[int, int, int, int]]]:
    regions = []
    group: list[tuple[int, float, bool]] = []

    def flush_group() -> None:
        if not group:
            return
        x0s, y0s, x1s, y1s = [], [], [], []
        for strip_index, _, _ in group:
            localized = boxes.get(strip_index) if motion_localize else None
            if localized is not None:
                x0, y0, x1, y1 = localized
            else:
                x0, x1 = strip_bounds(
                    width, height, strip_index, num_strips=num_strips
                )
                y0, y1 = 0, height - 1
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
        regions.append(
            (
                [item[0] for item in group],
                max(item[1] for item in group),
                group[0][2],
                (min(x0s), min(y0s), max(x1s), max(y1s)),
            )
        )

    for detection in detections:
        strip_index, _, is_drip = detection
        if (
            is_drip
            and group
            and group[-1][2]
            and strip_index == group[-1][0] + 1
        ):
            group.append(detection)
            continue
        flush_group()
        group = [detection]
    flush_group()
    return regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-strips", type=int, default=3)
    parser.add_argument("--all-strips", action="store_true")
    parser.add_argument("--motion-localize", action="store_true")
    parser.add_argument("--allow-no-drip", action="store_true")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    results_csv = args.results_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    detections = result_strips(
        results_csv, video.stem, args.threshold, args.all_strips
    )
    if not detections and not args.allow_no_drip:
        raise RuntimeError(f"No strips found for {video.stem}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output}")

    line_width = max(4, round(min(width, height) / 240))
    font_scale = max(1.0, min(width, height) / 900.0)
    label_thickness = max(2, line_width // 2)
    margin = max(12, line_width * 2)
    boxes = {}
    if args.motion_localize:
        for strip_index, _, is_drip in detections:
            if not is_drip:
                continue
            strip_video = None
            with results_csv.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    path = Path(row["path"])
                    if (
                        path.parent.name == video.stem
                        and path.stem == f"strip_{strip_index:02d}"
                    ):
                        strip_video = path
                        break
            if strip_video is not None:
                boxes[strip_index] = motion_bbox(
                    strip_video, width, height, strip_index, num_strips=args.num_strips
                )

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if not detections:
            label = "NO_DRIP"
            color = (0, 180, 0)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness
            )
            label_x = margin
            label_y = margin + text_height
            cv2.rectangle(
                frame,
                (label_x - margin // 2, label_y - text_height - margin // 2),
                (label_x + text_width + margin // 2, label_y + baseline + margin // 2),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA,
            )
        colors = [(255, 120, 0), (0, 0, 255), (0, 180, 0), (255, 0, 180)]
        regions = merged_regions(
            detections,
            width,
            height,
            args.num_strips,
            boxes,
            args.motion_localize,
        )
        for region_index, (strip_indexes, probability, is_drip, bounds) in enumerate(regions):
            x0, y0, x1, y1 = bounds
            color = colors[region_index % len(colors)]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, line_width)
            prediction = "DRIP" if is_drip else "NO_DRIP"
            if len(strip_indexes) == 1:
                strip_label = f"strip_{strip_indexes[0]:02d}"
            else:
                strip_label = f"strip_{strip_indexes[0]:02d}-{strip_indexes[-1]:02d}"
            label = f"{strip_label}  {prediction}  p={probability:.4f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness
            )
            label_x = min(max(x0 + margin, margin), width - text_width - margin)
            label_y = margin + text_height + region_index * (
                text_height + baseline + margin * 2
            )
            cv2.rectangle(
                frame,
                (label_x - margin // 2, label_y - text_height - margin // 2),
                (label_x + text_width + margin // 2, label_y + baseline + margin // 2),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA,
            )
        writer.write(frame)

    capture.release()
    writer.release()
    print(f"Output: {output}")
    for strip_index, probability, is_drip in detections:
        prediction = "drip" if is_drip else "no_drip"
        print(f"strip_{strip_index:02d}: {prediction}, drip={probability:.6f}")


if __name__ == "__main__":
    main()
