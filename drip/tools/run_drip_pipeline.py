#!/usr/bin/env python3
"""Run the complete drip-detection pipeline for one source video."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference
from preprocess import process_video_folder
from tools.split_videos import get_video_duration, split_video


DEFAULT_MODEL = PROJECT_ROOT / "checkpoints_v4.28.1" / "OKepoch_40.onnx"


def summarize_results(csv_path: Path, output_path: Path, threshold: float) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["pred_label"] == "ERROR":
                continue
            segment_name = Path(row["path"]).parent.name
            groups.setdefault(segment_name, []).append(row)

    summary = []
    for segment_name, rows in sorted(groups.items()):
        top = max(rows, key=lambda row: float(row["prob_drip"]))
        probabilities = [float(row["prob_drip"]) for row in rows]
        positive_count = sum(probability >= threshold for probability in probabilities)
        summary.append(
            {
                "segment": segment_name,
                "prediction": "drip" if positive_count else "no_drip",
                "max_prob_drip": f"{max(probabilities):.6f}",
                "mean_prob_drip": f"{sum(probabilities) / len(probabilities):.6f}",
                "positive_strips": positive_count,
                "total_strips": len(rows),
                "top_strip": Path(top["path"]).name,
            }
        )

    if not summary:
        raise RuntimeError("No successful strip predictions; inspect inference errors.")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5-second split -> strip preprocessing -> drip inference"
    )
    parser.add_argument("video", type=Path, help="Source video path")
    parser.add_argument("--output", type=Path, help="Pipeline output directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-strips", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.num_strips <= 0 or not 0 <= args.threshold <= 1:
        parser.error("num-strips must be positive; threshold must be within [0,1]")

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else video.parent / f"{video.stem}_drip_pipeline"
    )
    if output == video or output in video.parents:
        raise ValueError("Output must not contain the source video")
    if not args.model.expanduser().is_file():
        raise FileNotFoundError(f"Model not found: {args.model}; run scripts/download_models.py")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        shutil.rmtree(output)

    split_dir = output / "split"
    preprocessed_dir = output / "preprocessed"
    threshold_label = f"{args.threshold:g}"
    strip_csv = output / f"strip_results_threshold_{threshold_label}.csv"
    segment_csv = output / f"segment_results_threshold_{threshold_label}.csv"
    visualized_dir = output / "visualized_strip_boxes"

    duration = get_video_duration(video)
    if duration is None:
        raise RuntimeError(f"Cannot read video duration: {video}")
    if duration <= 5.2:
        split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, split_dir / f"{video.stem}_part_001{video.suffix}")
        success, failed = 1, 0
    else:
        success, failed = split_video(video, split_dir, duration=5)
    if failed or success == 0:
        raise RuntimeError(f"Video splitting failed: success={success}, failed={failed}")

    process_video_folder(
        str(split_dir),
        str(preprocessed_dir),
        video_extensions=[".mp4", ".avi", ".mov"],
        num_strips=args.num_strips,
        max_frames=None,
        short_window=10,
        long_window=30,
        blur_ksize=0,
        target_height=480,
        frame_stride=1,
        overlap=0.2,
    )

    inference.MODEL_PATH = str(args.model.expanduser().resolve())
    inference.INPUT_PATH = str(preprocessed_dir)
    inference.THRESHOLD = args.threshold
    inference.HIGH_THRESHOLD = args.threshold
    if args.device == "cuda" and not inference.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    inference.REQUIRE_CUDA = args.device == "cuda"
    inference.USE_CUDA_TORCH = args.device != "cpu" and inference.torch.cuda.is_available()
    inference.USE_CUDA_ONNX = inference.USE_CUDA_TORCH
    inference.SAVE_CSV = str(strip_csv)
    inference.SAVE_MINORITY_TXT = ""
    inference.MOVE_MINORITY_VIDEOS = False
    inference.LOG_MODE = "compact"
    inference.main()

    summary = summarize_results(strip_csv, segment_csv, args.threshold)

    visualizer = PROJECT_ROOT / "tools" / "visualize_drip_strips.py"
    visualized_dir.mkdir(parents=True, exist_ok=True)
    segment_videos = sorted(
        path
        for path in split_dir.iterdir()
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    )
    for segment_video in segment_videos:
        visualized_path = visualized_dir / f"{segment_video.stem}_visualized.mp4"
        subprocess.run(
            [
                sys.executable,
                str(visualizer),
                str(segment_video),
                str(strip_csv),
                str(visualized_path),
                "--threshold",
                str(args.threshold),
                "--num-strips",
                str(args.num_strips),
                "--allow-no-drip",
            ],
            check=True,
        )

    drip_count = sum(row["prediction"] == "drip" for row in summary)
    print(f"\nSegments: {len(summary)}, drip: {drip_count}, no_drip: {len(summary) - drip_count}")
    print(f"Segment results: {segment_csv}")
    print(f"Visualized videos: {visualized_dir}")


if __name__ == "__main__":
    main()
