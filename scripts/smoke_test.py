#!/usr/bin/env python3
"""Synthetic video smoke test; checks execution and outputs, not detection accuracy."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/smoke")
    args = parser.parse_args()
    import cv2
    import numpy as np
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = output / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create an mp4v video")
    for index in range(48):
        frame = np.full((120, 160, 3), 45, dtype=np.uint8)
        cv2.circle(frame, (65, 15 + index * 2 % 90), 4, (210, 210, 210), -1)
        writer.write(frame)
    writer.release()

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)

    for task in ("steam", "drip"):
        run(f"{task}/preprocess.py", "--input", video, "--output", output / task / "pre")
    run("drip/preprocess_fast.py", "--input", video, "--output", output / "drip-fast/pre")
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    results = []
    for entry in manifest["assets"]:
        if entry["role"] not in ("default", "legacy-deployment") or Path(entry["path"]).suffix not in (".pt", ".onnx"):
            continue
        task = entry["task"]
        csv_path = output / task / (Path(entry["asset"]).stem + Path(entry["path"]).suffix + ".csv")
        run(f"{task}/inference.py", "--input", output / task / "pre",
            "--model", ROOT / entry["path"], "--output", csv_path,
            "--device", args.device, "--no-tta")
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise AssertionError(f"No predictions: {entry['path']}")
        for row in rows:
            if row["pred_label"] == "ERROR":
                raise AssertionError(row)
            probability = float(row[f"prob_{task}"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise AssertionError(row)
        results.append({"model": entry["path"], "predictions": len(rows), "ok": True})
    (output / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Smoke test passed:", output / "report.json")


if __name__ == "__main__":
    main()
