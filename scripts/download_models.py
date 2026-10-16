#!/usr/bin/env python3
"""Download release assets using gh (also works with private repositories)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "steam", "drip"), default="all")
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repo", help="Override owner/repo for a fork")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    assets = [entry for entry in manifest["assets"]
              if (args.task == "all" or entry["task"] == args.task)
              and (args.include_historical or not entry["role"].startswith("historical"))]
    if not args.verify_only and not shutil.which("gh"):
        parser.error("Install GitHub CLI and run `gh auth login` first.")
    for entry in assets:
        target = ROOT / entry["path"]
        if target.is_file() and sha256(target) == entry["sha256"]:
            print("OK", entry["path"])
            continue
        if args.verify_only:
            raise SystemExit("Missing or checksum mismatch: " + entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            subprocess.run(["gh", "release", "download", manifest["tag"],
                            "--repo", args.repo or manifest["repository"],
                            "--pattern", entry["asset"], "--dir", temp], check=True)
            downloaded = Path(temp) / entry["asset"]
            if sha256(downloaded) != entry["sha256"]:
                raise SystemExit("Downloaded file checksum mismatch: " + entry["asset"])
            downloaded.replace(target)
        print("Downloaded and verified", entry["path"])


if __name__ == "__main__":
    main()
