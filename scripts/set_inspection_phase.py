#!/usr/bin/env python3
"""Atomic file bridge for testing the idle/reference/inspect signal contract."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('phase', choices=('idle', 'reference', 'inspect'))
    parser.add_argument('--cycle-id', required=True)
    args = parser.parse_args()
    if not args.cycle_id:
        parser.error('cycle-id cannot be empty')
    args.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=args.path.parent, delete=False, encoding='utf-8') as handle:
        temporary = Path(handle.name)
        json.dump(dict(cycle_id=args.cycle_id, phase=args.phase), handle)
    try:
        os.replace(temporary, args.path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
