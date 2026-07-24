#!/usr/bin/env python3
"""Download the released LiveLight weights from ModelScope."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


MODEL_ID = "wjm1029/LiveLight"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pretrained_weights" / "LiveLight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the released LiveLight weights from ModelScope."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modelscope_cli = shutil.which("modelscope")
    if modelscope_cli is None:
        raise SystemExit(
            "ModelScope is not installed. Install it first with: "
            "python -m pip install modelscope"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        modelscope_cli,
        "download",
        "--model",
        MODEL_ID,
        "--local_dir",
        str(output_dir),
    ]
    print(f"Downloading {MODEL_ID} to {output_dir}")
    subprocess.run(command, check=True)
    print(f"LiveLight weights are available at: {output_dir}")


if __name__ == "__main__":
    main()
