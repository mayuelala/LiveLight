import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.livelight_wrapper_perframe_ref import RelightPerFrameRefLive


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequence inference for stage3 per-frame reference refresh relighting."
    )
    parser.add_argument("--config-path", type=str, default="./configs/prompts/relight_perframe_ref.yaml")
    parser.add_argument(
        "--acceleration",
        type=str,
        default="xformers",
        choices=["none", "xformers"],
        help="Inference acceleration mode passed to the per-frame relight wrapper.",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing source/original video frames.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--depth-dir", type=str, default=None, help="Optional directory containing per-frame .npy depth maps.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--light-u", type=float, default=0.5)
    parser.add_argument("--light-v", type=float, default=0.35)
    parser.add_argument("--light-z-rel", type=float, default=1.0)
    parser.add_argument("--light-intensity", type=float, default=1.0)
    parser.add_argument("--light-r", type=float, default=1.0)
    parser.add_argument("--light-g", type=float, default=1.0)
    parser.add_argument("--light-b", type=float, default=1.0)
    return parser.parse_args()


def list_frames(input_dir: Path):
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES])


def maybe_load_depth(depth_dir: Path | None, frame_path: Path):
    if depth_dir is None:
        return None
    depth_path = depth_dir / f"{frame_path.stem}.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth map for frame {frame_path.name}: {depth_path}")
    return np.load(depth_path).astype(np.float32)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None

    frames = list_frames(input_dir)
    end_index = min(len(frames), args.start_frame + args.num_frames)
    frames = frames[args.start_frame:end_index]
    if not frames:
        raise ValueError("No input frames selected")

    output_frames_dir = output_dir / "frames"
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    runner = RelightPerFrameRefLive(args)
    try:
        images = [Image.open(frame_path).convert("RGB") for frame_path in frames]
        depth_maps = [maybe_load_depth(depth_dir, frame_path) for frame_path in frames]
        predictions, reports = runner.render_sequence(
            frame_images=images,
            overrides={
                "light_u": args.light_u,
                "light_v": args.light_v,
                "light_z_rel": args.light_z_rel,
                "light_intensity": args.light_intensity,
                "light_r": args.light_r,
                "light_g": args.light_g,
                "light_b": args.light_b,
            },
            depth_maps=depth_maps,
            frame_tags=[frame_path.name for frame_path in frames],
        )
        for frame_path, pred in zip(frames, predictions):
            pred.save(output_frames_dir / frame_path.name)
    finally:
        runner.reset()

    (output_dir / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "depth_dir": str(depth_dir) if depth_dir else None,
                "frame_count": len(frames),
                "light": {
                    "u": args.light_u,
                    "v": args.light_v,
                    "z_rel": args.light_z_rel,
                    "intensity": args.light_intensity,
                    "rgb": [args.light_r, args.light_g, args.light_b],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
