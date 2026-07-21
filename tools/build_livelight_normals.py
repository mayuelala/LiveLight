"""Construct surface normals from relight depth maps.

The relight pipeline already treats Pixel Perfect Depth outputs as a dense depth
field in the same camera-space convention used by `build_livelight_mpli.py`:

- `forward` increases away from the camera
- `right` increases with image x
- `up` increases against image y

This script keeps that convention exactly so the derived normals are geometric
supervision targets that are self-consistent with both the existing PPD depth
construction and the MPLI geometry code.

Important note on depth semantics:
Pixel Perfect Depth writes raw float32 predictions resized back to the input
image resolution. Those depths are not metric depths, but for normal recovery a
single global scale factor cancels out. Re-projecting them with the same scaled
intrinsics used by MPLI therefore yields stable camera-space normals.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_livelight_mpli import scaled_intrinsics


def parse_args():
    parser = argparse.ArgumentParser(description="Build relight normals from predicted depth maps.")
    parser.add_argument(
        "--label-root",
        type=str,
        required=True,
        help="Root directory containing per-sample relight label JSONs.",
    )
    parser.add_argument(
        "--depth-root",
        type=str,
        required=True,
        help="Root directory containing per-sample depth NPYs.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Root directory where normal NPYs and optional visualizations are written.",
    )
    parser.add_argument("--sample", type=str, help="Relative sample path like M01/B00/C0000.")
    parser.add_argument(
        "--sample-list",
        type=str,
        help="Optional sample list. Supports '<sample_rel>' and '<rootA>\\t<rootB>\\t<sample_rel>' rows.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save RGB visualizations with channels mapped from the signed normal vectors.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Output dtype for stored normal tensors.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a sample if all expected normal files already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of samples to process.",
    )
    return parser.parse_args()


def parse_sample_line(line: str) -> Path:
    parts = [part.strip() for part in line.split("\t")]
    if len(parts) == 1:
        return Path(parts[0])
    if len(parts) == 3:
        return Path(parts[2])
    raise ValueError(
        "Sample list lines must be either '<sample_rel>' or '<rootA>\\t<rootB>\\t<sample_rel>'."
    )


def discover_samples(depth_root: Path):
    samples = []
    for sample_dir in sorted(depth_root.rglob("*")):
        if not sample_dir.is_dir():
            continue
        if any(sample_dir.glob("*.npy")):
            samples.append(sample_dir.relative_to(depth_root))
    return samples


def resolve_samples(args, depth_root: Path):
    if args.sample:
        samples = [Path(args.sample)]
    elif args.sample_list:
        samples = []
        for raw_line in Path(args.sample_list).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            samples.append(parse_sample_line(line))
    else:
        samples = discover_samples(depth_root)

    if args.limit > 0:
        samples = samples[: args.limit]
    return samples


def find_label_json(label_root: Path, sample_rel: Path) -> Path:
    sample_dir = label_root / sample_rel
    candidates = sorted(sample_dir.glob("*.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one label JSON under {sample_dir}, found {len(candidates)}"
        )
    return candidates[0]


def load_camera(label_json_path: Path):
    data = json.loads(label_json_path.read_text(encoding="utf-8"))
    camera = data.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"Label JSON does not contain a camera dict: {label_json_path}")
    return camera


def depth_to_camera_points(depth: np.ndarray, intrinsics: dict) -> np.ndarray:
    height, width = depth.shape
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)

    forward = depth
    right = (uu - float(intrinsics["cx"])) / float(intrinsics["fx"]) * forward
    up = -(vv - float(intrinsics["cy"])) / float(intrinsics["fy"]) * forward
    return np.stack([forward, right, up], axis=-1).astype(np.float32)


def build_neighbor_valid_mask(valid_depth: np.ndarray) -> np.ndarray:
    center_valid = valid_depth[1:-1, 1:-1]
    center_valid &= valid_depth[1:-1, :-2]
    center_valid &= valid_depth[1:-1, 2:]
    center_valid &= valid_depth[:-2, 1:-1]
    center_valid &= valid_depth[2:, 1:-1]
    return center_valid


def depth_to_normals(depth: np.ndarray, intrinsics: dict, eps: float = 1e-8):
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be HxW, got shape {depth.shape}")

    depth = depth.astype(np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    points = depth_to_camera_points(depth, intrinsics)

    height, width = depth.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    valid_normals = np.zeros((height, width), dtype=bool)

    if height < 3 or width < 3:
        return normals, valid_normals

    tangent_u = points[:, 2:, :] - points[:, :-2, :]
    tangent_v = points[2:, :, :] - points[:-2, :, :]

    # Use the same camera-space axis order as MPLI: (forward, right, up).
    # cross(tangent_u, tangent_v) yields front-facing planes with a negative
    # forward component, i.e. normals that point back toward the camera.
    center_normals = np.cross(tangent_u[1:-1, :, :], tangent_v[:, 1:-1, :])
    center_norm = np.linalg.norm(center_normals, axis=-1, keepdims=True)
    center_valid = build_neighbor_valid_mask(valid_depth) & (center_norm[..., 0] > eps)

    center_normals = center_normals / np.maximum(center_norm, eps)

    # Keep visible surfaces consistently oriented toward the camera so the sign
    # convention is stable across datasets and matches standard view-space usage.
    flip_mask = center_normals[..., 0] > 0.0
    center_normals[flip_mask] *= -1.0

    normals[1:-1, 1:-1, :] = center_normals
    valid_normals[1:-1, 1:-1] = center_valid
    normals[~valid_normals] = 0.0
    return normals, valid_normals


def normals_to_vis(normals: np.ndarray, valid_normals: np.ndarray) -> Image.Image:
    vis = ((normals + 1.0) * 0.5).clip(0.0, 1.0)
    vis[~valid_normals] = 0.0
    return Image.fromarray((vis * 255.0).astype(np.uint8))


def save_array(path: Path, array: np.ndarray, dtype_name: str):
    if dtype_name == "float16":
        np.save(path, array.astype(np.float16))
    else:
        np.save(path, array.astype(np.float32))


def process_sample(
    sample_rel: Path,
    label_root: Path,
    depth_root: Path,
    out_root: Path,
    save_vis: bool,
    dtype_name: str,
):
    label_json_path = find_label_json(label_root, sample_rel)
    camera = load_camera(label_json_path)

    depth_sample_dir = depth_root / sample_rel
    depth_paths = sorted(depth_sample_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth NPY files found under {depth_sample_dir}")

    normal_sample_dir = out_root / sample_rel
    vis_sample_dir = None
    if save_vis:
        vis_sample_dir = out_root.parent / f"{out_root.name}_vis" / sample_rel

    normal_sample_dir.mkdir(parents=True, exist_ok=True)
    if vis_sample_dir is not None:
        vis_sample_dir.mkdir(parents=True, exist_ok=True)

    for depth_path in depth_paths:
        out_normal_path = normal_sample_dir / depth_path.name
        out_valid_path = normal_sample_dir / f"{depth_path.stem}_valid.npy"
        out_vis_path = None if vis_sample_dir is None else vis_sample_dir / f"{depth_path.stem}.png"

        depth = np.load(depth_path).astype(np.float32)
        intrinsics = scaled_intrinsics(camera, render_width=depth.shape[1], render_height=depth.shape[0])
        normals, valid_normals = depth_to_normals(depth, intrinsics)

        save_array(out_normal_path, normals, dtype_name)
        np.save(out_valid_path, valid_normals.astype(np.uint8))
        if out_vis_path is not None:
            normals_to_vis(normals, valid_normals).save(out_vis_path)

    return {
        "sample_rel": sample_rel.as_posix(),
        "depth_files": len(depth_paths),
        "normal_dir": str(normal_sample_dir),
        "vis_dir": None if vis_sample_dir is None else str(vis_sample_dir),
        "camera_convention": "(forward,right,up)",
        "normal_orientation": "camera_facing_when_visible",
    }


def main():
    args = parse_args()
    label_root = Path(args.label_root).resolve()
    depth_root = Path(args.depth_root).resolve()
    out_root = Path(args.out_root).resolve()
    samples = resolve_samples(args, depth_root)

    processed = []
    skipped = []
    failed = []

    for sample_rel in samples:
        depth_sample_dir = depth_root / sample_rel
        normal_sample_dir = out_root / sample_rel
        depth_paths = sorted(depth_sample_dir.glob("*.npy"))
        if args.skip_existing and depth_paths:
            expected = [normal_sample_dir / depth_path.name for depth_path in depth_paths]
            if expected and all(path.exists() for path in expected):
                skipped.append(sample_rel.as_posix())
                continue

        try:
            processed.append(
                process_sample(
                    sample_rel=sample_rel,
                    label_root=label_root,
                    depth_root=depth_root,
                    out_root=out_root,
                    save_vis=args.save_vis,
                    dtype_name=args.dtype,
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch scripts should continue past bad samples.
            failed.append({"sample_rel": sample_rel.as_posix(), "reason": str(exc)})

    report = {
        "label_root": str(label_root),
        "depth_root": str(depth_root),
        "out_root": str(out_root),
        "processed": len(processed),
        "skipped": len(skipped),
        "failed": len(failed),
        "normal_semantics": {
            "storage_order": ["forward", "right", "up"],
            "coordinate_system": "camera_space_matching_build_livelight_mpli",
            "depth_source": "raw_ppd_depth_resized_to_input_resolution",
        },
        "processed_samples": processed[:20],
        "failed_samples": failed[:20],
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        (out_root / "_failed.txt").write_text(
            "\n".join(f"{item['sample_rel']}\t{item['reason']}" for item in failed),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "processed": len(processed),
                "skipped": len(skipped),
                "failed": len(failed),
                "out_root": str(out_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
