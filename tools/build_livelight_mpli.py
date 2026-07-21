import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ABS_PLANE_DEPTHS = [64.0, 128.0, 256.0, 512.0]
DEFAULT_REL_PLANE_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0]


def parse_float_list(raw):
    if raw is None or raw == "":
        return None
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Build MPLI tensors for relight samples.")
    parser.add_argument(
        "--meta-root",
        type=str,
        default="./data/label",
    )
    parser.add_argument(
        "--depth-root",
        type=str,
        default="./data/depth",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="./data/image",
    )
    parser.add_argument("--sample", type=str, help="Relative sample path like M01/B00/C0000")
    parser.add_argument("--sample-list", type=str, help="Text file with one relative sample path per line")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="depth_relative",
        choices=["metadata_absolute", "depth_relative"],
        help="depth_relative is the train/inference-consistent path based on predicted depth units.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="grouped",
        choices=["grouped", "single_frame"],
        help="grouped keeps the original paper-style 4-frame layout, while single_frame exports one MPLI per lit frame.",
    )
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--drop-last-incomplete-group", action="store_true")
    parser.add_argument("--plane-depths", type=str, default="64,128,256,512")
    parser.add_argument("--plane-multipliers", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--s1", type=float, default=25000.0)
    parser.add_argument("--s2", type=float, default=1.0)
    parser.add_argument("--depth-reduction", type=str, default="median_patch", choices=["point", "median_patch"])
    parser.add_argument("--depth-patch-radius", type=int, default=3)
    parser.add_argument(
        "--depth-anchor",
        type=str,
        default="subject_projection",
        choices=["subject_projection", "image_center"],
        help="Reference point used to read the predicted depth that anchors the relative MPLI scale.",
    )
    parser.add_argument("--preview-groups", type=int, default=3)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    return parser.parse_args()


def resolve_samples(args):
    if args.sample:
        return [Path(args.sample)]
    if args.sample_list:
        lines = Path(args.sample_list).read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip()]
    raise ValueError("One of --sample or --sample-list must be provided")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scaled_intrinsics(camera, render_width, render_height):
    src_w = float(camera["image_width"])
    src_h = float(camera["image_height"])
    sx = render_width / src_w
    sy = render_height / src_h
    return {
        "fx": float(camera["fx"]) * sx,
        "fy": float(camera["fy"]) * sy,
        "cx": float(camera["cx"]) * sx,
        "cy": float(camera["cy"]) * sy,
    }


def scale_projection(projection, src_width, src_height, dst_width, dst_height):
    sx = float(dst_width) / max(float(src_width), 1e-8)
    sy = float(dst_height) / max(float(src_height), 1e-8)
    return np.asarray([float(projection[0]) * sx, float(projection[1]) * sy], dtype=np.float32)


def projection_depth_to_camera(projection, depth, intrinsics):
    u = float(projection[0])
    v = float(projection[1])
    forward = float(depth)
    right = (u - intrinsics["cx"]) / intrinsics["fx"] * forward
    up = -(v - intrinsics["cy"]) / intrinsics["fy"] * forward
    return np.asarray([forward, right, up], dtype=np.float32)


def resolve_depth_anchor_projection(metadata, depth_anchor):
    camera = metadata["camera"]
    if depth_anchor == "subject_projection":
        subject = metadata.get("subject") or {}
        projection = subject.get("projection_2d")
        if projection is not None:
            return projection, "subject_projection"
    return [camera["image_width"] * 0.5, camera["image_height"] * 0.5], "image_center"


def read_reference_depth(depth_map, anchor_projection, camera, reduction, patch_radius):
    if anchor_projection is None:
        x = depth_map.shape[1] // 2
        y = depth_map.shape[0] // 2
    else:
        scaled_projection = scale_projection(
            projection=anchor_projection,
            src_width=camera["image_width"],
            src_height=camera["image_height"],
            dst_width=depth_map.shape[1],
            dst_height=depth_map.shape[0],
        )
        x = int(round(float(scaled_projection[0])))
        y = int(round(float(scaled_projection[1])))
        x = min(max(x, 0), depth_map.shape[1] - 1)
        y = min(max(y, 0), depth_map.shape[0] - 1)

    if reduction == "point":
        value = float(depth_map[y, x])
    else:
        x0 = max(0, x - patch_radius)
        x1 = min(depth_map.shape[1], x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(depth_map.shape[0], y + patch_radius + 1)
        patch = depth_map[y0:y1, x0:x1]
        value = float(np.median(patch))

    if np.isfinite(value) and value > 0:
        return value

    fallback = float(np.median(depth_map[np.isfinite(depth_map)]))
    if fallback > 0:
        return fallback
    raise ValueError("Unable to derive a positive predicted reference depth")


def build_plane_geometry(plane_depths, intrinsics, render_width, render_height):
    plane_depths = np.asarray(plane_depths, dtype=np.float32)
    xs = np.arange(render_width, dtype=np.float32)
    ys = np.arange(render_height, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)

    ray_right = (uu - intrinsics["cx"]) / intrinsics["fx"]
    ray_up = -(vv - intrinsics["cy"]) / intrinsics["fy"]

    depth_volume = plane_depths[:, None, None]
    forward = np.broadcast_to(depth_volume, (len(plane_depths), render_height, render_width))
    right = ray_right[None, ...] * depth_volume
    up = ray_up[None, ...] * depth_volume
    return forward.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


def render_frame_mpli(light_cam, light_color, light_intensity, plane_geometry, s1, s2):
    forward, right, up = plane_geometry
    df = forward - float(light_cam[0])
    dr = right - float(light_cam[1])
    du = up - float(light_cam[2])
    dist2 = df * df + dr * dr + du * du
    denom = dist2 / max(float(s1), 1e-8) + float(s2)
    irradiance = float(light_intensity) / np.maximum(denom, 1e-8)
    plane = irradiance[..., None] * np.asarray(light_color, dtype=np.float32)[None, None, None, :]
    return plane.astype(np.float32)


def build_group_slices(num_frames, group_size, drop_last_incomplete_group):
    groups = []
    for start in range(0, num_frames, group_size):
        end = min(num_frames, start + group_size)
        if drop_last_incomplete_group and end - start < group_size:
            break
        groups.append((start, end))
    if not groups:
        raise ValueError("No frame groups were created. Check group-size / drop-last settings.")
    return groups


def make_preview_image(mpli_items, preview_groups):
    num_groups = min(preview_groups, mpli_items.shape[0])
    planes = mpli_items[:num_groups]
    rows = []
    for group_idx in range(num_groups):
        cols = []
        for plane_idx in range(planes.shape[1]):
            arr = planes[group_idx, plane_idx]
            arr = np.log1p(arr)
            arr = arr / max(float(arr.max()), 1e-8)
            img = (arr * 255.0).clip(0, 255).astype(np.uint8)
            cols.append(Image.fromarray(img))
        row = Image.new("RGB", (cols[0].width * len(cols), cols[0].height))
        for idx, img in enumerate(cols):
            row.paste(img, (idx * cols[0].width, 0))
        rows.append(row)

    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)))
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    return canvas


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": float(values.min()),
        "p5": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def build_mode_params(args, metadata, depth_root, sample_rel, intrinsics, abs_plane_depths, rel_plane_multipliers):
    lit_frames = [frame for frame in metadata["frames"] if frame.get("light_active")]
    camera = metadata["camera"]

    if args.mode == "metadata_absolute":
        subject = metadata.get("subject") or {}
        subject_depth = subject.get("depth")
        return {
            "plane_depths": np.asarray(abs_plane_depths, dtype=np.float32),
            "scale_note": "absolute camera-space uvz from metadata",
            "subject_depth_gt": None if subject_depth is None else float(subject_depth),
            "subject_ref_depth_pred": None,
            "depth_scale": 1.0,
            "depth_anchor_source": None,
            "depth_anchor_projection": None,
            "light_depth_summary": summarize([float(frame["light_depth"]) for frame in lit_frames]),
            "s1_value": float(args.s1),
        }

    subject = metadata.get("subject") or {}
    if subject.get("depth") is None:
        raise ValueError("depth_relative mode requires subject.depth in metadata")

    depth_path = depth_root / sample_rel / "0000.npy"
    depth_map = np.load(depth_path)
    depth_anchor_projection, depth_anchor_source = resolve_depth_anchor_projection(metadata, args.depth_anchor)
    subject_ref_depth_pred = read_reference_depth(
        depth_map=depth_map,
        anchor_projection=depth_anchor_projection,
        camera=camera,
        reduction=args.depth_reduction,
        patch_radius=args.depth_patch_radius,
    )
    subject_depth_gt = float(subject["depth"])
    depth_scale = subject_ref_depth_pred / max(subject_depth_gt, 1e-8)
    light_depth_pred_units = [float(frame["light_depth"]) * depth_scale for frame in lit_frames]
    return {
        "plane_depths": np.asarray(rel_plane_multipliers, dtype=np.float32) * subject_ref_depth_pred,
        "scale_note": "uvz converted into predicted-depth units using first-frame reference depth",
        "subject_depth_gt": subject_depth_gt,
        "subject_ref_depth_pred": float(subject_ref_depth_pred),
        "depth_scale": float(depth_scale),
        "depth_anchor_source": depth_anchor_source,
        "depth_anchor_projection": [float(x) for x in depth_anchor_projection],
        "light_depth_summary": summarize(light_depth_pred_units),
        "s1_value": float(args.s1) * float(depth_scale * depth_scale),
    }


def build_frame_light_params(frame, metadata, intrinsics, render_width, render_height, depth_scale):
    camera = metadata["camera"]
    light_projection_render = scale_projection(
        projection=frame["light_2d_projection"],
        src_width=camera["image_width"],
        src_height=camera["image_height"],
        dst_width=render_width,
        dst_height=render_height,
    )
    light_depth = float(frame["light_depth"]) * float(depth_scale)
    light_camera = projection_depth_to_camera(
        projection=light_projection_render,
        depth=light_depth,
        intrinsics=intrinsics,
    )
    return {
        "light_projection_render": light_projection_render.astype(np.float32),
        "light_depth": float(light_depth),
        "light_camera": light_camera.astype(np.float32),
    }


def compute_uvz_reconstruction_error(lit_frames, metadata, intrinsics, render_width, render_height):
    residuals = []
    for frame in lit_frames:
        reconstructed = build_frame_light_params(
            frame=frame,
            metadata=metadata,
            intrinsics=intrinsics,
            render_width=render_width,
            render_height=render_height,
            depth_scale=1.0,
        )["light_camera"]
        gt = np.asarray(frame["light_position_camera"], dtype=np.float32)
        residuals.append(float(np.linalg.norm(reconstructed - gt)))
    return summarize(residuals)


def cast_mpli_dtype(mpli, dtype_name):
    if dtype_name == "float16":
        return mpli.astype(np.float16)
    return mpli.astype(np.float32)


def save_grouped_sample(
    out_root,
    sample_rel,
    grouped_mpli,
    frame_group_ranges,
    grouped_light_camera,
    grouped_light_projection,
    grouped_light_depth,
    config,
    stats,
    save_preview,
    preview_groups,
    dtype_name,
):
    sample_out = out_root / sample_rel
    sample_out.mkdir(parents=True, exist_ok=True)

    mpli_to_save = cast_mpli_dtype(grouped_mpli, dtype_name)

    np.savez_compressed(
        sample_out / "mpli.npz",
        mpli=mpli_to_save,
        frame_group_ranges=frame_group_ranges.astype(np.int32),
        grouped_light_camera=grouped_light_camera.astype(np.float32),
        grouped_light_projection=grouped_light_projection.astype(np.float32),
        grouped_light_depth=grouped_light_depth.astype(np.float32),
        plane_depths=np.asarray(config["plane_depths"], dtype=np.float32),
        config_json=np.asarray(json.dumps(config)),
        stats_json=np.asarray(json.dumps(stats)),
    )

    if save_preview:
        preview = make_preview_image(grouped_mpli, preview_groups=preview_groups)
        preview.save(sample_out / "mpli_preview.png")


def save_single_frame_sample(
    out_root,
    sample_rel,
    frame_mpli,
    save_preview,
    preview_groups,
    dtype_name,
):
    sample_out = out_root / sample_rel
    sample_out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sample_out / "mpli.npz",
        mpli=cast_mpli_dtype(frame_mpli, dtype_name),
    )

    if save_preview:
        preview = make_preview_image(frame_mpli, preview_groups=preview_groups)
        preview.save(sample_out / "mpli_preview.png")


def build_for_sample(args, sample_rel):
    meta_root = Path(args.meta_root)
    depth_root = Path(args.depth_root)
    out_root = Path(args.out_root)

    meta_path = meta_root / sample_rel / f"Seq_{sample_rel.parts[0]}_{sample_rel.parts[1]}_{sample_rel.parts[2]}.json"
    metadata = load_json(meta_path)
    intrinsics = scaled_intrinsics(metadata["camera"], args.render_width, args.render_height)
    abs_plane_depths = parse_float_list(args.plane_depths) or DEFAULT_ABS_PLANE_DEPTHS
    rel_plane_multipliers = parse_float_list(args.plane_multipliers) or DEFAULT_REL_PLANE_MULTIPLIERS

    lit_frames = [frame for frame in metadata["frames"] if frame.get("light_active")]
    if not lit_frames:
        raise ValueError(f"No lit frames found in {meta_path}")

    mode_params = build_mode_params(
        args=args,
        metadata=metadata,
        depth_root=depth_root,
        sample_rel=sample_rel,
        intrinsics=intrinsics,
        abs_plane_depths=abs_plane_depths,
        rel_plane_multipliers=rel_plane_multipliers,
    )

    plane_depths = mode_params["plane_depths"]
    plane_geometry = build_plane_geometry(
        plane_depths=plane_depths,
        intrinsics=intrinsics,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    default_light_color = metadata["light"]["color"]
    default_light_intensity = float(metadata["light"]["intensity"])

    per_frame_mplis = []
    per_frame_light_depth = []
    for frame in lit_frames:
        light_params = build_frame_light_params(
            frame=frame,
            metadata=metadata,
            intrinsics=intrinsics,
            render_width=args.render_width,
            render_height=args.render_height,
            depth_scale=mode_params["depth_scale"],
        )
        frame_light_color = frame.get("light_color", default_light_color)
        frame_light_intensity = float(frame.get("light_intensity", default_light_intensity))
        frame_mpli = render_frame_mpli(
            light_cam=light_params["light_camera"],
            light_color=frame_light_color,
            light_intensity=frame_light_intensity,
            plane_geometry=plane_geometry,
            s1=mode_params["s1_value"],
            s2=args.s2,
        )
        per_frame_mplis.append(frame_mpli)
        per_frame_light_depth.append(light_params["light_depth"])

    per_frame_mpli = np.stack(per_frame_mplis, axis=0)
    per_frame_light_depth = np.asarray(per_frame_light_depth, dtype=np.float32)

    uvz_reconstruction_error = compute_uvz_reconstruction_error(
        lit_frames=lit_frames,
        metadata=metadata,
        intrinsics=intrinsics,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    config = {
        "mode": args.mode,
        "layout": args.layout,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "group_size": args.group_size,
        "drop_last_incomplete_group": args.drop_last_incomplete_group,
        "plane_depths": [float(x) for x in plane_depths],
        "abs_plane_depths": [float(x) for x in abs_plane_depths],
        "rel_plane_multipliers": [float(x) for x in rel_plane_multipliers],
        "s1": float(mode_params["s1_value"]),
        "s2": float(args.s2),
        "light_parameterization": "projection_depth_uvz",
        "scale_note": mode_params["scale_note"],
        "depth_anchor": args.depth_anchor,
        "depth_anchor_source": mode_params["depth_anchor_source"],
        "depth_anchor_projection": mode_params["depth_anchor_projection"],
        "depth_reduction": args.depth_reduction,
        "depth_patch_radius": int(args.depth_patch_radius),
        "depth_scale": float(mode_params["depth_scale"]),
        "subject_depth_gt": mode_params["subject_depth_gt"],
        "subject_ref_depth_pred": mode_params["subject_ref_depth_pred"],
        "light_depth_summary": mode_params["light_depth_summary"],
    }

    if args.layout == "single_frame":
        stats = {
            "shape": list(per_frame_mpli.shape),
            "min": float(per_frame_mpli.min()),
            "p50": float(np.percentile(per_frame_mpli, 50)),
            "p95": float(np.percentile(per_frame_mpli, 95)),
            "p99": float(np.percentile(per_frame_mpli, 99)),
            "max": float(per_frame_mpli.max()),
            "light_depth_summary": summarize(per_frame_light_depth),
            "uvz_reconstruction_error": uvz_reconstruction_error,
        }
        save_single_frame_sample(
            out_root=out_root,
            sample_rel=sample_rel,
            frame_mpli=per_frame_mpli,
            save_preview=args.save_preview,
            preview_groups=args.preview_groups,
            dtype_name=args.dtype,
        )
        return {
            "sample": str(sample_rel),
            "meta_path": str(meta_path),
            "frames": int(per_frame_mpli.shape[0]),
            "stats": stats,
            "config": config,
        }

    group_slices = build_group_slices(
        num_frames=len(lit_frames),
        group_size=args.group_size,
        drop_last_incomplete_group=args.drop_last_incomplete_group,
    )

    grouped_mplis = []
    grouped_light_camera = []
    grouped_light_projection = []
    grouped_light_depth = []
    frame_group_ranges = []

    for group_idx, (start, end) in enumerate(group_slices):
        chunk_frames = lit_frames[start:end]
        grouped_mplis.append(np.mean(per_frame_mpli[start:end], axis=0, dtype=np.float32))
        grouped_light_depth.append(float(np.mean(per_frame_light_depth[start:end])))

        chunk_light_camera = []
        chunk_light_projection = []
        for frame in chunk_frames:
            light_params = build_frame_light_params(
                frame=frame,
                metadata=metadata,
                intrinsics=intrinsics,
                render_width=args.render_width,
                render_height=args.render_height,
                depth_scale=mode_params["depth_scale"],
            )
            chunk_light_camera.append(light_params["light_camera"])
            chunk_light_projection.append(light_params["light_projection_render"])

        grouped_light_camera.append(np.mean(chunk_light_camera, axis=0, dtype=np.float32))
        grouped_light_projection.append(np.mean(chunk_light_projection, axis=0, dtype=np.float32))
        frame_group_ranges.append([int(chunk_frames[0]["frame_id"]), int(chunk_frames[-1]["frame_id"])])

    grouped_mpli = np.stack(grouped_mplis, axis=0)
    grouped_light_camera = np.stack(grouped_light_camera, axis=0)
    grouped_light_projection = np.stack(grouped_light_projection, axis=0)
    grouped_light_depth = np.asarray(grouped_light_depth, dtype=np.float32)
    frame_group_ranges = np.asarray(frame_group_ranges, dtype=np.int32)

    stats = {
        "shape": list(grouped_mpli.shape),
        "min": float(grouped_mpli.min()),
        "p50": float(np.percentile(grouped_mpli, 50)),
        "p95": float(np.percentile(grouped_mpli, 95)),
        "p99": float(np.percentile(grouped_mpli, 99)),
        "max": float(grouped_mpli.max()),
        "grouped_light_depth_summary": summarize(grouped_light_depth),
        "uvz_reconstruction_error": uvz_reconstruction_error,
    }

    save_grouped_sample(
        out_root=out_root,
        sample_rel=sample_rel,
        grouped_mpli=grouped_mpli,
        frame_group_ranges=frame_group_ranges,
        grouped_light_camera=grouped_light_camera,
        grouped_light_projection=grouped_light_projection,
        grouped_light_depth=grouped_light_depth,
        config=config,
        stats=stats,
        save_preview=args.save_preview,
        preview_groups=args.preview_groups,
        dtype_name=args.dtype,
    )

    return {
        "sample": str(sample_rel),
        "meta_path": str(meta_path),
        "groups": int(grouped_mpli.shape[0]),
        "stats": stats,
        "config": config,
    }


def main():
    args = parse_args()
    samples = resolve_samples(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for sample_rel in samples:
        report = build_for_sample(args, sample_rel)
        reports.append(report)
        print(json.dumps(report, indent=2))

    (out_root / "_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
