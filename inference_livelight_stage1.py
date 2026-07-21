import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, ControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision.utils import save_image

from src.models.light_guider import LightGuider
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.scheduler.scheduler_ddim import DDIMScheduler
from tools.build_livelight_mpli import (
    DEFAULT_REL_PLANE_MULTIPLIERS,
    build_plane_geometry,
    make_preview_image,
    projection_depth_to_camera,
    render_frame_mpli,
)


def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline inference for relight stage1 using the same depth-relative MPLI rule.")
    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--depth-npy", type=str, required=True, help="Predicted relative depth for the input image.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default="./exp_output/relight_stage1_tmux_bs1_20k")
    parser.add_argument("--train-config", type=str, default="./configs/train/relight_stage1_tmux.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-xformers", action="store_true")
    parser.add_argument("--controlnet-ckpt-dir", type=str, default=None,
                        help="If set, use ControlNet for light injection instead of LightGuider. "
                             "Point to the exp dir containing controlnet-*.pth.")
    parser.add_argument("--controlnet-train-config", type=str, default=None,
                        help="Training config used for ControlNet run (to find warm_start_dir for ref/denoising UNet). "
                             "If omitted, uses --train-config.")
    parser.add_argument("--force-single-block", action="store_true",
                        help="Force single-block injection (original PoseGuider style), "
                             "ignoring multiscale even if the config enables it.")

    parser.add_argument("--light-u", type=float, default=0.5, help="Normalized light x in [0, 1].")
    parser.add_argument("--light-v", type=float, default=0.35, help="Normalized light y in [0, 1].")
    parser.add_argument("--light-z-rel", type=float, default=0.66, help="Light depth as a multiplier of the anchor depth.")
    parser.add_argument("--light-intensity", type=float, default=1.0)
    parser.add_argument("--light-color", type=str, default="1.0,1.0,1.0")

    parser.add_argument("--anchor-u", type=float, default=0.5, help="Normalized anchor x in [0, 1].")
    parser.add_argument("--anchor-v", type=float, default=0.5, help="Normalized anchor y in [0, 1].")
    parser.add_argument("--depth-patch-radius", type=int, default=3)
    parser.add_argument(
        "--plane-multipliers",
        type=str,
        default=",".join(str(x) for x in DEFAULT_REL_PLANE_MULTIPLIERS),
    )
    parser.add_argument("--base-s1", type=float, default=25000.0)
    parser.add_argument(
        "--canonical-subject-depth",
        type=float,
        default=256.0,
        help="Explicit inference-side rule used to map predicted-depth anchor to depth_scale.",
    )
    parser.add_argument("--s2", type=float, default=1.0)
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help="Render intrinsics rule: fx = focal_scale * width, fy = focal_scale * height.",
    )
    return parser.parse_args()


def make_intrinsics(width: int, height: int, focal_scale: float):
    return {
        "fx": float(width) * float(focal_scale),
        "fy": float(height) * float(focal_scale),
        "cx": (float(width) - 1.0) * 0.5,
        "cy": (float(height) - 1.0) * 0.5,
    }


def normalized_to_pixel(u: float, v: float, width: int, height: int):
    x = float(u) * max(width - 1, 1)
    y = float(v) * max(height - 1, 1)
    return x, y


def read_anchor_depth(depth_map: np.ndarray, anchor_u: float, anchor_v: float, patch_radius: int):
    height, width = depth_map.shape[:2]
    x_f, y_f = normalized_to_pixel(anchor_u, anchor_v, width, height)
    x = min(max(int(round(x_f)), 0), width - 1)
    y = min(max(int(round(y_f)), 0), height - 1)

    x0 = max(0, x - patch_radius)
    x1 = min(width, x + patch_radius + 1)
    y0 = max(0, y - patch_radius)
    y1 = min(height, y + patch_radius + 1)
    patch = depth_map[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        raise ValueError("Depth anchor patch contains no finite values")
    value = float(np.median(patch))
    if value <= 0:
        raise ValueError("Depth anchor must be positive")
    return value


def pack_light(frame_mpli: np.ndarray):
    light = torch.from_numpy(frame_mpli.astype(np.float32))
    return light.permute(0, 3, 1, 2).reshape(-1, frame_mpli.shape[1], frame_mpli.shape[2])


def load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_latest_ckpt(ckpt_dir: Path, prefix: str):
    candidates = sorted(ckpt_dir.glob(f"{prefix}-*.pth"), key=lambda p: int(p.stem.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for prefix {prefix} under {ckpt_dir}")
    return candidates[-1]


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.train_config)
    infer_cfg = OmegaConf.load(cfg.inference_config)
    use_multiscale_light_injection = bool(cfg.model.get('enable_multiscale_light_injection', False))
    multiscale_light_injection_scale = float(cfg.model.get('multiscale_light_injection_scale', 0.25))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported weight dtype: {cfg.weight_dtype}")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    source_pil = Image.open(args.input_image).convert("RGB")
    depth_map = np.load(args.depth_npy).astype(np.float32)

    render_width = int(cfg.data.train_width)
    render_height = int(cfg.data.train_height)
    intrinsics = make_intrinsics(render_width, render_height, args.focal_scale)

    anchor_depth = read_anchor_depth(
        depth_map=depth_map,
        anchor_u=args.anchor_u,
        anchor_v=args.anchor_v,
        patch_radius=args.depth_patch_radius,
    )
    plane_multipliers = parse_float_list(args.plane_multipliers)
    plane_depths = np.asarray(plane_multipliers, dtype=np.float32) * anchor_depth
    light_depth = float(args.light_z_rel) * anchor_depth

    # Training used s1 = base_s1 * depth_scale^2, depth_scale = pred_anchor / subject_depth_gt.
    # For real-image inference, the worklog leaves the final user rule open, so we make it explicit:
    # use a fixed canonical subject depth to convert the anchor depth into the same depth_relative scaling pipeline.
    depth_scale = anchor_depth / max(float(args.canonical_subject_depth), 1e-8)
    s1_value = float(args.base_s1) * float(depth_scale * depth_scale)

    plane_geometry = build_plane_geometry(
        plane_depths=plane_depths,
        intrinsics=intrinsics,
        render_width=render_width,
        render_height=render_height,
    )
    light_u_px, light_v_px = normalized_to_pixel(args.light_u, args.light_v, render_width, render_height)
    light_camera = projection_depth_to_camera([light_u_px, light_v_px], light_depth, intrinsics)
    light_color = np.asarray(parse_float_list(args.light_color), dtype=np.float32)
    if light_color.shape != (3,):
        raise ValueError("--light-color must contain exactly 3 floats")

    frame_mpli = render_frame_mpli(
        light_cam=light_camera,
        light_color=light_color,
        light_intensity=float(args.light_intensity),
        plane_geometry=plane_geometry,
        s1=s1_value,
        s2=float(args.s2),
    )
    light_cond = pack_light(frame_mpli).unsqueeze(0).unsqueeze(2).to(device=device, dtype=weight_dtype)

    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_cfg.noise_scheduler_kwargs))
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    if torch.is_tensor(scheduler.final_alpha_cumprod):
        scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.to(device)

    use_controlnet = args.controlnet_ckpt_dir is not None

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=device, dtype=weight_dtype)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(device=device, dtype=weight_dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet").to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(infer_cfg.unet_additional_kwargs),
    ).to(device=device, dtype=weight_dtype)

    if use_controlnet:
        from diffusers import UNet2DConditionModel as DiffusersUNet2D
        cn_cfg_path = args.controlnet_train_config or args.train_config
        cn_cfg = OmegaConf.load(cn_cfg_path)
        cn_ckpt_dir = Path(args.controlnet_ckpt_dir)

        base_unet_for_init = DiffusersUNet2D.from_pretrained(cfg.base_model_path, subfolder="unet")
        controlnet = ControlNetModel.from_unet(base_unet_for_init, conditioning_channels=int(cn_cfg.model.light_channels))
        del base_unet_for_init
        controlnet_ckpt = find_latest_ckpt(cn_ckpt_dir, "controlnet")
        controlnet.load_state_dict(load_state_dict(controlnet_ckpt), strict=True)
        controlnet.to(device=device, dtype=weight_dtype)
        controlnet.eval()
        print(f"[controlnet] loaded {controlnet_ckpt}")

        warm_start_dir = cn_cfg.get("warm_start_dir", "")
        if warm_start_dir:
            ref_ckpt = find_latest_ckpt(Path(warm_start_dir), "reference_unet")
            den_ckpt = find_latest_ckpt(Path(warm_start_dir), "denoising_unet")
        else:
            ref_ckpt = find_latest_ckpt(cn_ckpt_dir, "reference_unet")
            den_ckpt = find_latest_ckpt(cn_ckpt_dir, "denoising_unet")
        reference_unet.load_state_dict(load_state_dict(ref_ckpt), strict=True)
        denoising_unet.load_state_dict(load_state_dict(den_ckpt), strict=False)
        print(f"[controlnet] ref_unet from {ref_ckpt}")
        print(f"[controlnet] den_unet from {den_ckpt}")
        light_guider = None
    else:
        light_guider = LightGuider(conditioning_channels=cfg.model.light_channels).to(device=device, dtype=weight_dtype)
        controlnet = None

        ckpt_dir = Path(args.ckpt_dir)
        reference_unet_ckpt = find_latest_ckpt(ckpt_dir, "reference_unet")
        denoising_unet_ckpt = find_latest_ckpt(ckpt_dir, "denoising_unet")
        light_guider_ckpt = find_latest_ckpt(ckpt_dir, "light_guider")

        reference_unet.load_state_dict(load_state_dict(reference_unet_ckpt), strict=True)
        denoising_unet.load_state_dict(load_state_dict(denoising_unet_ckpt), strict=False)
        light_guider.load_state_dict(load_state_dict(light_guider_ckpt), strict=True)
        light_guider.eval()

    if args.use_xformers:
        if not is_xformers_available():
            raise ValueError("xformers requested but not available")
        reference_unet.enable_xformers_memory_efficient_attention()
        denoising_unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None:
            controlnet.enable_xformers_memory_efficient_attention()

    vae.eval()
    image_enc.eval()
    reference_unet.eval()
    denoising_unet.eval()

    clip_processor = CLIPImageProcessor()
    image_transform = transforms.Compose(
        [
            transforms.Resize((render_height, render_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    source_tensor = image_transform(source_pil).unsqueeze(0).to(device=device, dtype=weight_dtype)
    clip_image = clip_processor(images=source_pil, return_tensors="pt").pixel_values.to(device=device, dtype=weight_dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.inference_mode():
        ref_image_latents = vae.encode(source_tensor).latent_dist.mean * 0.18215
        clip_embeds = image_enc(clip_image).image_embeds.unsqueeze(1)
        if not use_controlnet:
            light_emb = light_guider(light_cond)
        else:
            light_emb = None

        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        reference_unet(
            ref_image_latents,
            torch.zeros((1,), device=device, dtype=torch.long),
            encoder_hidden_states=clip_embeds,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer, drop_ratio=0.0)

        latent_h = render_height // 8
        latent_w = render_width // 8
        latents = torch.randn(
            (1, 4, 1, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        pred_original_sample = None
        for timestep in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, timestep).to(dtype=weight_dtype)
            # Original one-shot light injection path kept for reference:
            # model_pred = denoising_unet(
            #     latent_model_input,
            #     timestep,
            #     encoder_hidden_states=clip_embeds,
            #     pose_cond_fea=light_emb,
            #     skip_mm=True,
            # ).sample
            if use_controlnet:
                latent_2d = latent_model_input.squeeze(2)
                # light_cond is (1, 12, 1, H, W) for LightGuider path; squeeze to (1, 12, H, W) for ControlNet
                cn_cond = light_cond.squeeze(2) if light_cond.dim() == 5 else light_cond
                down_residuals, mid_residual = controlnet(
                    sample=latent_2d,
                    timestep=timestep,
                    encoder_hidden_states=clip_embeds,
                    controlnet_cond=cn_cond,
                    return_dict=False,
                )
                down_residuals = tuple(r.unsqueeze(2) for r in down_residuals)
                mid_residual = mid_residual.unsqueeze(2)
                model_pred = denoising_unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=clip_embeds,
                    pose_cond_fea=None,
                    skip_mm=True,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                ).sample
            else:
                model_pred = denoising_unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=clip_embeds,
                    pose_cond_fea=light_emb,
                    skip_mm=True,
                    use_multiscale_pose_cond=use_multiscale_light_injection,
                    pose_cond_scale=multiscale_light_injection_scale,
                ).sample
            latents, pred_original_sample = scheduler.step(
                model_pred,
                timestep,
                latents,
                return_dict=False,
            )

        reference_control_reader.clear()
        reference_control_writer.clear()
        pred_image = vae.decode((pred_original_sample[:, :, 0] / 0.18215).to(dtype=vae.dtype)).sample

    pred_image_01 = ((pred_image.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    source_image_01 = ((source_tensor.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    side_by_side = torch.cat([source_image_01, pred_image_01], dim=-1).cpu()

    save_image(pred_image_01.cpu(), output_dir / "relit.png")
    save_image(side_by_side, output_dir / "source_and_relit.png")
    make_preview_image(frame_mpli[None, ...], preview_groups=1).save(output_dir / "mpli_preview.png")

    report = {
        "input_image": str(Path(args.input_image).resolve()),
        "depth_npy": str(Path(args.depth_npy).resolve()),
        "rule": {
            "type": "depth_relative",
            "anchor_rule": "image_center_or_user_anchor + median_patch",
            "patch_radius": int(args.depth_patch_radius),
            "plane_multipliers": [float(x) for x in plane_multipliers],
            "light_parameterization": "uvz",
            "canonical_subject_depth": float(args.canonical_subject_depth),
            "depth_scale": float(depth_scale),
            "s1_formula": "base_s1 * depth_scale^2",
            "s2": float(args.s2),
        },
        "anchor_depth": float(anchor_depth),
        "plane_depths": [float(x) for x in plane_depths],
        "light_u": float(args.light_u),
        "light_v": float(args.light_v),
        "light_z_rel": float(args.light_z_rel),
        "light_depth": float(light_depth),
        "light_camera": [float(x) for x in light_camera],
        "light_color": [float(x) for x in light_color],
        "light_intensity": float(args.light_intensity),
        "s1": float(s1_value),
        "focal_scale": float(args.focal_scale),
        "mpli_stats": {
            "min": float(frame_mpli.min()),
            "p50": float(np.percentile(frame_mpli, 50)),
            "p95": float(np.percentile(frame_mpli, 95)),
            "max": float(frame_mpli.max()),
        },
        "checkpoints": {
            "reference_unet": str(ref_ckpt.resolve()) if use_controlnet else str(reference_unet_ckpt.resolve()),
            "denoising_unet": str(den_ckpt.resolve()) if use_controlnet else str(denoising_unet_ckpt.resolve()),
            "controlnet": str(controlnet_ckpt.resolve()) if use_controlnet else None,
            "light_guider": None if use_controlnet else str(light_guider_ckpt.resolve()),
        },
        "seed": int(args.seed),
        "num_inference_steps": int(args.num_inference_steps),
        "model_options": {
            "enable_multiscale_light_injection": bool(use_multiscale_light_injection),
            "multiscale_light_injection_scale": float(multiscale_light_injection_scale),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
