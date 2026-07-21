from __future__ import annotations

import shutil
import subprocess
import os
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.models.light_guider import LightGuider
from src.models.mutual_self_attention_perframe import PerFrameReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.scheduler.scheduler_ddim import DDIMScheduler
from tools.build_livelight_mpli import (
    DEFAULT_REL_PLANE_MULTIPLIERS,
    build_plane_geometry,
    projection_depth_to_camera,
    render_frame_mpli,
)


class NoOpMotionModule(torch.nn.Module):
    """Keep temporal modules active while removing PersonaLive motion modules."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


@dataclass
class LightParams:
    light_u: float
    light_v: float
    light_z_rel: float
    light_intensity: float
    light_r: float
    light_g: float
    light_b: float


@dataclass
class FrameState:
    """Cached per-frame inputs for per-frame reference refresh inference."""

    ref_latent: torch.Tensor
    clip_embed: torch.Tensor
    depth_map: np.ndarray
    light_params: LightParams
    frame_tag: Optional[str]
    depth_source: str
    packed_light: Optional[torch.Tensor] = None
    light_report: Optional[Dict[str, object]] = None


def map_device(device_or_str):
    return device_or_str if isinstance(device_or_str, torch.device) else torch.device(device_or_str)


def parse_float_list(raw) -> List[float]:
    if raw is None:
        return []
    if OmegaConf.is_list(raw):
        return [float(x) for x in list(raw)]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def make_intrinsics(width: int, height: int, focal_scale: float) -> Dict[str, float]:
    return {
        "fx": float(width) * float(focal_scale),
        "fy": float(height) * float(focal_scale),
        "cx": (float(width) - 1.0) * 0.5,
        "cy": (float(height) - 1.0) * 0.5,
    }


def normalized_to_pixel(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    x = float(u) * max(width - 1, 1)
    y = float(v) * max(height - 1, 1)
    return x, y


def read_anchor_depth(depth_map: np.ndarray, anchor_u: float, anchor_v: float, patch_radius: int) -> float:
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


def pack_light(frame_mpli: np.ndarray) -> torch.Tensor:
    light = torch.from_numpy(frame_mpli.astype(np.float32))
    return light.permute(0, 3, 1, 2).reshape(-1, frame_mpli.shape[1], frame_mpli.shape[2])


class RelightPerFrameRefLive:
    """Stage-3 inference wrapper with per-frame reference refresh.

    This matches the new training semantics more closely than the existing
    fixed-reference wrapper:
    - every incoming source/original frame is re-encoded
    - every render rebuilds a per-frame reference bank over the current 16-frame
      temporal chunk
    - light condition is rebuilt per frame from each frame's depth map

    The implementation is correctness-first. It does not apply HKM or depth
    reuse; each frame can provide either a precomputed depth map or trigger a
    fresh PPD depth inference.
    """

    def __init__(self, cfg_or_path, device=None):
        if isinstance(cfg_or_path, (str, Path)):
            cfg = OmegaConf.load(str(cfg_or_path))
            acceleration = None
        elif OmegaConf.is_config(cfg_or_path):
            cfg = cfg_or_path
            acceleration = None
        elif hasattr(cfg_or_path, "config_path"):
            cfg = OmegaConf.load(cfg_or_path.config_path)
            acceleration = getattr(cfg_or_path, "acceleration", None)
        else:
            raise TypeError("RelightPerFrameRefLive expects an OmegaConf config, config path, or args with config_path")

        infer_cfg = OmegaConf.load(cfg.inference_config)

        self.cfg = cfg
        self.device = map_device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if cfg.dtype == "fp16" else torch.float32

        self.width = int(cfg.width)
        self.height = int(cfg.height)
        self.num_inference_steps = int(cfg.num_inference_steps)
        self.stage3_history_length = int(cfg.get("stage3_history_length", 16))
        self.temporal_window_size = int(cfg.get("temporal_window_size", 4))
        self.temporal_adaptive_step = int(cfg.get("temporal_adaptive_step", 4))
        self.timesteps_list = [int(x) for x in parse_float_list(cfg.get("timesteps_list", [999, 666, 333, 0]))]
        self.use_stage3_sliding_rollout = bool(cfg.get("use_stage3_sliding_rollout", True))
        self.use_multiscale_light_injection = bool(cfg.get("use_multiscale_light_injection", True))
        self.multiscale_light_injection_scale = float(cfg.get("multiscale_light_injection_scale", 0.25))
        if self.stage3_history_length != self.temporal_window_size * self.temporal_adaptive_step:
            raise ValueError(
                "stage3_history_length must match temporal_window_size * temporal_adaptive_step "
                f"for the current per-frame Stage-3 inference path, got "
                f"{self.stage3_history_length} vs {self.temporal_window_size} * {self.temporal_adaptive_step}."
            )
        if len(self.timesteps_list) != self.temporal_adaptive_step:
            raise ValueError(
                "timesteps_list length must match temporal_adaptive_step for the current "
                f"micro-chunk Stage-3 inference path, got {len(self.timesteps_list)} vs {self.temporal_adaptive_step}."
            )

        self.anchor_u = float(cfg.get("anchor_u", 0.5))
        self.anchor_v = float(cfg.get("anchor_v", 0.5))
        self.depth_patch_radius = int(cfg.get("depth_patch_radius", 3))
        self.plane_multipliers = parse_float_list(cfg.get("plane_multipliers", DEFAULT_REL_PLANE_MULTIPLIERS))
        self.base_s1 = float(cfg.get("base_s1", 25000.0))
        self.canonical_subject_depth = float(cfg.get("canonical_subject_depth", 256.0))
        self.s2 = float(cfg.get("s2", 1.0))
        self.focal_scale = float(cfg.get("focal_scale", 1.0))
        self.intrinsics = make_intrinsics(self.width, self.height, self.focal_scale)

        self.ppd_root = Path(str(cfg.ppd_root)).resolve()
        self.ppd_python = Path(str(cfg.ppd_python)).resolve()
        self.cache_root = Path(str(cfg.cache_dir)).resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.scheduler = DDIMScheduler(**OmegaConf.to_container(infer_cfg.noise_scheduler_kwargs))
        step_length = self.timesteps_list[0] - self.timesteps_list[1] if len(self.timesteps_list) > 1 else 333
        self.scheduler.set_step_length(step_length)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        if torch.is_tensor(self.scheduler.final_alpha_cumprod):
            self.scheduler.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(self.device)

        self.vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=self.device, dtype=self.dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(
            device=self.device, dtype=self.dtype
        )
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path,
            subfolder="unet",
        ).to(device=self.device, dtype=self.dtype)
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_cfg.unet_additional_kwargs),
        ).to(device=self.device, dtype=self.dtype)
        self.light_guider = LightGuider(conditioning_channels=int(cfg.light_channels)).to(
            device=self.device,
            dtype=self.dtype,
        )

        self.reference_unet.load_state_dict(torch.load(cfg.reference_unet_weight_path, map_location="cpu"), strict=True)
        self.denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
        self.light_guider.load_state_dict(torch.load(cfg.light_guider_path, map_location="cpu"), strict=True)

        temporal_module_path = str(cfg.get("temporal_module_path", "") or "").strip()
        if not temporal_module_path:
            raise ValueError("Per-frame reference inference requires temporal_module_path to be set")
        self.denoising_unet.load_state_dict(torch.load(temporal_module_path, map_location="cpu"), strict=False)
        for module in self.denoising_unet.modules():
            motion_modules = getattr(module, "motion_modules", None)
            if isinstance(motion_modules, torch.nn.ModuleList):
                for idx in range(len(motion_modules)):
                    motion_modules[idx] = NoOpMotionModule()

        if acceleration == "xformers":
            if not is_xformers_available():
                raise ValueError("xformers acceleration requested but xformers is not available")
            self.reference_unet.enable_xformers_memory_efficient_attention()
            self.denoising_unet.enable_xformers_memory_efficient_attention()

        self.vae.eval()
        self.image_encoder.eval()
        self.reference_unet.eval()
        self.denoising_unet.eval()
        self.light_guider.eval()

        self.reference_control_writer = PerFrameReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
            per_frame_reference=True,
        )
        self.reference_control_reader = PerFrameReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
            per_frame_reference=True,
        )

        self.clip_processor = CLIPImageProcessor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(cfg.seed))
        self.default_light_params = LightParams(
            light_u=float(cfg.get("default_light_u", 0.5)),
            light_v=float(cfg.get("default_light_v", 0.35)),
            light_z_rel=float(cfg.get("default_light_z_rel", 1.0)),
            light_intensity=float(cfg.get("default_light_intensity", 1.0)),
            light_r=float(cfg.get("default_light_r", 1.0)),
            light_g=float(cfg.get("default_light_g", 1.0)),
            light_b=float(cfg.get("default_light_b", 1.0)),
        )
        self.reset()

    def reset(self):
        self.frame_history: deque[FrameState] = deque(maxlen=self.stage3_history_length)
        self.real_frame_count = 0
        self.chunk_index = 0
        self.stage3_chunk_latents: Optional[torch.Tensor] = None
        self.reference_bank_cache: Optional[List[List[torch.Tensor]]] = None
        self.last_report: Dict[str, object] = {}
        self.reference_control_writer.clear()
        self.reference_control_reader.clear()
        self.latent_h = self.height // 8
        self.latent_w = self.width // 8

    def _light_params_from_dict(self, overrides: Optional[Dict[str, float]] = None) -> LightParams:
        data = self.default_light_params.__dict__.copy()
        if overrides:
            data.update({k: float(v) for k, v in overrides.items()})
        return LightParams(**data)

    def _resize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        if depth_map.ndim != 2:
            raise ValueError(f"Depth map must be HxW, got shape {depth_map.shape}")
        if depth_map.shape == (self.height, self.width):
            return depth_map.astype(np.float32)
        depth_tensor = torch.from_numpy(depth_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        depth_tensor = F.interpolate(depth_tensor, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return depth_tensor[0, 0].cpu().numpy().astype(np.float32)

    def _predict_depth_once(self, resized_frame: Image.Image) -> np.ndarray:
        cache_dir = self.cache_root / uuid.uuid4().hex
        cache_dir.mkdir(parents=True, exist_ok=True)
        image_path = cache_dir / "reference.png"
        manifest_path = cache_dir / "manifest.txt"
        depth_root = cache_dir / "depth_npy"
        depth_vis_root = cache_dir / "depth_vis"

        try:
            resized_frame.save(image_path)
            manifest_path.write_text(str(image_path) + "\n", encoding="utf-8")
            cmd = [
                str(self.ppd_python),
                str(self.ppd_root / "run_relight_batch.py"),
                "--manifest",
                str(manifest_path),
                "--relative-root",
                str(cache_dir),
                "--outdir-vis",
                str(depth_vis_root),
                "--outdir-npy",
                str(depth_root),
                "--skip-existing",
            ]
            subprocess.run(cmd, cwd=str(self.ppd_root), check=True, capture_output=True, text=True)
            depth_path = depth_root / "reference.npy"
            if not depth_path.exists():
                raise FileNotFoundError(f"Expected depth output was not created: {depth_path}")
            return np.load(depth_path).astype(np.float32)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _encode_frames_batch(self, frame_images: List[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        if not frame_images:
            raise ValueError("frame_images must not be empty")

        source_tensor = torch.stack([self.image_transform(frame_image) for frame_image in frame_images], dim=0).to(
            device=self.device,
            dtype=self.dtype,
        )
        clip_image = self.clip_processor(images=frame_images, return_tensors="pt").pixel_values.to(
            device=self.device,
            dtype=self.dtype,
        )
        ref_latents = self._vae_encode(source_tensor)
        clip_embeds = self.image_encoder(clip_image).image_embeds.unsqueeze(1)
        return ref_latents, clip_embeds

    def _encode_frame(self, frame_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        ref_latents, clip_embeds = self._encode_frames_batch([frame_image])
        return ref_latents[:1], clip_embeds[:1]

    def _vae_encode(self, source_tensor: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(source_tensor).latent_dist.mean * 0.18215

    def _build_frame_states_batch(
        self,
        frame_images: List[Image.Image],
        overrides: Optional[Dict[str, float]] = None,
        depth_maps: Optional[List[Optional[np.ndarray]]] = None,
        frame_tags: Optional[List[Optional[str]]] = None,
        packed_lights: Optional[List[Optional[torch.Tensor]]] = None,
        precomputed_ref_latents: Optional[torch.Tensor] = None,
        precomputed_clip_embeds: Optional[torch.Tensor] = None,
    ) -> List[FrameState]:
        if depth_maps is None:
            depth_maps = [None] * len(frame_images)
        if frame_tags is None:
            frame_tags = [None] * len(frame_images)
        if packed_lights is None:
            packed_lights = [None] * len(frame_images)
        if not (len(frame_images) == len(depth_maps) == len(frame_tags) == len(packed_lights)):
            raise ValueError("frame_images/depth_maps/frame_tags/packed_lights must have the same length")

        if precomputed_ref_latents is not None or precomputed_clip_embeds is not None:
            if precomputed_ref_latents is None or precomputed_clip_embeds is None:
                raise ValueError("precomputed_ref_latents and precomputed_clip_embeds must be provided together")
            if precomputed_ref_latents.shape[0] != len(frame_images) or precomputed_clip_embeds.shape[0] != len(frame_images):
                raise ValueError("Precomputed feature count must match frame_images length")
            ref_latents = precomputed_ref_latents.to(device=self.device, dtype=self.dtype)
            clip_embeds = precomputed_clip_embeds.to(device=self.device, dtype=self.dtype)
        else:
            ref_latents = None
            clip_embeds = None

        frame_pils: List[Image.Image] = []
        depth_maps_out: List[np.ndarray] = []
        depth_sources: List[str] = []
        light_params: List[LightParams] = []
        packed_light_tensors: List[torch.Tensor] = []
        light_reports: List[Optional[Dict[str, object]]] = []

        for frame_image, depth_map, packed_light in zip(frame_images, depth_maps, packed_lights):
            frame_pil: Optional[Image.Image] = None
            if ref_latents is None or clip_embeds is None or (packed_light is None and depth_map is None):
                frame_pil = frame_image.convert("RGB").resize((self.width, self.height), Image.BILINEAR)
            if packed_light is None:
                if depth_map is None:
                    if frame_pil is None:
                        frame_pil = frame_image.convert("RGB").resize((self.width, self.height), Image.BILINEAR)
                    depth_map = self._predict_depth_once(frame_pil)
                    depth_source = "predicted"
                else:
                    depth_map = self._resize_depth_map(depth_map)
                    depth_source = "provided"
            else:
                if depth_map is None:
                    depth_map = np.ones((self.height, self.width), dtype=np.float32)
                else:
                    depth_map = self._resize_depth_map(depth_map)
                depth_source = "exported_mpli"

            params = self._light_params_from_dict(overrides)
            light_report = None
            if packed_light is not None:
                packed_light_tensor = packed_light.detach().to(dtype=torch.float32, device="cpu")
            else:
                frame_mpli, light_report = self._render_single_frame_mpli(depth_map, params)
                packed_light_tensor = pack_light(frame_mpli).detach().to(dtype=torch.float32, device="cpu")

            if frame_pil is not None:
                frame_pils.append(frame_pil)
            depth_maps_out.append(depth_map)
            depth_sources.append(depth_source)
            light_params.append(params)
            packed_light_tensors.append(packed_light_tensor)
            light_reports.append(light_report)

        if ref_latents is None or clip_embeds is None:
            ref_latents, clip_embeds = self._encode_frames_batch(frame_pils)
        states: List[FrameState] = []
        for idx, frame_tag in enumerate(frame_tags):
            states.append(
                FrameState(
                    ref_latent=ref_latents[idx : idx + 1],
                    clip_embed=clip_embeds[idx : idx + 1],
                    depth_map=depth_maps_out[idx],
                    light_params=light_params[idx],
                    frame_tag=frame_tag,
                    depth_source=depth_sources[idx],
                    packed_light=packed_light_tensors[idx],
                    light_report=light_reports[idx],
                )
            )
        return states

    def _sample_stage3_noise(self, frames: int) -> torch.Tensor:
        noise = torch.randn(
            (1, 4, frames, self.latent_h, self.latent_w),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        return noise * self.scheduler.init_noise_sigma

    def _build_frame_state(
        self,
        frame_image: Image.Image,
        overrides: Optional[Dict[str, float]] = None,
        depth_map: Optional[np.ndarray] = None,
        frame_tag: Optional[str] = None,
        packed_light: Optional[torch.Tensor] = None,
    ) -> FrameState:
        frame_pil = frame_image.convert("RGB").resize((self.width, self.height), Image.BILINEAR)
        if packed_light is None:
            if depth_map is None:
                depth_map = self._predict_depth_once(frame_pil)
                depth_source = "predicted"
            else:
                depth_map = self._resize_depth_map(depth_map)
                depth_source = "provided"
        else:
            if depth_map is None:
                depth_map = np.ones((self.height, self.width), dtype=np.float32)
            else:
                depth_map = self._resize_depth_map(depth_map)
            depth_source = "exported_mpli"

        params = self._light_params_from_dict(overrides)
        ref_latent, clip_embed = self._encode_frame(frame_pil)
        light_report = None
        if packed_light is not None:
            packed_light_tensor = packed_light.detach().to(dtype=torch.float32, device="cpu")
        else:
            frame_mpli, light_report = self._render_single_frame_mpli(depth_map, params)
            packed_light_tensor = pack_light(frame_mpli).detach().to(dtype=torch.float32, device="cpu")
        return FrameState(
            ref_latent=ref_latent,
            clip_embed=clip_embed,
            depth_map=depth_map,
            light_params=params,
            frame_tag=frame_tag,
            depth_source=depth_source,
            packed_light=packed_light_tensor,
            light_report=light_report,
        )

    def _history_with_padding(self) -> List[FrameState]:
        if not self.frame_history:
            raise RuntimeError("push_frame() must be called before rendering")
        history = list(self.frame_history)
        if len(history) < self.stage3_history_length:
            history = [history[0]] * (self.stage3_history_length - len(history)) + history
        return history

    def _render_single_frame_mpli(self, depth_map: np.ndarray, params: LightParams):
        anchor_depth = read_anchor_depth(depth_map, self.anchor_u, self.anchor_v, self.depth_patch_radius)
        plane_depths = np.asarray(self.plane_multipliers, dtype=np.float32) * anchor_depth
        light_depth = float(params.light_z_rel) * anchor_depth

        depth_scale = anchor_depth / max(self.canonical_subject_depth, 1e-8)
        s1_value = self.base_s1 * float(depth_scale * depth_scale)
        plane_geometry = build_plane_geometry(
            plane_depths=plane_depths,
            intrinsics=self.intrinsics,
            render_width=self.width,
            render_height=self.height,
        )
        light_u_px, light_v_px = normalized_to_pixel(params.light_u, params.light_v, self.width, self.height)
        light_camera = projection_depth_to_camera([light_u_px, light_v_px], light_depth, self.intrinsics)
        light_color = np.asarray([params.light_r, params.light_g, params.light_b], dtype=np.float32)

        frame_mpli = render_frame_mpli(
            light_cam=light_camera,
            light_color=light_color,
            light_intensity=float(params.light_intensity),
            plane_geometry=plane_geometry,
            s1=s1_value,
            s2=self.s2,
        )
        report = {
            "anchor_depth": float(anchor_depth),
            "light_depth": float(light_depth),
            "light_camera": [float(x) for x in light_camera],
        }
        return frame_mpli, report

    def _build_light_sequence(self, history: Iterable[FrameState]):
        light_frames = []
        reports = []
        for state in history:
            if state.packed_light is not None:
                light_frames.append(state.packed_light.to(dtype=torch.float32))
                reports.append(state.light_report or {})
                continue

            frame_mpli, report = self._render_single_frame_mpli(state.depth_map, state.light_params)
            light_frames.append(pack_light(frame_mpli))
            reports.append(report)
        light_seq = torch.stack(light_frames, dim=1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        return light_seq, reports

    def _stack_packed_light_history(self, history: Iterable[FrameState]) -> torch.Tensor:
        return torch.stack(
            [state.packed_light.to(dtype=torch.float32) for state in history],
            dim=1,
        ).unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _clone_reference_bank_cache(self, bank_cache: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        return [[tensor.clone() for tensor in module_bank] for module_bank in bank_cache]

    def _snapshot_reference_bank_cache(self) -> List[List[torch.Tensor]]:
        return self._clone_reference_bank_cache(
            [module.bank for module in self.reference_control_reader._get_reader_modules()]
        )

    def _restore_reference_bank_cache(self, bank_cache: List[List[torch.Tensor]]):
        reader_modules = self.reference_control_reader._get_reader_modules()
        if len(reader_modules) != len(bank_cache):
            raise ValueError("Reference bank cache/module count mismatch")
        for module, module_bank in zip(reader_modules, bank_cache):
            module.bank = [tensor.clone() for tensor in module_bank]

    _HOOK_TO_EXPLICIT = ["m", "u10", "u11", "u12", "u20", "u21", "u22", "u30", "u31", "u32"]

    def _export_explicit_bank(self) -> Dict[str, torch.Tensor]:
        reader_modules = self.reference_control_reader._get_reader_modules()
        bank_dict = {}
        for i, module in enumerate(reader_modules):
            bank_dict[self._HOOK_TO_EXPLICIT[i]] = module.bank[0]
        return bank_dict

    def _rebuild_reference_bank(self, history: List[FrameState], clip_image_embeds: torch.Tensor):
        ref_image_latents = torch.cat([state.ref_latent for state in history], dim=0)

        self.reference_control_writer.clear()
        self.reference_control_reader.clear()
        self.reference_unet(
            ref_image_latents,
            torch.zeros((ref_image_latents.shape[0],), device=self.device, dtype=torch.long),
            encoder_hidden_states=clip_image_embeds,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer, drop_ratio=0.0)
        self.reference_bank_cache = self._snapshot_reference_bank_cache()
        self.reference_control_writer.clear()

    def _merge_reference_bank_cache(self, shift_count: int) -> Optional[List[List[torch.Tensor]]]:
        if self.reference_bank_cache is None:
            return None

        writer_modules = self.reference_control_writer._get_writer_modules()
        if len(self.reference_bank_cache) != len(writer_modules):
            return None

        merged_cache: List[List[torch.Tensor]] = []
        for cached_module_bank, writer_module in zip(self.reference_bank_cache, writer_modules):
            if len(cached_module_bank) != len(writer_module.bank):
                return None

            merged_module_bank: List[torch.Tensor] = []
            for cached_tensor, new_tensor in zip(cached_module_bank, writer_module.bank):
                if cached_tensor.shape[0] < shift_count or new_tensor.shape[0] != shift_count:
                    return None
                merged_module_bank.append(
                    torch.cat(
                        [cached_tensor[shift_count:], new_tensor.clone().to(dtype=cached_tensor.dtype)],
                        dim=0,
                    )
                )
            merged_cache.append(merged_module_bank)
        return merged_cache

    def _prepare_reference_bank(self, history: Iterable[FrameState]):
        history = list(history)
        clip_image_embeds = torch.cat([state.clip_embed for state in history], dim=0)

        if self.reference_bank_cache is None:
            self._rebuild_reference_bank(history, clip_image_embeds)
            return clip_image_embeds

        new_states = history[-self.temporal_window_size :]
        new_ref_image_latents = torch.cat([state.ref_latent for state in new_states], dim=0)
        new_clip_image_embeds = torch.cat([state.clip_embed for state in new_states], dim=0)

        self.reference_control_writer.clear()
        self.reference_unet(
            new_ref_image_latents,
            torch.zeros((new_ref_image_latents.shape[0],), device=self.device, dtype=torch.long),
            encoder_hidden_states=new_clip_image_embeds,
            return_dict=False,
        )
        merged_cache = self._merge_reference_bank_cache(self.temporal_window_size)
        self.reference_control_writer.clear()

        if merged_cache is None:
            self._rebuild_reference_bank(history, clip_image_embeds)
            return clip_image_embeds

        self.reference_control_reader.clear()
        self._restore_reference_bank_cache(merged_cache)
        self.reference_bank_cache = self._clone_reference_bank_cache(merged_cache)
        return clip_image_embeds

    def _history_reference_latents(self, history: Iterable[FrameState]) -> torch.Tensor:
        ref_image_latents = torch.cat([state.ref_latent for state in history], dim=0)
        ref_image_latents = ref_image_latents.reshape(
            1,
            self.stage3_history_length,
            ref_image_latents.shape[1],
            ref_image_latents.shape[2],
            ref_image_latents.shape[3],
        )
        return ref_image_latents.permute(0, 2, 1, 3, 4).contiguous().to(dtype=self.dtype)

    def _initialize_stage3_chunk_latents(self, history: Iterable[FrameState]) -> torch.Tensor:
        # Training seeds the oldest 12 frames with lower-noise states and keeps
        # only the newest 4 frames at full noise. We cannot use GT target
        # latents online, so initialize that low-noise history from the aligned
        # source/reference latents instead of restarting all 16 slots from noise.
        ref_latents = self._history_reference_latents(history)
        init_frames = self.stage3_history_length - self.temporal_window_size
        timesteps = self._stage3_micro_chunk_timesteps(batch_size=1)[:init_frames]

        history_latents = ref_latents[:, :, :init_frames]
        history_latents_flat = history_latents.permute(0, 2, 1, 3, 4).reshape(
            -1,
            history_latents.shape[1],
            history_latents.shape[3],
            history_latents.shape[4],
        )
        history_noise = torch.randn(
            history_latents_flat.shape,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        noisy_history = self.scheduler.add_noise(history_latents_flat, history_noise, timesteps)
        noisy_history = noisy_history.reshape(
            1,
            init_frames,
            history_latents.shape[1],
            history_latents.shape[3],
            history_latents.shape[4],
        ).permute(0, 2, 1, 3, 4)

        return torch.cat([noisy_history.to(dtype=self.dtype), self._sample_stage3_noise(self.temporal_window_size)], dim=2)

    def _decode_latent_frames(self, pred_frames: torch.Tensor) -> List[Image.Image]:
        pred_frames = pred_frames.permute(0, 2, 1, 3, 4).reshape(
            -1,
            pred_frames.shape[1],
            pred_frames.shape[3],
            pred_frames.shape[4],
        )
        decoded = self.vae.decode((pred_frames / 0.18215).to(dtype=self.vae.dtype)).sample
        decoded = ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu()

        images = []
        for frame in decoded:
            frame_np = frame.permute(1, 2, 0).numpy()
            images.append(Image.fromarray(np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)))
        return images

    def _decode_frame_range(self, pred_original_sample: torch.Tensor, start_index: int, end_index: int) -> List[Image.Image]:
        return self._decode_latent_frames(pred_original_sample[:, :, start_index:end_index])

    def _stage3_micro_chunk_timesteps(self, batch_size: int) -> torch.Tensor:
        # Stage-3 training uses a micro-chunk schedule where the oldest frames in
        # the current temporal chunk carry lower noise and the newest frames carry
        # higher noise. Keep the same per-frame timestep layout here.
        timesteps = torch.tensor(self.timesteps_list[::-1], device=self.device, dtype=torch.long)
        timesteps = timesteps.repeat_interleave(self.temporal_window_size, dim=0)
        timesteps = torch.stack([timesteps] * batch_size, dim=0)
        return timesteps.reshape(-1)

    def _infer_stage3_micro_chunk(
        self,
        latents_input: torch.Tensor,
        clip_image_embeds: torch.Tensor,
        light_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Old incorrect path kept for auditability:
        # latents = self.base_latents_stage3.clone()
        # pred_original_sample = None
        # _, pred_original_sample = self._infer_stage3_micro_chunk(
        #     latents_input=self.base_latents_stage3.clone(),
        #     clip_image_embeds=clip_image_embeds,
        #     light_emb=light_emb,
        # )
        #
        # That old path had two coupled bugs:
        # 1. it re-opened the whole 16-frame chunk from a fixed noise tensor on
        #    every inference call, so the low-noise temporal history trained by
        #    Stage 3 was never carried forward
        # 2. it therefore evaluated the per-frame model under a different rollout
        #    than the training loop, even after the timestep-vector fix
        #
        # The corrected path takes the persisted chunk latents explicitly and
        # advances them with the same 4-frame stride used by training.
        latents = latents_input.to(device=self.device, dtype=self.dtype)
        timesteps = self._stage3_micro_chunk_timesteps(batch_size=latents.shape[0])
        model_pred = self.denoising_unet(
            latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
            pose_cond_fea=light_emb,
            skip_mm=False,
            use_multiscale_pose_cond=self.use_multiscale_light_injection,
            pose_cond_scale=self.multiscale_light_injection_scale,
        ).sample

        clip_length = model_pred.shape[2]
        mid_noise_pred = model_pred.permute(0, 2, 1, 3, 4).reshape(-1, model_pred.shape[1], model_pred.shape[3], model_pred.shape[4])
        mid_latents = latents.permute(0, 2, 1, 3, 4).reshape(-1, latents.shape[1], latents.shape[3], latents.shape[4])
        mid_latents, pred_original_sample = self.scheduler.step(
            mid_noise_pred,
            timesteps,
            mid_latents,
            return_dict=False,
        )
        mid_latents = mid_latents.reshape(latents.shape[0], clip_length, latents.shape[1], latents.shape[3], latents.shape[4]).permute(0, 2, 1, 3, 4)
        pred_original_sample = pred_original_sample.reshape(latents.shape[0], clip_length, latents.shape[1], latents.shape[3], latents.shape[4]).permute(0, 2, 1, 3, 4)
        latents = torch.cat(
            [
                pred_original_sample[:, :, : self.temporal_window_size],
                mid_latents[:, :, self.temporal_window_size :],
            ],
            dim=2,
        ).to(dtype=self.dtype)

        # Carry the overlapping 12 frames forward as the scheduler-updated
        # low-noise latents instead of re-noising pred_x0. This preserves the
        # current history directly across chunk boundaries while still appending
        # a fresh 4-frame noise block for the newly entered micro-chunk.
        next_chunk_latents = torch.cat(
            [
                mid_latents[:, :, self.temporal_window_size :],
                self._sample_stage3_noise(self.temporal_window_size),
            ],
            dim=2,
        ).to(dtype=self.dtype)
        return latents, next_chunk_latents, pred_original_sample

    def _infer_stage3_whole_block(
        self,
        latents_input: torch.Tensor,
        clip_image_embeds: torch.Tensor,
        light_emb: torch.Tensor,
    ) -> torch.Tensor:
        latents = latents_input.to(device=self.device, dtype=self.dtype)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        pred_original_sample = latents
        for timestep in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, timestep).to(dtype=self.dtype)
            model_pred = self.denoising_unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=clip_image_embeds,
                pose_cond_fea=light_emb,
                skip_mm=False,
                use_multiscale_pose_cond=self.use_multiscale_light_injection,
                pose_cond_scale=self.multiscale_light_injection_scale,
            ).sample
            latents, pred_original_sample = self.scheduler.step(model_pred, timestep, latents, return_dict=False)
        return pred_original_sample

    @torch.inference_mode()
    def _render_current_chunk_latents(self, real_output_count: int) -> tuple[torch.Tensor, List[Dict[str, object]]]:
        history = self._history_with_padding()
        light_cond, light_reports = self._build_light_sequence(history)
        clip_image_embeds = self._prepare_reference_bank(history)
        light_emb = self.light_guider(light_cond)

        if self.stage3_chunk_latents is None:
            latents_input = self._initialize_stage3_chunk_latents(history)
        else:
            latents_input = self.stage3_chunk_latents

        _, next_chunk_latents, pred_original_sample = self._infer_stage3_micro_chunk(
            latents_input=latents_input,
            clip_image_embeds=clip_image_embeds,
            light_emb=light_emb,
        )
        self.stage3_chunk_latents = next_chunk_latents.detach()

        self.reference_control_writer.clear()
        self.reference_control_reader.clear()

        output_start = 0
        output_end = output_start + self.temporal_window_size
        output_latents = pred_original_sample[:, :, output_start:output_end].detach()
        output_states = history[output_start:output_end]
        output_light_reports = light_reports[output_start:output_end]

        reports = []
        for output_offset, (state, light_report) in enumerate(zip(output_states, output_light_reports)):
            reports.append(
                {
                    "mode": "stage3_perframe_reference_refresh_chunked",
                    "inference_schedule": "micro_chunk_vector_timestep_chunk_stride4",
                    "chunk_index": self.chunk_index,
                    "chunk_output_offset": output_offset,
                    "history_length": len(history),
                    "history_real_frames": self.real_frame_count,
                    "frame_tag": state.frame_tag,
                    "depth_source": state.depth_source,
                    "light_u": float(state.light_params.light_u),
                    "light_v": float(state.light_params.light_v),
                    "light_z_rel": float(state.light_params.light_z_rel),
                    "light_intensity": float(state.light_params.light_intensity),
                    "light_color": [
                        float(state.light_params.light_r),
                        float(state.light_params.light_g),
                        float(state.light_params.light_b),
                    ],
                    "anchor_depth": float(light_report["anchor_depth"]) if "anchor_depth" in light_report else None,
                    "light_depth": float(light_report["light_depth"]) if "light_depth" in light_report else None,
                    "light_camera": [float(x) for x in light_report["light_camera"]] if "light_camera" in light_report else None,
                }
            )

        self.chunk_index += 1
        return output_latents[:, :, :real_output_count], reports[:real_output_count]

    @torch.inference_mode()
    def _render_current_whole_block_latents(self, history: List[FrameState], real_output_count: int) -> tuple[torch.Tensor, List[Dict[str, object]]]:
        light_cond, light_reports = self._build_light_sequence(history)
        clip_image_embeds = self._prepare_reference_bank(history)
        light_emb = self.light_guider(light_cond)
        latents_input = self._sample_stage3_noise(self.stage3_history_length)
        pred_original_sample = self._infer_stage3_whole_block(
            latents_input=latents_input,
            clip_image_embeds=clip_image_embeds,
            light_emb=light_emb,
        )

        self.reference_control_writer.clear()
        self.reference_control_reader.clear()

        output_latents = pred_original_sample[:, :, :real_output_count].detach()
        output_states = history[:real_output_count]
        output_light_reports = light_reports[:real_output_count]

        reports = []
        for output_offset, (state, light_report) in enumerate(zip(output_states, output_light_reports)):
            reports.append(
                {
                    "mode": "stage3_perframe_reference_refresh_whole_block",
                    "inference_schedule": "whole_block_independent_stage2like_ddim4",
                    "chunk_index": self.chunk_index,
                    "chunk_output_offset": output_offset,
                    "history_length": len(history),
                    "history_real_frames": real_output_count,
                    "frame_tag": state.frame_tag,
                    "depth_source": state.depth_source,
                    "light_u": float(state.light_params.light_u),
                    "light_v": float(state.light_params.light_v),
                    "light_z_rel": float(state.light_params.light_z_rel),
                    "light_intensity": float(state.light_params.light_intensity),
                    "light_color": [
                        float(state.light_params.light_r),
                        float(state.light_params.light_g),
                        float(state.light_params.light_b),
                    ],
                    "anchor_depth": float(light_report["anchor_depth"]) if "anchor_depth" in light_report else None,
                    "light_depth": float(light_report["light_depth"]) if "light_depth" in light_report else None,
                    "light_camera": [float(x) for x in light_report["light_camera"]] if "light_camera" in light_report else None,
                }
            )

        self.chunk_index += 1
        return output_latents, reports

    @torch.inference_mode()
    def _render_current_chunk(self, real_output_count: int) -> tuple[List[Image.Image], List[Dict[str, object]]]:
        output_latents, reports = self._render_current_chunk_latents(real_output_count=real_output_count)
        return self._decode_latent_frames(output_latents), reports

    @torch.inference_mode()
    def _render_sequence_whole_blocks(
        self,
        frame_images: List[Image.Image],
        overrides: Optional[Dict[str, float]],
        depth_maps: List[Optional[np.ndarray]],
        frame_tags: List[Optional[str]],
        packed_lights: List[Optional[torch.Tensor]],
        precomputed_ref_latents: Optional[torch.Tensor],
        precomputed_clip_embeds: Optional[torch.Tensor],
    ) -> tuple[List[Image.Image], List[Dict[str, object]]]:
        predictions: List[Image.Image] = []
        reports: List[Dict[str, object]] = []
        block_size = self.stage3_history_length

        for block_start in range(0, len(frame_images), block_size):
            block_end = min(len(frame_images), block_start + block_size)
            block_states = self._build_frame_states_batch(
                frame_images=frame_images[block_start:block_end],
                overrides=overrides,
                depth_maps=depth_maps[block_start:block_end],
                frame_tags=frame_tags[block_start:block_end],
                packed_lights=packed_lights[block_start:block_end],
                precomputed_ref_latents=None if precomputed_ref_latents is None else precomputed_ref_latents[block_start:block_end],
                precomputed_clip_embeds=None if precomputed_clip_embeds is None else precomputed_clip_embeds[block_start:block_end],
            )
            real_block_count = len(block_states)
            if real_block_count < block_size:
                pad_state = block_states[-1]
                block_states = block_states + [pad_state] * (block_size - real_block_count)

            block_latents, block_reports = self._render_current_whole_block_latents(block_states, real_output_count=real_block_count)
            predictions.extend(self._decode_latent_frames(block_latents))
            reports.extend(block_reports)

        if reports:
            self.last_report = dict(reports[-1])
        return predictions, reports

    @torch.inference_mode()
    def render_sequence(
        self,
        frame_images: List[Image.Image],
        overrides: Optional[Dict[str, float]] = None,
        depth_maps: Optional[List[Optional[np.ndarray]]] = None,
        frame_tags: Optional[List[Optional[str]]] = None,
        packed_lights: Optional[List[Optional[torch.Tensor]]] = None,
        precomputed_ref_latents: Optional[torch.Tensor] = None,
        precomputed_clip_embeds: Optional[torch.Tensor] = None,
        decode_batch_size: Optional[int] = None,
        noise_init_warmup: bool = False,
    ) -> tuple[List[Image.Image], List[Dict[str, object]]]:
        if not frame_images:
            return [], []

        if depth_maps is None:
            depth_maps = [None] * len(frame_images)
        if frame_tags is None:
            frame_tags = [None] * len(frame_images)
        if packed_lights is None:
            packed_lights = [None] * len(frame_images)

        if not (len(frame_images) == len(depth_maps) == len(frame_tags) == len(packed_lights)):
            raise ValueError("frame_images/depth_maps/frame_tags/packed_lights must have the same length")
        if precomputed_ref_latents is not None or precomputed_clip_embeds is not None:
            if precomputed_ref_latents is None or precomputed_clip_embeds is None:
                raise ValueError("precomputed_ref_latents and precomputed_clip_embeds must be provided together")
            if precomputed_ref_latents.shape[0] != len(frame_images) or precomputed_clip_embeds.shape[0] != len(frame_images):
                raise ValueError("Precomputed feature count must match frame_images length")
        if decode_batch_size is not None and int(decode_batch_size) <= 0:
            raise ValueError("decode_batch_size must be positive when provided")

        if not self.use_stage3_sliding_rollout:
            return self._render_sequence_whole_blocks(
                frame_images=frame_images,
                overrides=overrides,
                depth_maps=depth_maps,
                frame_tags=frame_tags,
                packed_lights=packed_lights,
                precomputed_ref_latents=precomputed_ref_latents,
                precomputed_clip_embeds=precomputed_clip_embeds,
            )

        chunk_decode_mode = decode_batch_size is None or int(decode_batch_size) <= self.temporal_window_size
        decode_batch_size = self.temporal_window_size if decode_batch_size is None else int(decode_batch_size)

        warmup_chunks = self.temporal_adaptive_step - 1
        warmup_frames = warmup_chunks * self.temporal_window_size if not noise_init_warmup else 0
        total_input_frames = len(frame_images)

        predictions: List[Image.Image] = []
        reports: List[Dict[str, object]] = []
        pending_latents: Optional[torch.Tensor] = None
        pending_reports: List[Dict[str, object]] = []

        tw = self.temporal_window_size

        if noise_init_warmup:
            first_chunk_end = min(len(frame_images), tw)
            first_chunk_states = self._build_frame_states_batch(
                frame_images=frame_images[:first_chunk_end],
                overrides=overrides,
                depth_maps=depth_maps[:first_chunk_end],
                frame_tags=frame_tags[:first_chunk_end],
                packed_lights=packed_lights[:first_chunk_end],
                precomputed_ref_latents=None if precomputed_ref_latents is None else precomputed_ref_latents[:first_chunk_end],
                precomputed_clip_embeds=None if precomputed_clip_embeds is None else precomputed_clip_embeds[:first_chunk_end],
            )
            pad_state = first_chunk_states[0]
            padding_count = warmup_chunks * tw
            for _ in range(padding_count):
                self.frame_history.append(pad_state)

            ref_latent = pad_state.ref_latent
            init_latents = ref_latent.unsqueeze(2).repeat(1, 1, padding_count, 1, 1).to(
                device=self.device, dtype=self.dtype
            )
            init_noise = torch.randn(
                init_latents.shape, generator=self.generator, device=self.device, dtype=self.dtype
            )
            init_timesteps = list(reversed(self.timesteps_list))
            init_timesteps_vec = torch.tensor(
                init_timesteps, device=self.device, dtype=torch.long
            ).repeat_interleave(tw, dim=0)[:padding_count]
            noisy_init = self.scheduler.add_noise(
                init_latents.permute(0, 2, 1, 3, 4).reshape(-1, init_latents.shape[1], init_latents.shape[3], init_latents.shape[4]),
                init_noise.permute(0, 2, 1, 3, 4).reshape(-1, init_noise.shape[1], init_noise.shape[3], init_noise.shape[4]),
                init_timesteps_vec,
            )
            noisy_init = noisy_init.reshape(1, padding_count, init_latents.shape[1], init_latents.shape[3], init_latents.shape[4]).permute(0, 2, 1, 3, 4)
            self.stage3_chunk_latents = torch.cat(
                [noisy_init.to(dtype=self.dtype), self._sample_stage3_noise(tw)], dim=2
            )

            for state in first_chunk_states:
                self.frame_history.append(state)
                self.real_frame_count += 1
            if len(first_chunk_states) < tw:
                extra_pad = first_chunk_states[-1]
                for _ in range(tw - len(first_chunk_states)):
                    self.frame_history.append(extra_pad)

            if chunk_decode_mode:
                chunk_predictions, chunk_reports = self._render_current_chunk(real_output_count=tw)
                predictions.extend(chunk_predictions)
                reports.extend(chunk_reports)
            else:
                chunk_latents, chunk_reps = self._render_current_chunk_latents(real_output_count=tw)
                pending_latents = chunk_latents
                pending_reports.extend(chunk_reps)
                while pending_latents is not None and pending_latents.shape[2] >= decode_batch_size:
                    decode_latents = pending_latents[:, :, :decode_batch_size]
                    predictions.extend(self._decode_latent_frames(decode_latents))
                    reports.extend(pending_reports[:decode_batch_size])
                    remaining_frames = pending_latents.shape[2] - decode_batch_size
                    pending_latents = pending_latents[:, :, decode_batch_size:] if remaining_frames > 0 else None
                    pending_reports = pending_reports[decode_batch_size:]

            loop_start = tw
        else:
            loop_start = 0

        for chunk_start in range(loop_start, len(frame_images), tw):
            chunk_end = min(len(frame_images), chunk_start + tw)
            chunk_states = self._build_frame_states_batch(
                frame_images=frame_images[chunk_start:chunk_end],
                overrides=overrides,
                depth_maps=depth_maps[chunk_start:chunk_end],
                frame_tags=frame_tags[chunk_start:chunk_end],
                packed_lights=packed_lights[chunk_start:chunk_end],
                precomputed_ref_latents=None if precomputed_ref_latents is None else precomputed_ref_latents[chunk_start:chunk_end],
                precomputed_clip_embeds=None if precomputed_clip_embeds is None else precomputed_clip_embeds[chunk_start:chunk_end],
            )
            real_chunk_count = len(chunk_states)
            for state in chunk_states:
                self.frame_history.append(state)
                self.real_frame_count += 1

            if real_chunk_count < tw:
                pad_state = chunk_states[-1]
                while len(chunk_states) < tw:
                    self.frame_history.append(pad_state)
                    chunk_states.append(pad_state)

            if chunk_decode_mode:
                chunk_predictions, chunk_reports = self._render_current_chunk(real_output_count=tw)
                predictions.extend(chunk_predictions)
                reports.extend(chunk_reports)
                continue

            chunk_latents, chunk_reports = self._render_current_chunk_latents(real_output_count=tw)
            pending_latents = chunk_latents if pending_latents is None else torch.cat([pending_latents, chunk_latents], dim=2)
            pending_reports.extend(chunk_reports)

            while pending_latents is not None and pending_latents.shape[2] >= decode_batch_size:
                decode_latents = pending_latents[:, :, :decode_batch_size]
                predictions.extend(self._decode_latent_frames(decode_latents))
                reports.extend(pending_reports[:decode_batch_size])
                remaining_frames = pending_latents.shape[2] - decode_batch_size
                pending_latents = pending_latents[:, :, decode_batch_size:] if remaining_frames > 0 else None
                pending_reports = pending_reports[decode_batch_size:]

        drain_chunks = warmup_chunks if not noise_init_warmup else 0
        drain_pad_state = list(self.frame_history)[-1]
        for _drain_i in range(drain_chunks):
            for _ in range(tw):
                self.frame_history.append(drain_pad_state)

            if chunk_decode_mode:
                drain_preds, drain_reps = self._render_current_chunk(real_output_count=tw)
                predictions.extend(drain_preds)
                reports.extend(drain_reps)
            else:
                drain_latents, drain_reps = self._render_current_chunk_latents(real_output_count=tw)
                pending_latents = drain_latents if pending_latents is None else torch.cat([pending_latents, drain_latents], dim=2)
                pending_reports.extend(drain_reps)

                while pending_latents is not None and pending_latents.shape[2] >= decode_batch_size:
                    decode_latents = pending_latents[:, :, :decode_batch_size]
                    predictions.extend(self._decode_latent_frames(decode_latents))
                    reports.extend(pending_reports[:decode_batch_size])
                    remaining_frames = pending_latents.shape[2] - decode_batch_size
                    pending_latents = pending_latents[:, :, decode_batch_size:] if remaining_frames > 0 else None
                    pending_reports = pending_reports[decode_batch_size:]

        if not chunk_decode_mode and pending_latents is not None and pending_latents.shape[2] > 0:
            predictions.extend(self._decode_latent_frames(pending_latents))
            reports.extend(pending_reports)

        predictions = predictions[warmup_frames:][:total_input_frames]
        reports = reports[warmup_frames:][:total_input_frames]

        if reports:
            self.last_report = dict(reports[-1])
        return predictions, reports

    @torch.inference_mode()
    def push_frame(
        self,
        frame_image: Image.Image,
        overrides: Optional[Dict[str, float]] = None,
        depth_map: Optional[np.ndarray] = None,
        frame_tag: Optional[str] = None,
        packed_light: Optional[torch.Tensor] = None,
    ) -> Image.Image:
        """Disabled on purpose because one-frame Stage-3 rollout is semantically wrong.

        The corrected Stage-3 inference path is `render_sequence(...)`, which
        advances the temporal state in 4-frame chunks and matches training much
        more closely. Keeping a single-frame convenience API here invites users
        to benchmark or judge quality on an invalid rollout path, so fail fast
        instead of silently doing the wrong thing.
        """

        raise RuntimeError(
            "RelightPerFrameRefLive.push_frame() is disabled. "
            "Use render_sequence(...) so Stage-3 inference runs with the "
            "training-aligned 4-frame chunked rollout."
        )
