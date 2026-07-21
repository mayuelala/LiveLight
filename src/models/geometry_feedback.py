import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_state_dict_compat(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class FrozenPPDDepthEstimator(nn.Module):
    """Frozen PixelPerfectDepth wrapper that preserves gradients to the RGB input."""

    def __init__(
        self,
        pixel_perfect_depth_root,
        semantics_model,
        semantics_ckpt_path,
        model_ckpt_path,
        sampling_steps=4,
        input_resolution=256,
        latent_init="zeros",
    ):
        super().__init__()
        ppd_root = Path(pixel_perfect_depth_root)
        if str(ppd_root) not in sys.path:
            sys.path.insert(0, str(ppd_root))

        from ppd.models.ppd import PixelPerfectDepth

        self.model = PixelPerfectDepth(
            semantics_model=semantics_model,
            semantics_pth=semantics_ckpt_path,
            sampling_steps=int(sampling_steps),
        )
        missing = self.model.load_state_dict(load_state_dict_compat(model_ckpt_path), strict=False)
        filtered_missing = [key for key in missing.missing_keys if not key.startswith("sem_encoder.")]
        if filtered_missing or missing.unexpected_keys:
            raise RuntimeError(
                f"PPD checkpoint mismatch: missing={filtered_missing}, unexpected={missing.unexpected_keys}"
            )

        self.model.eval()
        self.model.requires_grad_(False)
        self.input_resolution = int(input_resolution)
        self.latent_init = str(latent_init)
        self.default_target_area = 1024 * 768

    def _init_latent(self, cond):
        if self.latent_init == "zeros":
            return torch.zeros(cond.shape[0], 1, cond.shape[2], cond.shape[3], device=cond.device, dtype=cond.dtype)
        if self.latent_init == "randn":
            return torch.randn(cond.shape[0], 1, cond.shape[2], cond.shape[3], device=cond.device, dtype=cond.dtype)
        raise ValueError(f"Unsupported latent_init: {self.latent_init}")

    def _infer_resize_hw(self, src_hw):
        src_h, src_w = src_hw
        if self.input_resolution > 0:
            return self.input_resolution, self.input_resolution

        scale = (float(self.default_target_area) / float(src_h * src_w)) ** 0.5
        resize_h = max(16, int(round(src_h * scale / 16.0)) * 16)
        resize_w = max(16, int(round(src_w * scale / 16.0)) * 16)
        return resize_h, resize_w

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image to have shape [B, 3, H, W], got {tuple(image.shape)}")

        image = image.float().clamp(0.0, 1.0)
        src_hw = image.shape[-2:]
        resize_hw = self._infer_resize_hw(src_hw)
        if src_hw != resize_hw:
            image = F.interpolate(
                image,
                size=resize_hw,
                mode="bilinear",
                align_corners=False,
            )

        cond = image - 0.5
        semantics = self.model.sem_encoder.forward_semantics(image)
        latent = self._init_latent(cond)

        for timestep in self.model.sampling_timesteps:
            timestep_value = int(timestep.item())
            timestep_batch = torch.full(
                (image.shape[0],),
                timestep_value,
                device=image.device,
                dtype=torch.int,
            )
            pred = self.model.dit(
                x=torch.cat([latent, cond], dim=1),
                semantics=semantics,
                timestep=timestep_batch,
            )
            latent = self.model.sampler.step(pred=pred, x_t=latent, t=timestep_batch)

        depth = latent + 0.5
        if depth.shape[-2:] != src_hw:
            depth = F.interpolate(depth, size=src_hw, mode="bilinear", align_corners=False)
        return depth.clamp_min(0.0)
