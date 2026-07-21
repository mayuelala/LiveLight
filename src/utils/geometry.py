import torch
import torch.nn.functional as F


def scale_intrinsics(intrinsics: torch.Tensor, src_hw, dst_hw) -> torch.Tensor:
    """Scale pinhole intrinsics from one raster size to another."""
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0)
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    scaled = intrinsics.clone().float()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    scaled[:, 3] *= scale_y
    return scaled


def depth_to_normals_from_camera_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
):
    """Match the offline normal builder's (forward, right, up) convention."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth to have shape [B, 1, H, W], got {tuple(depth.shape)}")

    batch_size, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    fx = intrinsics[:, 0].view(batch_size, 1, 1, 1)
    fy = intrinsics[:, 1].view(batch_size, 1, 1, 1)
    cx = intrinsics[:, 2].view(batch_size, 1, 1, 1)
    cy = intrinsics[:, 3].view(batch_size, 1, 1, 1)

    xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    ys = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)

    forward = depth
    right = (xs - cx) / fx.clamp_min(eps) * forward
    up = -(ys - cy) / fy.clamp_min(eps) * forward
    points = torch.cat([forward, right, up], dim=1)

    dx = points[:, :, 1:-1, 2:] - points[:, :, 1:-1, :-2]
    dy = points[:, :, 2:, 1:-1] - points[:, :, :-2, 1:-1]
    normals_inner_raw = torch.cross(dx, dy, dim=1)
    normals_inner_norm = torch.linalg.vector_norm(normals_inner_raw, dim=1, keepdim=True)
    normals_inner = normals_inner_raw / normals_inner_norm.clamp_min(eps)

    normals = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)
    normals[:, :, 1:-1, 1:-1] = normals_inner

    if valid_mask is None:
        valid_mask = depth > eps
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    inner_valid = (
        valid_mask[:, :, 1:-1, 1:-1]
        & valid_mask[:, :, 1:-1, 2:]
        & valid_mask[:, :, 1:-1, :-2]
        & valid_mask[:, :, 2:, 1:-1]
        & valid_mask[:, :, :-2, 1:-1]
        & (normals_inner_norm > eps)
    )
    normal_valid = torch.zeros(batch_size, 1, height, width, device=device, dtype=torch.bool)
    normal_valid[:, :, 1:-1, 1:-1] = inner_valid
    normals = torch.where(normal_valid.expand(-1, 3, -1, -1), normals, torch.zeros_like(normals))

    # Offline normals are kept camera-facing by requiring a non-positive
    # forward component in the (forward, right, up) basis.
    flip_mask = normals[:, 0:1] > 0
    normals = torch.where(flip_mask, -normals, normals)
    return normals, normal_valid


def masked_relative_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-element relative L2 averaged only over valid pixels."""
    prediction = prediction.float()
    target = target.float()
    rel_error = (prediction - target).pow(2) / target.pow(2).clamp_min(eps)

    if mask is None:
        return rel_error.mean()

    mask = mask.float()
    while mask.ndim < rel_error.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and rel_error.shape[1] != 1:
        mask = mask.expand(-1, rel_error.shape[1], -1, -1)

    denom = mask.sum().clamp_min(1.0)
    return (rel_error * mask).sum() / denom


def masked_relative_map_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Relative L2 on the whole masked vector field."""
    prediction = prediction.float()
    target = target.float()

    if mask is None:
        numerator = (prediction - target).pow(2).flatten(1).sum(dim=1)
        denominator = target.pow(2).flatten(1).sum(dim=1).clamp_min(eps)
        return (numerator / denominator).mean()

    mask = mask.float()
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)

    masked_sq_error = ((prediction - target).pow(2) * mask).flatten(1).sum(dim=1)
    masked_sq_target = (target.pow(2) * mask).flatten(1).sum(dim=1)
    valid_samples = mask.flatten(1).sum(dim=1) > 0
    if not valid_samples.any():
        return prediction.new_tensor(0.0)

    sample_loss = masked_sq_error[valid_samples] / masked_sq_target[valid_samples].clamp_min(eps)
    return sample_loss.mean()
