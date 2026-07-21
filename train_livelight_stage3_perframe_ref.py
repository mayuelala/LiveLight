import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from src.dataset.livelight_video_pair import RelightVideoPairDataset
from src.models.light_guider import LightGuider
from src.models.motion_module import zero_module
from src.models.mutual_self_attention_perframe import PerFrameReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.utils.util import delete_additional_ckpt, seed_everything


def load_state_dict_compat(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

try:
    import mlflow
except ImportError:
    mlflow = None

logger = get_logger(__name__, log_level="INFO")


class NoOpMotionModule(nn.Module):
    """Keep temporal modules active while removing PersonaLive motion inputs."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


def reset_module_parameters(module: nn.Module) -> int:
    """Reset temporal modules while restoring their zero-init residual output."""

    reset_count = 0

    # Old generic reset path kept for auditability. It random-reset every child
    # module that exposed reset_parameters(), which also re-randomized the
    # temporal residual proj_out that was intentionally zero-initialized at
    # construction time.
    #
    # def _reset(submodule: nn.Module):
    #     nonlocal reset_count
    #     if submodule is module:
    #         return
    #     reset_fn = getattr(submodule, "reset_parameters", None)
    #     if callable(reset_fn):
    #         reset_fn()
    #         reset_count += 1

    def _reset(submodule: nn.Module):
        nonlocal reset_count
        if submodule is module:
            return
        reset_fn = getattr(submodule, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()
            reset_count += 1

    module.apply(_reset)

    # VanillaTemporalModule uses zero_module(...) on temporal_transformer.proj_out
    # so the temporal residual starts from an identity-like zero contribution.
    # After the generic child resets above, restore that designed zero-init state
    # explicitly instead of leaving a random residual bias.
    temporal_transformer = getattr(module, "temporal_transformer", None)
    proj_out = getattr(temporal_transformer, "proj_out", None)
    if proj_out is not None:
        zero_module(proj_out)

    return reset_count


class Net(nn.Module):
    """Stage 3 wrapper with per-frame reference refresh."""

    def __init__(
        self,
        denoising_unet: UNet3DConditionModel,
        reference_control_reader,
        enable_multiscale_light_injection: bool = False,
        multiscale_light_injection_scale: float = 0.25,
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.reference_control_reader = reference_control_reader
        self.enable_multiscale_light_injection = enable_multiscale_light_injection
        self.multiscale_light_injection_scale = multiscale_light_injection_scale

    def forward(self, noisy_latents, timesteps, clip_image_embeds, light_emb):
        return self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
            pose_cond_fea=light_emb,
            skip_mm=False,
            use_multiscale_pose_cond=self.enable_multiscale_light_injection,
            pose_cond_scale=self.multiscale_light_injection_scale,
        ).sample


def get_x0_from_eps(ddim, sample: torch.FloatTensor, model_output: torch.FloatTensor, timesteps: torch.IntTensor):
    video_length = sample.shape[2]
    sample = rearrange(sample, "b c f h w -> (b f) c h w")
    model_output = rearrange(model_output, "b c f h w -> (b f) c h w")
    alpha_prod = ddim.alphas_cumprod.to(sample.device)
    alpha_prod_t = alpha_prod[timesteps]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)

    pred_original_sample = torch.sqrt(1.0 / alpha_prod_t) * sample - torch.sqrt(1.0 / alpha_prod_t - 1.0) * model_output
    pred_original_sample = rearrange(pred_original_sample, "(b f) c h w -> b c f h w", f=video_length)
    return pred_original_sample


def decode_latents(vae: AutoencoderKL, latents, decode_chunk_size=4):
    latents = latents.to(vae.dtype)
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    images = []
    for frame_idx in range(0, latents.shape[0], decode_chunk_size):
        images.append(vae.decode(latents[frame_idx : frame_idx + decode_chunk_size]).sample)
    images = torch.cat(images)
    images = rearrange(images, "(b f) c h w -> b c f h w", f=video_length)
    return images


def save_preview(vae, reference_videos, target_videos, pred_latents, save_dir, global_step):
    preview_dir = osp.join(save_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    with torch.no_grad():
        pred_videos = decode_latents(vae, pred_latents[:1].detach())

    preview_path = osp.join(preview_dir, f"step_{global_step:06d}.png")
    rows = []
    for video in [
        reference_videos[:1].float().cpu(),
        target_videos[:1].float().cpu(),
        pred_videos.float().cpu(),
    ]:
        frames = rearrange(video[0], "c t h w -> t c h w")
        frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0)
        rows.append(torchvision.utils.make_grid(frames, nrow=frames.shape[0], padding=2))

    torchvision.utils.save_image(torch.cat(rows, dim=1), preview_path)


def save_temporal_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")
    if total_limit is not None:
        checkpoints = [d for d in os.listdir(save_dir) if d.startswith(prefix)]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))
        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            for removing_checkpoint in checkpoints[0:num_to_remove]:
                os.remove(os.path.join(save_dir, removing_checkpoint))

    temporal_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "temporal_modules" in key:
            temporal_state_dict[key] = state_dict[key]
    torch.save(temporal_state_dict, save_path)


def encode_latent_frames(vae, frames, max_batch_frames):
    latents = []
    for i in range(0, frames.shape[0], max_batch_frames):
        batch_latents = vae.encode(frames[i : i + max_batch_frames]).latent_dist.sample()
        latents.append(batch_latents)
    return torch.cat(latents, dim=0) * 0.18215


def encode_clip_frames(image_enc, clip_frames, max_batch_frames):
    embeds = []
    for i in range(0, clip_frames.shape[0], max_batch_frames):
        embeds.append(image_enc(clip_frames[i : i + max_batch_frames]).image_embeds)
    return torch.cat(embeds, dim=0).unsqueeze(1)


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow" if mlflow is not None else None,
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    infer_config = OmegaConf.load(cfg.inference_config)
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported weight dtype: {cfg.weight_dtype}")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    if len(cfg.timesteps_list) > 1:
        train_noise_scheduler.set_step_length(int(cfg.timesteps_list[0]) - int(cfg.timesteps_list[1]))

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(
        dtype=weight_dtype,
        device="cuda",
    )

    reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet").to(
        device="cuda",
        dtype=weight_dtype,
    )
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
    ).to(device="cuda")
    light_guider = LightGuider(conditioning_channels=cfg.model.light_channels).to(
        device="cuda",
        dtype=weight_dtype,
    )

    lpips_weight = float(cfg.loss.get("lpips_weight", 0.0))
    mse_weight = float(cfg.loss.get("mse_weight", 1.0))
    net_lpips = None
    if lpips_weight > 0:
        import lpips

        net_lpips = lpips.LPIPS(net="vgg").cuda()
        net_lpips.requires_grad_(False)

    denoising_unet.load_state_dict(load_state_dict_compat(cfg.denoising_unet_path), strict=False)
    temporal_module_path = str(getattr(cfg, "temporal_module_path", "") or "").strip()
    if temporal_module_path and Path(temporal_module_path).is_file():
        denoising_unet.load_state_dict(load_state_dict_compat(temporal_module_path), strict=False)
        logger.info(f"Warm-started temporal modules from {temporal_module_path}")
    elif temporal_module_path:
        logger.warning(
            "temporal_module_path=%s was not found. Falling back to temporal weights already stored in denoising_unet_path.",
            temporal_module_path,
        )
    reference_unet.load_state_dict(load_state_dict_compat(cfg.reference_unet_path), strict=True)
    light_guider.load_state_dict(load_state_dict_compat(cfg.light_guider_path), strict=True)

    for module in denoising_unet.modules():
        motion_modules = getattr(module, "motion_modules", None)
        if isinstance(motion_modules, nn.ModuleList):
            for idx in range(len(motion_modules)):
                motion_modules[idx] = NoOpMotionModule()

    if bool(cfg.get("reset_temporal_modules", True)):
        total_reset_submodules = 0
        for module in denoising_unet.modules():
            temporal_modules = getattr(module, "temporal_modules", None)
            if isinstance(temporal_modules, nn.ModuleList):
                for temporal_module in temporal_modules:
                    total_reset_submodules += reset_module_parameters(temporal_module)
        logger.info(
            "Reinitialized temporal modules from scratch for per-frame-reference relight Stage 3 "
            "(reset %d submodules).",
            total_reset_submodules,
        )

    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    light_guider.requires_grad_(False)
    denoising_unet.requires_grad_(False)

    for name, module in denoising_unet.named_modules():
        if "temporal_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = PerFrameReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
        per_frame_reference=True,
    )
    reference_control_reader = PerFrameReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        per_frame_reference=True,
    )

    net = Net(
        denoising_unet=denoising_unet,
        reference_control_reader=reference_control_reader,
        enable_multiscale_light_injection=bool(cfg.model.get("enable_multiscale_light_injection", False)),
        multiscale_light_injection_scale=float(cfg.model.get("multiscale_light_injection_scale", 0.25)),
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    vae.decoder.gradient_checkpointing = True
    vae.decoder.training = True

    learning_rate = cfg.solver.learning_rate if not cfg.solver.scale_lr else (
        cfg.solver.learning_rate
        * cfg.solver.gradient_accumulation_steps
        * cfg.data.train_bs
        * accelerator.num_processes
    )

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, net.parameters())),
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = RelightVideoPairDataset(
        image_root=cfg.data.image_root,
        mpli_root=cfg.data.mpli_root,
        sample_list_path=cfg.data.sample_list_path,
        img_size=(cfg.data.train_width, cfg.data.train_height),
        n_sample_frames=cfg.data.n_sample_frames,
        reference_frame_start=cfg.data.reference_frame_start,
        target_frame_start=cfg.data.target_frame_start,
        random_clip_start=bool(cfg.data.get("random_clip_start", False)),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.solver.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process and mlflow is not None:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(cfg.exp_name, init_kwargs={"mlflow": {"run_name": run_time}})
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")
    elif accelerator.is_main_process and mlflow is None:
        logger.warning("mlflow is not installed in the current environment, so tracker logging is disabled.")

    logger.info("***** Running relight stage3 training with per-frame reference refresh *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size = "
        f"{cfg.data.train_bs * accelerator.num_processes * cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    first_epoch = 0
    if cfg.resume_from_checkpoint:
        resume_dir = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint != "latest" else save_dir
        dirs = [d for d in os.listdir(resume_dir) if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    temporal_window_size = int(cfg.temporal_window_size)
    temporal_adaptive_step = int(cfg.temporal_adaptive_step)
    n_sample_frames = int(cfg.data.n_sample_frames)
    if n_sample_frames % temporal_window_size != 0:
        raise ValueError(
            f"n_sample_frames={n_sample_frames} must be divisible by temporal_window_size={temporal_window_size}"
        )
    if len(cfg.timesteps_list) % temporal_adaptive_step != 0:
        raise ValueError(
            f"num_inference_steps={len(cfg.timesteps_list)} must be divisible by "
            f"temporal_adaptive_step={temporal_adaptive_step}"
        )
    windows = n_sample_frames // temporal_window_size

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            reference_video = batch["reference_video"].to(weight_dtype)
            target_video = batch["target_video"].to(weight_dtype)
            light_seq = batch["light_seq"].to(weight_dtype)
            clip_reference_video = batch["reference_clip_video"].to(dtype=image_enc.dtype, device=image_enc.device)

            video_length = target_video.shape[1]
            target_video_flat = rearrange(target_video, "b f c h w -> (b f) c h w")
            with torch.no_grad():
                latents = encode_latent_frames(
                    vae,
                    target_video_flat.to(dtype=vae.dtype, device=vae.device),
                    cfg.data.max_batch_frames,
                )
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                light_emb = light_guider(light_seq.to(dtype=light_guider.dtype, device=light_guider.device))

            pixel_values = rearrange(target_video_flat, "(b f) c h w -> b c f h w", f=video_length)
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            base_timesteps = torch.tensor(list(cfg.timesteps_list)[::-1], device=latents.device)
            base_timesteps = base_timesteps.repeat_interleave(temporal_window_size, dim=0).long()
            timesteps = base_timesteps.repeat(batch_size)

            uncond_fwd = random.random() < cfg.uncond_ratio
            if uncond_fwd:
                light_emb = torch.zeros_like(light_emb)

            init_frames = (temporal_adaptive_step - 1) * temporal_window_size
            noisy_latents = train_noise_scheduler.add_noise(
                latents[:, :, :init_frames],
                noise[:, :, :init_frames],
                base_timesteps[:init_frames].repeat(batch_size),
            )
            noisy_latents_all = torch.cat([noisy_latents, noise[:, :, init_frames:]], dim=2)

            mse_value = torch.tensor(0.0, device=latents.device)
            lpips_value = torch.tensor(0.0, device=latents.device)
            preview_reference_chunk = None
            preview_target_chunk = None
            preview_pred_chunk = None
            ref_latents_cache = None
            clip_embeds_cache = None
            chunk_frames = temporal_adaptive_step * temporal_window_size

            for i in range(windows - temporal_adaptive_step + 1):
                l = i * temporal_window_size
                r = (i + temporal_adaptive_step) * temporal_window_size

                reference_chunk = reference_video[:, l:r]
                clip_reference_chunk = clip_reference_video[:, l:r]
                with torch.no_grad():
                    if ref_latents_cache is None or clip_embeds_cache is None:
                        # The first temporal chunk has no cached reference state,
                        # so we encode the whole 16-frame chunk once.
                        reference_chunk_flat = rearrange(reference_chunk, "b f c h w -> (b f) c h w")
                        clip_reference_chunk_flat = rearrange(clip_reference_chunk, "b f c h w -> (b f) c h w")
                        ref_latents_cache = rearrange(
                            encode_latent_frames(
                                vae,
                                reference_chunk_flat.to(dtype=vae.dtype, device=vae.device),
                                cfg.data.max_batch_frames,
                            ),
                            "(b f) c h w -> b f c h w",
                            b=reference_chunk.shape[0],
                            f=chunk_frames,
                        )
                        clip_embeds_cache = rearrange(
                            encode_clip_frames(
                                image_enc,
                                clip_reference_chunk_flat,
                                cfg.data.max_batch_frames,
                            ),
                            "(b f) n c -> b f n c",
                            b=clip_reference_chunk.shape[0],
                            f=chunk_frames,
                        )
                    else:
                        # Subsequent windows overlap by 12 frames. Reuse those
                        # cached encodings and only append the newly entered 4
                        # frames before rebuilding the current chunk bank.
                        new_l = r - temporal_window_size
                        new_reference = reference_video[:, new_l:r]
                        new_clip_reference = clip_reference_video[:, new_l:r]
                        new_reference_flat = rearrange(new_reference, "b f c h w -> (b f) c h w")
                        new_clip_reference_flat = rearrange(new_clip_reference, "b f c h w -> (b f) c h w")
                        new_ref_latents = rearrange(
                            encode_latent_frames(
                                vae,
                                new_reference_flat.to(dtype=vae.dtype, device=vae.device),
                                cfg.data.max_batch_frames,
                            ),
                            "(b f) c h w -> b f c h w",
                            b=new_reference.shape[0],
                            f=temporal_window_size,
                        )
                        new_clip_embeds = rearrange(
                            encode_clip_frames(
                                image_enc,
                                new_clip_reference_flat,
                                cfg.data.max_batch_frames,
                            ),
                            "(b f) n c -> b f n c",
                            b=new_clip_reference.shape[0],
                            f=temporal_window_size,
                        )
                        ref_latents_cache = torch.cat(
                            [ref_latents_cache[:, temporal_window_size:], new_ref_latents],
                            dim=1,
                        )
                        clip_embeds_cache = torch.cat(
                            [clip_embeds_cache[:, temporal_window_size:], new_clip_embeds],
                            dim=1,
                        )

                    ref_image_latents_chunk = rearrange(
                        ref_latents_cache,
                        "b f c h w -> (b f) c h w",
                    )
                    clip_image_embeds_chunk = rearrange(
                        clip_embeds_cache,
                        "b f n c -> (b f) n c",
                    )

                reference_control_reader.clear()
                reference_control_writer.clear()
                if not uncond_fwd:
                    reference_unet(
                        ref_image_latents_chunk,
                        torch.zeros(
                            (ref_image_latents_chunk.shape[0],),
                            device=ref_image_latents_chunk.device,
                            dtype=torch.long,
                        ),
                        encoder_hidden_states=clip_image_embeds_chunk,
                        return_dict=False,
                    )
                    reference_control_reader.update(
                        reference_control_writer,
                        drop_ratio=float(cfg.get("reference_drop_ratio", 0.0)),
                    )

                noisy_latents_chunk = noisy_latents_all[:, :, l:r].detach()
                light_emb_chunk = light_emb[:, :, l:r]

                if i % (temporal_adaptive_step - 1) != 0:
                    with torch.no_grad():
                        model_pred = net(
                            noisy_latents_chunk,
                            timesteps,
                            clip_image_embeds_chunk,
                            light_emb_chunk,
                        )
                        latents_pred = get_x0_from_eps(train_noise_scheduler, noisy_latents_chunk, model_pred, timesteps)
                        clip_length = model_pred.shape[2]
                        mid_noise_pred = rearrange(model_pred, "b c f h w -> (b f) c h w")
                        mid_latents = rearrange(noisy_latents_chunk, "b c f h w -> (b f) c h w")
                        prev_latents, _ = train_noise_scheduler.step(
                            mid_noise_pred,
                            timesteps,
                            mid_latents,
                            return_dict=False,
                        )
                        prev_latents = rearrange(
                            prev_latents,
                            "(b f) c h w -> b c f h w",
                            b=batch_size,
                            f=clip_length,
                        )
                        noisy_latents_all[:, :, l:r] = torch.cat(
                            [latents_pred[:, :, :temporal_window_size], prev_latents[:, :, temporal_window_size:]],
                            dim=2,
                        ).detach()
                else:
                    with accelerator.accumulate(net):
                        model_pred = net(
                            noisy_latents_chunk,
                            timesteps,
                            clip_image_embeds_chunk,
                            light_emb_chunk,
                        )
                        latents_pred = get_x0_from_eps(train_noise_scheduler, noisy_latents_chunk, model_pred, timesteps)
                        image_pred = decode_latents(vae, latents_pred)
                        pixel_values_tgt = pixel_values[:, :, l:r]

                        mse_value = F.mse_loss(image_pred.float(), pixel_values_tgt.float(), reduction="mean")
                        loss = mse_value * mse_weight

                        if net_lpips is not None and lpips_weight > 0:
                            image_pred_frames = rearrange(image_pred, "b c f h w -> (b f) c h w")
                            pixel_values_tgt_frames = rearrange(pixel_values_tgt, "b c f h w -> (b f) c h w")
                            lpips_value = net_lpips(image_pred_frames.float(), pixel_values_tgt_frames.float()).mean()
                            loss = loss + lpips_value * lpips_weight
                        else:
                            lpips_value = torch.tensor(0.0, device=latents.device)

                        preview_reference_chunk = rearrange(reference_chunk.detach(), "b f c h w -> b c f h w")
                        preview_target_chunk = pixel_values_tgt.detach()
                        preview_pred_chunk = latents_pred.detach()

                        clip_length = model_pred.shape[2]
                        mid_noise_pred = rearrange(model_pred, "b c f h w -> (b f) c h w")
                        mid_latents = rearrange(noisy_latents_chunk, "b c f h w -> (b f) c h w")
                        prev_latents, _ = train_noise_scheduler.step(
                            mid_noise_pred,
                            timesteps,
                            mid_latents,
                            return_dict=False,
                        )
                        prev_latents = rearrange(
                            prev_latents,
                            "(b f) c h w -> b c f h w",
                            b=batch_size,
                            f=clip_length,
                        )
                        noisy_latents_all[:, :, l:r] = torch.cat(
                            [latents_pred[:, :, :temporal_window_size], prev_latents[:, :, temporal_window_size:]],
                            dim=2,
                        )

                        avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                        train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                list(filter(lambda p: p.requires_grad, net.parameters())),
                                cfg.solver.max_grad_norm,
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                reference_control_reader.clear()
                reference_control_writer.clear()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "mse": mse_value.detach().item(),
                        "lpips": lpips_value.detach().item(),
                    },
                    step=global_step,
                )
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    delete_additional_ckpt(save_dir, 1)
                    accelerator.save_state(save_path)
                    save_temporal_checkpoint(
                        accelerator.unwrap_model(net).denoising_unet,
                        save_dir,
                        "temporal_module",
                        global_step,
                        total_limit=2,
                    )

                if (
                    accelerator.is_main_process
                    and hasattr(cfg, "preview_every_steps")
                    and cfg.preview_every_steps > 0
                    and global_step % cfg.preview_every_steps == 0
                    and preview_reference_chunk is not None
                    and preview_target_chunk is not None
                    and preview_pred_chunk is not None
                ):
                    save_preview(
                        vae=vae,
                        reference_videos=preview_reference_chunk,
                        target_videos=preview_target_chunk,
                        pred_latents=preview_pred_chunk,
                        save_dir=save_dir,
                        global_step=global_step,
                    )

            logs = {
                "loss": float(loss.detach().item()),
                "mse": float(mse_value.detach().item()),
                "lpips": float(lpips_value.detach().item()),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            if global_step >= cfg.solver.max_train_steps:
                break

        if global_step >= cfg.solver.max_train_steps:
            break

    if accelerator.is_main_process and bool(cfg.get("save_final_model", True)):
        save_temporal_checkpoint(
            accelerator.unwrap_model(net).denoising_unet,
            save_dir,
            "temporal_module",
            global_step,
            total_limit=2,
        )

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(description="Relight Stage 3 training with per-frame reference refresh")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    main(config)
