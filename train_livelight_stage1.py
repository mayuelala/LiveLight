import argparse
import logging
import math
import os
import os.path as osp
import warnings
from datetime import datetime

import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
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
from torchvision.utils import save_image

from src.dataset.livelight_image import RelightImageDataset
from src.models.light_guider import LightGuider
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything


def load_state_dict_compat(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def find_latest_component_ckpt(ckpt_dir, prefix):
    candidates = sorted(
        [p for p in os.listdir(ckpt_dir) if p.startswith(f"{prefix}-") and p.endswith('.pth')],
        key=lambda name: int(name.split('-')[-1].split('.')[0]),
    )
    if not candidates:
        raise FileNotFoundError(f'No {prefix} checkpoint found under {ckpt_dir}')
    return os.path.join(ckpt_dir, candidates[-1])

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

try:
    import mlflow
except ImportError:
    mlflow = None

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        light_guider: LightGuider,
        reference_control_writer,
        reference_control_reader,
        enable_multiscale_light_injection: bool = False,
        multiscale_light_injection_scale: float = 0.25,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.light_guider = light_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.enable_multiscale_light_injection = enable_multiscale_light_injection
        self.multiscale_light_injection_scale = multiscale_light_injection_scale

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        light_cond,
        uncond_fwd: bool = False,
    ):
        light_emb = self.light_guider(light_cond)

        if uncond_fwd:
            light_emb = torch.zeros_like(light_emb)
        else:
            self.reference_unet(
                ref_image_latents,
                torch.zeros_like(timesteps[: noisy_latents.shape[0]]),
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer, drop_ratio=0.0)

        # Original one-shot light injection path kept for reference:
        # model_pred = self.denoising_unet(
        #     noisy_latents,
        #     timesteps,
        #     encoder_hidden_states=clip_image_embeds,
        #     pose_cond_fea=light_emb,
        #     skip_mm=True,
        # ).sample
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
            pose_cond_fea=light_emb,
            skip_mm=True,
            use_multiscale_pose_cond=self.enable_multiscale_light_injection,
            pose_cond_scale=self.multiscale_light_injection_scale,
        ).sample
        return model_pred


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


def save_preview(vae, source_images, target_images, pred_latents, save_dir, global_step):
    preview_dir = osp.join(save_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    with torch.no_grad():
        pred_images = vae.decode((pred_latents[:, :, 0] / 0.18215).to(dtype=vae.dtype)).sample

    triptych = torch.cat(
        [
            ((source_images[:1].float() + 1.0) / 2.0).clamp(0.0, 1.0),
            ((target_images[:1].float() + 1.0) / 2.0).clamp(0.0, 1.0),
            ((pred_images[:1].float() + 1.0) / 2.0).clamp(0.0, 1.0),
        ],
        dim=-1,
    ).cpu()
    save_image(triptych, osp.join(preview_dir, f"step_{global_step:06d}.png"))


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dtype=weight_dtype, device="cuda")

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
    ).to(device="cuda")
    light_guider = LightGuider(conditioning_channels=cfg.model.light_channels).to(device="cuda")

    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(True)
    denoising_unet.requires_grad_(True)
    light_guider.requires_grad_(True)

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

    enable_multiscale_light_injection = bool(cfg.model.get('enable_multiscale_light_injection', False))
    multiscale_light_injection_scale = float(cfg.model.get('multiscale_light_injection_scale', 0.25))

    # Original constructor kept for reference:
    # net = Net(
    #     reference_unet=reference_unet,
    #     denoising_unet=denoising_unet,
    #     light_guider=light_guider,
    #     reference_control_writer=reference_control_writer,
    #     reference_control_reader=reference_control_reader,
    # )
    net = Net(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        light_guider=light_guider,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        enable_multiscale_light_injection=enable_multiscale_light_injection,
        multiscale_light_injection_scale=multiscale_light_injection_scale,
    )

    warm_start_dir = cfg.get('warm_start_dir', '')
    warm_start_step = int(cfg.get('warm_start_step', 0))
    if warm_start_dir:
        reference_unet_ckpt = find_latest_component_ckpt(warm_start_dir, 'reference_unet')
        denoising_unet_ckpt = find_latest_component_ckpt(warm_start_dir, 'denoising_unet')
        light_guider_ckpt = find_latest_component_ckpt(warm_start_dir, 'light_guider')

        reference_unet.load_state_dict(load_state_dict_compat(reference_unet_ckpt), strict=True)
        denoising_missing = denoising_unet.load_state_dict(
            load_state_dict_compat(denoising_unet_ckpt), strict=False
        )
        light_guider.load_state_dict(load_state_dict_compat(light_guider_ckpt), strict=True)

        logger.info(f"Warm-started reference_unet from {reference_unet_ckpt}")
        logger.info(f"Warm-started denoising_unet from {denoising_unet_ckpt}")
        logger.info(f"Warm-started light_guider from {light_guider_ckpt}")
        logger.info(
            "Warm-start denoising_unet missing keys: %s ; unexpected keys: %s",
            denoising_missing.missing_keys,
            denoising_missing.unexpected_keys,
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

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
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

    train_dataset = RelightImageDataset(
        image_root=cfg.data.image_root,
        mpli_root=cfg.data.mpli_root,
        sample_list_path=cfg.data.sample_list_path,
        img_size=(cfg.data.train_height, cfg.data.train_width),
        source_frame_id=cfg.data.source_frame_id,
        lit_frame_start=cfg.data.lit_frame_start,
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
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")
    elif accelerator.is_main_process:
        logger.warning("mlflow is not installed in the current environment, so tracker logging is disabled.")

    total_batch_size = cfg.data.train_bs * accelerator.num_processes * cfg.solver.gradient_accumulation_steps
    logger.info("***** Running relight stage1 training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    first_epoch = 0
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch
    elif warm_start_step > 0:
        global_step = warm_start_step
        first_epoch = global_step // num_update_steps_per_epoch
        accelerator.print(f"Warm-starting model weights from step {warm_start_step}")

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                pixel_values = batch["target_img"].to(weight_dtype)
                source_images = batch["source_img"].to(weight_dtype)
                light_cond = batch["light"].to(device=pixel_values.device, dtype=weight_dtype).unsqueeze(2)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample().unsqueeze(2)
                    latents = latents * 0.18215
                    ref_image_latents = vae.encode(source_images).latent_dist.sample() * 0.18215

                    image_enc_device = next(image_enc.parameters()).device
                    clip_img = batch["clip_image"].to(dtype=image_enc.dtype, device=image_enc_device)
                    clip_image_embeds = image_enc(clip_img.to(image_enc_device, dtype=weight_dtype)).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                uncond_fwd = torch.rand(1).item() < cfg.uncond_ratio

                noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)

                reference_control_reader.clear()
                reference_control_writer.clear()
                model_pred = net(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    ref_image_latents=ref_image_latents,
                    clip_image_embeds=image_prompt_embeds,
                    light_cond=light_cond,
                    uncond_fwd=uncond_fwd,
                )
                latents_pred = get_x0_from_eps(train_noise_scheduler, noisy_latents, model_pred, timesteps)
                loss = F.mse_loss(latents_pred.float(), latents.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    delete_additional_ckpt(save_dir, 1)
                    accelerator.save_state(save_path)

                if (
                    accelerator.is_main_process
                    and hasattr(cfg, "preview_every_steps")
                    and cfg.preview_every_steps > 0
                    and global_step % cfg.preview_every_steps == 0
                ):
                    save_preview(
                        vae=vae,
                        source_images=source_images,
                        target_images=pixel_values,
                        pred_latents=latents_pred,
                        save_dir=save_dir,
                        global_step=global_step,
                    )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= cfg.solver.max_train_steps:
                break

        if (epoch + 1) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=2)
            save_checkpoint(unwrap_net.reference_unet, save_dir, "reference_unet", global_step, total_limit=2)
            save_checkpoint(unwrap_net.light_guider, save_dir, "light_guider", global_step, total_limit=2)

    unwrap_net = accelerator.unwrap_model(net)
    save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=2)
    save_checkpoint(unwrap_net.reference_unet, save_dir, "reference_unet", global_step, total_limit=2)
    save_checkpoint(unwrap_net.light_guider, save_dir, "light_guider", global_step, total_limit=2)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))
        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            for removing_checkpoint in checkpoints[:num_to_remove]:
                os.remove(os.path.join(save_dir, removing_checkpoint))

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/relight_stage1.yaml")
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        config = OmegaConf.load(args.config)
    elif args.config.endswith(".py"):
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Unsupported config format")
    main(config)
