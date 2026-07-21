from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.models.attention import TemporalBasicTransformerBlock
from .attention import BasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class PerFrameReferenceAttentionControl:
    """Reference controller that supports one reference frame per video frame.

    The original PersonaLive controller assumes a single static reference image
    and repeats its bank across the temporal dimension in read mode. This
    variant keeps the same hook structure but allows writer banks that are
    already flattened as `(batch * frames, tokens, channels)`.
    """

    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
        cache_kv=False,
        per_frame_reference=False,
    ) -> None:
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.cache_kv = cache_kv
        self.per_frame_reference = per_frame_reference
        self._hooked_modules = []
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
            cache_kv=self.cache_kv,
            per_frame_reference=self.per_frame_reference,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
        cache_kv=False,
        per_frame_reference=False,
    ):
        mode_flag = mode
        fusion_blocks = fusion_blocks
        cache_kv = cache_kv

        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def _align_bank_batch(bank_tensor: torch.Tensor, target_batch: int, video_length: int) -> torch.Tensor:
            if per_frame_reference:
                if bank_tensor.shape[0] == target_batch:
                    return bank_tensor
                if bank_tensor.shape[0] > 0 and target_batch % bank_tensor.shape[0] == 0:
                    repeat_factor = target_batch // bank_tensor.shape[0]
                    return bank_tensor.repeat_interleave(repeat_factor, dim=0)
                raise ValueError(
                    f"Per-frame reference bank batch mismatch: bank={bank_tensor.shape[0]}, "
                    f"target={target_batch}"
                )

            return rearrange(
                bank_tensor.unsqueeze(1).repeat(1, video_length, 1, 1),
                "b t l c -> (b t) l c",
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if mode_flag == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

                if mode_flag == "read":
                    bank_fea = [
                        _align_bank_batch(d, norm_hidden_states.shape[0], video_length)
                        for d in self.bank
                    ]

                    if self.kv_bank is not None and cache_kv:
                        if per_frame_reference:
                            ahead_fea = _align_bank_batch(
                                rearrange(self.kv_bank, "b n l c -> b (n l) c"),
                                norm_hidden_states.shape[0],
                                video_length,
                            )
                        else:
                            ahead_fea = self.kv_bank.unsqueeze(1).repeat(1, video_length, 1, 1, 1)
                            ahead_fea = rearrange(ahead_fea, "b t n l c -> (b t) (n l) c")
                        bank_fea.append(ahead_fea)

                    modify_norm_hidden_states = torch.cat([norm_hidden_states] + bank_fea, dim=1)

                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )

                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        if self.kv_bank is not None:
                            modify_norm_hidden_states = torch.cat([norm_hidden_states, ahead_fea], dim=1)
                        else:
                            modify_norm_hidden_states = norm_hidden_states
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=modify_norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    if self.attn2 is not None:
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )

                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states

        if self.reference_attn:
            if fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks))
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            # Persist the exact module objects that were patched here. Stage 3
            # later clears and updates reference banks repeatedly; recomputing
            # the module list from the whole UNet can pick up temporal blocks
            # that were never hooked, which then do not own a `bank` field.
            self._hooked_modules = list(dict.fromkeys(attn_modules))

            for i, module in enumerate(self._hooked_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )
                module.bank = []
                module.kv_bank = None
                module.attn_weight = float(i) / float(len(self._hooked_modules))

    def _get_reader_modules(self):
        return [
            module
            for module in self._hooked_modules
            if isinstance(module, TemporalBasicTransformerBlock)
        ]

    def _get_writer_modules(self):
        return [
            module
            for module in self._hooked_modules
            if isinstance(module, BasicTransformerBlock)
        ]

    def update(self, writer, dtype=torch.float16, drop_ratio=0.0):
        if self.reference_attn:
            reader_attn_modules = self._get_reader_modules()
            writer_attn_modules = writer._get_writer_modules()
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                if drop_ratio > 0:
                    r.bank = []
                    for v in w.bank:
                        n, l, d = v.shape
                        len_keep = int(l * (1 - drop_ratio))
                        noise = torch.rand(n, l)
                        ids_shuffle = torch.argsort(noise, dim=1)
                        ids_keep = ids_shuffle[:, :len_keep].to(v.device)
                        visible_tokens = torch.gather(
                            v.clone(),
                            dim=1,
                            index=ids_keep.unsqueeze(-1).repeat(1, 1, d),
                        )
                        r.bank.append(visible_tokens.to(dtype))
                else:
                    r.bank = [v.clone().to(dtype) for v in w.bank]

    def update_hkf(self, writer, dtype=torch.float16, drop_ratio=0.0):
        if self.reference_attn:
            reader_attn_modules = self._get_reader_modules()
            writer_attn_modules = writer._get_writer_modules()
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                if r.kv_bank is None:
                    r.kv_bank = torch.cat([v.clone().unsqueeze(1).to(dtype) for v in w.bank], dim=1)
                else:
                    r.kv_bank = torch.cat(
                        [r.kv_bank] + [v.clone().unsqueeze(1).to(dtype) for v in w.bank],
                        dim=1,
                    ).to(dtype)

    def clear(self):
        if self.reference_attn:
            for r in self._hooked_modules:
                r.bank.clear()
                if self.cache_kv:
                    r.kv_bank = None
