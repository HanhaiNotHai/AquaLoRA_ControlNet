from typing import Dict

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttentionProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import ImageProjection


class CustomUNet2DConditionLoadersMixin:

    def custom_load_ip_adapter_weights(self, state_dict):
        from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        for name in self.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]
            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = (
                    AttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else AttnProcessor
                )
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = (
                    CustomIPAdapterAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else CustomIPAdapterAttnProcessor
                )
                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
                ).to(dtype=self.dtype, device=self.device)

                value_dict = {}
                for k, w in attn_procs[name].state_dict().items():
                    value_dict.update({f"{k}": state_dict["ip_adapter"][f"{key_id}.{k}"]})

                attn_procs[name].load_state_dict(value_dict)
                key_id += 2

        self.set_attn_processor(attn_procs)

        # create image projection layers.
        clip_embeddings_dim = state_dict["image_proj"]["proj.weight"].shape[-1]
        cross_attention_dim = state_dict["image_proj"]["proj.weight"].shape[0] // 4

        image_projection = ImageProjection(
            cross_attention_dim=cross_attention_dim,
            image_embed_dim=clip_embeddings_dim,
            num_image_text_embeds=4,
        )
        image_projection.to(dtype=self.dtype, device=self.device)

        # load image projection layer weights
        image_proj_state_dict = {}
        image_proj_state_dict.update(
            {
                "image_embeds.weight": state_dict["image_proj"]["proj.weight"],
                "image_embeds.bias": state_dict["image_proj"]["proj.bias"],
                "norm.weight": state_dict["image_proj"]["norm.weight"],
                "norm.bias": state_dict["image_proj"]["norm.bias"],
            }
        )

        image_projection.load_state_dict(image_proj_state_dict)

        self.encoder_hid_proj = image_projection.to(device=self.device, dtype=self.dtype)
        self.config.encoder_hid_dim_type = "ip_image_proj"


class CustomUNet2DConditionModel(UNet2DConditionModel, CustomUNet2DConditionLoadersMixin):

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(
                    return_deprecated_lora=False
                )

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors


class CustomIPAdapterAttnProcessor(IPAdapterAttnProcessor):

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # split hidden states
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomIPAdapterAttnProcessor2_0(IPAdapterAttnProcessor2_0):

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # split hidden states
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
