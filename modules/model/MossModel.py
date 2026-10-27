from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.model.moss.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
)
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import AutoencoderKLFlux2, DiffusionPipeline
from transformers import Qwen2Tokenizer, Qwen3_5Model

# The final pretraining stage's text limit: the whole chat-templated condition.
PROMPT_MAX_LENGTH = 4096

# Pixels per transformer latent cell: the FLUX.2 VAE downsamples by 8 and the
# 2x2 cells are packed into one 128-channel token.
LATENT_DOWNSCALE = 16


def format_input(text: str):
    return [
        {"role": "user", "content": text},
    ]


class MossModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: MOSSImageScheduler | None
    text_encoder: Qwen3_5Model | None
    vae: AutoencoderKLFlux2 | None
    transformer: MOSSImageTransformer | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent lora training data
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
        self,
        model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [
            a
            for a in [
                self.transformer_lora,
            ]
            if a is not None
        ]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if (
                self.text_encoder_offload_conductor is not None
                and self.text_encoder_offload_conductor.layer_offload_activated()
            ):
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if (
            self.transformer_offload_conductor is not None
            and self.transformer_offload_conductor.layer_offload_activated()
        ):
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return MOSSImagePipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(
        self,
        train_device: torch.device,
        batch_size: int = 1,
        rand: Random | None = None,
        text: str | list[str] = None,
        tokens: Tensor = None,
        tokens_mask: Tensor = None,
        text_encoder_dropout_probability: float | None = None,
        text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """Final-layer text hidden states and their bool attention mask, both
        cut to the batch's longest prompt. ``text`` is templated in place.

        Text is tokenized to its own length rather than PROMPT_MAX_LENGTH: the
        encoder is causal and padding sits on the right, so the real-token
        states are unchanged and the ~4000 pad tokens are never encoded."""
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            for i, prompt_item in enumerate(text):
                text[i] = self.tokenizer.apply_chat_template(
                    format_input(prompt_item),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )

            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                text_encoder_output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError  # https://github.com/Nerogar/OneTrainer/issues/957

        text_encoder_output = text_encoder_output.to(device=train_device)
        bool_attention_mask = tokens_mask.to(device=train_device).bool()

        # padding is on the right, so the batch can be cut to its longest prompt
        seq_lengths = bool_attention_mask.sum(dim=1)
        max_seq_length = int(seq_lengths.max().item())
        if max_seq_length == 0:
            raise ValueError("MOSS needs at least one text token per sample; the chat template always adds some.")

        # round up to a multiple of 16 so the attention kernels see fewer distinct shapes
        if max_seq_length % 16 != 0 and (seq_lengths != seq_lengths[0]).any():
            max_seq_length += 16 - max_seq_length % 16

        if text_encoder_output.shape[1] >= max_seq_length:
            text_encoder_output = text_encoder_output[:, :max_seq_length]
            bool_attention_mask = bool_attention_mask[:, :max_seq_length]
        else:
            pad = max_seq_length - text_encoder_output.shape[1]
            text_encoder_output = F.pad(text_encoder_output, (0, 0, 0, pad))
            bool_attention_mask = F.pad(bool_attention_mask, (0, pad))

        return text_encoder_output, bool_attention_mask

    def scale_latents(self, latents: Tensor) -> Tensor:
        # the FLUX.2 VAE emits 32-channel latents at 1/8 scale; the transformer consumes
        # 2x2-packed 128-channel latents at 1/16 scale, normalized with the VAE's 128-channel
        # batch-norm running statistics.
        b, c, h, w = latents.shape
        packed = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        mean = bn.running_mean.to(device=packed.device, dtype=packed.dtype).view(1, -1, 1, 1)
        std = (bn.running_var + eps).sqrt().to(device=packed.device, dtype=packed.dtype).view(1, -1, 1, 1)
        return (packed - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        # inverse of scale_latents: denormalize in packed space, then unpack the 2x2 cells
        b, c, h, w = latents.shape
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        mean = bn.running_mean.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        std = (bn.running_var + eps).sqrt().to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        denormed = latents * std + mean
        return denormed.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)

    def calculate_timestep_shift(self, latent_width: int, latent_height: int) -> float:
        # MOSS was pretrained without a resolution-dependent shift: the final stage
        # (technical report, Table 6) samples on the unshifted grid at every
        # resolution, which the scheduler records as training_shift. Its shift=5.0
        # is for inference only.
        return float(self.noise_scheduler.config.training_shift)
