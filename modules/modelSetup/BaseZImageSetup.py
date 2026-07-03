from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_qwen3_encoder_layers,
    enable_checkpointing_for_z_image_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseZImageSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "full": [],
        "blocks": ["layers"],
        "attn-mlp": {
            "patterns": ["^(?=.*attention)(?!.*refiner).*", "^(?=.*feed_forward)(?!.*refiner).*"],
            "regex": True,
        },
        "attn-only": {"patterns": ["^(?=.*attention)(?!.*refiner).*"], "regex": True},
    }

    def setup_optimizations(
        self,
        model: ZImageModel,
        config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = enable_checkpointing_for_z_image_transformer(
                model.transformer, config
            )
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3_encoder_layers(
                    model.text_encoder, config
                )

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        # TODO necessary if we don't train it?
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def predict(
        self,
        model: ZImageModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            batch_size = batch["latent_image"].shape[0]
            distill_conditioning = batch.get("_distill_conditioning", "conditional")
            if distill_conditioning == "unconditional":
                negative_prompt = batch.get("distill_negative_prompt", batch.get("negative_prompt", ""))
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * batch_size
                text_encoder_output = model.encode_text(
                    train_device=self.train_device,
                    batch_size=batch_size,
                    rand=rand,
                    text=list(negative_prompt),
                    text_encoder_dropout_probability=None,
                )
            else:
                text_encoder_output = model.encode_text(
                    train_device=self.train_device,
                    batch_size=batch_size,
                    rand=rand,
                    tokens=batch.get("tokens"),
                    tokens_mask=batch.get("tokens_mask"),
                    text_encoder_output=batch.get("text_encoder_hidden_state"),
                    text_encoder_dropout_probability=config.text_encoder.dropout_probability
                    if not deterministic
                    else None,
                )
            scaled_latent_image = model.scale_latents(batch["latent_image"])

            if "_distill_latent_input" in batch:
                scaled_noisy_latent_image = batch["_distill_latent_input"].to(
                    device=scaled_latent_image.device, dtype=scaled_latent_image.dtype
                )
                sigma = batch["_distill_sigma"].to(device=scaled_latent_image.device, dtype=torch.float32).flatten()
                latent_input = scaled_noisy_latent_image.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype())
                latent_input_list = list(latent_input.unbind(dim=0))
                # CONCURRENT distill mode fuses several latents per prompt into
                # one forward; tile the per-sample embeddings to match.
                if len(latent_input_list) != len(text_encoder_output):
                    repeat, remainder = divmod(len(latent_input_list), len(text_encoder_output))
                    if remainder != 0:
                        raise RuntimeError(
                            f"Distill latent batch ({len(latent_input_list)}) is not a multiple of the "
                            f"conditioning batch ({len(text_encoder_output)})."
                        )
                    text_encoder_output = list(text_encoder_output) * repeat
                transformer_timestep = 1.0 - sigma

                output_list = model.transformer(
                    latent_input_list, transformer_timestep, text_encoder_output, return_dict=True
                ).sample

                predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)
                while sigma.dim() < predicted_flow.dim():
                    sigma = sigma.unsqueeze(-1)
                predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                return {
                    "loss_type": "target",
                    "timestep": transformer_timestep,
                    "predicted": predicted_flow,
                    "target": torch.zeros_like(predicted_flow),
                    "latent_image": scaled_latent_image,
                    "noisy_latent_image": scaled_noisy_latent_image,
                    "sigma": sigma,
                    "predicted_latent": predicted_scaled_latent_image,
                }

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config["num_train_timesteps"],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            if not deterministic:
                scaled_noisy_latent_image, latent_noise = self._apply_ciop(
                    scaled_noisy_latent_image,
                    latent_noise,
                    config,
                    generator,
                    rand,
                )

            latent_input = scaled_noisy_latent_image.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype())
            latent_input_list = list(latent_input.unbind(dim=0))

            output_list = model.transformer(
                latent_input_list, (1000 - timestep) / 1000, text_encoder_output, return_dict=True
            ).sample

            predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)

            flow = latent_noise - scaled_latent_image
            predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
            model_output_data = {
                "loss_type": "target",
                "timestep": timestep,
                "predicted": predicted_flow,
                "target": flow,
                "latent_image": scaled_latent_image,
                "noisy_latent_image": scaled_noisy_latent_image,
                "sigma": sigma,
                "predicted_latent": predicted_scaled_latent_image,
            }

            if config.debug_mode:
                with torch.no_grad():
                    self._save_tokens("7-prompt", batch["tokens"], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
        self,
        model: ZImageModel,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    @staticmethod
    def _lpips_vae_colocated(vae_device: torch.device, train_device: torch.device) -> bool:
        """Whether the VAE sits on the compute device, so LPIPS can decode.

        Compares by device *type*, treating an unindexed device (``cuda``) as
        compatible with its indexed form (``cuda:0``). A strict ``!=`` skips LPIPS
        even after the VAE is moved on-device for validation, because
        ``torch.device("cuda") != torch.device("cuda:0")``. Distinct indices
        (``cuda:1`` vs ``cuda:0``) and a CPU-offloaded VAE still count as not
        co-located.
        """
        if vae_device.type != train_device.type:
            return False
        if vae_device.index is None or train_device.index is None:
            return True
        return vae_device.index == train_device.index

    def _distill_perceptual_loss(
        self,
        model: ZImageModel,
        generated_scaled_latent: Tensor,
        target_scaled_latent: Tensor,
        config: TrainConfig,
    ) -> Tensor | None:
        """Optional image-space LPIPS term for the teacher-match / endpoint loss.

        Decodes both latents through the VAE and runs LPIPS. Skips (returns None)
        when the VAE is offloaded, and — for any transient error — skips this call
        but retries next time, so a fixable issue self-heals. Only a missing lpips
        package hard-disables the term for the run. Never aborts training.
        """
        if getattr(self, "_distill_lpips_disabled", False):
            return None
        try:
            vae_device = next(model.vae.parameters()).device
            if not self._lpips_vae_colocated(vae_device, self.train_device):
                # VAE is offloaded (latent caching); decoding here would thrash
                # devices every step. Skip rather than pay that cost silently.
                return None

            lpips_net = getattr(self, "_distill_lpips_net", None)
            if lpips_net is None:
                try:
                    import lpips
                except ImportError:
                    # Not installed and won't change mid-run: hard-disable.
                    print("Distill LPIPS term disabled: the 'lpips' package is not installed.")
                    self._distill_lpips_disabled = True
                    return None

                lpips_net = lpips.LPIPS(net="vgg").to(self.train_device)
                lpips_net.eval()
                for param in lpips_net.parameters():
                    param.requires_grad_(False)
                self._distill_lpips_net = lpips_net

            # Decode in the VAE's own dtype. The cached latents / train_dtype can be
            # bf16 while the VAE weights are float32 (or vice versa), which would
            # otherwise raise "Input type and bias type should be the same".
            vae_dtype = next(model.vae.parameters()).dtype
            generated_image = model.vae.decode(
                model.unscale_latents(generated_scaled_latent.to(dtype=vae_dtype)),
                return_dict=False,
            )[0]
            with torch.no_grad():
                target_image = model.vae.decode(
                    model.unscale_latents(target_scaled_latent.to(dtype=vae_dtype)),
                    return_dict=False,
                )[0]

            return lpips_net(generated_image.float().clamp(-1, 1), target_image.float().clamp(-1, 1)).mean()
        except Exception as exception:  # noqa: BLE001 - never let LPIPS abort a run
            # Transient/other error: skip this round but retry next validation
            # rather than silently killing the metric for the whole run.
            print(f"Distill LPIPS term skipped after error: {exception}")
            return None

    def prepare_text_caching(self, model: ZImageModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
