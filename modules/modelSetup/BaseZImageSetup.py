from abc import ABCMeta
from collections.abc import Callable
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupPerceptualLossMixin import ModelSetupPerceptualLossMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.attention_backend_util import apply_attention_backend
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
from modules.util.transfer_util import (
    TransferForkState,
    high_frequency_ratio,
    run_late_stack,
    run_prefix_and_capture,
    split_layer_key,
    swap_module_weights,
)

import torch
from torch import Tensor

# Noise level of the transfer sharpness probe: high enough that the one-step
# prediction is dominated by what the model invents rather than what it is shown.
TRANSFER_SHARPNESS_SIGMA = 0.9


class BaseZImageSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupPerceptualLossMixin,
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
        apply_attention_backend(model.transformer, config.attention_backend)

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

    def _prepare_flow_inputs(
        self,
        model: ZImageModel,
        config: TrainConfig,
        deterministic: bool,
        generator: torch.Generator,
        rand: Random,
        scaled_latent_image: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
        """The standard training branch's noise/timestep preamble, shared with
        transfer_fork_forward so transfer activations stay exactly on the
        training distribution. Behavior-preserving extraction from predict().

        Returns (latent_noise, timestep, sigma, scaled_noisy_latent_image,
        latent_input_list)."""
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
        return latent_noise, timestep, sigma, scaled_noisy_latent_image, latent_input_list

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

            latent_noise, timestep, sigma, scaled_noisy_latent_image, latent_input_list = self._prepare_flow_inputs(
                model, config, deterministic, generator, rand, scaled_latent_image
            )

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

    def _transfer_swap_modules(self, model: ZImageModel, config: TrainConfig) -> dict[str, torch.nn.Module]:
        """Bare-stem -> orig Linear for EVERY late adapter-covered module (all
        stems the adapter wraps at layers >= transfer_late_start, families or
        not -- the teacher education must also land on e.g. attention.to_out.0).
        Cached: the module set is fixed for the run."""
        cache = getattr(self, "_transfer_swap_modules_cache", None)
        if cache is None:
            cache = {}
            for name, peft_module in model.transformer_lora.lora_modules.items():
                parsed = split_layer_key(name)
                if parsed is not None and parsed[0] >= config.transfer_late_start:
                    cache[parsed[1]] = peft_module.orig_module
            self._transfer_swap_modules_cache = cache
        return cache

    def transfer_fork_forward(
        self,
        model: ZImageModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
        seed: int | None = None,
        latent_input: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> TransferForkState:
        """Run the shared prefix (embedders, refiners, layers < late_start) once
        and capture the fork block's inputs. The student adapter stays hooked:
        its early/mid parameters are the transplanted education, which is
        exactly correct there, so this prefix equals the teacher's prefix.

        ``latent_input``/``sigma`` (both or neither) inject an explicit noisy
        latent at an explicit sigma instead of noising the batch's image latent
        -- redistill's student-trajectory samples, using the same timestep
        convention as predict()'s ``_distill_latent_input`` path."""
        if (latent_input is None) != (sigma is None):
            raise ValueError("latent_input and sigma must be provided together")
        with model.autocast_context:
            if seed is not None:
                batch_seed = seed
            elif deterministic:
                batch_seed = 0
            else:
                batch_seed = train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            batch_size = batch["latent_image"].shape[0]
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch_size,
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_dropout_probability=None,
            )
            if latent_input is not None:
                latent_input_list = list(
                    latent_input.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype()).unbind(dim=0)
                )
                transformer_timestep = 1.0 - sigma.to(device=latent_input.device, dtype=torch.float32).flatten()
            else:
                scaled_latent_image = model.scale_latents(batch["latent_image"])
                _, timestep, _, _, latent_input_list = self._prepare_flow_inputs(
                    model, config, deterministic, generator, rand, scaled_latent_image
                )
                transformer_timestep = (1000 - timestep) / 1000

            fork_block = model.transformer.layers[config.transfer_late_start]
            args, kwargs = run_prefix_and_capture(
                lambda: model.transformer(
                    latent_input_list, transformer_timestep, text_encoder_output, return_dict=True
                ),
                fork_block,
            )

        # Stack call is positional: (unified, unified_mask, unified_freqs,
        # adaln_input, unified_noise_tensor, t_noisy, t_clean); the mask is
        # (batch, tokens) bool or None when all sequences share a length.
        mask = args[1] if len(args) > 1 and isinstance(args[1], Tensor) else kwargs.get("attn_mask")
        return TransferForkState(block_args=args, block_kwargs=kwargs, mask=mask)

    def transfer_sharpness_probe(self, model: ZImageModel, batch: dict, config: TrainConfig, *, seed: int) -> float:
        """One-step clean-latent prediction from a heavily noised latent, student
        vs the raw variant (every adapter hook off), compared by high-frequency
        energy. Caller holds no_grad; the seed fixes the noise so successive
        validations are comparable."""
        sigma = TRANSFER_SHARPNESS_SIGMA
        with model.autocast_context:
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(seed)
            batch_size = batch["latent_image"].shape[0]
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch_size,
                rand=Random(seed),
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_dropout_probability=None,
            )
            scaled_latent_image = model.scale_latents(batch["latent_image"])
            latent_noise = self._create_noise(scaled_latent_image, config, generator)
            scaled_noisy_latent_image = (1.0 - sigma) * scaled_latent_image + sigma * latent_noise
            latent_input_list = list(
                scaled_noisy_latent_image.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype()).unbind(dim=0)
            )
            transformer_timestep = torch.full(
                (batch_size,), 1.0 - sigma, device=scaled_latent_image.device, dtype=torch.float32
            )

            def predict_clean_latent() -> Tensor:
                output_list = model.transformer(
                    latent_input_list, transformer_timestep, text_encoder_output, return_dict=True
                ).sample
                predicted_flow = -torch.stack(output_list, dim=0).squeeze(dim=2)
                return scaled_noisy_latent_image.float() - predicted_flow.float() * sigma

            student = predict_clean_latent()
            with self.prior_model(model, config):
                variant = predict_clean_latent()
        return high_frequency_ratio(student, variant)

    def transfer_late_forward(
        self,
        model: ZImageModel,
        state: TransferForkState,
        config: TrainConfig,
        *,
        weights: dict[str, Tensor] | Callable[[str], Tensor] | None = None,
    ) -> list[Tensor]:
        transformer = model.transformer
        blocks = list(transformer.layers[config.transfer_late_start :])
        checkpointing_fn = (
            transformer._gradient_checkpointing_func
            if torch.is_grad_enabled() and getattr(transformer, "gradient_checkpointing", False)
            else None
        )
        with model.autocast_context:
            if callable(weights):
                # Lazy weights (stem -> tensor): materialize, swap and release one
                # block at a time, so the pass never holds a whole extra late stack.
                swap_modules = self._transfer_swap_modules(model, config)
                outputs = []
                x = state.block_args[0]
                for offset, block in enumerate(blocks):
                    prefix = f"layers.{config.transfer_late_start + offset}."
                    block_modules = {stem: module for stem, module in swap_modules.items() if stem.startswith(prefix)}
                    block_weights = {stem: weights(stem) for stem in block_modules}
                    with swap_module_weights(block_modules, block_weights):
                        x = run_late_stack([block], x, state.block_args[1:], state.block_kwargs, checkpointing_fn)[0]
                    del block_weights
                    outputs.append(x)
                return outputs
            if weights is not None:
                with swap_module_weights(self._transfer_swap_modules(model, config), weights):
                    return run_late_stack(
                        blocks, state.block_args[0], state.block_args[1:], state.block_kwargs, checkpointing_fn
                    )
            return run_late_stack(
                blocks, state.block_args[0], state.block_args[1:], state.block_kwargs, checkpointing_fn
            )

    def calculate_loss(
        self,
        model: ZImageModel,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ) -> Tensor:
        losses = self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        )
        # per-sample, before the mean, so loss_weight / scaler semantics match MSE
        losses = self._add_perceptual_loss(model, batch, data, config, losses)
        return losses.mean()

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
