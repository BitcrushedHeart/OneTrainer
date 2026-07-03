import os
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig, TrainModelPartConfig
from modules.util.distill_dmd2_util import (
    build_distill_sigma_matrix,
    cfg_baked_velocity,
    dmd2_endpoint_loss,
    dmd2_fake_score_loss,
    dmd2_kl_pseudo_loss,
    euler_flow_step,
)
from modules.util.distill_metadata_util import extract_distill_metadata
from modules.util.enum.DistillCfgMode import DistillCfgMode
from modules.util.enum.DistillVramMode import DistillVramMode
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter


class BaseModelSetup(
    TimedActionMixin,
    metaclass=ABCMeta,
):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        debug_mode: bool,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode
        self.frozen_parameters = {}
        self._dpo_ref_params = None
        self._last_dpo_metrics = None
        self._dpo_paired_half = None  # read by ModelSetupNoiseMixin._apply_dpo_paired_rng
        self._dpo_runtime_beta = None
        self._last_distill_metrics = None
        self._last_grad_step_index = None  # student backprop step chosen this update (debug/tests)
        self._last_distill_x0_teacher = None  # paired teacher target from the last student update

    @abstractmethod
    def create_parameters(
        self,
        model: BaseModel,
        config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        pass

    @abstractmethod
    def setup_optimizations(
        self,
        model: BaseModel,
        config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_model(
        self,
        model: BaseModel,
        config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_train_device(
        self,
        model: BaseModel,
        config: TrainConfig,
    ):
        pass

    @abstractmethod
    def predict(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
        self,
        model: BaseModel,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(
        self,
        model: BaseModel,
        config: TrainConfig,
        train_progress: TrainProgress,
    ):
        pass

    def report_to_tensorboard(
        self,
        model: BaseModel,
        config: TrainConfig,
        scheduler: LRScheduler,
        tensorboard: SummaryWriter,
    ):
        lrs = scheduler.get_last_lr()
        parameters = model.parameters.display_name_mapping

        reported_learning_rates = {}

        if any("optim_type" in g for g in model.optimizer.param_groups):
            for group in model.optimizer.param_groups:
                name = group.get("name")
                if not name or not group["params"]:
                    continue
                optim_type = group.get("optim_type", "unknown")
                unique_name = f"{name}_{optim_type}"
                if unique_name not in reported_learning_rates:
                    reported_learning_rates[unique_name] = group["lr"]
        else:
            for lr, parameter in zip(lrs, parameters, strict=True):
                name = parameter.split("/")[0]

                if name not in reported_learning_rates:
                    reported_learning_rates[name] = lr

        reported_learning_rates = config.optimizer.optimizer.maybe_adjust_lrs(reported_learning_rates, model.optimizer)

        for name, lr in reported_learning_rates.items():
            tensorboard.add_scalar(f"lr/{name}", lr, model.train_progress.global_step)

        if hasattr(model.optimizer, "kourkoutas_helper") and model.optimizer.kourkoutas_helper is not None:
            stats = model.optimizer.kourkoutas_helper.last_beta2_stats
            if stats:
                tensorboard.add_scalar("kourkoutas/beta2_mean", stats["mean"], model.train_progress.global_step)

    @staticmethod
    def _is_dpo_rejected_key(key: str) -> bool:
        return key.endswith("_rejected")

    @classmethod
    def _create_dpo_batched_batch(cls, batch: dict) -> tuple[dict, int]:
        # Returns a batch where every <key>/<key>_rejected pair is concatenated
        # as [chosen; rejected] on dim 0, shared per-sample tensors are duplicated
        # by self-concat, and non-batched values pass through. The chosen half is
        # always the first B entries of the result.
        chosen_b = batch["latent_image"].shape[0]
        batched: dict = {}
        for key, value in batch.items():
            if cls._is_dpo_rejected_key(key):
                continue
            rejected_key = key + "_rejected"
            if rejected_key in batch:
                rejected_value = batch[rejected_key]
                # Non-tensor paired metadata (crop_resolution tuples,
                # image_path strings, etc.) can't be concatenated and isn't
                # consumed by predict(); pass the chosen side through.
                if isinstance(value, torch.Tensor) and isinstance(rejected_value, torch.Tensor):
                    batched[key] = torch.cat([value, rejected_value], dim=0)
                else:
                    batched[key] = value
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == chosen_b:
                batched[key] = torch.cat([value, value], dim=0)
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == chosen_b for v in value)
            ):
                # SDXL-style per-field tensor lists (original_resolution / crop_resolution /
                # crop_offset collate a per-sample tuple into a list of [B] tensors). The
                # rejected image shares the chosen sample's bucket, so duplicate each field
                # elementwise to reach 2B and keep add_time_ids aligned with the latents.
                batched[key] = type(value)(torch.cat([v, v], dim=0) for v in value)
            else:
                batched[key] = value
        return batched, chosen_b

    @staticmethod
    def _split_dpo_batched_output(output: dict, chosen_b: int) -> tuple[dict, dict]:
        # Splits a model output dict whose batched tensors have leading dim 2B
        # into chosen-only (first B) and rejected-only (last B) dicts.
        chosen_out: dict = {}
        rejected_out: dict = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 2 * chosen_b:
                chosen_out[key] = value[:chosen_b]
                rejected_out[key] = value[chosen_b:]
            else:
                chosen_out[key] = value
                rejected_out[key] = value
        return chosen_out, rejected_out

    def get_last_dpo_metrics(self) -> dict[str, float]:
        return self._last_dpo_metrics or {}

    def get_last_distill_metrics(self) -> dict[str, float | str]:
        return self._last_distill_metrics or {}

    @staticmethod
    def _distill_update_mode(config: TrainConfig, train_progress: TrainProgress) -> str:
        if train_progress.global_step < config.distill_fake_warmup_steps:
            return "fake"
        ratio = max(1, int(config.distill_ttur_ratio))
        relative_step = train_progress.global_step - config.distill_fake_warmup_steps
        return "student" if relative_step % (ratio + 1) == ratio else "fake"

    @staticmethod
    def _distill_batch_size(batch: dict) -> int:
        latent = batch["latent_image"]
        return latent.shape[0] if isinstance(latent, Tensor) and latent.ndim > 0 else 1

    @staticmethod
    def _distill_image_paths(batch: dict) -> list[str]:
        paths = batch.get("image_path", [])
        if isinstance(paths, str):
            return [paths]
        if isinstance(paths, (list, tuple)):
            return [str(path) for path in paths]
        return []

    def _distill_sidecar_values(self, batch: dict, attr: str) -> list[float | int | None]:
        values = []
        for path in self._distill_image_paths(batch):
            metadata = extract_distill_metadata(path)
            values.append(getattr(metadata, attr) if metadata is not None else None)
        return values

    def _distill_sidecar_text_values(self, batch: dict, attr: str, default: str = "") -> list[str]:
        values = []
        for path in self._distill_image_paths(batch):
            metadata = extract_distill_metadata(path)
            value = getattr(metadata, attr) if metadata is not None else None
            values.append(default if value is None else str(value))
        return values

    def _distill_tensor_from_batch_or_sidecar(
        self,
        batch: dict,
        batch_key: str,
        sidecar_attr: str,
        default: float | int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = self._distill_batch_size(batch)
        if batch_key in batch:
            value = batch[batch_key]
            if isinstance(value, Tensor):
                tensor = value.to(device=device, dtype=dtype).flatten()
            elif isinstance(value, (list, tuple)):
                tensor = torch.tensor(value, device=device, dtype=dtype).flatten()
            else:
                tensor = torch.tensor([value], device=device, dtype=dtype)
        else:
            sidecar_values = self._distill_sidecar_values(batch, sidecar_attr)
            if sidecar_values:
                tensor = torch.tensor(
                    [default if value is None else value for value in sidecar_values],
                    device=device,
                    dtype=dtype,
                )
            else:
                tensor = torch.tensor([default], device=device, dtype=dtype)

        if tensor.numel() == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size)
        if tensor.numel() != batch_size:
            raise RuntimeError(
                f"{batch_key} has {tensor.numel()} value(s), but the distillation batch has {batch_size} sample(s)."
            )
        return tensor

    def _distill_cfg_scale(self, batch: dict, config: TrainConfig, device: torch.device) -> Tensor:
        if config.distill_cfg_mode == DistillCfgMode.FORCE_1:
            return torch.ones(self._distill_batch_size(batch), device=device, dtype=torch.float32)
        return self._distill_tensor_from_batch_or_sidecar(
            batch,
            "distill_cfg_scale",
            "cfg_scale",
            1.0,
            device,
            torch.float32,
        )

    def _distill_teacher_steps(self, batch: dict, config: TrainConfig, device: torch.device) -> Tensor:
        default = config.distill_teacher_steps
        if not config.distill_teacher_steps_auto:
            return torch.full((self._distill_batch_size(batch),), default, device=device, dtype=torch.long)
        return self._distill_tensor_from_batch_or_sidecar(
            batch,
            "distill_teacher_steps",
            "steps",
            default,
            device,
            torch.long,
        )

    def predict_distill_velocity(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        latent: Tensor,
        sigma: Tensor,
        *,
        conditioning: str = "conditional",
    ) -> Tensor:
        distill_batch = dict(batch)
        distill_batch["_distill_latent_input"] = latent
        distill_batch["_distill_sigma"] = sigma
        distill_batch["_distill_conditioning"] = conditioning
        if conditioning == "unconditional" and "distill_negative_prompt" not in distill_batch:
            negative_prompts = self._distill_sidecar_text_values(batch, "negative_prompt")
            if negative_prompts:
                distill_batch["distill_negative_prompt"] = negative_prompts
        return self.predict(model, distill_batch, config, train_progress, deterministic=True)["predicted"]

    def _cfg_baked_distill_velocity(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        latent: Tensor,
        sigma: Tensor,
        cfg_scale: Tensor,
    ) -> Tensor:
        conditional = self.predict_distill_velocity(
            model, batch, config, train_progress, latent, sigma, conditioning="conditional"
        )
        if bool(torch.all(cfg_scale == 1)):
            return conditional
        unconditional = self.predict_distill_velocity(
            model, batch, config, train_progress, latent, sigma, conditioning="unconditional"
        )
        return cfg_baked_velocity(conditional, unconditional, cfg_scale)

    def _distill_renoise(
        self,
        x0: Tensor,
        train_progress: TrainProgress,
        *,
        seed_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-noise a student sample x0 to a random interior sigma.

        Returns ``(x_sigma, sigma, v_target)`` where ``x_sigma`` follows the
        rectified-flow path ``(1 - sigma) * x0 + sigma * eps`` and ``v_target``
        is the flow velocity ``eps - x0``. Used by both the DMD2 fake-score
        update and the student KL update so the distribution-matching gradient
        is evaluated at a genuinely noised point (never the sigma=0 endpoint).
        """
        batch_size = x0.shape[0]
        generator = torch.Generator(device=x0.device)
        generator.manual_seed(train_progress.global_step * 2 + seed_offset)
        sigma = torch.rand(batch_size, generator=generator, device=x0.device, dtype=torch.float32)
        sigma = sigma.clamp(0.02, 0.98)
        eps = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        sigma_b = sigma.view(batch_size, *([1] * (x0.dim() - 1))).to(dtype=x0.dtype)
        x_sigma = (1.0 - sigma_b) * x0 + sigma_b * eps
        v_target = eps - x0
        return x_sigma, sigma, v_target

    def _distill_perceptual_loss(
        self,
        model: BaseModel,
        generated_scaled_latent: Tensor,
        target_scaled_latent: Tensor,
        config: TrainConfig,
    ) -> Tensor | None:
        """Optional LPIPS term in image space. Default: not available.

        Model-specific setups (BaseZImageSetup) override this to decode both
        latents and run LPIPS. Returns None when unavailable so the caller skips
        the term.
        """
        return None

    def _distill_initial_latent(self, batch: dict, train_progress: TrainProgress) -> Tensor:
        """Shared initial noise ``z`` for the student/teacher pairing.

        Honors an explicit ``batch["distill_initial_latent"]`` (e.g. tests / a
        future paired-noise cache), otherwise samples fresh noise seeded by the
        global step so the same ``z`` can be handed to both rollouts this step.
        """
        reference = batch["latent_image"]
        latent = batch.get("distill_initial_latent")
        if latent is not None:
            return latent.to(device=reference.device, dtype=reference.dtype)
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(train_progress.global_step)
        return torch.randn(reference.shape, generator=generator, device=reference.device, dtype=reference.dtype)

    def _student_onestep_x0(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        initial_latent: Tensor,
        sigmas: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Roll the student to a random step ``k`` and return its one-step ``x0``.

        Steps ``0..k-1`` run under ``no_grad`` (detached ``x_k``); step ``k`` takes
        a single grad-bearing forward ``v_k`` and the clean estimate is
        ``x0 = x_k - sigma_k * v_k``. Exactly one grad-bearing adapter forward
        keeps the in-place OFT/DoRA rotation buffers safe for autograd, while a
        per-step random ``k`` (seeded by the global step) spreads the learning
        signal across all denoising steps. ``distill_backprop_random_step=False``
        pins ``k`` to the final step (the previous behavior).
        """
        total_steps = config.distill_target_steps
        if config.distill_backprop_random_step and total_steps > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(train_progress.global_step)
            grad_step = int(torch.randint(0, total_steps, (1,), generator=generator).item())
        else:
            grad_step = total_steps - 1

        latent = initial_latent
        with torch.no_grad():
            for step_index in range(grad_step):
                sigma = sigmas[:, step_index]
                next_sigma = sigmas[:, step_index + 1]
                velocity = self.predict_distill_velocity(
                    model, batch, config, train_progress, latent, sigma, conditioning="conditional"
                )
                latent = euler_flow_step(latent, velocity, sigma, next_sigma)
        x_k = latent.detach()
        sigma_step = sigmas[:, grad_step]
        v_k = self.predict_distill_velocity(
            model, batch, config, train_progress, x_k, sigma_step, conditioning="conditional"
        )
        sigma_step_b = sigma_step.view(x_k.shape[0], *([1] * (x_k.dim() - 1))).to(dtype=x_k.dtype)
        x0 = x_k - sigma_step_b * v_k
        return x0, x_k, sigma_step, grad_step

    def calculate_distill_loss(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
    ) -> Tensor:
        mode = self._distill_update_mode(config, train_progress)
        device = batch["latent_image"].device
        cfg_scale = self._distill_cfg_scale(batch, config, device)
        teacher_steps = self._distill_teacher_steps(batch, config, device)
        sigmas = build_distill_sigma_matrix(
            config.distill_target_steps,
            teacher_steps,
            config.distill_timestep_grid_mode,
            device=device,
            dtype=torch.float32,
            shift=config.distill_sigma_shift,
        )

        if mode == "fake":
            # DMD2 fake-score update: model the student's *output* distribution
            # via denoising score-matching on a re-noised student sample. The
            # sample is drawn exactly the way the student update draws the x0
            # its KL is evaluated at (random-k one-step estimate when
            # distill_backprop_random_step is on; with it off, the final-step
            # one-step estimate equals the full rollout endpoint), so the
            # critic trains on the same distribution it is queried at — DMD2's
            # multi-step backward simulation. This also skips the rollout
            # steps after k.
            z = self._distill_initial_latent(batch, train_progress)
            with self.adapter_state(model, config, "student"), torch.no_grad():
                x0, _, _, _ = self._student_onestep_x0(model, batch, config, train_progress, z, sigmas)
            x0 = x0.detach()
            x_sigma, sigma_k, v_target = self._distill_renoise(x0, train_progress, seed_offset=0)
            with self.adapter_state(model, config, "fake"):
                # Conditional only: the student bakes CFG into its weights, so its
                # samples are conditional and mu_fake models that distribution.
                fake_predicted = self.predict_distill_velocity(
                    model, batch, config, train_progress, x_sigma.detach(), sigma_k, conditioning="conditional"
                )

            fake_loss = dmd2_fake_score_loss(fake_predicted, v_target)
            self._last_distill_metrics = {
                "mode": "fake",
                "loss": fake_loss.detach().item(),
                "fake_loss": fake_loss.detach().item(),
                "kl_loss": 0.0,
                "endpoint_loss": 0.0,
                "lpips_loss": 0.0,
                "target_steps": config.distill_target_steps,
                "teacher_steps": teacher_steps.float().mean().item(),
                "cfg_scale": cfg_scale.float().mean().item(),
            }
            return fake_loss

        # Student update (DMD2 generator step). From a shared initial noise z, roll
        # the student to a random interior step k, take ONE grad-bearing forward
        # there, and form the one-step clean estimate x0 = x_k - sigma_k * v_k. The
        # teacher's one-step denoise at the SAME x_k is the paired regression target
        # (well-posed: each z/k gives a distinct target, so diversity is preserved
        # instead of collapsing to a noise-independent mean). The distribution-
        # matching KL is then evaluated at a random re-noise of x0.
        z = self._distill_initial_latent(batch, train_progress)
        with self.adapter_state(model, config, "student"):
            x0, x_k, sigma_step, grad_step = self._student_onestep_x0(model, batch, config, train_progress, z, sigmas)
        self._last_grad_step_index = grad_step

        # The KL is evaluated at a random re-noise of the student estimate;
        # drawing it up front lets CONCURRENT mode fuse both teacher passes.
        x_sigma, sigma_k, _ = self._distill_renoise(x0, train_progress, seed_offset=1)

        # Two frozen-teacher predictions are needed: the paired teacher-match
        # one-step denoise at x_k, and the real-score velocity at x_sigma for the
        # KL. CONCURRENT batches them into one double-batch forward (same
        # weights, no grad) instead of two passes; SPLIT keeps them separate.
        sigma_step_b = sigma_step.view(x_k.shape[0], *([1] * (x_k.dim() - 1))).to(dtype=x_k.dtype)
        if config.distill_vram_mode == DistillVramMode.CONCURRENT:
            batch_size = x_k.shape[0]
            fused_latent = torch.cat([x_k, x_sigma.detach().to(dtype=x_k.dtype)], dim=0)
            fused_sigma = torch.cat([sigma_step, sigma_k], dim=0)
            fused_cfg = torch.cat([cfg_scale, cfg_scale], dim=0)
            with self.adapter_state(model, config, "teacher"), torch.no_grad():
                fused_velocity = self._cfg_baked_distill_velocity(
                    model, batch, config, train_progress, fused_latent, fused_sigma, fused_cfg
                )
            teacher_velocity = fused_velocity[:batch_size]
            real_predicted = fused_velocity[batch_size:]
        else:
            with self.adapter_state(model, config, "teacher"), torch.no_grad():
                teacher_velocity = self._cfg_baked_distill_velocity(
                    model, batch, config, train_progress, x_k, sigma_step, cfg_scale
                )
                real_predicted = self._cfg_baked_distill_velocity(
                    model, batch, config, train_progress, x_sigma.detach(), sigma_k, cfg_scale
                )

        # Paired teacher-match: teacher one-step denoise at the same x_k.
        x0_teacher = (x_k - sigma_step_b * teacher_velocity).detach()
        self._last_distill_x0_teacher = x0_teacher

        endpoint_loss = dmd2_endpoint_loss(x0, x0_teacher, weight=config.distill_endpoint_loss_weight)
        lpips_loss = None
        if config.distill_endpoint_lpips_weight > 0.0:
            lpips_term = self._distill_perceptual_loss(model, x0, x0_teacher, config)
            if lpips_term is not None:
                lpips_loss = config.distill_endpoint_lpips_weight * lpips_term
                endpoint_loss = endpoint_loss + lpips_loss

        # Distribution-matching KL at the re-noised student estimate.
        with self.adapter_state(model, config, "fake"), torch.no_grad():
            fake_predicted = self.predict_distill_velocity(
                model, batch, config, train_progress, x_sigma.detach(), sigma_k, conditioning="conditional"
            )

        kl_loss = dmd2_kl_pseudo_loss(
            x0,
            x_sigma,
            sigma_k,
            real_predicted,
            fake_predicted,
            weight=config.distill_kl_loss_weight,
            normalize=config.distill_kl_normalize,
        )

        loss = kl_loss + endpoint_loss
        self._last_distill_metrics = {
            "mode": "student",
            "loss": loss.detach().item(),
            "fake_loss": 0.0,
            "kl_loss": kl_loss.detach().item(),
            "endpoint_loss": endpoint_loss.detach().item(),
            "lpips_loss": lpips_loss.detach().item() if lpips_loss is not None else 0.0,
            "target_steps": config.distill_target_steps,
            "teacher_steps": teacher_steps.float().mean().item(),
            "cfg_scale": cfg_scale.float().mean().item(),
        }
        return loss

    # ------------------------------------------------------------------
    # Distill teacher-match validation (no held-out samples required: the
    # frozen teacher generates its own references, computed once and reused).
    # ------------------------------------------------------------------
    def capture_distill_val_batch(self, batch: dict, config: TrainConfig):
        """Stash the first N training batches' conditioning as the validation
        set. No images are held out from training; we only reuse the prompt
        conditioning + latent shape, paired later with fixed validation seeds."""
        if not getattr(config, "distill_validation", False):
            return
        cache = getattr(self, "_distill_val_batches", None)
        if cache is None:
            cache = []
            self._distill_val_batches = cache
            self._distill_teacher_ref_cache = {}
        if len(cache) >= max(1, int(config.distill_validation_prompts)):
            return
        keep: dict = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                keep[key] = value.detach().clone()
            elif isinstance(value, (list, tuple, str, int, float)) or value is None:
                keep[key] = value
        cache.append(keep)

    def _distill_val_noise(self, batch: dict, seed: int) -> Tensor:
        latent = batch["latent_image"]
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(seed)
        return torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)

    def _distill_run_trajectory(
        self, model, batch, config, train_progress, sigmas, cfg_scale, initial_latent, state
    ) -> Tensor:
        latent = initial_latent
        for step_index in range(sigmas.shape[1] - 1):
            sigma = sigmas[:, step_index]
            next_sigma = sigmas[:, step_index + 1]
            if state == "teacher":
                velocity = self._cfg_baked_distill_velocity(
                    model, batch, config, train_progress, latent, sigma, cfg_scale
                )
            else:
                velocity = self.predict_distill_velocity(
                    model, batch, config, train_progress, latent, sigma, conditioning="conditional"
                )
            latent = euler_flow_step(latent, velocity, sigma, next_sigma)
        return latent

    def _distill_diversity_ratio(
        self, model, batch, config, train_progress, student_sigmas, teacher_sigmas, cfg_scale, index, cache
    ) -> float | None:
        seeds = [2_000_003 + index * 10 + s for s in range(3)]

        def generate(state, sigmas):
            outs = []
            with self.adapter_state(model, config, state):
                for seed in seeds:
                    noise = self._distill_val_noise(batch, seed)
                    outs.append(
                        self._distill_run_trajectory(
                            model, batch, config, train_progress, sigmas, cfg_scale, noise, state
                        )
                    )
            return outs

        def mean_pairwise(outs):
            dists = [
                (outs[a].float() - outs[b].float()).pow(2).mean().item()
                for a in range(len(outs))
                for b in range(a + 1, len(outs))
            ]
            return sum(dists) / len(dists) if dists else 0.0

        teacher_key = ("div", index)
        teacher_div = cache.get(teacher_key)
        if teacher_div is None:
            teacher_div = mean_pairwise(generate("teacher", teacher_sigmas))
            cache[teacher_key] = teacher_div
        student_div = mean_pairwise(generate("student", student_sigmas))
        return (student_div / teacher_div) if teacher_div > 1e-8 else None

    def run_distill_validation(self, model, config, train_progress) -> dict[str, float]:
        """Teacher-match (latent MSE + LPIPS) + diversity ratio, teacher refs
        cached because the teacher is frozen. Only the student re-runs."""
        cache_batches = getattr(self, "_distill_val_batches", None)
        if not cache_batches:
            return {}
        ref_cache = self._distill_teacher_ref_cache
        device = cache_batches[0]["latent_image"].device
        mses: list[float] = []
        lpips_vals: list[float] = []
        div_ratios: list[float] = []
        with torch.no_grad():
            for index, batch in enumerate(cache_batches):
                cfg_scale = self._distill_cfg_scale(batch, config, device)
                teacher_steps = self._distill_teacher_steps(batch, config, device)
                teacher_target = int(teacher_steps.flatten()[0].item())
                student_sigmas = build_distill_sigma_matrix(
                    config.distill_target_steps,
                    teacher_steps,
                    config.distill_timestep_grid_mode,
                    device=device,
                    shift=config.distill_sigma_shift,
                )
                teacher_sigmas = build_distill_sigma_matrix(
                    teacher_target, teacher_steps, "AUTO", device=device, shift=config.distill_sigma_shift
                )
                noise = self._distill_val_noise(batch, seed=1_000_003 + index)

                ref = ref_cache.get(index)
                if ref is None:
                    with self.adapter_state(model, config, "teacher"):
                        ref = self._distill_run_trajectory(
                            model, batch, config, train_progress, teacher_sigmas, cfg_scale, noise, "teacher"
                        ).detach()
                    ref_cache[index] = ref
                with self.adapter_state(model, config, "student"):
                    student = self._distill_run_trajectory(
                        model, batch, config, train_progress, student_sigmas, cfg_scale, noise, "student"
                    )

                mses.append((student.float() - ref.float()).pow(2).mean().item())
                lpips_term = self._distill_perceptual_loss(model, student, ref, config)
                if lpips_term is not None:
                    lpips_vals.append(float(lpips_term))
                if index < 2:
                    ratio = self._distill_diversity_ratio(
                        model,
                        batch,
                        config,
                        train_progress,
                        student_sigmas,
                        teacher_sigmas,
                        cfg_scale,
                        index,
                        ref_cache,
                    )
                    if ratio is not None:
                        div_ratios.append(ratio)

        metrics: dict[str, float] = {}
        if mses:
            metrics["teacher_match_mse"] = sum(mses) / len(mses)
        if lpips_vals:
            metrics["teacher_match_lpips"] = sum(lpips_vals) / len(lpips_vals)
        if div_ratios:
            metrics["diversity_ratio"] = sum(div_ratios) / len(div_ratios)
        return metrics

    def set_dpo_runtime_beta(self, beta: float | None):
        # Adaptive-beta override from the trainer. The logged reward metrics
        # are computed before beta is applied, so adapting beta from them does
        # not create a feedback loop.
        self._dpo_runtime_beta = beta

    def calculate_dpo_loss(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
    ) -> Tensor:
        if "latent_image_rejected" not in batch:
            raise RuntimeError(
                "RLHF DPO requires paired chosen/rejected batches, but the dataloader did not provide rejected samples."
            )

        def mse_per_sample(pred, target):
            # fp32 accumulation in the reduction instead of upcasting the full
            # [2B,C,H,W] tensors - avoids four fp32 copies per step.
            return (pred - target).pow(2).mean(dim=list(range(1, pred.ndim)), dtype=torch.float32)

        beta = config.rlhf_dpo_beta if self._dpo_runtime_beta is None else self._dpo_runtime_beta
        supervised_loss = None

        # 2 forwards: 1 batched ref (no_grad) + 1 batched policy, each over the
        # [chosen; rejected] batch. Both halves share per-pair timestep+noise via
        # _dpo_paired_half, and ref/policy share them too because predict()
        # seeds its generator from global_step. Note for torch.compile users:
        # supervised/validation batches are B-sized while DPO batches are
        # 2B-sized, so mixing them in one session compiles two graphs.
        batched_input, chosen_b = self._create_dpo_batched_batch(batch)

        # Conditioning (caption) dropout is drawn per-sample inside encode_text.
        # On the [chosen; rejected] batch that independently zeroes the prompt on
        # one half of a pair (~2p(1-p) of pairs), comparing a prompted sample
        # against an unconditional one and silently corrupting the preference
        # margin. Neutralize it for the paired forward so both halves see
        # identical conditioning; restore afterwards. (Diffusion-DPO trains on the
        # given prompts, not CFG-dropped ones.)
        te_configs = (config.text_encoder, config.text_encoder_2, config.text_encoder_3, config.text_encoder_4)
        saved_dropout = [te.dropout_probability for te in te_configs]

        self._dpo_paired_half = chosen_b
        try:
            for te in te_configs:
                te.dropout_probability = 0.0
            with torch.no_grad(), self.reference_model(model, config):
                ref_output = self.predict(model, batched_input, config, train_progress)
                ref_predicted = ref_output["predicted"]
                ref_target = ref_output["target"]
                ref_chosen_logp = -mse_per_sample(ref_predicted[:chosen_b], ref_target[:chosen_b])
                ref_rejected_logp = -mse_per_sample(ref_predicted[chosen_b:], ref_target[chosen_b:])
                del ref_output, ref_predicted, ref_target

            policy_output = self.predict(model, batched_input, config, train_progress)
        finally:
            self._dpo_paired_half = None
            for te, saved in zip(te_configs, saved_dropout, strict=True):
                te.dropout_probability = saved
        policy_timestep = policy_output.get("timestep")
        policy_predicted = policy_output["predicted"]
        policy_target = policy_output["target"]
        policy_chosen_logp = -mse_per_sample(policy_predicted[:chosen_b], policy_target[:chosen_b])
        policy_rejected_logp = -mse_per_sample(policy_predicted[chosen_b:], policy_target[chosen_b:])
        if config.rlhf_supervised_mix > 0:
            chosen_output, _ = self._split_dpo_batched_output(policy_output, chosen_b)
            supervised_loss = self.calculate_loss(model, batch, chosen_output, config)
            del chosen_output
        del policy_output, policy_predicted, policy_target

        chosen_ratio = policy_chosen_logp - ref_chosen_logp.detach()
        rejected_ratio = policy_rejected_logp - ref_rejected_logp.detach()
        margin = chosen_ratio - rejected_ratio

        if config.rlhf_dpo_objective == DPOObjective.IPO:
            # IPO regresses the raw margin toward the fixed target 1/(2*tau)
            # instead of pushing it to infinity, which structurally resists
            # reward hacking. tau plays beta's role; label smoothing and beta
            # do not apply.
            dpo_loss = (margin - 1.0 / (2.0 * config.rlhf_dpo_ipo_tau)).pow(2).mean()
            loss = dpo_loss
        else:
            logits = beta * margin
            dpo_loss = -F.logsigmoid(logits).mean()
            loss = dpo_loss

            if config.rlhf_dpo_label_smoothing > 0:
                s = config.rlhf_dpo_label_smoothing
                loss = (1 - s) * loss + s * (-F.logsigmoid(-logits).mean())

        if supervised_loss is not None:
            loss = loss + config.rlhf_supervised_mix * supervised_loss
            del supervised_loss

        # Stack the six scalar metrics and sync them in a single .tolist() rather
        # than one .item() (one GPU->CPU sync) per metric.
        metric_values = torch.stack(
            [
                loss.detach().float(),
                dpo_loss.detach().float(),
                chosen_ratio.detach().mean(),
                rejected_ratio.detach().mean(),
                margin.detach().mean(),
                (chosen_ratio > rejected_ratio).float().mean(),
            ]
        ).tolist()
        self._last_dpo_metrics = dict(
            zip(
                ("loss", "dpo_loss", "chosen_reward", "rejected_reward", "reward_margin", "accuracy"),
                metric_values,
                strict=True,
            )
        )

        if config.rlhf_dpo_timestep_margin_logging and policy_timestep is not None:
            # Per-sample raw margins bucketed by the chosen half's timestep
            # quartile. Sums and counts are emitted for every quartile so the
            # trainer's accumulation always sees the same key set.
            raw_t = policy_timestep[:chosen_b].detach()
            if torch.is_floating_point(raw_t):
                # Continuous-timestep models already emit t in (0, 1].
                t = raw_t.float()
            else:
                # Discrete schedulers emit integer steps in [0, num_train_timesteps);
                # normalize by the model's actual count rather than sniffing the
                # batch max (which misbuckets batches that only sample steps {0,1})
                # or assuming 1000.
                num_train_timesteps = model.noise_scheduler.config["num_train_timesteps"]
                t = raw_t.float() / num_train_timesteps
            quartile_index = (t * 4).long().clamp(0, 3)
            per_sample_margin = margin.detach().float()
            # bincount for counts + scatter_add for sums, then one sync for all
            # eight quartile scalars instead of eight .item() calls.
            counts = torch.bincount(quartile_index, minlength=4).float()
            sums = torch.zeros(4, device=per_sample_margin.device, dtype=per_sample_margin.dtype)
            sums.scatter_add_(0, quartile_index, per_sample_margin)
            sums_l, counts_l = torch.stack([sums, counts]).tolist()
            for quartile in range(4):
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_sum"] = sums_l[quartile]
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_count"] = counts_l[quartile]

        return loss

    def stop_embedding_training_elapsed(
        self,
        config: TrainEmbeddingConfig,
        train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_embedding_training_" + str(config.uuid),
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    def __stop_model_part_training_elapsed(
        self,
        unique_name: str,
        config: TrainModelPartConfig,
        train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_" + unique_name + "_training",
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    @contextmanager
    def distillation_teacher_model(self, model: BaseModel, config: TrainConfig):
        """Swap student adapters out and teacher adapters in for a forward pass.

        The teacher wrapper is constructed but not hooked at setup time; this
        ctx mgr is the only place it gets hooked, and it's restored on exit.
        Mirrors :meth:`prior_model` but toggles between two adapter sets
        rather than disabling all of them.
        """
        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError("Distillation is only available with LoRA training")

        teachers = model.teacher_adapters()
        if len(teachers) == 0:
            raise RuntimeError("distillation_teacher_model called but no teacher adapters attached to the model.")

        students = model.adapters()
        for adapter in students:
            adapter.remove_hook_from_module()
        for adapter in teachers:
            adapter.hook_to_module()
        try:
            yield
        finally:
            for adapter in teachers:
                adapter.remove_hook_from_module()
            for adapter in students:
                adapter.hook_to_module()

    @contextmanager
    def adapter_state(self, model: BaseModel, config: TrainConfig, state: str):
        """Activate one named adapter state for a forward/backward section.

        DMD2 uses three mutually-exclusive states: ``teacher`` is the base model
        with no adapters, ``student`` is the trainable policy adapter, and
        ``fake`` is the trainable fake-score adapter.
        """
        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError("Adapter states are only available with LoRA training")

        students = model.adapters()
        teachers = model.teacher_adapters()
        fakes = model.fake_adapters()
        all_adapters = [*students, *teachers, *fakes]

        if state not in {"teacher", "student", "fake"}:
            raise ValueError(f"Unsupported adapter state: {state}")
        if state == "student" and len(students) == 0:
            raise RuntimeError("student adapter state requested but no student adapters are attached to the model.")
        if state == "fake" and len(fakes) == 0:
            raise RuntimeError("fake adapter state requested but no fake-score adapters are attached to the model.")

        for adapter in all_adapters:
            adapter.remove_hook_from_module()
        if state == "student":
            for adapter in students:
                adapter.hook_to_module()
        elif state == "fake":
            for adapter in fakes:
                adapter.hook_to_module()

        try:
            yield
        finally:
            for adapter in all_adapters:
                adapter.remove_hook_from_module()
            for adapter in students:
                adapter.hook_to_module()

    @contextmanager
    def prior_model(self, model: BaseModel, config: TrainConfig):
        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError("Prior model is only available with LoRA training")

        for adapter in model.adapters():
            adapter.remove_hook_from_module()
        try:
            yield
        finally:
            for adapter in model.adapters():
                adapter.hook_to_module()

    @contextmanager
    def reference_model(self, model: BaseModel, config: TrainConfig):
        adapters = model.adapters()

        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError(
                "RLHF DPO reference modes are currently only implemented for adapter training in the LoRA tab."
            )
        if len(adapters) == 0:
            raise RuntimeError(
                "RLHF DPO requires active adapters, but no trainable adapters are attached to the current model."
            )

        ref_mode = config.effective_dpo_ref_mode()

        if ref_mode == DPORefMode.NEW_ADAPTER:
            for adapter in adapters:
                adapter.remove_hook_from_module()
            try:
                yield
            finally:
                for adapter in adapters:
                    adapter.hook_to_module()
        elif ref_mode == DPORefMode.EXISTING_ADAPTER:
            # Reference params are captured once at first call and frozen for the
            # entire training run. This is intentional: DPO requires a fixed
            # reference policy from the start of training.
            if self._dpo_ref_params is None:
                self._dpo_ref_params = [[p.data.clone() for p in adapter.parameters()] for adapter in adapters]

            policy_data = [[p.data for p in adapter.parameters()] for adapter in adapters]
            try:
                for adapter, ref_params in zip(adapters, self._dpo_ref_params, strict=True):
                    for i, (param, ref_data) in enumerate(zip(adapter.parameters(), ref_params, strict=True)):
                        # Reference params restored from a backup live on CPU; move
                        # them onto the live param's device/dtype once and cache the
                        # moved tensor (a no-op for the lazily-cloned in-run case).
                        if ref_data.device != param.data.device or ref_data.dtype != param.data.dtype:
                            ref_data = ref_data.to(device=param.data.device, dtype=param.data.dtype)
                            ref_params[i] = ref_data
                        param.data = ref_data
                yield
            finally:
                for adapter, policy_ptrs in zip(adapters, policy_data, strict=True):
                    for param, policy_ptr in zip(adapter.parameters(), policy_ptrs, strict=True):
                        param.data = policy_ptr
        else:
            raise ValueError(f"Unsupported DPO reference mode: {ref_mode}")

    _DPO_REFERENCE_FILENAME = "dpo_reference.pt"

    def save_dpo_reference(self, backup_path: str) -> None:
        # Persist the frozen EXISTING_ADAPTER reference alongside a backup so it
        # survives resume. Without this the reference is re-captured from the
        # resumed (already DPO-trained) adapter weights, silently moving the KL
        # anchor. NEW_ADAPTER mode never captures params, so nothing is written.
        if self._dpo_ref_params is None:
            return
        try:
            payload = [[t.detach().to("cpu", copy=True) for t in adapter] for adapter in self._dpo_ref_params]
            torch.save(payload, os.path.join(backup_path, self._DPO_REFERENCE_FILENAME))
        except OSError:
            pass

    def load_dpo_reference(self, backup_path: str, model: BaseModel) -> bool:
        # Restore the frozen reference saved next to the backup we are resuming
        # from. Returns True on success; on any structural mismatch it leaves
        # _dpo_ref_params unset so the caller falls back to lazy capture rather
        # than adopting a wrong reference.
        path = os.path.join(backup_path, self._DPO_REFERENCE_FILENAME)
        if not os.path.isfile(path):
            return False
        saved = torch.load(path, map_location="cpu", weights_only=True)
        adapters = model.adapters()
        if not isinstance(saved, list) or len(saved) != len(adapters):
            return False
        restored: list[list[Tensor]] = []
        for adapter, adapter_saved in zip(adapters, saved, strict=True):
            params = list(adapter.parameters())
            if len(adapter_saved) != len(params):
                return False
            if any(tuple(t.shape) != tuple(p.shape) for t, p in zip(adapter_saved, params, strict=True)):
                return False
            restored.append(list(adapter_saved))
        self._dpo_ref_params = restored
        return True

    def _create_model_part_parameters(
        self,
        parameter_group_collection: NamedParameterGroupCollection,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        freeze: list[ModuleFilter] | None = None,
        debug: bool = False,
    ):
        if not config.train:
            return

        if freeze is not None and len(freeze) > 0:
            selected = []
            deselected = []
            parameters = []
            self.frozen_parameters[unique_name] = []
            for name, param in model.named_parameters():
                if any(f.matches(name) for f in freeze):
                    parameters.append(param)
                    selected.append(name)
                else:
                    self.frozen_parameters[unique_name].append(param)
                    deselected.append(name)

            if debug:
                print(f"Selected layers: {selected}")
                print(f"Deselected layers: {deselected}")
            else:
                print(f"Selected layers: {len(selected)}")
                print(f"Deselected layers: {len(deselected)}")
                print("Note: Enable Debug mode to see the full list of layer names")
        else:
            parameters = model.parameters()

        parameter_group_collection.add_group(
            NamedParameterGroup(
                unique_name=unique_name,
                parameters=parameters,
                learning_rate=config.learning_rate,
            )
        )

    def _setup_model_part_requires_grad(
        self,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        train_progress: TrainProgress,
    ):
        if model is not None:
            train_model_part = config.train and not self.__stop_model_part_training_elapsed(
                unique_name, config, train_progress
            )
            model.requires_grad_(train_model_part)

            if unique_name in self.frozen_parameters:
                for param in self.frozen_parameters[unique_name]:
                    param.requires_grad_(False)
