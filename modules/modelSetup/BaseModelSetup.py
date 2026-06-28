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
        if config.distill_cfg_mode == "FORCE_1":
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

    @staticmethod
    def _distill_scaled_image_latent(model: BaseModel, batch: dict) -> Tensor:
        # The student trajectory lives in scaled-latent space (the transformer
        # operates there), but the cached image latent is unscaled. Scale it so
        # the endpoint regression compares like with like. Models without a
        # scale_latents hook (e.g. the unit-test toy model) pass through.
        image_latent = batch.get("target_latent", batch["latent_image"])
        scale = getattr(model, "scale_latents", None)
        return scale(image_latent) if callable(scale) else image_latent

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

    def _simulate_student_distill_trajectory(
        self,
        model: BaseModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        sigmas: Tensor,
    ) -> list[Tensor]:
        latent = batch.get("distill_initial_latent")
        if latent is None:
            generator = torch.Generator(device=self.train_device)
            generator.manual_seed(train_progress.global_step)
            latent = torch.randn(
                batch["latent_image"].shape,
                generator=generator,
                device=batch["latent_image"].device,
                dtype=batch["latent_image"].dtype,
            )
        else:
            latent = latent.to(device=batch["latent_image"].device, dtype=batch["latent_image"].dtype)

        trajectory = [latent]
        for step_index in range(config.distill_target_steps):
            sigma = sigmas[:, step_index]
            next_sigma = sigmas[:, step_index + 1]
            velocity = self.predict_distill_velocity(
                model, batch, config, train_progress, latent, sigma, conditioning="conditional"
            )
            latent = euler_flow_step(latent, velocity, sigma, next_sigma)
            trajectory.append(latent)
        return trajectory

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
        )

        if mode == "fake":
            # DMD2 fake-score update: model the student's *output* distribution
            # via denoising score-matching on a re-noised student sample.
            with self.adapter_state(model, config, "student"), torch.no_grad():
                student_trajectory = self._simulate_student_distill_trajectory(
                    model, batch, config, train_progress, sigmas
                )
            x0 = student_trajectory[-1].detach()
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

        # Student update: generate x0 through the student trajectory (gradient
        # retained), then evaluate the distribution-matching gradient at a random
        # re-noised point x_sigma. Gradient flows (v_fake - v_real).detach() *
        # x_sigma -> x0 -> student adapter. This is robust for target_steps=1..N;
        # the clean sigma=0 endpoint is never used for the KL term.
        with self.adapter_state(model, config, "student"):
            student_trajectory = self._simulate_student_distill_trajectory(model, batch, config, train_progress, sigmas)
        x0 = student_trajectory[-1]
        x_sigma, sigma_k, _ = self._distill_renoise(x0, train_progress, seed_offset=1)

        with self.adapter_state(model, config, "teacher"), torch.no_grad():
            real_predicted = self._cfg_baked_distill_velocity(
                model, batch, config, train_progress, x_sigma.detach(), sigma_k, cfg_scale
            )
        with self.adapter_state(model, config, "fake"), torch.no_grad():
            fake_predicted = self.predict_distill_velocity(
                model, batch, config, train_progress, x_sigma.detach(), sigma_k, conditioning="conditional"
            )

        kl_loss = dmd2_kl_pseudo_loss(
            x_sigma,
            real_predicted,
            fake_predicted,
            weight=config.distill_kl_loss_weight,
        )
        scaled_image_latent = self._distill_scaled_image_latent(model, batch)
        endpoint_loss = dmd2_endpoint_loss(
            x0,
            scaled_image_latent,
            weight=config.distill_endpoint_loss_weight,
        )
        lpips_loss = None
        if config.distill_endpoint_lpips_weight > 0.0:
            lpips_term = self._distill_perceptual_loss(model, x0, scaled_image_latent, config)
            if lpips_term is not None:
                lpips_loss = config.distill_endpoint_lpips_weight * lpips_term
                endpoint_loss = endpoint_loss + lpips_loss

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

        self._dpo_paired_half = chosen_b
        try:
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

        self._last_dpo_metrics = {
            "loss": loss.detach().item(),
            "dpo_loss": dpo_loss.detach().item(),
            "chosen_reward": chosen_ratio.detach().mean().item(),
            "rejected_reward": rejected_ratio.detach().mean().item(),
            "reward_margin": margin.detach().mean().item(),
            "accuracy": (chosen_ratio > rejected_ratio).float().mean().item(),
        }

        if config.rlhf_dpo_timestep_margin_logging and policy_timestep is not None:
            # Per-sample raw margins bucketed by the chosen half's timestep
            # quartile. Sums and counts are emitted for every quartile so the
            # trainer's accumulation always sees the same key set.
            t = policy_timestep[:chosen_b].detach().float()
            if t.numel() > 0 and t.max() > 1.0:
                t = t / 1000.0  # discrete schedulers train on 1000 timesteps
            quartile_index = (t * 4).long().clamp(0, 3)
            per_sample_margin = margin.detach()
            for quartile in range(4):
                mask = quartile_index == quartile
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_sum"] = per_sample_margin[mask].sum().item()
                self._last_dpo_metrics[f"margin_t_q{quartile + 1}_count"] = float(mask.sum().item())

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
                    for param, ref_data in zip(adapter.parameters(), ref_params, strict=True):
                        param.data = ref_data
                yield
            finally:
                for adapter, policy_ptrs in zip(adapters, policy_data, strict=True):
                    for param, policy_ptr in zip(adapter.parameters(), policy_ptrs, strict=True):
                        param.data = policy_ptr
        else:
            raise ValueError(f"Unsupported DPO reference mode: {ref_mode}")

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
