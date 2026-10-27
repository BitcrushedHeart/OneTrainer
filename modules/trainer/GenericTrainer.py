import contextlib
import copy
import gc
import json
import math
import os
import random
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.oft_utils import set_rotation_cache_seed
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.compile_util import init_compile, save_precompile_cache, set_precompile_cache
from modules.util.concept_probe_util import (
    DEFAULT_PROBE_TIMESTEPS,
    DIAGNOSTICS_DIRNAME,
    LEDGER_FILENAME,
    PROBES_FILENAME,
    ConceptVisitLedger,
    ProbeSet,
    move_sample,
)
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dataset_fingerprint import compute_concept_fingerprint
from modules.util.dpo_addon_util import (
    CyclingBatchSource,
    accumulate_dpo_metrics,
    derive_dpo_addon_interval,
    dpo_ref_rebase_due,
    is_dpo_addon_step,
)
from modules.util.dpo_beta_controller import DPOBetaController, controller_concept_terms, controller_margin_terms
from modules.util.dpo_live_saturation import (
    LIVE_DIRNAME,
    DPOConceptInfo,
    DPOConceptResolver,
    DPOConceptWindow,
    LiveSaturationLedger,
    dpo_concept_infos,
)
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PatienceMode import PatienceMode
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.grad_attribution_util import (
    GENERIC_CONCEPT_FOLDERS,
    ConceptWindowStats,
    GradAttribution,
    TimestepBinNormalizer,
    concept_groups,
    concept_tag,
    sample_mask,
)
from modules.util.profiling_util import TorchMemoryRecorder, TorchProfiler
from modules.util.rlhf_schedule_util import KIND_ORDER, IndependentSchedule, MicroStepKind, wind_data_loader
from modules.util.text_reward_util import TextRewardScheduler, derive_text_reward_interval
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.transfer_util import TransferStopCriterion

from mgds.pipelineModules.SmartDiskCache import CachingStoppedException

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
import numpy as np
from requests.exceptions import ConnectionError
from tqdm import tqdm


def _latent_shape(batch) -> tuple[int, ...] | None:
    """The ``latent_image`` shape of a batch dict, if it has one (config log_vram)."""
    latent = batch.get("latent_image") if isinstance(batch, dict) else None
    return tuple(latent.shape) if isinstance(latent, Tensor) and latent.ndim >= 3 else None


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader
    # Second validation loader of a run that has BOTH standard validation (val
    # loss on the plain VALIDATION concepts -> patience) and DPO validation
    # (accuracy/margin on the chosen/rejected VALIDATION concepts -> DPO
    # patience). None when the run has at most one kind; then
    # validation_data_loader is that one loader (DPO or standard).
    dpo_validation_data_loader: BaseDataLoader | None

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    # Per-concept DPO metrics and the live saturation ledger, set up in start().
    # Class-level defaults, so end() and __backup() are safe on a trainer whose
    # start() never ran.
    _dpo_concept_resolver: DPOConceptResolver | None = None
    _dpo_concept_window: DPOConceptWindow | None = None
    _dpo_live_ledger: LiveSaturationLedger | None = None
    _dpo_unmapped_pair_warned: bool = False
    # Per-image learning signals of the gradient diagnostics, likewise.
    _visit_ledger: ConceptVisitLedger | None = None
    _probe_set: ProbeSet | None = None
    _diagnostics_dir: str | None = None
    _diagnostics_saved_at: float = 0.0

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        distill_error = self.config.validate_distill_startup()
        if distill_error is not None:
            raise RuntimeError(distill_error)

        transfer_error = self.config.validate_transfer_startup()
        if transfer_error is not None:
            raise RuntimeError(transfer_error)

        redistill_error = self.config.validate_redistill_startup()
        if redistill_error is not None:
            raise RuntimeError(redistill_error)

        prompt_only_error = self.config.validate_prompt_only_startup()
        if prompt_only_error is not None:
            raise RuntimeError(prompt_only_error)

        rlhf_error = self.config.validate_rlhf_startup()
        if rlhf_error is not None:
            raise RuntimeError(rlhf_error)

        text_reward_error = self.config.validate_text_reward_startup()
        if text_reward_error is not None:
            raise RuntimeError(text_reward_error)

        rotation_cache_error = self.config.validate_oft_rotation_cache_startup()
        if rotation_cache_error is not None:
            raise RuntimeError(rotation_cache_error)

        grad_diagnostics_error = self.config.validate_grad_diagnostics_startup()
        if grad_diagnostics_error is not None:
            raise RuntimeError(grad_diagnostics_error)
        if self.config.oft_rotation_cache and not self.config.oft_rotation_cache_applies():
            print(
                "Warning: the OFT rotation cache is enabled, but this run does not train OFT v2 adapters "
                "(LoRA tab, PEFT type OFT v2); ignoring it."
            )

        if multi.is_master():
            # TB writer creation is deferred to start() so a backup's tensorboard_subdir
            # can be reused and logging continues in the same TB run on resume.
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()
        self.tensorboard = None

        self.model = None
        self.one_step_trained = False
        self._precompile_cache_saved = False
        self.grad_hook_handles = []
        self.dpo_validation_data_loader = None
        # Unpaired KTO (KTO concepts): the third loader, or the main loader
        # when a DPO-mode run has KTO concepts and no pairs.
        self.kto_source = None
        self._kto_addon_interval = 1
        self._kto_primary = False
        self.dpo_addon_source = None
        self._dpo_addon_interval = 1

        # Loop-local state mirrored on the trainer instance so the
        # save path (__backup / __save) can read it without having to
        # be threaded through the training loop. Written by the loop
        # at lines around the accumulator-reset and update-step
        # boundaries; consumed by _stage_accumulator_state().
        self._loop_accumulated_loss: float = 0.0
        # Live tensor reference -- preferred over the float when present,
        # because the tensor lives on train_device and we only need to
        # .item() it at save time (rare event, one host sync each).
        self._loop_accumulated_loss_tensor: torch.Tensor | None = None
        self._loop_accumulated_dpo_metrics: dict | None = None
        # On-image text reward add-on: this epoch's pre-encoded prompt bank
        # and the micro-batch scheduler that spreads its budget over the epoch.
        self._text_reward_bank = None
        self._text_reward_scheduler = TextRewardScheduler(0)
        self._text_reward_epoch_samples = 0
        # rlhf_addon_schedule INDEPENDENT: this epoch's micro-step sequence
        # (standard / DPO / KTO / text reward), rebuilt at every epoch start.
        # None under PIGGYBACK, where the main loader alone drives the epoch.
        self._independent_schedule: IndependentSchedule | None = None
        # Gradient attribution diagnostics (config log_grad_components), set up
        # at the start of train(); all None while the diagnostics are off.
        self._grad_attribution: GradAttribution | None = None
        self._concept_attribution: GradAttribution | None = None
        self._t_normalizer: TimestepBinNormalizer | None = None
        self._concept_stats: ConceptWindowStats | None = None
        self._sft_concept_resolver: DPOConceptResolver | None = None
        self._probe_dpo_resolver: DPOConceptResolver | None = None
        # Per-concept adaptive beta: one controller per adaptive DPO concept
        # (keyed by its TensorBoard tag, created on its first update) and the
        # ratio each one last produced. The keys are fixed at start so every
        # rank reduces the same per-concept terms.
        self._dpo_adaptive_keys: list[str] = []
        self._dpo_beta_controllers: dict[str, DPOBetaController] = {}
        self._dpo_beta_ratios: dict[str, float] = {}
        self._concept_split_window = False
        # Per-image visit ledger and the fixed probe set (concept_probe_util),
        # persisted under <workspace>/diagnostics with the backups.
        self._visit_ledger: ConceptVisitLedger | None = None
        self._probe_set: ProbeSet | None = None
        self._diagnostics_dir: str | None = None
        self._diagnostics_saved_at = 0.0
        self._loop_scaler = None
        # Config log_vram: the latent shape an add-on micro-step ran at (its
        # batch is fetched inside the helper, so the loop never sees it).
        self._vram_addon_shape: tuple[int, ...] | None = None

        # Defaults for state start() re-initializes; __advance_redistill_phase
        # must be safe to reach even if start() was bypassed (tests).
        self._transfer_best_backup_path: str | None = None
        self._transfer_val_failures = 0

        # Redistill phase state: Phase A (transfer toward the old variant's
        # late tensors) until the step budget or convergence, then Phase B
        # (DMD2 distill). The marker json makes the boundary resume-proof.
        self._redistill_phase_b = False
        self._redistill_resume_start_step: int | None = None
        self._redistill_phase_marker_path = os.path.join(self.config.workspace_dir, "backup", "redistill-phase.json")
        if self.config.redistill_enabled:
            if self.config.continue_last_backup and os.path.isfile(self._redistill_phase_marker_path):
                try:
                    with open(self._redistill_phase_marker_path) as marker_file:
                        marker = json.load(marker_file)
                    self._redistill_resume_start_step = int(marker["phase_b_start_step"])
                    print(f"Resuming redistill in Phase B (started at step {self._redistill_resume_start_step}).")
                except Exception:
                    traceback.print_exc()
                    print("Could not read the redistill phase marker; Phase A state will be re-derived.")
            elif not self.config.continue_last_backup and os.path.isfile(self._redistill_phase_marker_path):
                try:
                    os.remove(self._redistill_phase_marker_path)
                except OSError:
                    traceback.print_exc()

    def start(self):
        # torch._dynamo.config overrides are thread-local. The web backend constructs the
        # trainer on the API request thread but runs start()/train() on a dedicated training
        # thread, so this must happen here (not in __init__) or the recompile limit stays at
        # the default of 8 and fullgraph=True training hard-fails on the 9th recompile.
        set_precompile_cache(self.config.compile and self.config.compile_cache)
        init_compile()
        phase_start = time.monotonic()

        if multi.is_master():
            self.__save_config_to_workspace()

            if self.config.clear_cache_before_training and self.config.latent_caching:
                self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        last_backup_path = None
        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(token=self.config.secrets.huggingface_token)

        self.callbacks.on_update_status("loading the model")

        if self.config.quantization.cache_dir is None:
            self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
            quantization=self.config.quantization,
        )
        self.model.train_config = self.config
        phase_start = self.__log_startup_phase("model load", phase_start)

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        # Adapters are loaded, dtype-cast and on the train device here; the cache is
        # filled once now and follows later device moves through _apply.
        self.model_setup.setup_adapter_rotation_cache(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        if self.config.rlhf_enabled and self.config.training_method != TrainingMethod.LORA:
            raise NotImplementedError("RLHF DPO is currently implemented for adapter training in the LoRA tab only.")

        # Resuming a DPO run: restore the frozen reference saved with the backup so
        # it is not re-captured from the already-trained adapter weights - unless
        # the user asked for a rebase, in which case the stale reference is left
        # behind and the adapter as it stands at the first DPO step becomes the
        # new anchor (the same lazy capture a fresh run does).
        if self.config.rlhf_enabled and last_backup_path:
            if self.config.rlhf_dpo_ref_rebase:
                print(
                    "DPO reference rebase: ignoring the reference saved with the backup; "
                    "the adapter as it stands at the first DPO step becomes the new reference."
                )
            else:
                self.model_setup.load_dpo_reference(last_backup_path, self.model, self.config)
        self._dpo_ref_rebase_bucket = 0

        if self.config.rlhf_sft_primary() and self.__count_sft_anchor_concepts() == 0:
            raise RuntimeError(
                "SFT + DPO mode needs at least one enabled STANDARD/PRIOR_PREDICTION concept without DPO "
                "patterns to drive the main loader; use RLHF mode DPO for a pairs-only run."
            )
        kto_concepts = self.__count_kto_concepts() if self.config.rlhf_enabled else 0
        dpo_pair_concepts = self.__count_dpo_pair_concepts() if self.config.rlhf_enabled else 0
        # A DPO-mode run whose only preference data are KTO concepts (Liked /
        # Disliked singles): the KTO loader becomes the main loader.
        self._kto_primary = self.config.rlhf_dpo_primary() and dpo_pair_concepts == 0 and kto_concepts > 0
        if self.config.rlhf_sft_primary() and dpo_pair_concepts == 0 and kto_concepts == 0:
            raise RuntimeError(
                "SFT + DPO mode needs preference data: an enabled concept with DPO chosen/rejected patterns, "
                "or a KTO concept of Liked or Disliked images."
            )

        phase_start = self.__log_startup_phase("model setup (quantize, adapters, optimizer)", phase_start)
        self.callbacks.on_update_status("creating the data loader/caching")

        # In SFT + DPO mode the main loader is the plain standard pipeline with
        # the DPO pattern concepts excluded (the same construction as the
        # SFT-anchor loader); the pairs come from a secondary loader below.
        if self._kto_primary:
            print(
                f"KTO: {kto_concepts} KTO concept(s) and no DPO pairs; the Liked / Disliked singles drive the "
                "run (loss = kto_loss, plus the SFT anchor when standard concepts are present)."
            )
        self.data_loader = self.create_data_loader(
            self.model,
            self.model_setup,
            self.model.train_progress,
            is_sft_anchor=self.config.rlhf_sft_primary(),
            is_kto=self._kto_primary,
        )
        self.data_loader.stop_check_fun = self.commands.get_stop_command
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()

        # One-shot guard for the mid-step re-cache warning (see __restore_train_device).
        self.__warned_midstep_recache = False

        # SFT-anchored DPO (RLHF mode DPO): when a DPO run also contains
        # STANDARD/PRIOR_PREDICTION concepts (without DPO patterns), run those
        # samples through a parallel non-DPO loader and add a weighted
        # supervised term to the DPO loss. Without this, the DPO pipeline drops
        # non-pattern concepts.
        self.sft_anchor_source = None
        if self.config.rlhf_dpo_primary() and self.config.rlhf_sft_anchor_weight > 0:
            sft_anchor_count = self.__count_sft_anchor_concepts()
            if sft_anchor_count > 0:
                print(
                    f"SFT-anchored DPO: {sft_anchor_count} STANDARD/PRIOR_PREDICTION "
                    f"concept(s) detected; loss = dpo_loss + "
                    f"{self.config.rlhf_sft_anchor_weight} * sft_loss."
                )
                self.callbacks.on_update_status("creating the SFT-anchor data loader/caching")
                sft_anchor_data_loader = self.create_data_loader(
                    self.model,
                    self.model_setup,
                    self.model.train_progress,
                    is_sft_anchor=True,
                )
                sft_anchor_data_loader.stop_check_fun = self.commands.get_stop_command
                self.sft_anchor_source = CyclingBatchSource(
                    sft_anchor_data_loader, "SFT-anchor", on_epoch_restart=self.__restore_train_device
                )

        if self.config.rlhf_dpo_light_primary_applies() and not self._kto_primary:
            print(
                "DPO light path: every pair micro-step forwards its chosen and rejected halves one at a time "
                "(calculate_dpo_loss_light) instead of one fused [chosen; rejected] forward; same loss and "
                "gradient, about half the peak activation memory."
            )

        # SFT + DPO (RLHF mode SFT_DPO): the standard loss stays at full weight
        # and a DPO pair from this secondary loader is interleaved every
        # N standard micro-batches; loss = sft_loss + weight * dpo_loss on
        # those micro-batches. N is derived per epoch from the loader lengths
        # unless rlhf_dpo_addon_interval overrides it.
        self.dpo_addon_source = None
        self._dpo_addon_interval = 1
        independent = self.config.rlhf_addon_independent()
        if self.config.rlhf_sft_primary() and dpo_pair_concepts > 0:
            if independent:
                print(
                    f"SFT + DPO (independent schedule): every DPO pair batch gets its own micro-step, "
                    f"loss = {self.config.rlhf_dpo_addon_weight} * dpo_loss there; standard micro-steps keep "
                    "the plain loss."
                )
            else:
                print(
                    f"SFT + DPO: loss = sft_loss + {self.config.rlhf_dpo_addon_weight} * dpo_loss "
                    "on interleaved DPO micro-batches."
                )
            self.callbacks.on_update_status("creating the DPO add-on data loader/caching")
            dpo_addon_data_loader = self.create_data_loader(self.model, self.model_setup, self.model.train_progress)
            dpo_addon_data_loader.stop_check_fun = self.commands.get_stop_command
            self.dpo_addon_source = CyclingBatchSource(
                dpo_addon_data_loader, "DPO add-on", on_epoch_restart=self.__restore_train_device
            )

        # Unpaired KTO concepts next to DPO pairs: a third loader. In DPO mode a
        # KTO batch is added to every pair micro-batch at full weight (the same
        # footing as the pairs); in SFT + DPO mode it is interleaved like the
        # DPO add-on, with its own interval and the add-on weight.
        self.kto_source = None
        self._kto_addon_interval = 1
        if kto_concepts > 0 and not self._kto_primary:
            if self.config.rlhf_sft_primary() and independent:
                print(
                    f"KTO: {kto_concepts} KTO concept(s); every KTO batch gets its own micro-step, "
                    f"loss = {self.config.rlhf_dpo_addon_weight} * kto_loss there."
                )
            elif self.config.rlhf_sft_primary():
                print(
                    f"KTO: {kto_concepts} KTO concept(s); loss = sft_loss + {self.config.rlhf_dpo_addon_weight} "
                    "* kto_loss on interleaved KTO micro-batches."
                )
            else:
                print(f"KTO: {kto_concepts} KTO concept(s); loss = dpo_loss + kto_loss on every micro-batch.")
            self.callbacks.on_update_status("creating the KTO data loader/caching")
            kto_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_kto=True
            )
            kto_data_loader.stop_check_fun = self.commands.get_stop_command
            self.kto_source = CyclingBatchSource(kto_data_loader, "KTO", on_epoch_restart=self.__restore_train_device)

        if self.config.text_reward_enabled and independent:
            print(
                f"Text reward (independent schedule): {self.config.text_reward_samples_per_epoch} OCR-rewarded "
                f"rollouts per epoch, each on its own micro-step with loss = {self.config.text_reward_weight} * ctc."
            )
        elif self.config.text_reward_enabled:
            print(
                f"Text reward: {self.config.text_reward_samples_per_epoch} OCR-rewarded rollouts per epoch, "
                f"loss += {self.config.text_reward_weight} * ctc (never on a DPO add-on micro-batch)."
            )

        if self.config.rlhf_enabled:
            self.model_setup.set_dpo_chosen_anchor_overridden(self.__dpo_chosen_anchor_overridden())
        self.__setup_dpo_pair_logging()
        self.__setup_adaptive_beta_concepts()

        if self.config.patience and not self.config.validation:
            # Flipped before the loaders are built: config.validation also
            # switches on the per-concept outputs the validation-loss pass reads.
            print("Warning: Patience enabled without Validation. Auto-enabling Validation.")
            self.config.validation = True

        self.dpo_validation_data_loader = None
        if self.config.validation and self.config.dpo_validation_active():
            # Both kinds of validation: the plain loader reads the VALIDATION
            # concepts without DPO patterns (val loss -> patience), the DPO
            # loader reads the ones with patterns (accuracy/margin -> DPO
            # patience). They are different pipelines, so one loader each.
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True, dpo_validation=False
            )
            self.dpo_validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True, dpo_validation=True
            )
        elif self.config.validation or self.config.dpo_validation_active():
            # One kind only: the loader follows the config (DPO pipeline when
            # DPO validation is active, plain validation loss otherwise).
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

        self.__log_startup_phase("data loaders", phase_start)

        self._patience_counter = 0
        self._patience_checks = 0
        self._patience_best_loss = float("inf")
        self._patience_best_step = -1
        self._patience_best_backup_path: str | None = None
        self._patience_alt_backup_path: str | None = None
        self._patience_alt_step = -1
        self._patience_second_backup_path: str | None = None
        self._patience_second_loss = float("inf")
        self._patience_second_step = -1

        self._dpo_patience_counter = 0
        self._dpo_best_accuracy = float("-inf")
        self._dpo_best_margin = float("-inf")
        self._dpo_best_backup_path: str | None = None
        self._dpo_beta_controller: DPOBetaController | None = None

        self._transfer_stop = TransferStopCriterion(
            patience=self.config.transfer_patience_checks if self.config.transfer_patience_enabled else 0,
            min_relative_improvement=self.config.transfer_min_relative_improvement,
            target_residual=self.config.transfer_target_residual,
        )
        self._transfer_best_backup_path: str | None = None
        self._transfer_val_failures = 0

        if self.config.transfer_enabled or self.config.redistill_enabled:
            # The criterion state must survive resume, or the first post-resume
            # validation "improves" on inf and clobbers a genuinely better
            # pre-resume snapshot. A sidecar json carries the best residual.
            best_path = os.path.join(self.config.workspace_dir, "backup", "transfer-best.pt")
            meta_path = best_path + ".json"
            if self.config.continue_last_backup and os.path.isfile(best_path) and os.path.isfile(meta_path):
                try:
                    with open(meta_path) as meta_file:
                        best_meta = json.load(meta_file)
                    self._transfer_stop.best_residual = float(best_meta["residual"])
                    self._transfer_stop.best_tag = best_meta.get("step")
                    self._transfer_best_backup_path = best_path
                    print(
                        f"Resuming transfer best state: residual {self._transfer_stop.best_residual:.6f} "
                        f"from step {self._transfer_stop.best_tag}"
                    )
                except Exception:
                    traceback.print_exc()
                    print("Could not read transfer best sidecar; starting the criterion fresh.")
            else:
                # Fresh run: stale best files from a previous run in this
                # workspace must not leak into end()'s restore.
                for stale in (best_path, meta_path, best_path + ".tmp"):
                    if os.path.isfile(stale):
                        try:
                            os.remove(stale)
                        except OSError:
                            traceback.print_exc()

        if multi.is_master():
            self._init_tensorboard_writer()

    def _init_tensorboard_writer(self):
        """Construct the SummaryWriter.

        On resume (model.resumed_tensorboard_subdir is set and the directory still
        exists under workspace_dir/tensorboard), reuse the existing subdir so the
        TB UI shows a single continuous run across stop/resume cycles. Use
        purge_step to trim any scalars written after the backup point by a
        previous session that crashed mid-log.
        """
        tensorboard_log_dir = os.path.join(self.config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)

        resumed = getattr(self.model, "resumed_tensorboard_subdir", None)
        reuse_path = None
        if resumed:
            candidate = os.path.join(tensorboard_log_dir, resumed)
            if os.path.isdir(candidate):
                reuse_path = candidate

        if reuse_path is not None:
            self.model.tensorboard_subdir = resumed
            self.tensorboard = SummaryWriter(
                reuse_path,
                purge_step=self.model.train_progress.global_step,
            )
        else:
            subdir = f"{self.config.save_filename_prefix}{get_string_timestamp()}"
            self.model.tensorboard_subdir = subdir
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, subdir))

    def __restore_train_device(self):
        """Put the model back on the train device after a secondary loader
        restarted its epoch mid-step.

        A mid-epoch restart of the SFT-anchor / DPO add-on loader re-runs that
        loader's caching pass. If the pass actually has to encode anything, it
        moves the whole model to the temp device and pulls the VAE (latents) or
        the text encoder (prompts) up onto the train device instead; nothing
        put the model back, so the next forward ran against a CPU-resident
        transformer and died with "Expected all tensors to be on the same
        device".

        The device check makes this a strict no-op for the common case: a
        restart served entirely from an already-complete cache never moves
        anything, so this must not touch the model, print, or churn memory.
        When it does fire, the run has just paid for a mid-training encode
        pass -- worth saying out loud once, because the usual cause is a
        secondary concept whose variations exceed what was cached up front.
        """
        # The trained parameters follow the transformer (transformer_to moves
        # the adapters with it), so they report where the model actually is.
        # Compare device TYPE only: config's train device may be an indexless
        # "cuda" while a parameter reports "cuda:0".
        if not self.parameters or self.parameters[0].device.type == self.train_device.type:
            return

        if not self.__warned_midstep_recache:
            self.__warned_midstep_recache = True
            print(
                "Warning: a secondary (SFT-anchor / DPO add-on) loader re-cached mid-epoch, which moved the "
                "model off the train device. Restoring it. This costs a full model round-trip per restart -- "
                "cache that concept's variations up front to avoid it."
            )

        self.model_setup.setup_train_device(self.model, self.config)

    def __start_text_reward_epoch(self):
        """Draw and encode this epoch's reward prompts, and size the interval
        so ``text_reward_samples_per_epoch`` rollouts fit the primary epoch."""
        epoch = self.model.train_progress.epoch
        self.callbacks.on_update_status("encoding text reward prompts")
        self._text_reward_bank = self.model_setup.build_text_reward_bank(self.model, self.config, epoch)
        primary_batches = self.data_loader.get_data_set().approximate_length()
        # Offline images: every dataset image once per epoch; online: the configured budget.
        samples = (
            len(self._text_reward_bank) * max(1, self.config.text_reward_dataset_repeats)
            if self._text_reward_bank.image_paths is not None
            else self.config.text_reward_samples_per_epoch
        )
        self._text_reward_epoch_samples = samples
        if self.config.rlhf_addon_independent():
            # Every rollout is its own micro-step; the schedule below places them.
            self._text_reward_scheduler.start_epoch(0)
            return
        interval = derive_text_reward_interval(primary_batches, samples)
        self._text_reward_scheduler.start_epoch(interval)
        if multi.is_master():
            print(
                f"Text reward: one rollout every {interval} standard micro-batch(es) "
                f"({len(self._text_reward_bank)} prompts vs {primary_batches} standard batches per epoch)."
            )

    def __start_secondary_epochs(self):
        """Advance the SFT-anchor / DPO add-on loader epochs in lockstep with
        the main loader and re-derive the add-on interval from the new lengths
        (PIGGYBACK), or build this epoch's micro-step schedule (INDEPENDENT)."""
        if self.sft_anchor_source is not None:
            self.sft_anchor_source.start_next_epoch()
        if self.config.text_reward_enabled:
            self.__start_text_reward_epoch()
        if self.config.rlhf_addon_independent():
            if self.dpo_addon_source is not None:
                self.dpo_addon_source.start_next_epoch()
            if self.kto_source is not None:
                self.kto_source.start_next_epoch()
            self.__start_independent_epoch()
            return
        if self.dpo_addon_source is not None:
            self.dpo_addon_source.start_next_epoch()
            sft_batches = self.data_loader.get_data_set().approximate_length()
            dpo_batches = self.dpo_addon_source.approximate_length()
            self._dpo_addon_interval = derive_dpo_addon_interval(
                sft_batches, dpo_batches, self.config.rlhf_dpo_addon_interval
            )
            if multi.is_master():
                print(
                    f"SFT + DPO: one DPO pair every {self._dpo_addon_interval} standard micro-batch(es) "
                    f"({dpo_batches} pair batches vs {sft_batches} standard batches per epoch)."
                )
        if self.kto_source is not None:
            self.kto_source.start_next_epoch()
            if self.config.rlhf_sft_primary():
                sft_batches = self.data_loader.get_data_set().approximate_length()
                kto_batches = self.kto_source.approximate_length()
                self._kto_addon_interval = derive_dpo_addon_interval(
                    sft_batches, kto_batches, self.config.rlhf_dpo_addon_interval
                )
                if multi.is_master():
                    print(
                        f"SFT + KTO: one KTO batch every {self._kto_addon_interval} standard micro-batch(es) "
                        f"({kto_batches} KTO batches vs {sft_batches} standard batches per epoch)."
                    )

    def __start_independent_epoch(self):
        """Build this epoch's INDEPENDENT micro-step schedule from the loader
        lengths (every loader has just started its epoch, so the lengths are
        current) and, on a mid-epoch resume, wind every source forward to
        where the schedule says it was.

        Resume rule: ``epoch_step`` is the position in the schedule. The
        schedule is a pure function of the per-kind counts, so a backup
        written under this schedule resumes exactly where it stopped: each
        loader skips the batches the schedule consumed before that position
        and the text bank's cursor moves with it. A backup written under
        PIGGYBACK (or a different dataset) carries an ``epoch_step`` that
        counted standard batches instead; it is reinterpreted as a position
        in this schedule -- the rest of the epoch trains under the new
        schedule, a few batches may be seen twice or skipped once in that
        epoch, nothing crashes and nothing is skipped wholesale. A position
        past the end resumes at the end (the epoch is over, the next one
        starts fresh).
        """
        train_progress = self.model.train_progress
        counts = {MicroStepKind.SFT: self.data_loader.get_data_set().approximate_length()}
        if self.dpo_addon_source is not None:
            counts[MicroStepKind.DPO] = self.dpo_addon_source.approximate_length()
        if self.kto_source is not None and self.config.rlhf_sft_primary():
            counts[MicroStepKind.KTO] = self.kto_source.approximate_length()
        if self._text_reward_bank is not None:
            counts[MicroStepKind.TEXT] = self._text_reward_epoch_samples
        if all(count <= 0 for kind, count in counts.items() if kind is not MicroStepKind.SFT):
            # Nothing to interleave: the main loader drives the epoch as usual.
            self._independent_schedule = None
            return
        schedule = IndependentSchedule(counts)
        self._independent_schedule = schedule
        if multi.is_master():
            print(f"Independent schedule: {schedule.describe(self.config.gradient_accumulation_steps)}.")

        if train_progress.epoch_step <= 0:
            return
        position = schedule.clamp(train_progress.epoch_step)
        consumed = schedule.consumed_before(position)
        batch_size = self.config.batch_size
        wind_data_loader(self.data_loader, consumed.get(MicroStepKind.SFT, 0), batch_size)
        if self.dpo_addon_source is not None:
            wind_data_loader(self.dpo_addon_source.data_loader, consumed.get(MicroStepKind.DPO, 0), batch_size)
        if MicroStepKind.KTO in counts:
            wind_data_loader(self.kto_source.data_loader, consumed.get(MicroStepKind.KTO, 0), batch_size)
        if self._text_reward_bank is not None:
            self._text_reward_bank.cursor = consumed.get(MicroStepKind.TEXT, 0)
        # epoch_sample counts standard samples (the main loader's own resume
        # offset); keep it consistent with the position the schedule resumes at.
        train_progress.epoch_sample = consumed.get(MicroStepKind.SFT, 0) * batch_size
        if multi.is_master():
            done = ", ".join(f"{count} {kind}" for kind, count in consumed.items())
            print(f"Independent schedule: resuming at micro-step {position} of {len(schedule)} ({done} already done).")

    def __load_concepts(self) -> list:
        """The run's concepts, from the config or else its concept file.

        Mirrors the loading logic in DataLoaderMgdsMixin._create_mgds.
        """
        from modules.util.config.ConceptConfig import ConceptConfig

        concepts = self.config.concepts
        if concepts is None:
            concept_file = self.config.concept_file_name
            if not concept_file or not os.path.isfile(concept_file):
                return []
            with open(concept_file, "r") as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]
        return concepts

    @staticmethod
    def __log_startup_phase(name: str, started_at: float) -> float:
        """Print how long a startup phase took and return the next phase's start time."""
        now = time.monotonic()
        print(f"[startup] {name}: {now - started_at:.1f}s", flush=True)
        return now

    def __count_sft_anchor_concepts(self) -> int:
        """Return the number of enabled STANDARD/PRIOR_PREDICTION concepts.

        Used to decide whether the parallel SFT-anchor data loader is needed.
        """
        concepts = self.__load_concepts()
        # DPO pattern concepts are typed STANDARD but feed the DPO pipeline;
        # exclude them so their chosen images aren't double-trained as anchors.
        anchor_types = {ConceptType.STANDARD, ConceptType.PRIOR_PREDICTION}
        return sum(1 for c in concepts if c.enabled and ConceptType(c.type) in anchor_types and not c.is_dpo())

    def __count_kto_concepts(self) -> int:
        """Enabled KTO concepts (single Liked / Disliked images)."""
        return sum(1 for c in self.__load_concepts() if c.enabled and ConceptType(c.type) == ConceptType.KTO)

    def __count_dpo_pair_concepts(self) -> int:
        """Enabled concepts with chosen/rejected patterns that the DPO pair loader reads."""
        pair_types = {ConceptType.STANDARD, ConceptType.PRIOR_PREDICTION}
        return sum(1 for c in self.__load_concepts() if c.enabled and ConceptType(c.type) in pair_types and c.is_dpo())

    def __dpo_chosen_anchor_overridden(self) -> bool:
        """Whether an enabled DPO concept sets its own chosen anchor > 0
        (dpo_chosen_anchor_override), which turns the DPOP anchor on for its
        pairs even with rlhf_dpo_chosen_anchor at 0. Read from the concept
        file: a sourceless run whose cache snapshot was not synced may differ."""
        return any(
            c.enabled and c.is_dpo() and getattr(c, "dpo_chosen_anchor_override", -1.0) > 0
            for c in self.__load_concepts()
        )

    def __setup_dpo_pair_logging(self):
        """Per-concept DPO metrics (TensorBoard) and the live saturation ledger
        (<workspace>/dpo_live/), both fed from the per-pair records the model
        setup keeps for the last DPO micro-batch."""
        self._dpo_concept_resolver = None
        self._dpo_concept_window = None
        self._dpo_live_ledger = None
        if not self.config.rlhf_enabled or not (
            self.config.rlhf_dpo_concept_logging or self.config.rlhf_dpo_live_saturation
        ):
            return
        infos = dpo_concept_infos(self.__load_concepts(), self.config.rlhf_supervised_mix)
        if not infos:
            return
        self._dpo_concept_resolver = DPOConceptResolver(infos)
        sigmoid = self.config.rlhf_dpo_objective == DPOObjective.SIGMOID
        if self.config.rlhf_dpo_concept_logging:
            self._dpo_concept_window = DPOConceptWindow(
                with_weight=sigmoid,
                with_anchor=(self.config.rlhf_dpo_chosen_anchor > 0 or self.__dpo_chosen_anchor_overridden())
                and self.config.rlhf_dpo_objective != DPOObjective.KTO,
            )
        if self.config.rlhf_dpo_live_saturation and multi.is_master():
            if not sigmoid:
                print(
                    "DPO live saturation needs the sigmoid objective (IPO has no beta to saturate, KTO saturates "
                    "per half rather than per pair); not tracking."
                )
                return
            out_dir = path_util.canonical_join(self.config.workspace_dir, LIVE_DIRNAME)
            self._dpo_live_ledger = LiveSaturationLedger(
                out_dir,
                infos,
                global_beta=self.config.rlhf_dpo_beta,
                start_step=self.model.train_progress.global_step,
            )
            print(f"DPO live saturation: tracking {len(infos)} concepts in {out_dir}")

    def __collect_dpo_pairs(self, train_progress: TrainProgress):
        """Hands the last DPO micro-batch's per-pair records to the per-concept
        window and the live saturation ledger."""
        if self._dpo_concept_resolver is None:
            return
        records = self.model_setup.get_last_dpo_pair_records()
        if not records:
            return
        # Only the ledger needs it, and it is cached per reference version.
        reference = self.model_setup.dpo_reference_fingerprint() if self._dpo_live_ledger is not None else None
        for record in records:
            info = self._dpo_concept_resolver.resolve(record["path"])
            if info is None:
                if not self._dpo_unmapped_pair_warned:
                    self._dpo_unmapped_pair_warned = True
                    print(
                        f"DPO pair logging: '{record['path']}' is under no DPO concept's chosen folder; "
                        "pairs like it are left out of the per-concept metrics."
                    )
                continue
            if self._dpo_concept_window is not None:
                self._dpo_concept_window.add(info, record)
            if self._dpo_live_ledger is not None:
                self._dpo_live_ledger.record(train_progress.global_step, info, record, reference)

    def __setup_adaptive_beta_concepts(self):
        """Per-concept adaptive beta: the DPO concepts are kept apart because
        their margins live on different scales, so one pooled controller would
        steer every concept's beta by the dominant concept's margins. Each
        adaptive concept gets its own controller, fed only by its own samples.

        Samples are attributed to concepts the way the per-concept metrics are
        (chosen image path under the concept's chosen folder, cloud remap
        included), keyed by the concept's TensorBoard tag: unique per concept
        even when names repeat. The sorted key list is fixed here so every rank
        reduces the same per-concept terms. Opted-out concepts get no key (and
        no controller); a sample under no adaptive concept follows the pooled
        controller as before."""
        self._dpo_adaptive_keys = []
        self._dpo_beta_controllers = {}
        self._dpo_beta_ratios = {}
        if not (
            self.config.rlhf_enabled
            and self.config.rlhf_dpo_adaptive_beta
            and self.config.rlhf_dpo_objective == DPOObjective.SIGMOID
        ):
            return
        infos = dpo_concept_infos(self.__load_concepts(), self.config.rlhf_supervised_mix)
        adaptive = [info for info in infos if info.adaptive]
        if not adaptive:
            return
        resolver = self._dpo_concept_resolver or DPOConceptResolver(infos)
        self._dpo_adaptive_keys = sorted(info.tag for info in adaptive)

        def key_of_path(path: str | None) -> str | None:
            info = resolver.resolve(path)
            return info.tag if info is not None and info.adaptive else None

        self.model_setup.set_dpo_concept_keys(self._dpo_adaptive_keys, key_of_path)
        print(f"DPO adaptive beta: one controller per concept for {', '.join(self._dpo_adaptive_keys)}.")

    # ------------------------------------------------------------------
    # Add-on terms (SFT + DPO pairs / KTO batches, text reward rollouts).
    # Each one pulls its own batch, backwards its term scaled by the add-on
    # weight / gradient accumulation (the scaler applied when AMP is on) and
    # returns the detached scaled value for the loss accumulator. Shared by
    # the PIGGYBACK path (riding on a standard micro-step) and the
    # INDEPENDENT path (a micro-step of their own); the arithmetic is the
    # same in both, only what else sits in the micro-step differs.
    # ------------------------------------------------------------------

    def __dpo_addon_term(self, scaler, train_progress: TrainProgress) -> Tensor:
        """One DPO pair batch, evaluated half by half with one graph alive at a
        time (calculate_dpo_loss_light); its gradients land through the
        backward callback, so the caller must not backward anything for it."""
        dpo_batch = self.dpo_addon_source.next()
        self._vram_addon_shape = _latent_shape(dpo_batch)
        self.__capture_probe_samples(MicroStepKind.DPO, dpo_batch)
        dpo_scale = self.config.rlhf_dpo_addon_weight / self.config.gradient_accumulation_steps

        def _addon_backward(term, _scale=dpo_scale, _scaler=scaler, **probe):
            # probe: the half's prediction and per-pair values, passed by the
            # light path only while the gradient diagnostics are on.
            self.__backward_term(term * _scale, _scaler, kind=MicroStepKind.DPO, scale=_scale, **probe)

        dpo_value = self.model_setup.calculate_dpo_loss_light(
            self.model, dpo_batch, self.config, train_progress, _addon_backward
        )
        return dpo_value.detach() * dpo_scale

    def __dpo_primary_light_term(self, batch: dict, scaler, train_progress: TrainProgress) -> Tensor:
        """A DPO-primary pair micro-step on the light path
        (rlhf_dpo_light_primary): the main loader's pair batch evaluated half
        by half with one graph alive at a time (calculate_dpo_loss_light),
        scaled by 1 / gradient accumulation exactly like the fused path's
        ``loss / GA``. Its gradients land through the backward callback, so
        the caller skips the common backward for this micro-step. Returns the
        detached scaled loss (preference term, anchor and supervised mix): the
        scalar the fused path would have logged."""
        scale = 1.0 / self.config.gradient_accumulation_steps

        def _primary_backward(term, _scale=scale, _scaler=scaler, **probe):
            self.__backward_term(term * _scale, _scaler, kind=MicroStepKind.DPO, scale=_scale, **probe)

        dpo_value = self.model_setup.calculate_dpo_loss_light(
            self.model, batch, self.config, train_progress, _primary_backward
        )
        return dpo_value.detach() * scale

    def __dpo_primary_extra_term(self, term: Tensor, scaler) -> Tensor:
        """A term the fused DPO-primary path folds into its one loss (a KTO
        batch, the weighted SFT anchor), backwarded on its own under the light
        path with the same 1 / gradient accumulation scaling. Returns the
        detached scaled value for the loss accumulator."""
        scale = 1.0 / self.config.gradient_accumulation_steps
        self.__backward_term(term * scale, scaler, kind=MicroStepKind.DPO, scale=scale)
        return term.detach() * scale

    def __kto_addon_term(self, scaler, train_progress: TrainProgress) -> Tensor:
        """One KTO batch: a no-grad reference forward and one policy forward,
        backwarded here with the add-on weight like the pair (one graph alive)."""
        kto_batch = self.kto_source.next()
        self._vram_addon_shape = _latent_shape(kto_batch)
        kto_scale = self.config.rlhf_dpo_addon_weight / self.config.gradient_accumulation_steps
        kto_value = self.model_setup.calculate_kto_loss(self.model, kto_batch, self.config, train_progress)
        kto_term = kto_value * kto_scale
        if scaler:
            scaler.scale(kto_term).backward()
        else:
            kto_term.backward()
        detached = kto_value.detach() * kto_scale
        del kto_value, kto_term
        return detached

    def __text_reward_term(self, scaler, train_progress: TrainProgress) -> tuple[Tensor, dict]:
        """One text reward rollout; the CTC term is backwarded inside
        calculate_text_reward_loss through the callback. Returns the detached
        scaled value and the rollout's metrics."""
        text_scale = self.config.text_reward_weight / self.config.gradient_accumulation_steps

        def _text_backward(term, _scale=text_scale, _scaler=scaler, **_probe):
            term = term * _scale
            if _scaler:
                _scaler.scale(term).backward()
            else:
                term.backward()

        text_value = self.model_setup.calculate_text_reward_loss(
            self.model, self._text_reward_bank.next(), self.config, train_progress, _text_backward
        )
        return text_value.detach() * text_scale, self.model_setup.get_last_text_reward_metrics()

    # ------------------------------------------------------------------
    # Gradient attribution diagnostics (config log_grad_components).
    # See modules/util/grad_attribution_util.py for the method; this is the
    # wiring: where the snapshots are taken, how a term is backwarded with
    # its prediction probed and split by concept, and what gets logged.
    # ------------------------------------------------------------------

    def __setup_grad_diagnostics(self):
        if not self.config.log_grad_components or self._grad_attribution is not None:
            return
        interval = self.config.log_grad_components_interval
        self._grad_attribution = GradAttribution(self.parameters, order=[str(kind) for kind in KIND_ORDER])
        self._concept_attribution = GradAttribution(self.parameters) if interval > 0 else None
        self._t_normalizer = TimestepBinNormalizer()
        self._concept_stats = ConceptWindowStats()

        # Concept tags: the DPO scheme (folder name, TensorBoard-safe, clashes
        # suffixed) for the DPO pattern concepts, the same for the standard
        # concepts the main loader reads; both resolve a sample by its image
        # path (the DPO one by the chosen folder, including the cloud remap).
        concepts = self.__load_concepts()
        dpo_infos = dpo_concept_infos(concepts, self.config.rlhf_supervised_mix) if self.config.rlhf_enabled else []
        self._probe_dpo_resolver = self._dpo_concept_resolver or (DPOConceptResolver(dpo_infos) if dpo_infos else None)
        tags_seen: dict[str, int] = {info.tag: 1 for info in dpo_infos}
        sft_infos = []
        for concept in concepts:
            if not concept.enabled or concept.is_dpo() or ConceptType(concept.type) == ConceptType.VALIDATION:
                continue
            label = os.path.basename(os.path.normpath(concept.path)) or concept.name or "concept"
            if label.lower() in GENERIC_CONCEPT_FOLDERS:
                # "chosen" alone names nothing: say whose chosen folder it is
                label = f"{os.path.basename(os.path.dirname(os.path.normpath(concept.path)))} {label}"
            tag = concept_tag(label)
            seen = tags_seen.get(tag, 0) + 1
            tags_seen[tag] = seen
            if seen > 1:
                tag = f"{tag}_{seen}"
            sft_infos.append(
                DPOConceptInfo(
                    label=label,
                    tag=tag,
                    path=os.path.normpath(concept.path),
                    chosen_dir=concept.path,
                    beta_override=0.0,
                )
            )
        self._sft_concept_resolver = DPOConceptResolver(sft_infos) if sft_infos else None
        self.model_setup.set_grad_probe(True)

        # Per-image learning signals. Both are restored on a resume (a fresh
        # run starts them empty and drops a stale file); a missing or
        # unreadable file simply starts empty.
        self._diagnostics_dir = os.path.join(self.config.workspace_dir, DIAGNOSTICS_DIRNAME)
        ledger_path = os.path.join(self._diagnostics_dir, LEDGER_FILENAME)
        probes_path = os.path.join(self._diagnostics_dir, PROBES_FILENAME)
        resuming = self.model.train_progress.global_step > 0
        self._visit_ledger = ConceptVisitLedger.load(ledger_path) if resuming else ConceptVisitLedger()
        self._probe_set = ProbeSet.load(probes_path) if resuming and interval > 0 else ProbeSet()
        if not resuming:
            for stale in (ledger_path, probes_path):
                if os.path.isfile(stale):
                    try:
                        os.remove(stale)
                    except OSError:
                        traceback.print_exc()
        if multi.is_master():
            concept_count = len(sft_infos) + len(dpo_infos)
            per_probe = concept_count * self._probe_set.images_per_concept * len(DEFAULT_PROBE_TIMESTEPS)
            print(
                f"Gradient diagnostics: per-image visit ledger ({len(self._visit_ledger)} images restored)"
                + (
                    f"; fixed probes: up to {concept_count} concepts x {self._probe_set.images_per_concept} images x "
                    f"{len(DEFAULT_PROBE_TIMESTEPS)} timesteps = {per_probe} no-grad forwards every {interval} updates "
                    f"(about {per_probe / interval:.2f} forwards per update, DPO pairs cost two batched forwards each; "
                    f"{self._probe_set.total()} probe images restored)."
                    if interval > 0
                    else "; fixed probes off (concept conflict interval 0)."
                )
            )

        if multi.is_master():
            per_copy = sum(p.numel() for p in self._grad_attribution.parameters) * 4 / 2**30
            print(
                f"Gradient diagnostics: per-objective gradient norms and cosines every update "
                f"(float32 snapshot-diff buffers, {per_copy:.2f} GB per copy; one snapshot plus one per objective"
                f"{', plus one snapshot and one per concept for the conflict probe' if interval > 0 else ''}), "
                f"per-concept timestep-normalised loss and output-gradient every update"
                + (
                    f", per-concept parameter-gradient conflict every {interval} updates "
                    "(one extra backward per additional concept in each of that window's batches)."
                    if interval > 0
                    else "."
                )
                + (
                    " OFT rotation cache: its gradients are materialised after every micro-step while this is on."
                    if self.config.oft_rotation_cache_applies()
                    else ""
                )
            )

    def __attribute(self, attribution: GradAttribution | None, key: str):
        """Close the current attribution segment under ``key``. With the OFT
        rotation cache, land the cache's dL/dR in Q.grad first: the deltas
        must be read in parameter space, and this is a no-op on an empty
        accumulator, so the update step's own materialisation never double
        counts."""
        if attribution is None:
            return
        if self.config.oft_rotation_cache_applies():
            self.model_setup.before_clip_adapters(self.model, self.config)
        attribution.end(key)

    def __concept_split_due(self, train_progress: TrainProgress) -> bool:
        interval = self.config.log_grad_components_interval
        if self._concept_attribution is None or interval <= 0:
            return False
        return (train_progress.global_step // self.config.gradient_accumulation_steps) % interval == 0

    def __concept_tags(self, kind: MicroStepKind, image_paths, count: int) -> list[str | None]:
        resolver = self._sft_concept_resolver if kind is MicroStepKind.SFT else self._probe_dpo_resolver
        paths = list(image_paths or [])
        tags: list[str | None] = []
        for i in range(count):
            info = resolver.resolve(paths[i]) if resolver is not None and i < len(paths) else None
            tags.append(info.tag if info is not None else None)
        return tags

    def __backward_term(
        self,
        term: Tensor,
        scaler,
        *,
        kind: MicroStepKind,
        predicted: Tensor | None = None,
        image_paths=None,
        timestep=None,
        losses: Tensor | None = None,
        scale: float = 1.0,
        half: str | None = None,
    ):
        """Backward one already-scaled term. Without the diagnostics (or without
        a prediction to probe) this is exactly the plain backward. With them:
        one autograd.grad from the term to the prediction gives every sample's
        output gradient (the loss part of the graph only, no model backward),
        which feeds the per-concept stats; on a conflict-probe window the
        prediction is then backwarded once per concept group with the masked
        output gradient, so the parameters receive the same total gradient in
        per-concept pieces."""
        scaled = scaler.scale(term) if scaler else term
        probe = (
            self._grad_attribution is not None
            and kind in (MicroStepKind.SFT, MicroStepKind.DPO)
            and isinstance(predicted, torch.Tensor)
            and predicted.requires_grad
        )
        if not probe:
            scaled.backward()
            return
        dpred = torch.autograd.grad(scaled, predicted, retain_graph=True)[0]
        scaler_scale = float(scaler.get_scale()) if scaler else 1.0
        out_norms = dpred.detach().flatten(1).float().norm(dim=1) / (scale * scaler_scale)
        tags = self.__concept_tags(kind, image_paths, predicted.shape[0])
        self.__record_concept_samples(kind, half, tags, out_norms, losses, timestep, image_paths)

        groups = concept_groups(tags) if self._concept_split_window and self._concept_attribution is not None else None
        if not groups:
            scaled.backward()
            return
        if len(groups) == 1:
            scaled.backward()
            self.__attribute(self._concept_attribution, next(iter(groups)))
            return
        items = list(groups.items())
        for i, (tag, indices) in enumerate(items):
            predicted.backward(gradient=dpred * sample_mask(indices, dpred), retain_graph=i < len(items) - 1)
            self.__attribute(self._concept_attribution, tag)

    def __record_concept_samples(
        self, kind: MicroStepKind, half, tags, out_norms: Tensor, losses, timestep, image_paths=None
    ):
        """One host sync: per-sample output-gradient norms, losses and
        timestep fractions, then the timestep-bin normalisation on the host.
        Each sample's normalised loss also goes to the visit ledger under its
        own source path (a pair under its chosen path, once per pair)."""
        count = out_norms.shape[0]
        nan = torch.full_like(out_norms, float("nan"))
        rows = [out_norms]
        if isinstance(losses, torch.Tensor) and losses.numel() == count:
            rows.append(losses.detach().float().reshape(-1).to(out_norms.device))
        else:
            rows.append(nan)
        if isinstance(timestep, torch.Tensor) and timestep.numel() >= count:
            rows.append(
                self.model_setup.timestep_fraction(self.model, timestep.detach())[:count].float().to(out_norms.device)
            )
        else:
            rows.append(nan)
        norms, loss_values, ts = torch.stack(rows).tolist()
        ts = [None if t != t else t for t in ts]
        out_key = f"out:{kind}:{half or ''}"
        out_tnorm = self._t_normalizer.normalize(out_key, norms, ts)
        loss_tnorm = self._t_normalizer.normalize(f"loss:{kind}", loss_values, ts)
        for i, tag in enumerate(tags):
            if tag is None:
                continue
            self._concept_stats.add(tag, "out_norm", norms[i])
            self._concept_stats.add(tag, "out_norm_tnorm", out_tnorm[i])
            if half:
                self._concept_stats.add(tag, f"out_norm_{half}", norms[i])
            self._concept_stats.add(tag, "loss_tnorm", loss_tnorm[i])
            if self._visit_ledger is not None and image_paths is not None and i < len(image_paths):
                self._visit_ledger.record(tag, image_paths[i], loss_tnorm[i], kind=str(kind))

    def __capture_probe_samples(self, kind: MicroStepKind, batch: dict):
        """Fill the fixed probe set from the training stream: the first K
        distinct images per concept the loader yields (seed-deterministic from
        a fresh start; a resume restores the set from disk). Their tensors are
        sliced out of the collated batch and kept on the CPU, so a probe reads
        exactly what training reads from the cache."""
        probes = self._probe_set
        if probes is None or not isinstance(batch, dict):
            return
        paths = batch.get("image_path")
        if not isinstance(paths, (list, tuple)) or not paths:
            return
        tags = self.__concept_tags(kind, paths, len(paths))
        for tag in {t for t in tags if t is not None}:
            if any(probes.wants(tag, p) for t, p in zip(tags, paths, strict=True) if t == tag):
                mine = [p if t == tag else None for t, p in zip(tags, paths, strict=True)]
                probes.capture(tag, str(kind), batch, mine, len(paths))

    def __run_probes(self, train_progress: TrainProgress):
        """Every concept's fixed probe set at the fixed timesteps, no grad,
        after the optimizer step (and the rotation cache refresh) so the curve
        shows the model as it stands. Logs the probe value, its change since
        the first probe over the same images, and the image count."""
        probes = self._probe_set
        if probes is None or probes.total() == 0:
            return
        device = torch.device(self.config.train_device)
        step = train_progress.global_step
        for tag in probes.tags():
            samples = probes.samples(tag)
            kind = probes.kind(tag)
            values = []
            for _path, sample in samples:
                batch = move_sample(sample, device)
                for fraction in DEFAULT_PROBE_TIMESTEPS:
                    if kind == str(MicroStepKind.DPO):
                        values.append(
                            self.model_setup.probe_dpo_margin(self.model, batch, self.config, train_progress, fraction)
                        )
                    else:
                        values.append(
                            self.model_setup.probe_loss(self.model, batch, self.config, train_progress, fraction)
                        )
            value = sum(values) / len(values)
            delta = probes.delta_since_first(tag, value)
            if multi.is_master():
                prefix = (
                    f"dpo/concept/{tag}/probe_margin" if kind == str(MicroStepKind.DPO) else f"loss/concept/{tag}/probe"
                )
                self.tensorboard.add_scalar(prefix, value, step)
                if delta is not None:
                    self.tensorboard.add_scalar(f"{prefix}_delta", delta, step)
                self.tensorboard.add_scalar(f"{prefix}_n", float(len(samples)), step)
        if probes.dirty:
            # a changed set (new images or baselines) is persisted with the ledger
            self.__save_diagnostics_state()

    def __save_diagnostics_state(self, probes_only: bool = False):
        if self._diagnostics_dir is None or not multi.is_master():
            return
        try:
            if self._probe_set is not None and self._probe_set.dirty:
                self._probe_set.save(os.path.join(self._diagnostics_dir, PROBES_FILENAME))
            if not probes_only and self._visit_ledger is not None:
                self._visit_ledger.save(os.path.join(self._diagnostics_dir, LEDGER_FILENAME))
                self._diagnostics_saved_at = time.monotonic()
        except Exception:  # noqa: BLE001 - diagnostics must never take the run down
            traceback.print_exc()

    def __log_grad_diagnostics(self, train_progress: TrainProgress, scaler):
        """At the update step, after gradient reduction and before clipping:
        the per-objective norms / cosines, the per-concept window means and,
        on a conflict-probe window, the per-concept parameter-gradient norms
        and cosines against the rest. Resets everything for the next window."""
        scale = float(scaler.get_scale()) if scaler else 1.0
        if multi.is_master():
            step = train_progress.global_step
            for name, value in self._grad_attribution.scalars(scale).items():
                self.tensorboard.add_scalar(f"grad/{name}", value, step)
            for name, value in self._concept_stats.scalars().items():
                tag, quantity = name.split("/", 1)
                if quantity == "loss_tnorm":
                    self.tensorboard.add_scalar(f"loss/concept/{tag}/tnorm", value, step)
                else:
                    self.tensorboard.add_scalar(f"grad/concept/{tag}/{quantity}", value, step)
            if self._concept_attribution is not None and self._concept_split_window:
                for name, value in self._concept_attribution.rest_cosines(scale).items():
                    quantity, tag = (
                        name.split("_", 1) if name.startswith("norm_") else ("cos_rest", name[len("cos_rest_") :])
                    )
                    self.tensorboard.add_scalar(f"grad/concept/{tag}/{quantity}", value, step)
            if self._visit_ledger is not None:
                for name, value in self._visit_ledger.scalars().items():
                    tag, quantity = name.split("/", 1)
                    self.tensorboard.add_scalar(f"loss/concept/{tag}/{quantity}", value, step)
                if time.monotonic() - self._diagnostics_saved_at > 600:
                    self.__save_diagnostics_state()
        self._grad_attribution.reset()
        self._concept_stats.reset()
        if self._concept_attribution is not None:
            self._concept_attribution.reset()

    def __independent_micro_steps(self, schedule: IndependentSchedule, position: int):
        """The INDEPENDENT epoch driver: yields ``(kind, batch)`` from the
        schedule position on, pulling a standard batch from the main loader
        on SFT micro-steps (``batch`` is None for the other kinds; they pull
        their own batch when they run, so nothing is held across the
        sample/backup block at the top of the loop). The main loader's
        length is approximate (aspect buckets drop remainders), so it can run
        dry a few micro-steps early; the epoch ends there, exactly as it does
        under PIGGYBACK, and the last few scheduled add-on steps are dropped
        rather than clumped into a tail."""
        standard_batches = iter(self.data_loader.get_data_loader())
        for index, kind in enumerate(schedule.kinds_from(position), start=position):
            if kind is MicroStepKind.SFT:
                try:
                    batch = next(standard_batches)
                except StopIteration:
                    if multi.is_master():
                        print(
                            f"Independent schedule: the standard loader ran dry at micro-step {index} of "
                            f"{len(schedule)}; ending the epoch here."
                        )
                    return
                yield kind, batch
            else:
                yield kind, None

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.save_filename_prefix}{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f"Clearing cache directory {self.config.cache_dir}! "
            f"SmartCache validates files incrementally, so this is usually unnecessary. "
            f'Disable "Clear cache before training" to keep your validated cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith("epoch-") or filename in ["image", "text"]):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [
                    dirpath
                    for dirpath in os.listdir(backup_dirpath)
                    if os.path.isdir(os.path.join(backup_dirpath, dirpath))
                ],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
        self,
        train_progress: TrainProgress,
        train_device: torch.device,
        sample_config_list: list[SampleConfig],
        ema_applied: bool,
        folder_postfix: str = "",
        is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=not self.config.samples_to_tensorboard and not ema_applied,
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}",
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}",  # noqa: B023
                            pil_to_tensor(sampler_output.data),
                            train_progress.global_step,
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = (
                    self.callbacks.on_update_sample_custom_progress
                    if is_custom_sample
                    else self.callbacks.on_update_sample_default_progress
                )

                self.model.to(self.temp_device)
                self.model.eval()

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                print("Error during sampling, proceeding without sampling")

            torch_gc()

    def __sample_during_training(
        self,
        train_progress: TrainProgress,
        train_device: torch.device,
        sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("Sampling ...")

        is_custom_sample = False
        if sample_params_list:
            is_custom_sample = True
        elif self.config.samples is not None:
            sample_params_list = self.config.samples
        else:
            try:
                with open(self.config.sample_definition_file_name, "r") as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                    sample_params_list = samples
            # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
            except Exception:
                traceback.print_exc()
                print("Error during loading the sample definition file, proceeding without sampling")
                sample_params_list = []

        if self.model.ema:
            # the EMA model only exists in the master process, so EMA sampling is done on one GPU only
            # non-EMA sampling is done on all GPUs
            assert multi.is_master() and self.config.ema != EMAMode.OFF
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)
        if self.model.ema or self.config.optimizer.optimizer.is_schedule_free:
            self.__refresh_rotation_cache_after_param_swap()

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
            ema_applied=self.config.ema != EMAMode.OFF,
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)
            self.__refresh_rotation_cache_after_param_swap()

        # ema-less sampling, if ema is enabled:
        if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
                ema_applied=False,
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()
            self.__refresh_rotation_cache_after_param_swap()

        torch_gc()

    def __refresh_rotation_cache_after_param_swap(self):
        # EMA and schedule-free eval/train overwrite Q in place (param.data), which the
        # OFT rotation cache cannot see. Sampling only runs between optimizer updates,
        # so there are no pending rotation gradients; the spectral state keeps
        # tracking the training weights (warm=False).
        if self.config.oft_rotation_cache_applies():
            self.model_setup.refresh_adapters_after_param_restore(self.model, self.config, warm=False)

    def __validate_distill(self, train_progress: TrainProgress):
        # Teacher-match validation is self-contained (frozen teacher generates
        # its own references) and must never crash a training run.
        if not multi.is_master():
            return
        vae_restore_device = None
        try:
            self.callbacks.on_update_status("Distill validation")
            self.model_setup.setup_train_device(self.model, self.config)
            # Bring the VAE on-device for the validation window so teacher-match
            # LPIPS can decode (latent caching otherwise offloads it). Restored
            # in finally so training continues on the expected device layout.
            vae = getattr(self.model, "vae", None)
            if vae is not None:
                current_device = next(vae.parameters()).device
                if current_device != self.model_setup.train_device:
                    vae_restore_device = current_device
                    self.model.vae_to(self.model_setup.train_device)
            torch_gc()
            metrics = self.model_setup.run_distill_validation(self.model, self.config, train_progress)
            for key, value in (metrics or {}).items():
                self.tensorboard.add_scalar(f"distill_val/{key}", value, train_progress.global_step)
        except Exception as exception:  # noqa: BLE001 - validation must not abort training
            print(f"Distill validation skipped after error: {exception}")
        finally:
            if vae_restore_device is not None:
                self.model.vae_to(vae_restore_device)

    def __validate_transfer(self, train_progress: TrainProgress):
        # Teacher-match residual on cached batches; must never crash a run.
        if not multi.is_master():
            return
        try:
            self.callbacks.on_update_status("Transfer validation")
            self.model_setup.setup_train_device(self.model, self.config)
            torch_gc()
            metrics = self.model_setup.run_transfer_validation(self.model, self.config, train_progress)
            if not metrics:
                return
            self._transfer_val_failures = 0
            for key, value in metrics.items():
                self.tensorboard.add_scalar(f"transfer_val/{key}", value, train_progress.global_step)
            self.__check_transfer_convergence(metrics, train_progress)
        except Exception as exception:  # noqa: BLE001 - validation must not abort training
            self._transfer_val_failures += 1
            print(f"Transfer validation skipped after error: {exception}")
            if self._transfer_val_failures >= 3:
                print(
                    f"WARNING: transfer validation has failed {self._transfer_val_failures} times in a row -- "
                    "auto-stop and best-checkpoint tracking are NOT operating; the run will only end at the "
                    "epochs cap and the final save will use last-step parameters."
                )

    def __transfer_guard_violations(self, metrics: dict[str, float]) -> list[str]:
        """De-distillation guards tripped by this validation (transfer mode only;
        redistill's Phase A teacher IS the distilled variant)."""
        if not self.config.transfer_enabled or self.config.redistill_enabled:
            return []
        violations = []
        for key, minimum in (
            ("turbo_retention", self.config.transfer_min_turbo_retention),
            ("turbo_sharpness", self.config.transfer_min_turbo_sharpness),
        ):
            value = metrics.get(key)
            if minimum > 0.0 and value is not None and value < minimum:
                violations.append(f"{key} {value:.3f} < {minimum:g}")
        return violations

    def __check_transfer_convergence(self, metrics: dict[str, float], train_progress: TrainProgress):
        residual = metrics.get("residual")
        if residual is None:
            return
        violations = self.__transfer_guard_violations(metrics)
        guard_ok = not violations
        self.tensorboard.add_scalar("transfer/deturbo_guard", 0.0 if guard_ok else 1.0, train_progress.global_step)
        if violations:
            print(
                f"Transfer de-distillation guard tripped at step {train_progress.global_step} "
                f"({', '.join(violations)}): not a best-checkpoint candidate, counted against patience."
            )
        # Snapshot on every numeric improvement, BEFORE the criterion folds the
        # value into its best: the final save always uses the best-seen adapter.
        if guard_ok and residual < self._transfer_stop.best_residual:
            self._transfer_best_backup_path = self.__save_transfer_best(residual, train_progress)
        should_stop = self._transfer_stop.update(residual, tag=train_progress.global_step, guard_ok=guard_ok)
        self.tensorboard.add_scalar(
            "transfer/patience_counter", self._transfer_stop.counter, train_progress.global_step
        )
        self.tensorboard.add_scalar(
            "transfer/best_residual", self._transfer_stop.best_residual, train_progress.global_step
        )
        if should_stop:
            print(
                f"Transfer convergence reached at step {train_progress.global_step}. "
                f"Best residual {self._transfer_stop.best_residual:.6f} at step {self._transfer_stop.best_tag}."
            )
            if self.config.redistill_enabled:
                # Phase A converged early: advance to Phase B instead of stopping.
                self.__advance_redistill_phase(train_progress, restore_best=True)
            else:
                self.commands.stop()

    def __maybe_advance_redistill_phase(self, train_progress: TrainProgress):
        if not self.config.redistill_enabled or self._redistill_phase_b:
            return
        if self._redistill_resume_start_step is not None:
            # Resumed past the boundary: enter Phase B without a best-restore
            # (the resumed backup already carries the post-boundary weights).
            self.__advance_redistill_phase(
                train_progress, restore_best=False, start_step=self._redistill_resume_start_step
            )
        elif train_progress.global_step >= self.config.redistill_transfer_steps:
            self.__advance_redistill_phase(train_progress, restore_best=True)

    def __advance_redistill_phase(
        self, train_progress: TrainProgress, *, restore_best: bool, start_step: int | None = None
    ):
        if self._redistill_phase_b:
            return
        self._redistill_phase_b = True
        start_step = train_progress.global_step if start_step is None else start_step

        if restore_best and self._transfer_best_backup_path and os.path.isfile(self._transfer_best_backup_path):
            try:
                stop = getattr(self, "_transfer_stop", None)
                detail = (
                    f" (residual {stop.best_residual:.6f} at step {stop.best_tag})"
                    if stop is not None and stop.best_tag is not None
                    else ""
                )
                print(f"Redistill Phase A -> B at step {train_progress.global_step}: restoring transfer best{detail}.")
                best_state = torch.load(self._transfer_best_backup_path, map_location=self.temp_device)
                if self.config.oft_rotation_cache_applies():
                    # In-flight rotation gradients belong to the rotation being replaced.
                    self.model_setup.before_clip_adapters(self.model, self.config)
                for param, saved in zip(self.parameters, best_state, strict=True):
                    param.data.copy_(saved)
                del best_state
            except Exception:  # noqa: BLE001 - a corrupt snapshot must not abort the run
                traceback.print_exc()
                print("Could not restore the Phase A best snapshot; continuing with last-step parameters.")
            # The restored Q can be far from the tracked spectral state: re-warm it, and
            # refresh any rotation cache (also after a partial restore).
            self.model_setup.refresh_adapters_after_param_restore(self.model, self.config, warm=True)
        else:
            print(f"Redistill Phase A -> B at step {train_progress.global_step}.")

        # The Phase A teacher (~2-3 GB of late tensors) is dead weight in Phase B.
        if self.model is not None and getattr(self.model, "transfer_teacher_late_tensors", None) is not None:
            self.model.transfer_teacher_late_tensors = None
            self.model.transfer_base_recovery = None
            self.model.transfer_base_late_tensors = None
            torch_gc()

        # Phase B's fake-critic warmup and TTUR schedule count from the boundary.
        self.model_setup.set_redistill_phase_b_start(start_step)

        if self._redistill_resume_start_step is None:
            try:
                os.makedirs(os.path.dirname(self._redistill_phase_marker_path), exist_ok=True)
                with open(self._redistill_phase_marker_path, "w") as marker_file:
                    json.dump({"phase_b_start_step": start_step}, marker_file)
            except OSError:
                traceback.print_exc()
                print("Could not write the redistill phase marker; a resume would repeat Phase A.")

    def __save_transfer_best(self, residual: float, train_progress: TrainProgress) -> str:
        best_path = os.path.join(self.config.workspace_dir, "backup", "transfer-best.pt")
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        try:
            # Atomic: a crash mid-save must not leave a corrupt file that makes
            # end()'s restore fail after a whole night of training.
            state = [p.data.clone().cpu() for p in self.parameters]
            tmp_path = best_path + ".tmp"
            torch.save(state, tmp_path)
            os.replace(tmp_path, best_path)
            with open(best_path + ".json", "w") as meta_file:
                json.dump({"residual": residual, "step": train_progress.global_step}, meta_file)
            print(f"Saved transfer best checkpoint (residual improved) at step {train_progress.global_step}")
        except Exception:
            traceback.print_exc()
            print("Could not save transfer best checkpoint.")
            return self._transfer_best_backup_path or ""
        return best_path

    def __validate(self, train_progress: TrainProgress):
        if self.config.redistill_enabled:
            if not self.__needs_validate(train_progress):
                return
            if not self._redistill_phase_b:
                if self.config.transfer_validation:
                    self.__validate_transfer(train_progress)
            elif self.config.distill_validation:
                self.__validate_distill(train_progress)
            return
        if self.config.transfer_enabled:
            if self.config.transfer_validation and self.__needs_validate(train_progress):
                self.__validate_transfer(train_progress)
            return
        if self.config.distill_enabled and self.config.distill_validation:
            if self.__needs_validate(train_progress):
                self.__validate_distill(train_progress)
            return
        if not self.__needs_validate(train_progress):
            return

        # DPO validation (paired chosen/rejected samples) and standard
        # validation (plain validation loss) are different data pipelines, so
        # they never share a loader. A run can have one, the other, or both.
        if self.dpo_validation_data_loader is not None:
            dpo_loader, standard_loader = self.dpo_validation_data_loader, self.validation_data_loader
        elif self.config.dpo_validation_active():
            dpo_loader, standard_loader = self.validation_data_loader, None
        else:
            dpo_loader, standard_loader = None, self.validation_data_loader

        # Every loader advances its epoch before the model goes back on the
        # train device: start_next_epoch runs the caching pass, which parks the
        # model on the temp device whenever it has something left to encode.
        loaders = [loader for loader in (dpo_loader, standard_loader) if loader is not None]
        for loader in loaders:
            loader.get_data_set().start_next_epoch()
        if all(loader.get_data_set().approximate_length() == 0 for loader in loaders):
            return

        self.callbacks.on_update_status("Calculating validation loss")
        self.model_setup.setup_train_device(self.model, self.config)

        torch_gc()

        if dpo_loader is not None:
            self.__validate_dpo(dpo_loader, train_progress)
        if standard_loader is not None:
            self.__validate_standard(standard_loader, train_progress)

    def __validate_dpo(self, loader: BaseDataLoader, train_progress: TrainProgress):
        """Held-out ranking accuracy / reward margin on the DPO validation
        pairs; feeds the dpo/val_* scalars and DPO patience. The loader's
        epoch has already been started by __validate."""
        current_epoch_length_validation = loader.get_data_set().approximate_length()
        if current_epoch_length_validation == 0:
            return

        step_tqdm_validation = tqdm(
            loader.get_data_loader(),
            desc="dpo_validation_step",
            total=current_epoch_length_validation,
        )

        dpo_val_accuracy = []
        dpo_val_chosen_reward = []
        dpo_val_rejected_reward = []
        dpo_val_reward_margin = []

        for validation_batch in step_tqdm_validation:
            if self.__needs_gc(train_progress):
                torch_gc()

            with torch.no_grad():
                self.model_setup.calculate_dpo_loss(self.model, validation_batch, self.config, train_progress)
            dpo_metrics = self.model_setup.get_last_dpo_metrics()
            dpo_val_accuracy.append(dpo_metrics["accuracy"])
            dpo_val_chosen_reward.append(dpo_metrics["chosen_reward"])
            dpo_val_rejected_reward.append(dpo_metrics["rejected_reward"])
            dpo_val_reward_margin.append(dpo_metrics["reward_margin"])

        if dpo_val_accuracy:
            # Validation loss is intentionally not tracked: reward hacking
            # drives it toward zero, so a low val loss is not evidence of a
            # good model. Held-out ranking accuracy is hack-resistant.
            val_accuracy = sum(dpo_val_accuracy) / len(dpo_val_accuracy)
            val_chosen_reward = sum(dpo_val_chosen_reward) / len(dpo_val_chosen_reward)
            val_rejected_reward = sum(dpo_val_rejected_reward) / len(dpo_val_rejected_reward)
            val_reward_margin = sum(dpo_val_reward_margin) / len(dpo_val_reward_margin)

            self.tensorboard.add_scalar("dpo/val_accuracy", val_accuracy, train_progress.global_step)
            self.tensorboard.add_scalar("dpo/val_chosen_reward", val_chosen_reward, train_progress.global_step)
            self.tensorboard.add_scalar("dpo/val_rejected_reward", val_rejected_reward, train_progress.global_step)
            self.tensorboard.add_scalar("dpo/val_reward_margin", val_reward_margin, train_progress.global_step)
            self.__check_dpo_patience(val_accuracy, val_reward_margin, train_progress)

    def __validate_standard(self, loader: BaseDataLoader, train_progress: TrainProgress):
        """Plain validation loss per concept on the non-DPO VALIDATION
        concepts; feeds loss/validation_step/* and patience. The loader's
        epoch has already been started by __validate."""
        current_epoch_length_validation = loader.get_data_set().approximate_length()
        if current_epoch_length_validation == 0:
            return

        step_tqdm_validation = tqdm(
            loader.get_data_loader(),
            desc="validation_step",
            total=current_epoch_length_validation,
        )

        accumulated_loss_per_concept = {}
        concept_counts = {}
        mapping_seed_to_label = {}
        mapping_label_to_seed = {}

        for validation_batch in step_tqdm_validation:
            if self.__needs_gc(train_progress):
                torch_gc()

            with torch.no_grad():
                model_output_data = self.model_setup.predict(
                    self.model, validation_batch, self.config, train_progress, deterministic=True
                )
                loss_validation = self.model_setup.calculate_loss(
                    self.model, validation_batch, model_output_data, self.config
                )

            concept_name = validation_batch["concept_name"][0]
            concept_path = validation_batch["concept_path"][0]
            concept_seed = validation_batch["concept_seed"].item()
            loss = loss_validation.item()

            label = concept_name if concept_name else os.path.basename(concept_path)
            if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                suffix = 1
                new_label = f"{label}({suffix})"
                while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                    suffix += 1
                    new_label = f"{label}({suffix})"
                label = new_label

            if concept_seed not in mapping_seed_to_label:
                mapping_seed_to_label[concept_seed] = label
                mapping_label_to_seed[label] = concept_seed

            accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
            concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

        for concept_seed, total_loss in accumulated_loss_per_concept.items():
            average_loss = total_loss / concept_counts[concept_seed]
            label = mapping_seed_to_label[concept_seed]

            self.tensorboard.add_scalar(f"loss/validation_step/{label}", average_loss, train_progress.global_step)

        total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
        total_count = sum(concept_counts[key] for key in concept_counts)
        if total_count == 0:
            # approximate_length() estimated a non-empty validation set, but the
            # data loader yielded no usable batches (e.g. all samples filtered out
            # at runtime). Skip rather than crash the training run.
            print("Warning: validation produced no batches, skipping validation loss")
            return
        total_average_loss = total_loss / total_count

        if len(concept_counts) > 1:
            self.tensorboard.add_scalar(
                "loss/validation_step/total_average", total_average_loss, train_progress.global_step
            )

        if self.config.patience:
            self.__check_patience(total_average_loss, train_progress)

    def __check_patience(self, val_loss: float, train_progress: TrainProgress):
        self._patience_checks += 1
        if self._patience_checks <= self.config.patience_min_checks:
            # Warm-up window: neither the best checkpoint nor the no-improvement
            # counter is tracked yet, so an untrained early snapshot cannot win
            # on a noisy validation set and patience cannot fire before the
            # model has had a chance to learn anything.
            self.tensorboard.add_scalar("patience/counter", 0, train_progress.global_step)
            return

        is_new_best = val_loss < self._patience_best_loss
        stop_mode = self.config.patience_mode == PatienceMode.STOP_TRAINING

        if is_new_best:
            if self.config.patience_save_second_best and self._patience_best_backup_path:
                # The outgoing best is, by construction, at least as good as the
                # current runner-up: demote it instead of re-snapshotting.
                self.__promote_best_to_second()
            self._patience_best_backup_path = self.__save_patience_best(train_progress)
            self._patience_best_step = train_progress.global_step
            self._patience_best_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            if self.config.patience_save_second_best and val_loss < self._patience_second_loss:
                self._patience_second_backup_path = self.__save_patience_second(train_progress)
                self._patience_second_step = train_progress.global_step
                self._patience_second_loss = val_loss

        self.tensorboard.add_scalar("patience/counter", self._patience_counter, train_progress.global_step)
        self.tensorboard.add_scalar("patience/best_val_loss", self._patience_best_loss, train_progress.global_step)

        if stop_mode:
            if self._patience_counter >= self.config.patience_epochs:
                print(
                    f"Patience triggered at step {train_progress.global_step}. "
                    f"Best checkpoint from step {self._patience_best_step} "
                    f"(val_loss: {self._patience_best_loss:.6f})"
                )
                self.commands.stop()
        elif self._patience_counter == self.config.patience_epochs:
            # Preserve the threshold weights before end() restores the best checkpoint.
            # Both modes stop here; this mode also writes an alternative output.
            print(
                f"Patience threshold reached at step {train_progress.global_step} "
                f"(best val_loss {self._patience_best_loss:.6f} at step {self._patience_best_step}). "
                "Snapshotting alternative model and stopping training."
            )
            self._patience_alt_backup_path = self.__save_patience_alternative(train_progress)
            self._patience_alt_step = train_progress.global_step
            self.commands.stop()

    def __save_patience_extra_outputs(self, final_save_path: str):
        """Write the extra patience outputs next to the final model: the threshold
        weights ('-patience-alt', SAVE_ALTERNATIVE_AT_END mode) and the runner-up
        validation checkpoint ('-patience-second', when enabled)."""
        if not self.config.patience:
            return
        if self.config.patience_mode == PatienceMode.SAVE_ALTERNATIVE_AT_END and self._patience_alt_backup_path:
            self.__save_patience_snapshot_output(
                final_save_path,
                "-patience-alt",
                self._patience_alt_backup_path,
                f"patience alternative snapshot from step {self._patience_alt_step} "
                f"(best val_loss {self._patience_best_loss:.6f} at step {self._patience_best_step})",
            )
        if self.config.patience_save_second_best and self._patience_second_backup_path:
            self.__save_patience_snapshot_output(
                final_save_path,
                "-patience-second",
                self._patience_second_backup_path,
                f"second-best validation checkpoint from step {self._patience_second_step} "
                f"(val_loss {self._patience_second_loss:.6f}; best {self._patience_best_loss:.6f} "
                f"at step {self._patience_best_step})",
            )

    def __save_patience_snapshot_output(self, final_save_path: str, suffix: str, backup_path: str, label: str):
        if not os.path.isfile(backup_path):
            return
        if self.config.output_model_format.is_single_file():
            root, ext = os.path.splitext(final_save_path)
            save_path = f"{root}{suffix}{ext}"
        else:
            save_path = final_save_path.rstrip("/\\") + suffix

        try:
            print(f"Restoring {label}")
            self.callbacks.on_update_status(f"Saving the {suffix[1:]} model")
            state = torch.load(backup_path, map_location=self.temp_device)
            for param, saved in zip(self.parameters, state, strict=True):
                param.data.copy_(saved)
            del state

            print("Saving " + save_path)
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype(),
            )
        except Exception:  # noqa: BLE001 - an extra output must never take the final save down with it
            traceback.print_exc()
            print(f"Could not save the {suffix[1:]} model; the final model was saved normally.")

    def __promote_best_to_second(self):
        """The current best snapshot becomes the runner-up (a new best is about to replace it)."""
        second_path = os.path.join(self.config.workspace_dir, "backup", "patience-second.pt")
        try:
            os.replace(self._patience_best_backup_path, second_path)
        except OSError:
            traceback.print_exc()
            print("Could not keep the previous best checkpoint as the second-best snapshot.")
            return
        self._patience_second_backup_path = second_path
        self._patience_second_loss = self._patience_best_loss
        self._patience_second_step = self._patience_best_step

    def __save_patience_second(self, train_progress: TrainProgress) -> str:
        second_path = os.path.join(self.config.workspace_dir, "backup", "patience-second.pt")
        os.makedirs(os.path.dirname(second_path), exist_ok=True)
        try:
            state = [p.data.clone().cpu() for p in self.parameters]
            torch.save(state, second_path)
            print(f"Saved patience second-best checkpoint at step {train_progress.global_step}")
        except Exception:
            traceback.print_exc()
            print("Could not save patience second-best checkpoint.")
            return self._patience_second_backup_path or ""
        return second_path

    def __save_patience_alternative(self, train_progress: TrainProgress) -> str:
        alt_path = os.path.join(self.config.workspace_dir, "backup", "patience-alternative.pt")
        os.makedirs(os.path.dirname(alt_path), exist_ok=True)
        try:
            state = [p.data.clone().cpu() for p in self.parameters]
            torch.save(state, alt_path)
            print(f"Saved patience alternative snapshot at step {train_progress.global_step}")
        except Exception:
            traceback.print_exc()
            print("Could not save patience alternative snapshot.")
            return self._patience_alt_backup_path or ""
        return alt_path

    def __save_patience_best(self, train_progress: TrainProgress) -> str:
        best_path = os.path.join(self.config.workspace_dir, "backup", "patience-best.pt")
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        try:
            state = [p.data.clone().cpu() for p in self.parameters]
            torch.save(state, best_path)
            print(f"Saved patience best checkpoint (val_loss improved) at step {train_progress.global_step}")
        except Exception:
            traceback.print_exc()
            print("Could not save patience best checkpoint.")
            return self._patience_best_backup_path or ""
        return best_path

    def __check_dpo_patience(self, val_accuracy: float, val_reward_margin: float, train_progress: TrainProgress):
        rounded_accuracy = round(val_accuracy, 5)
        rounded_best_accuracy = round(self._dpo_best_accuracy, 5)

        is_new_best = rounded_accuracy > rounded_best_accuracy or (
            rounded_accuracy == rounded_best_accuracy and val_reward_margin > self._dpo_best_margin
        )

        if is_new_best and self.config.rlhf_dpo_save_best:
            self._dpo_best_backup_path = self.__save_dpo_best(val_accuracy, val_reward_margin, train_progress)

        if is_new_best:
            self._dpo_best_accuracy = val_accuracy
            self._dpo_best_margin = val_reward_margin

        if not self.config.rlhf_dpo_patience_enabled:
            return

        if is_new_best:
            self._dpo_patience_counter = 0
        else:
            self._dpo_patience_counter += 1

        self.tensorboard.add_scalar("dpo/patience_counter", self._dpo_patience_counter, train_progress.global_step)

        if self._dpo_patience_counter >= self.config.rlhf_dpo_patience_value:
            print(
                f"DPO early stopping triggered: patience exhausted after {self._dpo_patience_counter} "
                f"consecutive checks without improvement."
            )
            self.commands.stop()

    def __save_dpo_best(self, val_accuracy: float, val_reward_margin: float, train_progress: TrainProgress) -> str:
        best_path = os.path.join(self.config.workspace_dir, "backup", "dpo-best.pt")
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        try:
            state = [p.data.clone().cpu() for p in self.parameters]
            torch.save(state, best_path)
            print(
                f"Saved DPO best checkpoint (accuracy={val_accuracy:.4f}, "
                f"margin={val_reward_margin:.6f}) to {best_path}"
            )
        except Exception:
            traceback.print_exc()
            print("Could not save DPO best checkpoint.")
            return self._dpo_best_backup_path or ""
        return best_path

    def _collect_garbage_and_freeze(self):
        """Epoch-start full collection, then freeze the survivors.

        The data loaders' parsed cache indices (large dicts, alive for the whole
        epoch) move into the permanent generation, so the periodic torch_gc()
        calls stop re-traversing them. Unfreezing first keeps one full
        collection per epoch; end() unfreezes for later runs in this process.
        """
        gc.unfreeze()
        torch_gc()
        gc.freeze()

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def __backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self._stage_accumulator_state_for_save()
            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
            self.model_setup.save_dpo_reference(backup_path)
            if self._dpo_live_ledger is not None:
                self._dpo_live_ledger.flush(train_progress.global_step, force=True)
            self.__save_diagnostics_state()
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            self._clear_staged_accumulator_state()
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self._stage_accumulator_state_for_save()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype(),
            )
            self._clear_staged_accumulator_state()
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            self._clear_staged_accumulator_state()
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __log_vram(self, train_progress: TrainProgress, kind: MicroStepKind, batch) -> None:
        """Config log_vram: this micro-step's CUDA peaks (allocated, and reserved
        by the caching allocator, which is what nvidia-smi sees), what it was
        and the latent size it ran at, then reset the peaks for the next one.
        Host-side counters only, no sync. vram/probe marks the micro-steps that
        ran the concept conflict probe and fixed probes on top."""
        if not self.config.log_vram or not torch.cuda.is_available() or not multi.is_master():
            return
        gib = 1024**3
        step = train_progress.global_step
        shape = _latent_shape(batch) if batch is not None else self._vram_addon_shape
        self._vram_addon_shape = None
        self.tensorboard.add_scalar("vram/peak_allocated_gib", torch.cuda.max_memory_allocated() / gib, step)
        self.tensorboard.add_scalar("vram/peak_reserved_gib", torch.cuda.max_memory_reserved() / gib, step)
        self.tensorboard.add_scalar("vram/kind", KIND_ORDER.index(kind), step)
        if shape is not None:
            self.tensorboard.add_scalar("vram/latent_hw", shape[-2] * shape[-1], step)
            self.tensorboard.add_scalar("vram/batch", shape[0], step)
        if self._concept_split_window:
            self.tensorboard.add_scalar("vram/probe", 1, step)
        torch.cuda.reset_peak_memory_stats()

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    # ------------------------------------------------------------------
    # Fix B: accumulator-state staging for save, and restore on resume.
    # ------------------------------------------------------------------

    def _stage_accumulator_state_for_save(self):
        """Build the in-flight accumulator snapshot and attach it to the
        model so ``InternalModelSaverMixin._save_internal_data`` writes
        it out alongside the optimizer/EMA state.

        Captures:
          - accumulated_loss / accumulated_dpo_metrics from the loop locals
            mirrored on self at lines around the accumulator-reset block
          - per-trainable-parameter ``.grad`` tensors (CPU copies) keyed
            by the stable identifier from ``NamedParameterGroupCollection.iter_named_parameters``
          - scaler ``state_dict`` if AMP is in use
          - RNG state for torch / torch.cuda / python / numpy
          - fingerprint (acc_steps + dataset hash) for warn-only on resume

        Idempotent w.r.t. clean boundaries: at a fresh accumulator boundary
        the loop locals are zeroed and ``.grad`` is None, so the payload
        is an empty-by-construction snapshot -- correct to write and
        cheap to load.
        """
        if self.config.oft_rotation_cache_applies() and self.model is not None:
            # Accumulated rotation gradients are not persisted: land them in Q.grad,
            # which the snapshot below captures (a no-op between optimizer updates).
            self.model_setup.before_clip_adapters(self.model, self.config)

        if not multi.is_master():
            # The saver is master-only; non-master ranks have nothing to stage.
            self.model.accumulator_state = None
            return

        # Loop-local accumulators (mirrored from train()).
        if self._loop_accumulated_loss_tensor is not None and isinstance(
            self._loop_accumulated_loss_tensor, torch.Tensor
        ):
            try:
                acc_loss_f = float(self._loop_accumulated_loss_tensor.item())
            except Exception:
                acc_loss_f = float(self._loop_accumulated_loss)
        else:
            acc_loss_f = float(self._loop_accumulated_loss)

        # Per-parameter .grad CPU copies. Skip params with no grad to keep
        # the payload tight; an absent key on load means "no grad to restore".
        param_grads: dict[str, torch.Tensor] = {}
        if self.model is not None and self.model.parameters is not None:
            for key, p in self.model.parameters.iter_named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                param_grads[key] = p.grad.detach().to(device="cpu", copy=True)

        # Scaler state (None if AMP off).
        scaler_state = None
        if self._loop_scaler is not None:
            try:
                scaler_state = self._loop_scaler.state_dict()
            except Exception:
                scaler_state = None

        # RNG snapshots -- all four sources. CUDA list is None when CUDA
        # isn't available, e.g. CPU-only tests.
        rng: dict = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(legacy=True),
        }

        # Dataset fingerprint for warn-only on resume.
        fp_hash, fp_count = compute_concept_fingerprint(getattr(self.config, "concepts", None))
        fingerprint = {
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "dataset_hash": fp_hash,
            "concept_count": fp_count,
        }

        self.model.accumulator_state = {
            "accumulated_loss": acc_loss_f,
            "accumulated_dpo_metrics": (
                dict(self._loop_accumulated_dpo_metrics) if self._loop_accumulated_dpo_metrics is not None else None
            ),
            "param_grads": param_grads,
            "scaler": scaler_state,
            "rng": rng,
            "fingerprint": fingerprint,
        }

    def _clear_staged_accumulator_state(self):
        if self.model is not None:
            self.model.accumulator_state = None

    def _restore_accumulator_state(
        self,
        accumulated_loss: torch.Tensor,
        train_device: torch.device,
        scaler,
    ) -> tuple[torch.Tensor, dict | None, bool]:
        """Apply a previously-saved accumulator snapshot to the live trainer.

        Returns (accumulated_loss_tensor, accumulated_dpo_metrics, has_gradient).
        If no snapshot is attached to the model, returns the inputs unchanged
        and has_gradient=False. Mismatched fingerprints emit a warning but
        do NOT discard the snapshot -- losing in-flight gradient state from
        hours of training is worse than an off-spec effective batch.
        """
        if not multi.is_master():
            return accumulated_loss, None, False

        state = getattr(self.model, "accumulator_state", None)
        if state is None:
            return accumulated_loss, None, False

        # Fingerprint comparison (warn-only).
        fp = state.get("fingerprint", {})
        saved_acc = fp.get("gradient_accumulation_steps")
        if saved_acc is not None and saved_acc != self.config.gradient_accumulation_steps:
            print(
                f"Warning: gradient_accumulation_steps mismatch on resume: "
                f"saved={saved_acc} current={self.config.gradient_accumulation_steps}; "
                f"restoring partial accumulator state anyway (one off-spec optimizer step is "
                f"cheaper than discarding accumulated gradients)."
            )
        current_hash, current_count = compute_concept_fingerprint(getattr(self.config, "concepts", None))
        if fp.get("dataset_hash") and fp.get("dataset_hash") != current_hash:
            delta = current_count - int(fp.get("concept_count", current_count))
            print(
                f"Warning: dataset fingerprint mismatch on resume: "
                f"saved_concepts={fp.get('concept_count')} current_concepts={current_count} "
                f"(delta={delta}); restoring partial accumulator state anyway."
            )

        # Accumulators.
        acc_loss_f = float(state.get("accumulated_loss", 0.0) or 0.0)
        accumulated_loss = torch.tensor(acc_loss_f, device=train_device)
        accumulated_dpo_metrics = state.get("accumulated_dpo_metrics")
        if accumulated_dpo_metrics is not None:
            accumulated_dpo_metrics = dict(accumulated_dpo_metrics)

        # Per-parameter grads.
        saved_grads: dict = state.get("param_grads", {}) or {}
        if self.model is not None and self.model.parameters is not None:
            current_keys = {k for k, _ in self.model.parameters.iter_named_parameters()}
            missing = [k for k in saved_grads if k not in current_keys]
            if saved_grads and len(missing) / len(saved_grads) > 0.10:
                print(
                    f"Warning: {len(missing)} of {len(saved_grads)} saved grad keys are "
                    f"absent in the current model; skipping those grads."
                )
            applied = 0
            for key, p in self.model.parameters.iter_named_parameters():
                if not p.requires_grad:
                    continue
                if key in saved_grads:
                    g = saved_grads[key]
                    p.grad = g.to(device=p.device, dtype=p.dtype, non_blocking=True)
                    applied += 1
                else:
                    p.grad = None
            has_gradient = applied > 0
        else:
            has_gradient = False

        # Scaler.
        if scaler is not None and state.get("scaler") is not None:
            try:
                scaler.load_state_dict(state["scaler"])
            except Exception:
                # Backwards-compat: an older scaler state from a different
                # torch version isn't worth crashing over -- the next
                # optimizer step will re-establish a sane scale.
                print("Warning: could not restore GradScaler state; continuing with a fresh scaler.")

        # RNG -- restore all four; tolerate missing keys.
        rng = state.get("rng", {}) or {}
        if "torch_cpu" in rng and rng["torch_cpu"] is not None:
            torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            except Exception:
                pass
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            try:
                np.random.set_state(rng["numpy"])
            except Exception:
                pass

        # Consumed -- clear so a subsequent backup-without-stop doesn't
        # double-apply on a third session.
        self.model.accumulator_state = None
        return accumulated_loss, accumulated_dpo_metrics, has_gradient

    def __apply_fused_back_pass(self, scaler):
        fused_optimizer_step = (
            self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        )
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:

                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:

                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=__optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                __optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)

    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching")
                for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                    self.data_loader.get_data_set().start_next_epoch()
                    # The secondary loaders cache into the same cache dir; without this their concepts
                    # (DPO pairs, SFT anchors) are never cached and a sourceless run finds them empty.
                    for source in (self.dpo_addon_source, self.sft_anchor_source, self.kto_source):
                        if source is not None:
                            source.start_next_epoch()
                    # The validation loaders cache the same way __validate drives
                    # them (start_next_epoch, which runs their caching pass); a
                    # cache-only run would otherwise leave the VALIDATION concepts
                    # uncached and a sourceless run without validation data.
                    # Neither exists when validation is off.
                    for loader in (
                        getattr(self, "validation_data_loader", None),
                        getattr(self, "dpo_validation_data_loader", None),
                    ):
                        if loader is not None:
                            loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None
        # Mirror the scaler so the save-side staging can capture its state_dict.
        self._loop_scaler = scaler
        self.__setup_grad_diagnostics()

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        accumulated_dpo_metrics: dict[str, float] | None = None
        accumulated_kto_metrics: dict[str, float] | None = None
        accumulated_text_metrics: dict[str, float] | None = None
        # INDEPENDENT schedule only: the standard term on its own, so the
        # usual SFT curve survives windows that mix kinds (loss/sft_step).
        accumulated_sft_loss = 0.0
        accumulated_sft_count = 0
        # Periodic reference rebase counts from the resume step, so a resumed
        # run waits for the next interval boundary rather than rebasing at once.
        if self.config.rlhf_enabled and self.config.rlhf_dpo_ref_rebase_interval > 0:
            self._dpo_ref_rebase_bucket = train_progress.global_step // self.config.rlhf_dpo_ref_rebase_interval
        accumulated_distill_metrics: dict[str, float | str] | None = None
        accumulated_transfer_metrics: dict[str, float] | None = None
        ema_loss = None
        ema_loss_steps = 0
        # Mean of the (accumulation-averaged) training loss over the optimizer
        # steps of the current epoch; logged once per epoch when enabled.
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        def log_epoch_loss(min_steps: int = 1):
            """Log the epoch mean and reset. ``min_steps`` (batches) suppresses a
            near-empty partial epoch at an early stop, whose one-or-two-step mean
            would just be a noise spike at the end of the curve."""
            nonlocal epoch_loss_sum, epoch_loss_count
            steps_seen = epoch_loss_count * self.config.gradient_accumulation_steps
            if (
                self.config.tensorboard_epoch_loss
                and epoch_loss_count > 0
                and steps_seen >= min_steps
                and multi.is_master()
            ):
                self.tensorboard.add_scalar(
                    "loss/train_epoch", epoch_loss_sum / epoch_loss_count, train_progress.global_step
                )
            epoch_loss_sum, epoch_loss_count = 0.0, 0

        ema_reward_margin = None
        ema_reward_margin_steps = 0
        ema_chosen_reward = None
        ema_dpo_accuracy = None
        reward_hacking_streak = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        # Fix B: if a previous run was stopped mid-window and a backup was
        # written via end()'s backup_before_save path, restore the in-flight
        # gradient-accumulation state now. No-op for fresh runs and for
        # backups written at clean accumulation boundaries.
        accumulated_loss, restored_dpo_metrics, restored_has_grad = self._restore_accumulator_state(
            accumulated_loss, train_device, scaler
        )
        if restored_dpo_metrics is not None:
            accumulated_dpo_metrics = restored_dpo_metrics
        if restored_has_grad:
            has_gradient = True
        # Sync the mirrored tensor reference for the next save.
        self._loop_accumulated_loss_tensor = accumulated_loss
        self._loop_accumulated_loss = float(accumulated_loss.item()) if accumulated_loss is not None else 0.0
        self._loop_accumulated_dpo_metrics = accumulated_dpo_metrics

        for _epoch in tqdm(epochs, desc="epoch") if multi.is_master() else epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            try:
                for _ in multi.master_first():
                    # The secondary loaders (SFT-anchor / DPO add-on) cache
                    # too, and caching moves the model off the train device,
                    # so they must advance BEFORE setup_train_device() puts it
                    # back. Mid-epoch exhaustion (when the two sets differ in
                    # length) is handled by CyclingBatchSource restarting it.
                    if self.config.latent_caching:
                        self.data_loader.get_data_set().start_next_epoch()
                        self.__start_secondary_epochs()
                        self.model_setup.setup_train_device(self.model, self.config)
                    else:
                        self.model_setup.setup_train_device(self.model, self.config)
                        self.data_loader.get_data_set().start_next_epoch()
                        if (
                            self.sft_anchor_source is not None
                            or self.dpo_addon_source is not None
                            or self.kto_source is not None
                            or self.config.text_reward_enabled
                        ):
                            self.__start_secondary_epochs()
                            self.model_setup.setup_train_device(self.model, self.config)
            except CachingStoppedException:
                return

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            self._collect_garbage_and_freeze()

            # INDEPENDENT schedule: the epoch is the whole micro-step sequence
            # (standard + DPO + KTO + text reward), not just the main loader.
            schedule = self._independent_schedule
            if schedule is not None:
                current_epoch_length = len(schedule)
            else:
                current_epoch_length = self.data_loader.get_data_set().approximate_length()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=current_epoch_length,
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step,
                )

            if schedule is not None:
                micro_steps = self.__independent_micro_steps(schedule, schedule.clamp(train_progress.epoch_step))
            else:
                micro_steps = ((MicroStepKind.SFT, batch) for batch in self.data_loader.get_data_loader())
            if multi.is_master():
                batches = step_tqdm = tqdm(
                    micro_steps,
                    desc="step",
                    total=current_epoch_length,
                    initial=train_progress.epoch_step,
                )
            else:
                batches = micro_steps
            for kind, batch in batches:
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                if (
                    not self.commands.get_stop_command()
                    and self.__needs_sample(train_progress)
                    or self.commands.get_and_reset_sample_default_command()
                ):
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:

                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        # LoRA savers write only adapter/embedding tensors (copied to CPU one
                        # at a time), never base weights, so offloading the whole base model
                        # before a backup is pure stall. A save without a backup keeps the move;
                        # with both, __backup restores the train device before __save runs anyway.
                        if not (backup and self.config.training_method == TrainingMethod.LORA):
                            self.model.to(self.temp_device)
                        if backup:
                            self.__backup(train_progress, True, step_tqdm.write)
                        if save:
                            self.__save(train_progress, True, step_tqdm.write)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                with (
                    TorchMemoryRecorder(enabled=False, filename=f"memory-step{train_progress.global_step}.pickle"),
                    TorchProfiler(enabled=False, filename=f"profile-step{train_progress.global_step}.json"),
                ):
                    step_seed = train_progress.global_step
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)
                    if self.config.oft_rotation_cache_applies():
                        set_rotation_cache_seed(step_seed, train_device)

                    self.__maybe_advance_redistill_phase(train_progress)

                    # Gradient diagnostics: snapshot .grad before this
                    # micro-step's backward(s); each term closes its segment
                    # through __attribute. The standard path records its
                    # prediction here so the common backward below can probe it.
                    probe_output = None
                    if self._grad_attribution is not None:
                        self._grad_attribution.begin()
                        if self._concept_attribution is not None:
                            self._concept_attribution.begin()
                        self._concept_split_window = self.__concept_split_due(train_progress)

                    # Set on SFT + DPO add-on micro-steps: the standard loss is
                    # backwarded early (see below) and its detached value is
                    # folded into accumulated_loss alongside the DPO term.
                    addon_sft_loss = None
                    addon_dpo_loss = None
                    addon_kto_loss = None
                    addon_text_loss = None
                    # Set on DPO-primary micro-steps under rlhf_dpo_light_primary:
                    # every term was backwarded (scaled by 1 / GA) as it was
                    # computed and this holds their detached sum.
                    primary_light_loss = None
                    if kind is not MicroStepKind.SFT:
                        # INDEPENDENT schedule: this micro-step is one add-on
                        # term on its own, scaled exactly as when it rides on
                        # a standard step (weight / gradient accumulation) and
                        # backwarded inside its helper; the common backward
                        # below is skipped. One graph alive at a time.
                        if kind is MicroStepKind.DPO:
                            addon_dpo_loss = self.__dpo_addon_term(scaler, train_progress)
                            accumulated_dpo_metrics = accumulate_dpo_metrics(
                                accumulated_dpo_metrics, self.model_setup.get_last_dpo_metrics()
                            )
                            self._loop_accumulated_dpo_metrics = accumulated_dpo_metrics
                            self.__collect_dpo_pairs(train_progress)
                        elif kind is MicroStepKind.KTO:
                            addon_kto_loss = self.__kto_addon_term(scaler, train_progress)
                            accumulated_kto_metrics = accumulate_dpo_metrics(
                                accumulated_kto_metrics, self.model_setup.get_last_kto_metrics()
                            )
                        elif kind is MicroStepKind.TEXT:
                            addon_text_loss, text_metrics = self.__text_reward_term(scaler, train_progress)
                            accumulated_text_metrics = accumulate_dpo_metrics(accumulated_text_metrics, text_metrics)
                        else:
                            raise RuntimeError(f"unknown micro-step kind {kind}")
                        self.__attribute(self._grad_attribution, str(kind))
                    elif self.config.rlhf_dpo_primary():
                        sft_anchor_keys = ("sft_anchor_loss",) if self.sft_anchor_source is not None else ()
                        if self._kto_primary:
                            # Unpaired KTO run: the main loader yields Liked /
                            # Disliked singles and there are no pair metrics.
                            loss = self.model_setup.calculate_kto_loss(self.model, batch, self.config, train_progress)
                            accumulated_kto_metrics = accumulate_dpo_metrics(
                                accumulated_kto_metrics,
                                self.model_setup.get_last_kto_metrics(),
                                extra_keys=sft_anchor_keys,
                            )
                        else:
                            self.__capture_probe_samples(MicroStepKind.DPO, batch)
                            if self.config.rlhf_dpo_light_primary_applies():
                                # Light path: the pair's halves are forwarded one
                                # at a time and backwarded inside (same loss and
                                # gradient as the fused forward below, about half
                                # the peak activation memory); the common
                                # backward is skipped for this micro-step.
                                primary_light_loss = self.__dpo_primary_light_term(batch, scaler, train_progress)
                            else:
                                loss = self.model_setup.calculate_dpo_loss(
                                    self.model, batch, self.config, train_progress
                                )
                            # Accumulate per-micro-batch DPO metrics across the grad-accum window.
                            # Without this, only the final micro-batch's metric reaches TensorBoard —
                            # which produces 0.0/1.0 accuracy when batch_size=1 regardless of effective batch.
                            accumulated_dpo_metrics = accumulate_dpo_metrics(
                                accumulated_dpo_metrics,
                                self.model_setup.get_last_dpo_metrics(),
                                extra_keys=sft_anchor_keys,
                            )
                            self.__collect_dpo_pairs(train_progress)

                            if self.kto_source is not None:
                                # KTO singles ride on every pair micro-batch at
                                # full weight: preference data like the pairs.
                                kto_batch = self.kto_source.next()
                                kto_loss = self.model_setup.calculate_kto_loss(
                                    self.model, kto_batch, self.config, train_progress
                                )
                                if primary_light_loss is not None:
                                    primary_light_loss = primary_light_loss + self.__dpo_primary_extra_term(
                                        kto_loss, scaler
                                    )
                                else:
                                    loss = loss + kto_loss
                                accumulated_kto_metrics = accumulate_dpo_metrics(
                                    accumulated_kto_metrics, self.model_setup.get_last_kto_metrics()
                                )
                                del kto_loss

                        if self.sft_anchor_source is not None:
                            sft_batch = self.sft_anchor_source.next()
                            sft_output = self.model_setup.predict(self.model, sft_batch, self.config, train_progress)
                            sft_loss = self.model_setup.calculate_loss(self.model, sft_batch, sft_output, self.config)
                            if primary_light_loss is not None:
                                primary_light_loss = primary_light_loss + self.__dpo_primary_extra_term(
                                    self.config.rlhf_sft_anchor_weight * sft_loss, scaler
                                )
                                del sft_output
                            else:
                                loss = loss + self.config.rlhf_sft_anchor_weight * sft_loss
                            anchor_metrics = accumulated_kto_metrics if self._kto_primary else accumulated_dpo_metrics
                            anchor_metrics["sft_anchor_loss"] += sft_loss.detach().item()
                        # Mirror for Fix B save-side staging (after the SFT-anchor
                        # block so the mirror reflects the final dpo metrics dict).
                        self._loop_accumulated_dpo_metrics = accumulated_dpo_metrics
                    elif self.config.distill_enabled or (self.config.redistill_enabled and self._redistill_phase_b):
                        loss = self.model_setup.calculate_distill_loss(self.model, batch, self.config, train_progress)
                        self.model_setup.capture_distill_val_batch(batch, self.config)
                        micro_distill_metrics = self.model_setup.get_last_distill_metrics()
                        if accumulated_distill_metrics is None:
                            accumulated_distill_metrics = {
                                "mode": micro_distill_metrics.get("mode", ""),
                                "_count": 0,
                            }
                            for key, value in micro_distill_metrics.items():
                                if isinstance(value, int | float):
                                    accumulated_distill_metrics[key] = 0.0
                        accumulated_distill_metrics["mode"] = micro_distill_metrics.get("mode", "")
                        for key, value in micro_distill_metrics.items():
                            if isinstance(value, int | float):
                                accumulated_distill_metrics[key] += value
                        accumulated_distill_metrics["_count"] += 1
                    elif self.config.transfer_enabled or self.config.redistill_enabled:
                        if self.config.redistill_enabled:
                            loss = self.model_setup.calculate_redistill_transfer_loss(
                                self.model, batch, self.config, train_progress
                            )
                        else:
                            loss = self.model_setup.calculate_transfer_loss(
                                self.model, batch, self.config, train_progress
                            )
                        self.model_setup.capture_transfer_val_batch(batch, self.config)
                        micro_transfer_metrics = self.model_setup.get_last_transfer_metrics()
                        if accumulated_transfer_metrics is None:
                            accumulated_transfer_metrics = {"_count": 0}
                            for key, value in micro_transfer_metrics.items():
                                if isinstance(value, int | float):
                                    accumulated_transfer_metrics[key] = 0.0
                        for key, value in micro_transfer_metrics.items():
                            if isinstance(value, int | float):
                                accumulated_transfer_metrics[key] += value
                        accumulated_transfer_metrics["_count"] += 1
                    else:
                        # Standard training path
                        prior_pred_indices = [
                            i
                            for i in range(self.config.batch_size)
                            if ConceptType(batch["concept_type"][i]) == ConceptType.PRIOR_PREDICTION
                        ]
                        distillation_enabled = bool(self.config.distillation_teacher_lora_model_name)

                        if distillation_enabled and len(prior_pred_indices) > 0:
                            raise RuntimeError("Distillation and PRIOR_PREDICTION concepts are mutually exclusive.")

                        if distillation_enabled:
                            with self.model_setup.distillation_teacher_model(self.model, self.config), torch.no_grad():
                                teacher_output_data = self.model_setup.predict(
                                    self.model, batch, self.config, train_progress
                                )
                            base_output_data = None
                            if self.config.distillation_base_anchor_weight > 0:
                                with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                                    base_output_data = self.model_setup.predict(
                                        self.model, batch, self.config, train_progress
                                    )
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            # Replace the per-sample noise target with the teacher's
                            # epsilon prediction. Both forwards see identical noisy
                            # latents and timesteps because predict() seeds RNG from
                            # train_progress.global_step.
                            model_output_data["target"] = teacher_output_data["predicted"].to(
                                dtype=model_output_data["target"].dtype
                            )
                            loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)
                            if base_output_data is not None:
                                anchor_loss = torch.nn.functional.mse_loss(
                                    model_output_data["predicted"].float(),
                                    base_output_data["predicted"]
                                    .to(dtype=model_output_data["predicted"].dtype)
                                    .float(),
                                )
                                loss = loss + self.config.distillation_base_anchor_weight * anchor_loss
                        elif len(prior_pred_indices) > 0 or (
                            self.config.masked_training
                            and self.config.masked_prior_preservation_weight > 0
                            and self.config.training_method == TrainingMethod.LORA
                        ):
                            with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                                # do NOT create a subbatch using the indices, even though it would be more efficient:
                                # different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                                prior_model_output_data = self.model_setup.predict(
                                    self.model, batch, self.config, train_progress
                                )
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            prior_model_prediction = prior_model_output_data["predicted"].to(
                                dtype=model_output_data["target"].dtype
                            )
                            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                            model_output_data["prior_target"] = prior_model_prediction
                            loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)
                        else:
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)
                        probe_output = model_output_data
                        self.__capture_probe_samples(MicroStepKind.SFT, batch)

                        if schedule is not None:
                            # INDEPENDENT: the add-ons have micro-steps of their
                            # own; this standard step carries nothing else.
                            dpo_step = kto_step = text_step = False
                        else:
                            dpo_step = self.dpo_addon_source is not None and is_dpo_addon_step(
                                train_progress.epoch_step, self._dpo_addon_interval
                            )
                            kto_step = self.kto_source is not None and is_dpo_addon_step(
                                train_progress.epoch_step, self._kto_addon_interval
                            )
                            # The text reward never shares a micro-batch with a
                            # DPO pair or a KTO batch (peak memory); a blocked
                            # firing carries over.
                            text_step = self._text_reward_bank is not None and self._text_reward_scheduler.due(
                                train_progress.epoch_step, blocked=dpo_step or kto_step
                            )
                        if dpo_step or kto_step or text_step:
                            # Add-on micro-batch: the standard loss above stays
                            # at full weight; a DPO pair or a text-reward
                            # rollout rides along. Backward the standard loss
                            # FIRST so its graph is freed before the add-on's
                            # forwards: holding both graphs for one combined
                            # backward OOMs a 24 GB card. Gradients accumulate
                            # identically (d(a+b) = da + db), and fused_back_pass
                            # is not supported in this mode (two backwards per
                            # micro-step).
                            addon_sft_loss = loss / self.config.gradient_accumulation_steps
                            # On a text step, keep the standard loss's gradient
                            # on the velocity output: the text term reports its
                            # own on the same tensor, so the two norms compare
                            # the actual push each term lands per micro-step.
                            sft_predicted = model_output_data.get("predicted") if text_step else None
                            if isinstance(sft_predicted, torch.Tensor) and sft_predicted.requires_grad:
                                sft_predicted.retain_grad()
                            else:
                                sft_predicted = None
                            self.__backward_term(
                                addon_sft_loss,
                                scaler,
                                kind=MicroStepKind.SFT,
                                predicted=model_output_data.get("predicted"),
                                image_paths=batch.get("image_path"),
                                timestep=model_output_data.get("timestep"),
                                losses=self.model_setup.get_last_per_sample_losses(),
                                scale=1.0 / self.config.gradient_accumulation_steps,
                            )
                            self.__attribute(self._grad_attribution, str(MicroStepKind.SFT))
                            addon_sft_loss = addon_sft_loss.detach()
                            sft_grad_norm_v = None
                            if sft_predicted is not None and sft_predicted.grad is not None:
                                sft_grad_norm_v = float(sft_predicted.grad.detach().float().norm())
                            del loss, model_output_data, sft_predicted

                        if text_step:
                            addon_text_loss, text_metrics = self.__text_reward_term(scaler, train_progress)
                            if sft_grad_norm_v is not None:
                                text_metrics["sft_grad_norm_v"] = sft_grad_norm_v
                            accumulated_text_metrics = accumulate_dpo_metrics(accumulated_text_metrics, text_metrics)
                            self.__attribute(self._grad_attribution, str(MicroStepKind.TEXT))

                        if dpo_step:
                            # The pair itself is evaluated half by half with one
                            # graph alive at a time (calculate_dpo_loss_light);
                            # its gradients land through the helper's backward
                            # callback, so the common backward below is skipped
                            # on this step.
                            addon_dpo_loss = self.__dpo_addon_term(scaler, train_progress)
                            accumulated_dpo_metrics = accumulate_dpo_metrics(
                                accumulated_dpo_metrics, self.model_setup.get_last_dpo_metrics()
                            )
                            self._loop_accumulated_dpo_metrics = accumulated_dpo_metrics
                            self.__collect_dpo_pairs(train_progress)
                            self.__attribute(self._grad_attribution, str(MicroStepKind.DPO))

                        if kto_step:
                            addon_kto_loss = self.__kto_addon_term(scaler, train_progress)
                            accumulated_kto_metrics = accumulate_dpo_metrics(
                                accumulated_kto_metrics, self.model_setup.get_last_kto_metrics()
                            )
                            self.__attribute(self._grad_attribution, str(MicroStepKind.KTO))

                    if kind is not MicroStepKind.SFT:
                        # INDEPENDENT add-on micro-step: its one term was
                        # already backwarded above and is the whole step.
                        detached_loss = next(
                            term for term in (addon_dpo_loss, addon_kto_loss, addon_text_loss) if term is not None
                        )
                    elif primary_light_loss is not None:
                        # DPO-primary micro-step on the light path: the pair's
                        # halves, any KTO batch and the SFT anchor were each
                        # backwarded above with the 1 / GA scaling the fused
                        # path applies below, so only the attribution and the
                        # detached sum remain.
                        self.__attribute(self._grad_attribution, str(MicroStepKind.DPO))
                        detached_loss = primary_light_loss
                    elif addon_sft_loss is None:
                        loss = loss / self.config.gradient_accumulation_steps
                        # A DPO-primary micro-step (pairs, with the SFT anchor
                        # and any KTO batch folded in) is attributed to DPO as
                        # a whole; a standard one is probed per sample.
                        self.__backward_term(
                            loss,
                            scaler,
                            kind=MicroStepKind.DPO if self.config.rlhf_dpo_primary() else MicroStepKind.SFT,
                            predicted=probe_output.get("predicted") if probe_output is not None else None,
                            image_paths=batch.get("image_path") if probe_output is not None else None,
                            timestep=probe_output.get("timestep") if probe_output is not None else None,
                            losses=self.model_setup.get_last_per_sample_losses() if probe_output is not None else None,
                            scale=1.0 / self.config.gradient_accumulation_steps,
                        )
                        self.__attribute(
                            self._grad_attribution,
                            str(MicroStepKind.DPO if self.config.rlhf_dpo_primary() else MicroStepKind.SFT),
                        )
                        detached_loss = loss.detach()
                        if schedule is not None:
                            accumulated_sft_loss += detached_loss
                            accumulated_sft_count += 1
                    else:
                        # Add-on micro-step: every term was already backwarded
                        # (standard first, then the DPO pair and/or the text
                        # reward rollout).
                        detached_loss = addon_sft_loss
                        if addon_dpo_loss is not None:
                            detached_loss = detached_loss + addon_dpo_loss
                        if addon_kto_loss is not None:
                            detached_loss = detached_loss + addon_kto_loss
                        if addon_text_loss is not None:
                            detached_loss = detached_loss + addon_text_loss

                    has_gradient = True
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss
                    # Mirror the live accumulator tensor reference for the
                    # save-side staging (Fix B). No host sync here -- the
                    # .item() only happens at save time, which is rare.
                    self._loop_accumulated_loss_tensor = accumulated_loss

                    if self.__is_update_step(train_progress):
                        if self.config.fused_gradient_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        if self.config.oft_rotation_cache_applies():
                            # The rotation cache's accumulated dL/dR becomes Q.grad here, so
                            # unscale, clipping and the step all see the full gradient.
                            self.model_setup.before_clip_adapters(self.model, self.config)

                        if self._grad_attribution is not None:
                            # After reduction and the cache materialisation, before
                            # clipping: the window's gradient as the optimizer sees it.
                            self.__log_grad_diagnostics(train_progress, scaler)

                        # Pre-clip total gradient norm, logged below with the step's loss.
                        grad_norm = None
                        if (
                            scaler
                            and self.config.optimizer.optimizer.supports_fused_back_pass()
                            and self.config.optimizer.fused_back_pass
                        ):
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        # Periodic DPO reference rebase, applied right after the
                        # optimizer step so the reference never changes inside a
                        # gradient-accumulation window.
                        if self.config.rlhf_enabled and self.config.rlhf_dpo_ref_rebase_interval > 0:
                            rebase_due, self._dpo_ref_rebase_bucket = dpo_ref_rebase_due(
                                train_progress.global_step,
                                self.config.rlhf_dpo_ref_rebase_interval,
                                self._dpo_ref_rebase_bucket,
                            )
                            if rebase_due:
                                self.model_setup.rebase_dpo_reference(self.model, self.config)
                                print(
                                    f"DPO reference rebased to the current adapter at step {train_progress.global_step}."
                                )

                        # Adaptive beta must be applied on EVERY rank and driven by
                        # the global margin: otherwise non-master ranks keep the
                        # static beta and the averaged gradients mix losses at
                        # different beta scales. All ranks all-reduce the margin and
                        # run the same deterministic controller, so they stay in
                        # lockstep without a broadcast. reduce_tensor_sum is a
                        # collective, so this must sit outside the is_master() block.
                        # The margin is the mean over the samples whose concept takes
                        # part in adaptive beta (see controller_margin_terms); a step
                        # without such samples leaves the controller and the runtime
                        # beta where they were. The per-concept controllers ride in
                        # the same tensor (fixed key order, see
                        # __setup_adaptive_beta_concepts) and each updates only on a
                        # step with a sample of its own.
                        adaptive_beta = None
                        updated_ratio_keys: list[str] = []
                        if (
                            self.config.rlhf_enabled
                            and accumulated_dpo_metrics is not None
                            and self.config.rlhf_dpo_adaptive_beta
                            and self.config.rlhf_dpo_objective == DPOObjective.SIGMOID
                        ):
                            margin_terms = torch.tensor(
                                [
                                    *controller_margin_terms(accumulated_dpo_metrics),
                                    *controller_concept_terms(accumulated_dpo_metrics, self._dpo_adaptive_keys),
                                ],
                                device=train_device,
                            )
                            multi.reduce_tensor_sum(margin_terms)
                            margin_sum, included, *concept_terms = margin_terms.tolist()
                            if self._dpo_beta_controller is None:
                                self._dpo_beta_controller = DPOBetaController(self.config.rlhf_dpo_beta)
                            if included > 0:
                                adaptive_beta = self._dpo_beta_controller.update(margin_sum / included)
                                self.model_setup.set_dpo_runtime_beta(adaptive_beta)
                            for key, concept_sum, concept_count in zip(
                                self._dpo_adaptive_keys, concept_terms[::2], concept_terms[1::2], strict=True
                            ):
                                if concept_count <= 0:
                                    continue
                                controller = self._dpo_beta_controllers.get(key)
                                if controller is None:
                                    # beta0 = 1: the output is the ratio itself.
                                    controller = self._dpo_beta_controllers[key] = DPOBetaController(1.0)
                                self._dpo_beta_ratios[key] = controller.update(concept_sum / concept_count)
                                updated_ratio_keys.append(key)
                            if updated_ratio_keys:
                                self.model_setup.set_dpo_runtime_beta_ratios(self._dpo_beta_ratios)

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            accumulated_loss_cpu = accumulated_loss.item()
                            if math.isnan(accumulated_loss_cpu):
                                raise RuntimeError(
                                    "Training loss became NaN. This may be due to invalid parameters, precision issues, or a bug in the loss computation."
                                )

                            self.tensorboard.add_scalar(
                                "loss/train_step", accumulated_loss_cpu, train_progress.global_step
                            )
                            if accumulated_sft_count > 0:
                                # Mean standard loss over this window's standard
                                # micro-steps (the /GA scaling undone).
                                self.tensorboard.add_scalar(
                                    "loss/sft_step",
                                    float(accumulated_sft_loss)
                                    * self.config.gradient_accumulation_steps
                                    / accumulated_sft_count,
                                    train_progress.global_step,
                                )
                            if grad_norm is not None:
                                grad_norm_cpu = float(grad_norm)
                                self.tensorboard.add_scalar("grad/norm", grad_norm_cpu, train_progress.global_step)
                                self.tensorboard.add_scalar(
                                    "grad/clipped",
                                    float(grad_norm_cpu > self.config.clip_grad_norm),
                                    train_progress.global_step,
                                )
                            epoch_loss_sum += accumulated_loss_cpu
                            epoch_loss_count += 1
                            if self.config.rlhf_enabled and accumulated_dpo_metrics is not None:
                                count = accumulated_dpo_metrics.pop("_count")
                                dpo_metrics = {k: v / count for k, v in accumulated_dpo_metrics.items()}
                                self.tensorboard.add_scalar("loss/dpo", dpo_metrics["loss"], train_progress.global_step)
                                self.tensorboard.add_scalar(
                                    "dpo/raw_loss", dpo_metrics["dpo_loss"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/chosen_reward", dpo_metrics["chosen_reward"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/rejected_reward", dpo_metrics["rejected_reward"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/reward_margin", dpo_metrics["reward_margin"], train_progress.global_step
                                )
                                ema_reward_margin = ema_reward_margin or dpo_metrics["reward_margin"]
                                ema_reward_margin_steps += 1
                                ema_reward_margin_decay = min(0.99, 1 - (1 / ema_reward_margin_steps))
                                ema_reward_margin = (ema_reward_margin * ema_reward_margin_decay) + (
                                    dpo_metrics["reward_margin"] * (1 - ema_reward_margin_decay)
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/smooth_reward_margin", ema_reward_margin, train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/accuracy", dpo_metrics["accuracy"], train_progress.global_step
                                )
                                if "sft_anchor_loss" in dpo_metrics:
                                    self.tensorboard.add_scalar(
                                        "dpo/sft_anchor_loss",
                                        dpo_metrics["sft_anchor_loss"],
                                        train_progress.global_step,
                                    )
                                # Reward-hacking signature: the margin keeps growing while
                                # BOTH rewards go negative (the model degrades chosen and
                                # rejected alike) and held-out ranking saturates.
                                ema_chosen_reward = ema_chosen_reward or dpo_metrics["chosen_reward"]
                                ema_chosen_reward = (ema_chosen_reward * ema_reward_margin_decay) + (
                                    dpo_metrics["chosen_reward"] * (1 - ema_reward_margin_decay)
                                )
                                ema_dpo_accuracy = ema_dpo_accuracy or dpo_metrics["accuracy"]
                                ema_dpo_accuracy = (ema_dpo_accuracy * ema_reward_margin_decay) + (
                                    dpo_metrics["accuracy"] * (1 - ema_reward_margin_decay)
                                )
                                if ema_chosen_reward < 0 and ema_dpo_accuracy > 0.95:
                                    reward_hacking_streak += 1
                                else:
                                    reward_hacking_streak = 0
                                if reward_hacking_streak == 25:
                                    warning = (
                                        "DPO reward-hacking signature detected: chosen reward has stayed "
                                        "negative while training accuracy is saturated. The margin is likely "
                                        "growing by degrading both images of each pair. Lower the learning "
                                        "rate, raise beta, or switch to the IPO objective."
                                    )
                                    print(warning)
                                    self.tensorboard.add_text("dpo/warnings", warning, train_progress.global_step)

                                if adaptive_beta is not None:
                                    # Controller already ran on every rank above; here we
                                    # only log the (globally-consistent) beta it produced.
                                    # With per-concept controllers the pooled beta applies
                                    # to no sample of a keyed concept, so dpo/beta follows
                                    # the batch-mean beta actually applied instead.
                                    logged_beta = adaptive_beta
                                    if self._dpo_adaptive_keys and "beta_mean" in dpo_metrics:
                                        logged_beta = dpo_metrics["beta_mean"]
                                    self.tensorboard.add_scalar("dpo/beta", logged_beta, train_progress.global_step)
                                    self.tensorboard.add_scalar(
                                        "dpo/raw_margin_ema",
                                        self._dpo_beta_controller.margin_ema,
                                        train_progress.global_step,
                                    )
                                for key in updated_ratio_keys:
                                    self.tensorboard.add_scalar(
                                        f"dpo/concept/{key}/adaptive_ratio",
                                        self._dpo_beta_ratios[key],
                                        train_progress.global_step,
                                    )
                                if "beta_mean" in dpo_metrics:
                                    # Present only when concepts set per-concept beta
                                    # overrides. dpo/beta above tracks the global scalar;
                                    # this is the batch-mean beta actually applied, after
                                    # the adaptive controller's ratio.
                                    self.tensorboard.add_scalar(
                                        "dpo/beta_mean", dpo_metrics["beta_mean"], train_progress.global_step
                                    )
                                if "chosen_anchor_loss" in dpo_metrics:
                                    # Present only with the DPOP anchor on (rlhf_dpo_chosen_anchor
                                    # > 0, or a concept's dpo_chosen_anchor_override > 0).
                                    self.tensorboard.add_scalar(
                                        "dpo/chosen_anchor_loss",
                                        dpo_metrics["chosen_anchor_loss"],
                                        train_progress.global_step,
                                    )
                                    self.tensorboard.add_scalar(
                                        "dpo/chosen_below_ref",
                                        dpo_metrics["chosen_below_ref"],
                                        train_progress.global_step,
                                    )

                                if self.config.rlhf_dpo_timestep_margin_logging:
                                    for quartile in range(4):
                                        quartile_count = dpo_metrics.get(f"margin_t_q{quartile + 1}_count", 0.0)
                                        if quartile_count > 0:
                                            self.tensorboard.add_scalar(
                                                f"dpo/margin_by_t/q{quartile + 1}",
                                                dpo_metrics[f"margin_t_q{quartile + 1}_sum"] / quartile_count,
                                                train_progress.global_step,
                                            )
                                if self._dpo_concept_window is not None:
                                    for tag, value in self._dpo_concept_window.scalars().items():
                                        self.tensorboard.add_scalar(tag, value, train_progress.global_step)
                                if self._dpo_live_ledger is not None:
                                    self._dpo_live_ledger.flush(train_progress.global_step)
                            if accumulated_kto_metrics is not None:
                                kto_count = accumulated_kto_metrics.pop("_count")
                                self.tensorboard.add_scalar(
                                    "loss/kto", accumulated_kto_metrics["loss"] / kto_count, train_progress.global_step
                                )
                                for label in ("liked", "disliked"):
                                    label_count = accumulated_kto_metrics.get(f"{label}_count", 0.0)
                                    if label_count > 0:
                                        self.tensorboard.add_scalar(
                                            f"kto/{label}_reward",
                                            accumulated_kto_metrics[f"{label}_reward_sum"] / label_count,
                                            train_progress.global_step,
                                        )
                                self.tensorboard.add_scalar(
                                    "kto/accuracy",
                                    accumulated_kto_metrics["accuracy"] / kto_count,
                                    train_progress.global_step,
                                )
                                if "beta_mean" in accumulated_kto_metrics:
                                    self.tensorboard.add_scalar(
                                        "kto/beta_mean",
                                        accumulated_kto_metrics["beta_mean"] / kto_count,
                                        train_progress.global_step,
                                    )
                                if "sft_anchor_loss" in accumulated_kto_metrics:
                                    self.tensorboard.add_scalar(
                                        "kto/sft_anchor_loss",
                                        accumulated_kto_metrics["sft_anchor_loss"] / kto_count,
                                        train_progress.global_step,
                                    )
                            if accumulated_text_metrics is not None:
                                text_count = accumulated_text_metrics.pop("_count")
                                self.tensorboard.add_scalar(
                                    "loss/text_reward",
                                    accumulated_text_metrics.pop("ctc") / text_count * self.config.text_reward_weight,
                                    train_progress.global_step,
                                )
                                for key, value in accumulated_text_metrics.items():
                                    if key.startswith("family_ratio/"):
                                        # Summed only over the samples that drew
                                        # this family; the per-family count
                                        # rides along as its own key.
                                        continue
                                    if key.startswith("family_count/"):
                                        family = key.removeprefix("family_count/")
                                        if value > 0:
                                            self.tensorboard.add_scalar(
                                                f"text_reward/family/{family}",
                                                accumulated_text_metrics.get(f"family_ratio/{family}", 0.0) / value,
                                                train_progress.global_step,
                                            )
                                        continue
                                    self.tensorboard.add_scalar(
                                        f"text_reward/{key}", value / text_count, train_progress.global_step
                                    )
                                accumulated_text_metrics = None
                            pop_perceptual_metrics = getattr(self.model_setup, "pop_perceptual_metrics", None)
                            if pop_perceptual_metrics is not None:
                                # Averages over this optimiser step's micro-steps;
                                # empty unless the perceptual term ran.
                                for key, value in pop_perceptual_metrics().items():
                                    self.tensorboard.add_scalar(f"perceptual/{key}", value, train_progress.global_step)
                            if self.config.redistill_enabled:
                                self.tensorboard.add_scalar(
                                    "redistill/phase",
                                    1.0 if self._redistill_phase_b else 0.0,
                                    train_progress.global_step,
                                )
                            if accumulated_distill_metrics is not None:
                                count = accumulated_distill_metrics.pop("_count")
                                mode = accumulated_distill_metrics.pop("mode", "")
                                distill_metrics = {
                                    k: v / count
                                    for k, v in accumulated_distill_metrics.items()
                                    if isinstance(v, int | float)
                                }
                                self.tensorboard.add_scalar(
                                    "loss/distill", distill_metrics["loss"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "distill/fake_loss", distill_metrics["fake_loss"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "distill/kl_loss", distill_metrics["kl_loss"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "distill/endpoint_loss",
                                    distill_metrics["endpoint_loss"],
                                    train_progress.global_step,
                                )
                                self.tensorboard.add_scalar(
                                    "distill/mode_student",
                                    1.0 if mode == "student" else 0.0,
                                    train_progress.global_step,
                                )
                            if accumulated_transfer_metrics is not None:
                                count = accumulated_transfer_metrics.pop("_count")
                                for key, value in accumulated_transfer_metrics.items():
                                    tag = "loss/transfer" if key == "loss" else f"transfer/{key}"
                                    self.tensorboard.add_scalar(tag, value / count, train_progress.global_step)
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix(
                                {
                                    "loss": accumulated_loss_cpu,
                                    "smooth loss": ema_loss,
                                }
                            )
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                        accumulated_loss = 0.0
                        accumulated_dpo_metrics = None
                        if self._dpo_concept_window is not None:
                            self._dpo_concept_window.reset()
                        accumulated_kto_metrics = None
                        accumulated_text_metrics = None
                        accumulated_sft_loss = 0.0
                        accumulated_sft_count = 0
                        accumulated_distill_metrics = None
                        accumulated_transfer_metrics = None
                        # Reset Fix B mirrors at the accumulator boundary so a
                        # backup taken immediately after this point captures a
                        # clean zero state.
                        self._loop_accumulated_loss = 0.0
                        self._loop_accumulated_loss_tensor = None
                        self._loop_accumulated_dpo_metrics = None
                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)
                        # Once per update, on every step path (plain, scaler, fused back pass):
                        # advance the OFT spectral-clip state and refresh any rotation cache.
                        # After after_optimizer_step, which may change requires_grad (the cache
                        # syncs it); nothing between the optimizer step and here runs a model
                        # forward (DPO rebase copies params, adaptive beta and reporting are
                        # host-side; sampling/backup commands run at the next batch), and the
                        # EMA update and validation below come after.
                        self.model_setup.after_optimizer_step_adapters(self.model, self.config)
                        if self._probe_set is not None and self._concept_split_window:
                            # Fixed probes on the conflict-probe cadence, on the
                            # freshly updated (and cache-refreshed) adapter.
                            self.__run_probes(train_progress)

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay", self.model.ema.get_current_decay(update_step), train_progress.global_step
                            )
                            self.model.ema.step(self.parameters, update_step)

                        self.one_step_trained = True
                        if not self._precompile_cache_saved:
                            # Forward and checkpoint-recompute graphs all exist after
                            # the first optimizer step: persist them now so even a run
                            # that crashes later still warms the next one.
                            self._precompile_cache_saved = True
                            save_precompile_cache("after first step")

                if (
                    self.config.validation
                    or self.config.dpo_validation_active()
                    or (self.config.distill_enabled and self.config.distill_validation)
                    or (self.config.transfer_enabled and self.config.transfer_validation)
                ) and multi.is_master():
                    self.__validate(train_progress)

                self.__log_vram(train_progress, kind, batch)

                # epoch_sample counts standard samples: the main loader's resume
                # offset. An INDEPENDENT add-on micro-step advances the step
                # counters but not that.
                train_progress.next_step(self.config.batch_size if kind is MicroStepKind.SFT else 0)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    log_epoch_loss(min_steps=current_epoch_length // 2)  # final epoch only if mostly complete
                    return

            log_epoch_loss()
            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        # Thaw the heap frozen at epoch start first, so it stays collectable for
        # later runs in this process (UI, queue worker) even if the final save raises.
        gc.unfreeze()

        if self.one_step_trained:
            # Picks up graphs compiled after the first step (new buckets, sampling).
            save_precompile_cache("end of run")

        if self._dpo_live_ledger is not None:
            self._dpo_live_ledger.flush(self.model.train_progress.global_step, force=True)
        self.__save_diagnostics_state()

        if self.one_step_trained:
            if self.config.oft_rotation_cache_applies():
                # A mid-window stop leaves rotation gradients in the cache: land them in
                # Q.grad on the train device, before the offload and the backup below.
                self.model_setup.before_clip_adapters(self.model, self.config)
            self.model.to(self.temp_device)

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)

                if (
                    self.config.patience
                    and self._patience_best_backup_path
                    and os.path.isfile(self._patience_best_backup_path)
                ):
                    print(
                        f"Restoring patience best checkpoint from step {self._patience_best_step} "
                        f"(val_loss: {self._patience_best_loss:.6f})"
                    )
                    self.callbacks.on_update_status("Restoring best validation checkpoint")
                    best_state = torch.load(self._patience_best_backup_path, map_location=self.temp_device)
                    for param, saved in zip(self.parameters, best_state, strict=True):
                        param.data.copy_(saved)
                    del best_state

                # Restore DPO best AFTER EMA copy so it takes precedence
                if (
                    self.config.rlhf_enabled
                    and self.config.rlhf_dpo_save_best
                    and self._dpo_best_backup_path
                    and os.path.isfile(self._dpo_best_backup_path)
                ):
                    print(f"Restoring DPO best checkpoint from {self._dpo_best_backup_path}")
                    self.callbacks.on_update_status("Restoring best DPO checkpoint")
                    best_state = torch.load(self._dpo_best_backup_path, map_location=self.temp_device)
                    for param, saved in zip(self.parameters, best_state, strict=True):
                        param.data.copy_(saved)
                    del best_state

                # Transfer best AFTER the EMA copy so the best-seen adapter wins.
                if (
                    self.config.transfer_enabled
                    and self._transfer_best_backup_path
                    and os.path.isfile(self._transfer_best_backup_path)
                ):
                    try:
                        print(
                            f"Restoring transfer best checkpoint from step {self._transfer_stop.best_tag} "
                            f"(residual: {self._transfer_stop.best_residual:.6f})"
                        )
                        self.callbacks.on_update_status("Restoring best transfer checkpoint")
                        best_state = torch.load(self._transfer_best_backup_path, map_location=self.temp_device)
                        for param, saved in zip(self.parameters, best_state, strict=True):
                            param.data.copy_(saved)
                        del best_state
                    except Exception:  # noqa: BLE001 - a corrupt snapshot must not lose the final save
                        traceback.print_exc()
                        print("Could not restore transfer best checkpoint; saving last-step parameters instead.")
                if (
                    os.path.isdir(self.config.output_model_destination)
                    and self.config.output_model_format.is_single_file()
                ):
                    save_path = os.path.join(
                        self.config.output_model_destination,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}",
                    )
                else:
                    save_path = self.config.output_model_destination
                print("Saving " + save_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype(),
                )

                self.__save_patience_extra_outputs(save_path)

        if self.model is not None:
            self.model.to(self.temp_device)

        if multi.is_master():
            if self.tensorboard is not None:
                self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
        self.grad_hook_handles = []

        # Drop large object references so the next queue run starts from a clean slate.
        # Without this, model weights / optimizer state / dataloader caches live on the
        # trainer instance until the GC reclaims it, which can take multiple cycles when
        # cyclic refs are involved (callbacks ↔ trainer, hooks ↔ params).
        self.model = None
        self.model_loader = None
        self.model_setup = None
        self.model_saver = None
        self.model_sampler = None
        self.data_loader = None
        self.validation_data_loader = None
        self.dpo_validation_data_loader = None
        self.sft_anchor_source = None
        self.dpo_addon_source = None
        self.kto_source = None
        self._dpo_concept_resolver = None
        self._dpo_concept_window = None
        self._dpo_live_ledger = None
        self.parameters = None
        self.tensorboard = None
        self.sample_queue = []
