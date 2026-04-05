import contextlib
import threading
import tkinter as tk
from collections import OrderedDict
from enum import Enum
from tkinter import filedialog, messagebox

from modules.util.config.QueueConfig import QueueEntry
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.QueueEntryStatus import QueueEntryStatus
from modules.util.queue.QueueExecutor import QueueExecutor
from modules.util.queue.QueueManager import QueueManager
from modules.util.queue.QueueValidator import QueueValidator
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.type_util import issubclass_safe
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk

PAD = 10

STATUS_COLORS = {
    QueueEntryStatus.PENDING: "gray60",
    QueueEntryStatus.RUNNING: "#0d6efd",
    QueueEntryStatus.COMPLETED: "#198754",
    QueueEntryStatus.FAILED: "#dc3545",
    QueueEntryStatus.SKIPPED: "#fd7e14",
}

STATUS_SYMBOLS = {
    QueueEntryStatus.PENDING: "\u25cb",
    QueueEntryStatus.RUNNING: "\u25b6",
    QueueEntryStatus.COMPLETED: "\u2713",
    QueueEntryStatus.FAILED: "\u2717",
    QueueEntryStatus.SKIPPED: "\u2298",
}

EXCLUDED_FIELDS = {
    "concepts", "samples", "secrets", "cloud", "additional_embeddings",
    "embedding", "optimizer_defaults", "scheduler_params",
}

DISPLAY_NAMES = {
    "training_method": "Training Method", "model_type": "Model Type",
    "workspace_dir": "Workspace Directory", "cache_dir": "Cache Directory",
    "debug_mode": "Debug Mode", "debug_dir": "Debug Directory",
    "tensorboard": "Tensorboard", "tensorboard_expose": "Expose Tensorboard",
    "tensorboard_always_on": "Always-On Tensorboard", "tensorboard_port": "Tensorboard Port",
    "validation": "Validation", "validate_after": "Validate After",
    "validate_after_unit": "Validate After Unit",
    "patience": "Patience", "patience_epochs": "Patience Epochs",
    "continue_last_backup": "Continue From Last Backup", "prevent_overwrites": "Prevent Overwrites",
    "include_train_config": "Include Config",
    "multi_gpu": "Multi-GPU", "device_indexes": "Device Indexes",
    "gradient_reduce_precision": "Gradient Reduce Precision",
    "fused_gradient_reduce": "Fused Gradient Reduce",
    "async_gradient_reduce": "Async Gradient Reduce",
    "async_gradient_reduce_buffer": "Buffer Size (MB)",
    "base_model_name": "Base Model", "output_dtype": "Output Data Type",
    "output_model_format": "Output Format", "output_model_destination": "Output Destination",
    "gradient_checkpointing": "Gradient Checkpointing",
    "enable_async_offloading": "Async Offloading",
    "enable_activation_offloading": "Activation Offloading",
    "layer_offload_fraction": "Layer Offload Fraction",
    "force_circular_padding": "Force Circular Padding", "compile": "Compile",
    "concept_file_name": "Concept File",
    "aspect_ratio_bucketing": "Aspect Ratio Bucketing",
    "latent_caching": "Latent Caching", "clear_cache_before_training": "Clear Cache Before Training",
    "learning_rate_scheduler": "LR Scheduler", "custom_learning_rate_scheduler": "Custom LR Scheduler",
    "learning_rate": "Learning Rate", "learning_rate_warmup_steps": "LR Warmup Steps",
    "learning_rate_cycles": "LR Cycles", "learning_rate_min_factor": "LR Min Factor",
    "epochs": "Epochs", "batch_size": "Local Batch Size",
    "gradient_accumulation_steps": "Accumulation Steps",
    "ema": "EMA", "ema_decay": "EMA Decay", "ema_update_step_interval": "EMA Update Interval",
    "dataloader_threads": "Dataloader Threads", "train_device": "Train Device",
    "temp_device": "Temp Device", "train_dtype": "Train Data Type",
    "fallback_train_dtype": "Fallback Train Data Type",
    "enable_autocast_cache": "Autocast Cache", "only_cache": "Only Cache",
    "resolution": "Resolution", "frames": "Frames",
    "mse_strength": "MSE Strength", "mae_strength": "MAE Strength",
    "log_cosh_strength": "Log Cosh Strength", "huber_strength": "Huber Strength",
    "huber_delta": "Huber Delta", "vb_loss_strength": "VB Loss Strength",
    "loss_weight_fn": "Loss Weight Function", "loss_weight_strength": "Loss Weight Strength",
    "dropout_probability": "Dropout Probability",
    "loss_scaler": "Loss Scaler", "learning_rate_scaler": "LR Scaler",
    "clip_grad_norm": "Clip Grad Norm",
    "layer_filter": "Layer Filter", "layer_filter_preset": "Layer Filter Preset",
    "layer_filter_regex": "Regex Filter",
    "offset_noise_weight": "Offset Noise Weight", "generalized_offset_noise": "Generalized Offset Noise",
    "perturbation_noise_weight": "Perturbation Noise Weight",
    "rescale_noise_scheduler_to_zero_terminal_snr": "Rescale Noise to Zero Terminal SNR",
    "force_v_prediction": "Force V-Prediction", "force_epsilon_prediction": "Force Epsilon Prediction",
    "timestep_distribution": "Timestep Distribution",
    "min_noising_strength": "Min Noising Strength", "max_noising_strength": "Max Noising Strength",
    "noising_weight": "Noising Weight", "noising_bias": "Noising Bias",
    "timestep_shift": "Timestep Shift", "dynamic_timestep_shifting": "Dynamic Timestep Shifting",
    "masked_training": "Masked Training", "unmasked_probability": "Unmasked Probability",
    "unmasked_weight": "Unmasked Weight", "normalize_masked_area_loss": "Normalize Masked Area Loss",
    "masked_prior_preservation_weight": "Prior Preservation Weight",
    "custom_conditioning_image": "Custom Conditioning Image",
    "embedding_learning_rate": "Embedding Learning Rate",
    "preserve_embedding_norm": "Preserve Embedding Norm",
    "embedding_weight_dtype": "Embedding Data Type",
    "peft_type": "PEFT Type", "lora_model_name": "LoRA Model",
    "lora_rank": "LoRA Rank", "lora_alpha": "LoRA Alpha",
    "lora_decompose": "LoRA Decompose", "lora_decompose_norm_epsilon": "Decompose Norm Epsilon",
    "lora_decompose_output_axis": "Decompose Output Axis",
    "lora_weight_dtype": "LoRA Data Type", "bundle_additional_embeddings": "Bundle Embeddings",
    "oft_block_size": "OFT Block Size", "oft_coft": "OFT CoFT",
    "coft_eps": "CoFT Epsilon", "oft_block_share": "OFT Block Share",
    "sample_definition_file_name": "Sample Definition File",
    "sample_after": "Sample After", "sample_after_unit": "Sample After Unit",
    "sample_skip_first": "Skip First Samples", "sample_image_format": "Image Format",
    "sample_video_format": "Video Format", "sample_audio_format": "Audio Format",
    "samples_to_tensorboard": "Samples to Tensorboard", "non_ema_sampling": "Non-EMA Sampling",
    "backup_after": "Backup After", "backup_after_unit": "Backup After Unit",
    "rolling_backup": "Rolling Backup", "rolling_backup_count": "Rolling Backup Count",
    "backup_before_save": "Backup Before Save",
    "save_every": "Save Every", "save_every_unit": "Save Every Unit",
    "save_skip_first": "Skip First Saves", "save_filename_prefix": "Save Filename Prefix",
    "model_name": "Model Name", "include": "Include", "train": "Train",
    "stop_training_after": "Stop Training After", "stop_training_after_unit": "Stop After Unit",
    "weight_dtype": "Data Type", "train_embedding": "Train Embedding",
    "attention_mask": "Attention Mask", "guidance_scale": "Guidance Scale",
    "optimizer": "Optimizer", "weight_decay": "Weight Decay", "eps": "Epsilon",
    "text_encoder_layer_skip": "TE1 Layer Skip",
    "text_encoder_sequence_length": "TE1 Sequence Length",
    "text_encoder_2_layer_skip": "TE2 Layer Skip",
    "text_encoder_2_sequence_length": "TE2 Sequence Length",
    "text_encoder_3_layer_skip": "TE3 Layer Skip",
    "text_encoder_4_layer_skip": "TE4 Layer Skip",
}

PATH_FIELDS = {
    "workspace_dir": "dir", "cache_dir": "dir", "debug_dir": "dir",
    "base_model_name": "file", "output_model_destination": "file",
    "concept_file_name": "file", "lora_model_name": "file",
    "sample_definition_file_name": "file", "model_name": "file",
}

# fork authors: prefix custom fields (e.g. myfork_*) for auto-grouped sections
SECTION_MAP = OrderedDict([
    ("General", [
        "training_method", "model_type", "workspace_dir", "cache_dir",
        "continue_last_backup", "prevent_overwrites",
        "tensorboard", "tensorboard_expose", "tensorboard_always_on", "tensorboard_port",
        "validation", "validate_after", "validate_after_unit",
        "patience", "patience_epochs",
        "debug_mode", "debug_dir", "only_cache", "include_train_config",
    ]),
    ("Multi-GPU", [
        "multi_gpu", "device_indexes",
        "gradient_reduce_precision", "fused_gradient_reduce",
        "async_gradient_reduce", "async_gradient_reduce_buffer",
    ]),
    ("Model", [
        "base_model_name", "output_model_destination", "output_dtype", "output_model_format",
        "gradient_checkpointing", "enable_async_offloading", "enable_activation_offloading",
        "layer_offload_fraction", "force_circular_padding", "compile",
    ]),
    ("Concepts", ["concept_file_name"]),
    ("Data", [
        "aspect_ratio_bucketing", "latent_caching", "clear_cache_before_training",
    ]),
    ("Training", [
        "learning_rate", "epochs", "batch_size", "gradient_accumulation_steps",
        "learning_rate_scheduler", "custom_learning_rate_scheduler",
        "learning_rate_warmup_steps", "learning_rate_cycles", "learning_rate_min_factor",
        "ema", "ema_decay", "ema_update_step_interval",
        "dataloader_threads", "train_device", "temp_device",
        "train_dtype", "fallback_train_dtype", "enable_autocast_cache",
        "resolution", "frames",
        "mse_strength", "mae_strength", "log_cosh_strength",
        "huber_strength", "huber_delta", "vb_loss_strength",
        "loss_weight_fn", "loss_weight_strength",
        "dropout_probability", "loss_scaler", "learning_rate_scaler", "clip_grad_norm",
    ]),
    ("Layer Filter", ["layer_filter", "layer_filter_preset", "layer_filter_regex"]),
    ("Noise", [
        "offset_noise_weight", "generalized_offset_noise", "perturbation_noise_weight",
        "rescale_noise_scheduler_to_zero_terminal_snr",
        "force_v_prediction", "force_epsilon_prediction",
        "timestep_distribution", "min_noising_strength", "max_noising_strength",
        "noising_weight", "noising_bias", "timestep_shift", "dynamic_timestep_shifting",
    ]),
    ("Model Parts", [
        "unet", "prior", "transformer", "quantization",
        "text_encoder", "text_encoder_layer_skip", "text_encoder_sequence_length",
        "text_encoder_2", "text_encoder_2_layer_skip", "text_encoder_2_sequence_length",
        "text_encoder_3", "text_encoder_3_layer_skip",
        "text_encoder_4", "text_encoder_4_layer_skip",
        "vae", "effnet_encoder", "decoder",
        "decoder_text_encoder", "decoder_vqgan",
    ]),
    ("Masked Training", [
        "masked_training", "unmasked_probability", "unmasked_weight",
        "normalize_masked_area_loss", "masked_prior_preservation_weight",
        "custom_conditioning_image",
    ]),
    ("Embedding", ["embedding_learning_rate", "preserve_embedding_norm", "embedding_weight_dtype"]),
    ("LoRA / OFT", [
        "peft_type", "lora_model_name", "lora_rank", "lora_alpha",
        "lora_decompose", "lora_decompose_norm_epsilon", "lora_decompose_output_axis",
        "lora_weight_dtype", "bundle_additional_embeddings",
        "oft_block_size", "oft_coft", "coft_eps", "oft_block_share",
    ]),
    ("Optimizer", ["optimizer"]),
    ("Sampling", [
        "sample_definition_file_name", "sample_after", "sample_after_unit",
        "sample_skip_first", "sample_image_format", "sample_video_format",
        "sample_audio_format", "samples_to_tensorboard", "non_ema_sampling",
    ]),
    ("Backup", [
        "backup_after", "backup_after_unit", "rolling_backup", "rolling_backup_count",
        "backup_before_save", "save_every", "save_every_unit",
        "save_skip_first", "save_filename_prefix",
    ]),
])


class _FieldInfo:
    __slots__ = ("value_var", "widget", "field_type", "nullable", "path",
                 "indicator", "reset_btn", "is_overridden")

    def __init__(self, value_var, widget, field_type, nullable, path, indicator, reset_btn):
        self.value_var = value_var
        self.widget = widget
        self.field_type = field_type
        self.nullable = nullable
        self.path = path
        self.indicator = indicator
        self.reset_btn = reset_btn
        self.is_overridden = False


class QueueWindow(ctk.CTkToplevel):
    def __init__(self, parent, train_config: TrainConfig):
        super().__init__(parent)
        self.parent = parent
        self.train_config = train_config
        self.queue_manager = QueueManager()
        self._selected_entry_id: str | None = None
        self._fields: dict[str, _FieldInfo] = {}
        self._section_frames: dict[str, ctk.CTkFrame] = {}
        self._section_toggle_vars: dict[str, tk.BooleanVar] = {}
        self._entry_widgets: list[tuple[str, ctk.CTkFrame]] = []
        self._executor: QueueExecutor | None = None
        self._executor_thread: threading.Thread | None = None
        self._suppress_save = False

        self.title("Training Queue")
        self.geometry("1100x750")
        self.resizable(True, True)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0, minsize=250)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main_panel()
        self._build_footer()
        self._refresh_entry_list()
        self.after(200, lambda: set_window_icon(self))

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(PAD, 0))
        for i in range(5):
            btn_frame.grid_columnconfigure(i, weight=1)
        ctk.CTkButton(btn_frame, text="+", width=38, command=self._add_entry).grid(row=0, column=0, padx=2)
        ctk.CTkButton(btn_frame, text="\u2212", width=38, command=self._remove_entry).grid(row=0, column=1, padx=2)
        ctk.CTkButton(btn_frame, text="\u2398", width=38, command=self._copy_entry).grid(row=0, column=2, padx=2)
        ctk.CTkButton(btn_frame, text="\u25b2", width=38, command=self._move_up).grid(row=0, column=3, padx=2)
        ctk.CTkButton(btn_frame, text="\u25bc", width=38, command=self._move_down).grid(row=0, column=4, padx=2)

        ctk.CTkFrame(sidebar, height=1, fg_color="gray40").grid(row=1, column=0, sticky="ew", padx=PAD, pady=PAD)

        self._entry_list_frame = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self._entry_list_frame.grid(row=2, column=0, sticky="nsew", padx=PAD, pady=0)
        self._entry_list_frame.grid_columnconfigure(0, weight=1)

        self._build_sidebar_settings(sidebar)

    def _build_sidebar_settings(self, sidebar):
        ctk.CTkFrame(sidebar, height=1, fg_color="gray40").grid(row=3, column=0, sticky="ew", padx=PAD, pady=PAD)
        sf = ctk.CTkFrame(sidebar, fg_color="transparent")
        sf.grid(row=4, column=0, sticky="ew", padx=PAD, pady=(0, PAD))
        sf.grid_columnconfigure(1, weight=1)

        s = self.queue_manager.settings
        self._retry_var = tk.BooleanVar(value=s.retry_on_error)
        self._retry_backup_var = tk.BooleanVar(value=s.retry_from_backup)
        self._max_retries_var = tk.StringVar(value=str(s.max_retries))
        self._oom_threshold_var = tk.StringVar(value=str(s.oom_skip_threshold))

        r = 0
        ctk.CTkLabel(sf, text="Queue Settings", font=ctk.CTkFont(weight="bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(0, PAD))
        r += 1
        ctk.CTkCheckBox(sf, text="Retry on error", variable=self._retry_var,
                         command=self._save_settings).grid(row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1
        ctk.CTkCheckBox(sf, text="Retry from backup", variable=self._retry_backup_var,
                         command=self._save_settings).grid(row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1
        ctk.CTkLabel(sf, text="Max retries").grid(row=r, column=0, sticky="w", pady=2)
        ctk.CTkEntry(sf, textvariable=self._max_retries_var, width=50).grid(row=r, column=1, sticky="w", padx=PAD)
        r += 1
        ctk.CTkLabel(sf, text="OOM threshold").grid(row=r, column=0, sticky="w", pady=2)
        ctk.CTkEntry(sf, textvariable=self._oom_threshold_var, width=50).grid(row=r, column=1, sticky="w", padx=PAD)
        r += 1
        btn_row = ctk.CTkFrame(sf, fg_color="transparent")
        btn_row.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(PAD, 0))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(btn_row, text="Export", width=80, command=self._export_queue).grid(row=0, column=0, padx=2)
        ctk.CTkButton(btn_row, text="Import", width=80, command=self._import_queue).grid(row=0, column=1, padx=2)

        for var in (self._max_retries_var, self._oom_threshold_var):
            var.trace_add("write", lambda *_: self._save_settings())

    def _save_settings(self):
        s = self.queue_manager.settings
        s.retry_on_error = self._retry_var.get()
        s.retry_from_backup = self._retry_backup_var.get()
        with contextlib.suppress(ValueError):
            s.max_retries = int(self._max_retries_var.get())
        with contextlib.suppress(ValueError):
            s.oom_skip_threshold = int(self._oom_threshold_var.get())
        self.queue_manager.save()

    def _refresh_entry_list(self):
        for _, w in self._entry_widgets:
            w.destroy()
        self._entry_widgets.clear()
        for entry in self.queue_manager.entries:
            frame = ctk.CTkFrame(self._entry_list_frame, fg_color="transparent", cursor="hand2")
            frame.grid(sticky="ew", pady=2)
            frame.grid_columnconfigure(1, weight=1)
            color = STATUS_COLORS.get(entry.status, "gray60")
            symbol = STATUS_SYMBOLS.get(entry.status, "?")
            status_lbl = ctk.CTkLabel(frame, text=symbol, text_color=color, width=22,
                                       font=ctk.CTkFont(size=14))
            status_lbl.grid(row=0, column=0, padx=(PAD, 4))
            name_lbl = ctk.CTkLabel(frame, text=entry.name or "(unnamed)", anchor="w")
            name_lbl.grid(row=0, column=1, sticky="ew", pady=3)
            eid = entry.id
            for widget in (frame, status_lbl, name_lbl):
                widget.bind("<Button-1>", lambda _ev, _id=eid: self._select_entry(_id))
            if entry.id == self._selected_entry_id:
                frame.configure(fg_color=("gray80", "gray30"))
            self._entry_widgets.append((entry.id, frame))

    def _select_entry(self, entry_id: str):
        self._selected_entry_id = entry_id
        self._refresh_entry_list()
        self._load_entry_into_editor(entry_id)

    def _add_entry(self):
        idx = len(self.queue_manager.entries) + 1
        entry = self.queue_manager.add_entry(name=f"Run {idx}")
        self._select_entry(entry.id)

    def _remove_entry(self):
        if self._selected_entry_id is None:
            return
        self.queue_manager.remove_entry(self._selected_entry_id)
        self._selected_entry_id = None
        self._refresh_entry_list()
        self._clear_editor()

    def _copy_entry(self):
        if self._selected_entry_id is None:
            return
        new = self.queue_manager.duplicate_entry(self._selected_entry_id)
        if new:
            self._select_entry(new.id)

    def _move_up(self):
        if self._selected_entry_id:
            self.queue_manager.move_up(self._selected_entry_id)
            self._refresh_entry_list()

    def _move_down(self):
        if self._selected_entry_id:
            self.queue_manager.move_down(self._selected_entry_id)
            self._refresh_entry_list()

    def _build_main_panel(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        name_frame = ctk.CTkFrame(main, fg_color="transparent")
        name_frame.grid(row=0, column=0, sticky="ew", padx=PAD, pady=PAD)
        name_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(name_frame, text="Run Name:", font=ctk.CTkFont(size=13)).grid(row=0, column=0, padx=(0, PAD))
        self._name_var = tk.StringVar()
        self._name_var.trace_add("write", lambda *_: self._on_name_change())
        ctk.CTkEntry(name_frame, textvariable=self._name_var, font=ctk.CTkFont(size=13)).grid(
            row=0, column=1, sticky="ew")

        self._editor_scroll = ctk.CTkScrollableFrame(main, fg_color="transparent")
        self._editor_scroll.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=(0, PAD))
        self._editor_scroll.grid_columnconfigure(0, weight=1)

        self._build_override_sections()

    def _build_override_sections(self):
        ref_config = TrainConfig.default_values()
        from modules.util.config.BaseConfig import BaseConfig
        assigned = set()
        section_row = 0
        for section_name, field_names in SECTION_MAP.items():
            section_container, content, toggle_fn = self._make_section(section_name, section_row)
            field_row = 0
            for field_name in field_names:
                if field_name not in ref_config.types:
                    continue
                ft = ref_config.types[field_name]
                if issubclass_safe(ft, BaseConfig):
                    sub_config = getattr(ref_config, field_name)
                    display = field_name.replace("_", " ").title()
                    ctk.CTkLabel(content, text=display, font=ctk.CTkFont(size=12, weight="bold")).grid(
                        row=field_row, column=0, columnspan=5, sticky="w", padx=PAD, pady=(PAD, 2))
                    field_row += 1
                    for sub_name in sub_config.types:
                        path = f"{field_name}.{sub_name}"
                        self._create_field_row(content, path, sub_config.types[sub_name],
                                               sub_config.nullables[sub_name], field_row)
                        field_row += 1
                    assigned.add(field_name)
                else:
                    self._create_field_row(content, field_name, ft, ref_config.nullables[field_name], field_row)
                    field_row += 1
                    assigned.add(field_name)
            section_row += 1

        other_fields = [n for n in ref_config.types if n not in assigned and n not in EXCLUDED_FIELDS]
        if other_fields:
            _container, content, _toggle = self._make_section("Other", section_row)
            field_row = 0
            for field_name in other_fields:
                ft = ref_config.types[field_name]
                if issubclass_safe(ft, BaseConfig):
                    continue
                self._create_field_row(content, field_name, ft, ref_config.nullables[field_name], field_row)
                field_row += 1

    def _make_section(self, section_name, section_row):
        container = ctk.CTkFrame(self._editor_scroll)
        container.grid(row=section_row, column=0, sticky="ew", pady=(0, 4))
        container.grid_columnconfigure(0, weight=1)
        toggle_var = tk.BooleanVar(value=False)
        self._section_toggle_vars[section_name] = toggle_var

        header = ctk.CTkFrame(container, fg_color=("gray85", "gray25"), corner_radius=6, cursor="hand2")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        arrow_lbl = ctk.CTkLabel(header, text="\u25b6", width=20, font=ctk.CTkFont(size=11))
        arrow_lbl.grid(row=0, column=0, padx=(PAD, 0), pady=4)
        ctk.CTkLabel(header, text=section_name, font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").grid(row=0, column=1, sticky="w", padx=PAD, pady=4)

        content = ctk.CTkFrame(container, fg_color="transparent")
        content.grid_columnconfigure(0, weight=0, minsize=18)
        content.grid_columnconfigure(1, weight=0, minsize=180)
        content.grid_columnconfigure(2, weight=1)
        content.grid_columnconfigure(3, weight=0)
        content.grid_columnconfigure(4, weight=0)
        self._section_frames[section_name] = content

        sn = section_name

        def make_toggle(cont, arrow, sn_ref=sn):
            def toggle(_event=None):
                v = self._section_toggle_vars[sn_ref]
                v.set(not v.get())
                if v.get():
                    cont.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=(2, PAD))
                    arrow.configure(text="\u25bc")
                else:
                    cont.grid_remove()
                    arrow.configure(text="\u25b6")
            return toggle

        toggle_fn = make_toggle(content, arrow_lbl)
        header.bind("<Button-1>", toggle_fn)
        for child in header.winfo_children():
            child.bind("<Button-1>", toggle_fn)
        content.grid_remove()
        return container, content, toggle_fn

    def _create_field_row(self, parent, path: str, field_type: type, nullable: bool, row: int):
        leaf = path.split(".")[-1]
        label_text = DISPLAY_NAMES.get(leaf, DISPLAY_NAMES.get(path, leaf.replace("_", " ").title()))

        indicator = ctk.CTkLabel(parent, text="  ", width=16, font=ctk.CTkFont(size=10))
        indicator.grid(row=row, column=0, padx=(2, 0), pady=3, sticky="w")

        ctk.CTkLabel(parent, text=label_text, anchor="w").grid(
            row=row, column=1, padx=(0, PAD), pady=3, sticky="w")

        if field_type is bool:
            value_var = tk.BooleanVar(value=False)
            widget = ctk.CTkSwitch(parent, variable=value_var, text="", width=40)
            widget.grid(row=row, column=2, padx=PAD, pady=3, sticky="w")
        elif issubclass_safe(field_type, Enum):
            value_var = tk.StringVar()
            values = [str(e.value) for e in field_type]
            widget = ctk.CTkOptionMenu(parent, variable=value_var, values=values, width=160)
            widget.grid(row=row, column=2, padx=PAD, pady=3, sticky="ew")
        else:
            value_var = tk.StringVar()
            widget = ctk.CTkEntry(parent, textvariable=value_var, width=160)
            widget.grid(row=row, column=2, padx=PAD, pady=3, sticky="ew")

        browse_col = 3
        if leaf in PATH_FIELDS:
            mode = PATH_FIELDS[leaf]
            def _browse(m=mode, v=value_var):
                if m == "dir":
                    chosen = filedialog.askdirectory()
                else:
                    chosen = filedialog.askopenfilename(filetypes=[
                        ("All Files", "*.*"), ("Safetensors", "*.safetensors"),
                        ("Checkpoint", "*.ckpt *.pt *.bin"), ("JSON", "*.json"),
                    ])
                if chosen:
                    v.set(str(chosen))
            ctk.CTkButton(parent, text="...", width=30, height=26, command=_browse).grid(
                row=row, column=browse_col, padx=(0, 2), pady=3)
            browse_col = 4

        reset_btn = ctk.CTkButton(parent, text="\u21ba", width=30, height=26, fg_color="transparent",
                                   text_color=("gray40", "gray60"), hover_color=("gray80", "gray30"),
                                   command=lambda: self._reset_field(path))
        reset_btn.grid(row=row, column=browse_col if leaf not in PATH_FIELDS else 4, padx=(0, PAD), pady=3)

        self._fields[path] = _FieldInfo(value_var, widget, field_type, nullable, path, indicator, reset_btn)
        value_var.trace_add("write", lambda *_, p=path: self._on_value_change(p))

    def _on_value_change(self, path: str):
        if self._suppress_save:
            return
        fi = self._fields.get(path)
        if fi and self._selected_entry_id:
            fi.is_overridden = True
            self._update_override_indicator(fi, True)
            self._write_override(path)

    def _on_name_change(self):
        if self._suppress_save or not self._selected_entry_id:
            return
        entry = self.queue_manager.get_entry(self._selected_entry_id)
        if entry:
            entry.name = self._name_var.get()
            self.queue_manager.save()
            self._refresh_entry_list()

    def _reset_field(self, path: str):
        fi = self._fields.get(path)
        if not fi:
            return
        fi.is_overridden = False
        self._update_override_indicator(fi, False)
        self._remove_override(path)
        self._set_field_to_global(path)

    def _update_override_indicator(self, fi: _FieldInfo, is_overridden: bool):
        if is_overridden:
            fi.indicator.configure(text="\u25cf", text_color="#0d6efd")
            fi.reset_btn.configure(text_color=("#0d6efd", "#6ea8fe"))
        else:
            fi.indicator.configure(text="  ", text_color="transparent")
            fi.reset_btn.configure(text_color=("gray40", "gray60"))

    def _write_override(self, path: str):
        if not self._selected_entry_id:
            return
        entry = self.queue_manager.get_entry(self._selected_entry_id)
        if not entry:
            return
        fi = self._fields[path]
        value = self._get_field_value(fi)
        parts = path.split(".")
        target = entry.overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        self.queue_manager.save()

    def _remove_override(self, path: str):
        if not self._selected_entry_id:
            return
        entry = self.queue_manager.get_entry(self._selected_entry_id)
        if not entry:
            return
        parts = path.split(".")
        target = entry.overrides
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                return
            target = target[part]
        target.pop(parts[-1], None)
        for i in range(len(parts) - 2, -1, -1):
            parent = entry.overrides
            for p in parts[:i]:
                parent = parent.get(p, {})
            if isinstance(parent.get(parts[i]), dict) and not parent[parts[i]]:
                parent.pop(parts[i], None)
        self.queue_manager.save()

    def _get_field_value(self, fi: _FieldInfo):
        if fi.field_type is bool:
            return fi.value_var.get()
        if issubclass_safe(fi.field_type, Enum):
            return fi.value_var.get()
        raw = fi.value_var.get()
        if fi.nullable and raw == "":
            return None
        if fi.field_type is int:
            with contextlib.suppress(ValueError):
                return int(raw)
            return raw
        if fi.field_type is float:
            with contextlib.suppress(ValueError):
                return float(raw)
            return raw
        return raw

    def _set_field_to_global(self, path: str):
        fi = self._fields.get(path)
        if not fi:
            return
        self._set_field_display(fi, self._get_global_value(path))

    def _get_global_value(self, path: str):
        parts = path.split(".")
        obj = self.train_config
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _set_field_display(self, fi: _FieldInfo, value):
        self._suppress_save = True
        try:
            if fi.field_type is bool:
                fi.value_var.set(bool(value) if value is not None else False)
            elif issubclass_safe(fi.field_type, Enum):
                fi.value_var.set(str(value) if value is not None else "")
            else:
                fi.value_var.set(str(value) if value is not None else "")
        finally:
            self._suppress_save = False

    def _load_entry_into_editor(self, entry_id: str):
        entry = self.queue_manager.get_entry(entry_id)
        if not entry:
            return
        self._suppress_save = True
        try:
            self._name_var.set(entry.name)
            for path, fi in self._fields.items():
                override_value = self._get_override_value(entry.overrides, path)
                if override_value is not _SENTINEL:
                    fi.is_overridden = True
                    self._update_override_indicator(fi, True)
                    self._set_field_display(fi, override_value)
                else:
                    fi.is_overridden = False
                    self._update_override_indicator(fi, False)
                    self._set_field_display(fi, self._get_global_value(path))
        finally:
            self._suppress_save = False

    def _clear_editor(self):
        self._suppress_save = True
        try:
            self._name_var.set("")
            for fi in self._fields.values():
                fi.is_overridden = False
                self._update_override_indicator(fi, False)
                if isinstance(fi.value_var, tk.BooleanVar):
                    fi.value_var.set(False)
                else:
                    fi.value_var.set("")
        finally:
            self._suppress_save = False

    @staticmethod
    def _get_override_value(overrides: dict, path: str):
        parts = path.split(".")
        target = overrides
        for part in parts:
            if not isinstance(target, dict) or part not in target:
                return _SENTINEL
            target = target[part]
        return target

    def _build_footer(self):
        footer = ctk.CTkFrame(self, corner_radius=0)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=0)

        left = ctk.CTkFrame(footer, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=PAD, pady=PAD)
        left.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(left, text="Idle", anchor="w", font=ctk.CTkFont(size=12))
        self._status_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        pf = ctk.CTkFrame(left, fg_color="transparent")
        pf.grid(row=1, column=0, sticky="ew")
        pf.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(pf, text="Step").grid(row=0, column=0, padx=(0, PAD))
        self._step_progress = ctk.CTkProgressBar(pf)
        self._step_progress.grid(row=0, column=1, sticky="ew", padx=(0, PAD))
        self._step_progress.set(0)
        self._step_label = ctk.CTkLabel(pf, text="0/0", width=80)
        self._step_label.grid(row=0, column=2)

        ctk.CTkLabel(pf, text="Epoch").grid(row=1, column=0, padx=(0, PAD))
        self._epoch_progress = ctk.CTkProgressBar(pf)
        self._epoch_progress.grid(row=1, column=1, sticky="ew", padx=(0, PAD))
        self._epoch_progress.set(0)
        self._epoch_label = ctk.CTkLabel(pf, text="0/0", width=80)
        self._epoch_label.grid(row=1, column=2)

        btn_frame = ctk.CTkFrame(footer, fg_color="transparent")
        btn_frame.grid(row=0, column=1, padx=PAD, pady=PAD)

        self._start_btn = ctk.CTkButton(btn_frame, text="Start Queue", fg_color="#198754",
                                         hover_color="#146c43", command=self._start_queue)
        self._start_btn.grid(row=0, column=0, padx=3)
        self._stop_run_btn = ctk.CTkButton(btn_frame, text="Stop Run", fg_color="#fd7e14",
                                            hover_color="#e06c0a", command=self._stop_run)
        self._stop_run_btn.grid(row=0, column=1, padx=3)
        self._stop_run_btn.grid_remove()
        self._stop_queue_btn = ctk.CTkButton(btn_frame, text="Stop Queue", fg_color="#dc3545",
                                              hover_color="#bb2d3b", command=self._stop_queue)
        self._stop_queue_btn.grid(row=0, column=2, padx=3)
        self._stop_queue_btn.grid_remove()
        self._reset_btn = ctk.CTkButton(btn_frame, text="Reset All", command=self._reset_all)
        self._reset_btn.grid(row=0, column=3, padx=3)

    def _start_queue(self):
        if self.parent.training_thread is not None:
            messagebox.showwarning("Training Running", "Cannot start queue while single training is running.")
            return
        if self._executor_thread is not None:
            return
        if not self.queue_manager.entries:
            messagebox.showinfo("Empty Queue", "Add at least one run to the queue.")
            return

        results = QueueValidator.validate_queue(self.queue_manager, self.train_config)
        if QueueValidator.has_critical_errors(results):
            msg_parts = []
            for eid, (errors, _warnings) in results.items():
                entry = self.queue_manager.get_entry(eid)
                name = entry.name if entry else eid
                if errors:
                    msg_parts.append(f"{name}:\n" + "\n".join(f"  \u2022 {e}" for e in errors))
            if not messagebox.askyesno("Validation Errors",
                                        "Critical errors found:\n\n" + "\n\n".join(msg_parts) + "\n\nStart anyway?"):
                return
        elif results:
            msg_parts = []
            for eid, (_errors, warnings) in results.items():
                entry = self.queue_manager.get_entry(eid)
                name = entry.name if entry else eid
                if warnings:
                    msg_parts.append(f"{name}:\n" + "\n".join(f"  \u2022 {w}" for w in warnings))
            if msg_parts:
                messagebox.showinfo("Validation Warnings", "\n\n".join(msg_parts))

        self._executor = QueueExecutor(
            queue_manager=self.queue_manager, global_config=self.train_config,
            on_entry_start=lambda e, i, t: self.after(0, lambda: self._on_entry_start(e, i, t)),
            on_entry_complete=lambda e: self.after(0, lambda: self._on_entry_done(e)),
            on_entry_failed=lambda e, m: self.after(0, lambda: self._on_entry_done(e)),
            on_entry_skipped=lambda e: self.after(0, lambda: self._on_entry_done(e)),
            on_progress=lambda p, s, ep: self.after(0, lambda: self._on_progress(p, s, ep)),
            on_status=lambda s: self.after(0, lambda: self._on_status(s)),
            on_queue_complete=lambda: self.after(0, self._on_queue_complete),
        )
        self.parent._queue_executor = self._executor
        self._set_running_ui()
        self._executor_thread = threading.Thread(target=self._executor.run, daemon=True)
        self._executor_thread.start()
        if hasattr(self.parent, "training_button") and self.parent.training_button:
            self.parent.training_button.configure(state="disabled", text="Queue Running")

    def _stop_run(self):
        if self._executor:
            self._executor.stop_current_run()

    def _stop_queue(self):
        if self._executor:
            self._executor.stop_queue()

    def _reset_all(self):
        self.queue_manager.reset_all_to_pending()
        self._refresh_entry_list()

    def _set_running_ui(self):
        self._start_btn.grid_remove()
        self._reset_btn.grid_remove()
        self._stop_run_btn.grid(row=0, column=1, padx=3)
        self._stop_queue_btn.grid(row=0, column=2, padx=3)

    def _set_idle_ui(self):
        self._stop_run_btn.grid_remove()
        self._stop_queue_btn.grid_remove()
        self._start_btn.grid(row=0, column=0, padx=3)
        self._reset_btn.grid(row=0, column=3, padx=3)
        self._status_label.configure(text="Idle")
        self._step_progress.set(0)
        self._epoch_progress.set(0)
        self._step_label.configure(text="0/0")
        self._epoch_label.configure(text="0/0")

    def _on_entry_start(self, entry: QueueEntry, run_idx: int, total: int):
        self._status_label.configure(text=f"Run {run_idx} of {total} \u2014 {entry.name}")
        self._refresh_entry_list()

    def _on_entry_done(self, _entry: QueueEntry):
        self._refresh_entry_list()

    def _on_progress(self, train_progress: TrainProgress, max_step: int, max_epoch: int):
        if max_step > 0:
            self._step_progress.set(train_progress.epoch_step / max_step)
            self._step_label.configure(text=f"{train_progress.epoch_step}/{max_step}")
        if max_epoch > 0:
            self._epoch_progress.set(train_progress.epoch / max_epoch)
            self._epoch_label.configure(text=f"{train_progress.epoch}/{max_epoch}")

    def _on_status(self, status: str):
        self._status_label.configure(text=status)

    def _on_queue_complete(self):
        self._executor = None
        self._executor_thread = None
        self.parent._queue_executor = None
        if hasattr(self.parent, "_set_training_button_idle"):
            self.parent._set_training_button_idle()
        self._set_idle_ui()
        self._refresh_entry_list()
        torch_gc()

    def _export_queue(self):
        path = filedialog.asksaveasfilename(
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
            initialfile="queue.json", defaultextension=".json",
        )
        if path:
            self.queue_manager.export_to_file(path)

    def _import_queue(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            warnings = self.queue_manager.import_from_file(path)
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
            return
        self._refresh_entry_list()
        if warnings:
            messagebox.showwarning("Import Warnings",
                                   f"{len(warnings)} path warning(s):\n\n" + "\n".join(warnings[:10]))

    def _on_close(self):
        if self._executor_thread is not None:
            if not messagebox.askyesno("Queue Running", "Queue is still running. Stop and close?"):
                return
            if self._executor:
                self._executor.stop_queue_immediate()
        self.withdraw()

    def show(self):
        self.queue_manager.load()
        self._refresh_entry_list()
        self.deiconify()
        self.focus_set()


_SENTINEL = object()
