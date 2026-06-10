"""Queue entry diff helpers - port of CTK QueueWindow.py:45-225."""

from collections import OrderedDict

EXCLUDED_FIELDS: set[str] = {
    "concepts",
    "samples",
    "secrets",
    "cloud",
    "additional_embeddings",
    "embedding",
    "optimizer_defaults",
    "scheduler_params",
}

DISPLAY_NAMES: dict[str, str] = {
    "training_method": "Training Method",
    "model_type": "Model Type",
    "workspace_dir": "Workspace Directory",
    "cache_dir": "Cache Directory",
    "debug_mode": "Debug Mode",
    "debug_dir": "Debug Directory",
    "tensorboard": "Tensorboard",
    "tensorboard_expose": "Expose Tensorboard",
    "tensorboard_always_on": "Always-On Tensorboard",
    "tensorboard_port": "Tensorboard Port",
    "validation": "Validation",
    "validate_after": "Validate After",
    "validate_after_unit": "Validate After Unit",
    "patience": "Patience",
    "patience_epochs": "Patience Epochs",
    "continue_last_backup": "Continue From Last Backup",
    "prevent_overwrites": "Prevent Overwrites",
    "include_train_config": "Include Config",
    "multi_gpu": "Multi-GPU",
    "device_indexes": "Device Indexes",
    "gradient_reduce_precision": "Gradient Reduce Precision",
    "fused_gradient_reduce": "Fused Gradient Reduce",
    "async_gradient_reduce": "Async Gradient Reduce",
    "async_gradient_reduce_buffer": "Buffer Size (MB)",
    "base_model_name": "Base Model",
    "output_dtype": "Output Data Type",
    "output_model_format": "Output Format",
    "output_model_destination": "Output Destination",
    "gradient_checkpointing": "Gradient Checkpointing",
    "enable_async_offloading": "Async Offloading",
    "enable_activation_offloading": "Activation Offloading",
    "layer_offload_fraction": "Layer Offload Fraction",
    "force_circular_padding": "Force Circular Padding",
    "compile": "Compile",
    "concept_file_name": "Concept File",
    "aspect_ratio_bucketing": "Aspect Ratio Bucketing",
    "latent_caching": "Latent Caching",
    "clear_cache_before_training": "Clear Cache Before Training",
    "skip_cache_validation": "Skip Cache Validation",
    "learning_rate_scheduler": "LR Scheduler",
    "custom_learning_rate_scheduler": "Custom LR Scheduler",
    "learning_rate": "Learning Rate",
    "learning_rate_warmup_steps": "LR Warmup Steps",
    "learning_rate_cycles": "LR Cycles",
    "learning_rate_min_factor": "LR Min Factor",
    "epochs": "Epochs",
    "batch_size": "Local Batch Size",
    "gradient_accumulation_steps": "Accumulation Steps",
    "ema": "EMA",
    "ema_decay": "EMA Decay",
    "ema_update_step_interval": "EMA Update Interval",
    "dataloader_threads": "Dataloader Threads",
    "train_device": "Train Device",
    "temp_device": "Temp Device",
    "train_dtype": "Train Data Type",
    "fallback_train_dtype": "Fallback Train Data Type",
    "enable_autocast_cache": "Autocast Cache",
    "only_cache": "Only Cache",
    "resolution": "Resolution",
    "frames": "Frames",
    "mse_strength": "MSE Strength",
    "mae_strength": "MAE Strength",
    "log_cosh_strength": "Log Cosh Strength",
    "huber_strength": "Huber Strength",
    "huber_delta": "Huber Delta",
    "vb_loss_strength": "VB Loss Strength",
    "loss_weight_fn": "Loss Weight Function",
    "loss_weight_strength": "Loss Weight Strength",
    "dropout_probability": "Dropout Probability",
    "loss_scaler": "Loss Scaler",
    "learning_rate_scaler": "LR Scaler",
    "clip_grad_norm": "Clip Grad Norm",
    "layer_filter": "Layer Filter",
    "layer_filter_preset": "Layer Filter Preset",
    "layer_filter_regex": "Regex Filter",
    "offset_noise_weight": "Offset Noise Weight",
    "generalized_offset_noise": "Generalized Offset Noise",
    "perturbation_noise_weight": "Perturbation Noise Weight",
    "ciop_noise_weight": "I/O Noise Weight",
    "ciop_p": "I/O Noise Probability",
    "rescale_noise_scheduler_to_zero_terminal_snr": "Rescale Noise to Zero Terminal SNR",
    "force_v_prediction": "Force V-Prediction",
    "force_epsilon_prediction": "Force Epsilon Prediction",
    "timestep_distribution": "Timestep Distribution",
    "min_noising_strength": "Min Noising Strength",
    "max_noising_strength": "Max Noising Strength",
    "noising_weight": "Noising Weight",
    "noising_bias": "Noising Bias",
    "timestep_shift": "Timestep Shift",
    "dynamic_timestep_shifting": "Dynamic Timestep Shifting",
    "masked_training": "Masked Training",
    "unmasked_probability": "Unmasked Probability",
    "unmasked_weight": "Unmasked Weight",
    "normalize_masked_area_loss": "Normalize Masked Area Loss",
    "masked_prior_preservation_weight": "Prior Preservation Weight",
    "custom_conditioning_image": "Custom Conditioning Image",
    "embedding_learning_rate": "Embedding Learning Rate",
    "preserve_embedding_norm": "Preserve Embedding Norm",
    "embedding_weight_dtype": "Embedding Data Type",
    "peft_type": "PEFT Type",
    "lora_model_name": "LoRA Model",
    "lora_rank": "LoRA Rank",
    "lora_alpha": "LoRA Alpha",
    "lora_decompose": "LoRA Decompose",
    "lora_decompose_norm_epsilon": "Decompose Norm Epsilon",
    "lora_decompose_output_axis": "Decompose Output Axis",
    "lora_weight_dtype": "LoRA Data Type",
    "bundle_additional_embeddings": "Bundle Embeddings",
    "oft_block_size": "OFT Block Size",
    "oft_scaled": "Scaled OFT",
    "oft_clipped_norm": "Spectral Norm Clipping",
    "dora_oft": "DoRA OFT",
    "oft_block_share": "OFT Block Share",
    "sample_definition_file_name": "Sample Definition File",
    "sample_after": "Sample After",
    "sample_after_unit": "Sample After Unit",
    "sample_skip_first": "Skip First Samples",
    "sample_image_format": "Image Format",
    "sample_video_format": "Video Format",
    "sample_audio_format": "Audio Format",
    "samples_to_tensorboard": "Samples to Tensorboard",
    "non_ema_sampling": "Non-EMA Sampling",
    "backup_after": "Backup After",
    "backup_after_unit": "Backup After Unit",
    "rolling_backup": "Rolling Backup",
    "rolling_backup_count": "Rolling Backup Count",
    "backup_before_save": "Backup Before Save",
    "save_every": "Save Every",
    "save_every_unit": "Save Every Unit",
    "save_skip_first": "Skip First Saves",
    "save_filename_prefix": "Save Filename Prefix",
    "model_name": "Model Name",
    "include": "Include",
    "train": "Train",
    "stop_training_after": "Stop Training After",
    "stop_training_after_unit": "Stop After Unit",
    "weight_dtype": "Data Type",
    "train_embedding": "Train Embedding",
    "attention_mask": "Attention Mask",
    "guidance_scale": "Guidance Scale",
    "optimizer": "Optimizer",
}

SECTION_MAP: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        (
            "General",
            [
                "training_method",
                "model_type",
                "workspace_dir",
                "cache_dir",
                "continue_last_backup",
                "prevent_overwrites",
                "tensorboard",
                "tensorboard_expose",
                "tensorboard_always_on",
                "tensorboard_port",
                "validation",
                "validate_after",
                "validate_after_unit",
                "patience",
                "patience_epochs",
                "debug_mode",
                "debug_dir",
                "only_cache",
                "include_train_config",
            ],
        ),
        (
            "Multi-GPU",
            [
                "multi_gpu",
                "device_indexes",
                "gradient_reduce_precision",
                "fused_gradient_reduce",
                "async_gradient_reduce",
                "async_gradient_reduce_buffer",
            ],
        ),
        (
            "Model",
            [
                "base_model_name",
                "output_model_destination",
                "output_dtype",
                "output_model_format",
                "gradient_checkpointing",
                "enable_async_offloading",
                "enable_activation_offloading",
                "layer_offload_fraction",
                "force_circular_padding",
                "compile",
            ],
        ),
        ("Concepts", ["concept_file_name"]),
        (
            "Data",
            [
                "aspect_ratio_bucketing",
                "latent_caching",
                "clear_cache_before_training",
                "skip_cache_validation",
            ],
        ),
        (
            "Training",
            [
                "learning_rate",
                "epochs",
                "batch_size",
                "gradient_accumulation_steps",
                "learning_rate_scheduler",
                "custom_learning_rate_scheduler",
                "learning_rate_warmup_steps",
                "learning_rate_cycles",
                "learning_rate_min_factor",
                "ema",
                "ema_decay",
                "ema_update_step_interval",
                "dataloader_threads",
                "train_device",
                "temp_device",
                "train_dtype",
                "fallback_train_dtype",
                "enable_autocast_cache",
                "resolution",
                "frames",
                "mse_strength",
                "mae_strength",
                "log_cosh_strength",
                "huber_strength",
                "huber_delta",
                "vb_loss_strength",
                "loss_weight_fn",
                "loss_weight_strength",
                "dropout_probability",
                "loss_scaler",
                "learning_rate_scaler",
                "clip_grad_norm",
            ],
        ),
        ("Layer Filter", ["layer_filter", "layer_filter_preset", "layer_filter_regex"]),
        (
            "Noise",
            [
                "offset_noise_weight",
                "generalized_offset_noise",
                "perturbation_noise_weight",
                "ciop_noise_weight",
                "ciop_p",
                "rescale_noise_scheduler_to_zero_terminal_snr",
                "force_v_prediction",
                "force_epsilon_prediction",
                "timestep_distribution",
                "min_noising_strength",
                "max_noising_strength",
                "noising_weight",
                "noising_bias",
                "timestep_shift",
                "dynamic_timestep_shifting",
            ],
        ),
        (
            "Masked Training",
            [
                "masked_training",
                "unmasked_probability",
                "unmasked_weight",
                "normalize_masked_area_loss",
                "masked_prior_preservation_weight",
                "custom_conditioning_image",
            ],
        ),
        ("Embedding", ["embedding_learning_rate", "preserve_embedding_norm", "embedding_weight_dtype"]),
        (
            "LoRA / OFT",
            [
                "peft_type",
                "lora_model_name",
                "lora_rank",
                "lora_alpha",
                "lora_decompose",
                "lora_decompose_norm_epsilon",
                "lora_decompose_output_axis",
                "lora_weight_dtype",
                "bundle_additional_embeddings",
                "oft_block_size",
                "oft_scaled",
                "oft_clipped_norm",
                "dora_oft",
                "oft_block_share",
            ],
        ),
        ("Optimizer", ["optimizer"]),
        (
            "Sampling",
            [
                "sample_definition_file_name",
                "sample_after",
                "sample_after_unit",
                "sample_skip_first",
                "sample_image_format",
                "sample_video_format",
                "sample_audio_format",
                "samples_to_tensorboard",
                "non_ema_sampling",
            ],
        ),
        (
            "Backup",
            [
                "backup_after",
                "backup_after_unit",
                "rolling_backup",
                "rolling_backup_count",
                "backup_before_save",
                "save_every",
                "save_every_unit",
                "save_skip_first",
                "save_filename_prefix",
            ],
        ),
    ]
)

# Build reverse lookup from field name to section
_FIELD_TO_SECTION: dict[str, str] = {}
for section, fields in SECTION_MAP.items():
    for f in fields:
        _FIELD_TO_SECTION[f] = section


def label_for(field: str) -> str:
    """Human-readable label for a field, falling back to the field name."""
    return DISPLAY_NAMES.get(field, field)


def section_for(field: str) -> str:
    """Section a field belongs to, or 'Other' if not classified."""
    return _FIELD_TO_SECTION.get(field, "Other")


def diff_section_grouped(overrides: dict, defaults: dict) -> dict:
    """Group overrides by section and return {section: [{field, label, current, default}]}.

    Excludes EXCLUDED_FIELDS. Returns empty dict if no overrides.
    """
    grouped: dict[str, list[dict]] = {}
    for field, value in overrides.items():
        if field in EXCLUDED_FIELDS:
            continue
        section = section_for(field)
        grouped.setdefault(section, []).append(
            {
                "field": field,
                "label": label_for(field),
                "current": value,
                "default": defaults.get(field),
            }
        )

    # Re-order sections to match SECTION_MAP order, with 'Other' last
    ordered: dict[str, list[dict]] = {}
    for section in SECTION_MAP:
        if section in grouped:
            ordered[section] = grouped[section]
    if "Other" in grouped:
        ordered["Other"] = grouped["Other"]
    return ordered


def diff_full_config(full_config: dict, defaults: dict) -> dict:
    """Compute overrides as the subset of full_config that differs from defaults.

    Excludes EXCLUDED_FIELDS.
    """
    overrides: dict = {}
    for field, value in full_config.items():
        if field in EXCLUDED_FIELDS:
            continue
        if field not in defaults or defaults.get(field) != value:
            overrides[field] = value
    return overrides
