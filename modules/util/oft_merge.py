"""Orchestrate baking an OFT / DoRA-OFT adapter into a base model checkpoint.

Routes through OneTrainer's training math (``OFTModule.apply_to_module``) so
the merged checkpoint is bit-equivalent to what training would produce at
strength=1.0. Hard-fails on any verification gate: orthogonality drift,
NaN/Inf, new dead rows, or DoRA invariant violation.

The training-time TrainConfig is reconstructed from the adapter's
``ot_config`` metadata, then base_model_name / transformer_model_name /
vae_model_name are inherited from the active Train UI's TrainConfig
(the same fields populated by the Model tab). This guarantees the merge
sees the exact same load + wrapper-creation state training would.
"""

from __future__ import annotations

import gc
import json
import os

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
from modules.util.create import create_model_loader, create_model_saver
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.oft_verify import (
    MergeReport,
    MergeVerificationError,
    VerificationFailure,
    collect_oft_modules,
    count_zero_rows,
    gate_orthogonality,
    gate_post_merge,
)

import torch

from safetensors import safe_open


def _train_config_from_adapter(oft_adapter_path: str) -> TrainConfig:
    """Reconstruct the training-time TrainConfig from the adapter's ot_config.

    This is the cleanest way to mirror what training saw: every hyperparameter
    (peft_type, layer_filter, oft_block_size, dora_oft, ...) is restored to
    its training value via ``from_dict``. Path fields are then overridden by
    the caller with the merge-time selections.
    """
    with safe_open(oft_adapter_path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
    raw = md.get("ot_config")
    if not raw:
        raise ValueError(
            f"Adapter at {oft_adapter_path} is missing ot_config metadata. Cannot determine training-time TrainConfig."
        )
    cfg_dict = json.loads(raw)

    train_config = TrainConfig.default_values()
    train_config.from_dict(cfg_dict)
    return train_config


def merge_oft_adapter(
    oft_adapter_path: str,
    output_path: str,
    output_dtype: DataType,
    output_format: ModelFormat,
    *,
    ui_train_config: TrainConfig,
    strength: float = 1.0,
    compute_device: torch.device | str | None = None,
) -> MergeReport:
    """Bake an OFT/DoRA-OFT adapter into a base checkpoint.

    ``ui_train_config`` is the live TrainConfig from the active Train UI.
    The merge inherits the user's currently-selected base_model_name,
    transformer_model_name, and vae_model_name from it -- so the user
    picks the merge target in the Model tab and opens this tool with
    those settings already in place.

    ``strength`` interpolates between the base and the fully-baked weights
    (same semantics as ComfyUI's LoRA strength). DoRA row-norm invariant
    check is skipped at strength != 1.0.
    """
    if compute_device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(compute_device)

    if not os.path.exists(oft_adapter_path):
        raise FileNotFoundError(f"OFT adapter not found: {oft_adapter_path}")

    train_config = _train_config_from_adapter(oft_adapter_path)
    model_type = train_config.model_type

    if train_config.peft_type != PeftType.OFT_2:
        raise ValueError(f"Adapter peft_type is {train_config.peft_type}, expected OFT_2")
    if output_format is not ModelFormat.SAFETENSORS:
        raise NotImplementedError(f"Output format {output_format} not supported yet. Use SAFETENSORS.")

    # Inherit the merge target from the live Train UI: same Model tab fields
    # the user fills in for training. The training-time values in ot_config
    # would point at HF repos or training paths that may or may not exist now.
    # Modern TrainConfig nests per-component model names under sub-configs
    # (transformer.model_name, vae.model_name); only base_model_name is top-level.
    train_config.base_model_name = ui_train_config.base_model_name
    train_config.transformer.model_name = ui_train_config.transformer.model_name
    train_config.vae.model_name = ui_train_config.vae.model_name
    train_config.lora_model_name = oft_adapter_path
    train_config.distillation_teacher_lora_model_name = ""
    train_config.train_device = str(compute_device)
    train_config.temp_device = str(compute_device)

    if not train_config.base_model_name:
        raise ValueError(
            "Base Model not set in the Train UI's Model tab. The merge tool "
            "inherits that field so it knows where to load the tokenizer / "
            "text encoder / VAE / scheduler from (e.g. 'Tongyi-MAI/Z-Image')."
        )

    # Compare against the output path so we don't accidentally overwrite the
    # transformer the user is merging into.
    overwrite_candidates = [
        ("base_model_name", train_config.base_model_name),
        ("transformer.model_name", train_config.transformer.model_name),
    ]
    for label, path in overwrite_candidates:
        if path and os.path.exists(path) and os.path.abspath(path) == os.path.abspath(output_path):
            raise ValueError(f"Output path must differ from {label} ({path}). Refusing to overwrite.")

    loader = create_model_loader(model_type, TrainingMethod.LORA)
    if loader is None:
        raise RuntimeError(f"No LoRA model loader registered for {model_type}")

    print(f"[merge_oft] loading {model_type.name}: base={train_config.base_model_name!r}")
    if train_config.transformer.model_name:
        print(f"[merge_oft]   transformer override={train_config.transformer.model_name!r}")

    model = loader.load(
        model_type=model_type,
        model_names=ModelNames(
            base_model=train_config.base_model_name,
            transformer_model=train_config.transformer.model_name,
            vae_model=train_config.vae.model_name,
            lora=oft_adapter_path,
        ),
        weight_dtypes=ModelWeightDtypes.from_single_dtype(output_dtype),
        quantization=QuantizationConfig.default_values(),
    )

    if model.lora_state_dict is None:
        raise RuntimeError(f"Loader did not populate lora_state_dict for {oft_adapter_path}")

    # Reproduce just the wrapper-creation portion of <ModelType>LoRASetup.setup_model:
    # build the LoRAModuleWrapper against the loaded transformer with the
    # training-exact TrainConfig, load the adapter into it, hook. We skip the
    # rest of setup_model (parameter group collection, requires_grad, optimizer
    # init) since none of it is needed to bake the weights.
    print("[merge_oft] wiring adapter to transformer")
    model.transformer_lora = LoRAModuleWrapper(
        model.transformer,
        "transformer",
        train_config,
        train_config.layer_filter.split(",") if train_config.layer_filter else None,
    )
    model.transformer_lora.load_state_dict(model.lora_state_dict)
    model.lora_state_dict = None
    model.transformer_lora.set_dropout(train_config.dropout_probability)
    model.transformer_lora.to(dtype=train_config.lora_weight_dtype.torch_dtype())
    model.transformer_lora.hook_to_module()

    adapters = model.adapters()
    if not adapters:
        raise RuntimeError("No adapters wired after setup; nothing to merge")

    # Move transformer + wrapper onto the compute device.
    print(f"[merge_oft] moving compute to {compute_device}")
    if hasattr(model, "transformer_to"):
        model.transformer_to(compute_device)
    else:
        for wrapper in adapters:
            wrapper.to(compute_device)

    # -------- pre-merge gates --------
    print("[merge_oft] pre-merge gates: orthogonality + baseline zero-row count")
    pre_failures: list = []
    pre_zero_rows: dict[str, int] = {}
    base_row_norms: dict[str, torch.Tensor] = {}
    ortho_max_by_key: dict[str, float] = {}
    base_snapshots: dict[str, torch.Tensor] = {}

    for wrapper in adapters:
        oft_modules, base_modules, _ = collect_oft_modules(wrapper)
        failures, max_err = gate_orthogonality(oft_modules)
        pre_failures.extend(failures)
        ortho_max_by_key.update(max_err)
        for key, base_module in base_modules.items():
            w = base_module.weight.data
            pre_zero_rows[key] = count_zero_rows(w)
            # Pre-merge per-output-row L2 norm -- baked DoRA-OFT invariant is
            # ||merged_row|| == |dora_multiplier| * ||base_row||.
            base_row_norms[key] = w.reshape(w.shape[0], -1).to(torch.float32).norm(dim=1)
            if strength != 1.0:
                # Keep the pre-merge snapshot on CPU -- holding all base weights on
                # the compute device doubles VRAM (~2x model size) and OOMs the bake.
                base_snapshots[key] = w.detach().to("cpu", copy=True)

    if pre_failures:
        raise MergeVerificationError(pre_failures)

    # -------- apply --------
    print(f"[merge_oft] applying adapter to {len(pre_zero_rows)} module(s) at strength={strength}")
    for wrapper in adapters:
        wrapper.apply_to_module()

    if strength != 1.0:
        for wrapper in adapters:
            _, base_modules, _ = collect_oft_modules(wrapper)
            for key, base_module in base_modules.items():
                merged = base_module.weight.data
                # Move the CPU snapshot back one layer at a time (minimal extra VRAM).
                base = base_snapshots[key].to(merged.device, torch.float32)
                mixed = (1.0 - strength) * base + strength * merged.to(torch.float32)
                base_module.weight.data.copy_(mixed.to(merged.dtype))
        base_snapshots.clear()

    # -------- post-merge gates --------
    print("[merge_oft] post-merge gates: NaN/Inf, dead-row delta, DoRA invariant")
    all_base_modules: dict = {}
    all_dora_multipliers: dict = {}
    for wrapper in adapters:
        _, base_modules, dora_multipliers = collect_oft_modules(wrapper)
        all_base_modules.update(base_modules)
        all_dora_multipliers.update(dora_multipliers)

    # DoRA invariant only holds at full strength (strength<1 blends baked with
    # base, breaking ||merged_row|| == |mult|*||base_row||).
    dora_mult_for_gate = all_dora_multipliers if strength == 1.0 else None
    base_norms_for_gate = base_row_norms if strength == 1.0 else None
    post_failures, report = gate_post_merge(all_base_modules, pre_zero_rows, dora_mult_for_gate, base_norms_for_gate)
    report.orthogonality_max_err = ortho_max_by_key
    if post_failures:
        raise MergeVerificationError(post_failures)

    # Move back to CPU for the saver.
    if compute_device.type != "cpu":
        print("[merge_oft] moving compute back to cpu for save")
        if hasattr(model, "transformer_to"):
            model.transformer_to(torch.device("cpu"))
        else:
            for wrapper in adapters:
                wrapper.to(torch.device("cpu"))

    # Unhook so the saved weights are pure base + baked OFT.
    for wrapper in adapters:
        wrapper.remove_hook_from_module()

    # Clear adapter so FINE_TUNE saver doesn't try to serialize it.
    if hasattr(model, "transformer_lora"):
        model.transformer_lora = None

    # -------- save --------
    print(f"[merge_oft] saving merged checkpoint to {output_path}")
    saver = create_model_saver(model_type, TrainingMethod.FINE_TUNE)
    if saver is None:
        raise RuntimeError(f"No FINE_TUNE saver registered for {model_type}")
    saver.save(
        model=model,
        model_type=model_type,
        output_model_format=output_format,
        output_model_destination=output_path,
        dtype=output_dtype.torch_dtype(),
    )

    # -------- final sanity: re-read and NaN-scan the saved file --------
    # Free the in-memory model first: the merged weights are already on disk, and
    # holding the full transformer in RAM while memory-mapping the multi-GB output
    # for read-back exhausts the Windows commit limit (paging-file OSError 1455).
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[merge_oft] final sanity scan of saved file")
    if output_format is ModelFormat.SAFETENSORS:
        with safe_open(output_path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118  -- safe_open is not a dict
                t = f.get_tensor(key)
                if torch.isnan(t).any() or torch.isinf(t).any():
                    raise MergeVerificationError(
                        [
                            VerificationFailure(
                                gate="post_save_nan_inf",
                                location=key,
                                detail="NaN/Inf detected in saved file",
                            )
                        ]
                    )

    print("[merge_oft] done")
    return report
