"""Mechanics for the knowledge-transfer (activation-matching) training mode.

Transfer mode teaches a distilled variant's late blocks the education a
Base-trained OFT / DoRA-OFT adapter carries, without touching the variant's
distillation: the student (variant + conjugated warm-start adapter, only
late-family adapter params trainable) matches its late-stack activations to a
teacher built IN MEMORY from Base's late tensors with the ORIGINAL adapter's
education applied -- ``W_t = m * (W_b @ R^T)`` per output row, the same math
``OFTModule.apply_to_module`` bakes with. No merged checkpoint is ever written.

Because Base and the variants are bit-identical outside the late tensors and
the adapter is identical on early/mid layers, teacher and student activations
coincide up to the fork layer: the transformer forward runs ONCE to the fork,
then the late stack runs twice (teacher weights swapped in / student). The
helpers here are pure and CPU-testable; model wiring lives in the setup
classes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from modules.util.adapter_io import read_layer_weights_buffered, read_safetensors_buffered
from modules.util.oft_align import AdapterOFTConfig, merged_full, reconstruct_oft_rotation
from modules.util.oft_families import DEFAULT_LATE_FAMILY_STEMS

import torch
from torch import nn

_LAYER_KEY_RE = re.compile(r"^(?:transformer\.)?layers\.(\d+)\.(.+)$")


@dataclass
class TransferForkState:
    """Inputs captured at the fork block: everything the late stack needs."""

    block_args: tuple  # (x, *positional extras) exactly as the stack loop passes them
    block_kwargs: dict
    mask: torch.Tensor | None  # per-token validity mask for the loss, when available


def split_layer_key(module_name: str) -> tuple[int, str] | None:
    """(layer index, bare 'layers.N.stem' key) for a layer-stack module path,
    tolerating the optional ``transformer.`` prefix and ``.checkpoint.`` level."""
    m = _LAYER_KEY_RE.match(module_name.replace(".checkpoint.", "."))
    if not m:
        return None
    return int(m.group(1)), f"layers.{m.group(1)}.{m.group(2)}"


def parse_transfer_families(families_csv: str) -> dict[str, tuple[str, ...]]:
    """Subset of DEFAULT_LATE_FAMILY_STEMS selected by a csv like 'qkv,mlp'."""
    names = [n.strip() for n in families_csv.split(",") if n.strip()]
    if not names:
        raise ValueError("transfer_families must name at least one family (e.g. 'qkv,mlp')")
    unknown = [n for n in names if n not in DEFAULT_LATE_FAMILY_STEMS]
    if unknown:
        raise ValueError(f"unknown transfer families {unknown}; available: {sorted(DEFAULT_LATE_FAMILY_STEMS)}")
    return {n: DEFAULT_LATE_FAMILY_STEMS[n] for n in names}


def is_late_family_key(module_name: str, late_start: int, stems: dict[str, tuple[str, ...]]) -> str | None:
    """Family name if the module path is a late-family member, else None.

    Tolerates an optional ``transformer.`` prefix and the ``.checkpoint.``
    level gradient checkpointing splices into module names.
    """
    m = _LAYER_KEY_RE.match(module_name.replace(".checkpoint.", "."))
    if not m or int(m.group(1)) < late_start:
        return None
    stem = m.group(2)
    for fam, fam_stems in stems.items():
        if stem in fam_stems:
            return fam
    return None


def collect_late_weight_modules(
    root: nn.Module, late_start: int, stems: dict[str, tuple[str, ...]]
) -> dict[str, nn.Module]:
    """Late-family Linear modules under ``root``, keyed by normalized relative name."""
    out = {}
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        normalized = name.replace(".checkpoint.", ".")
        if is_late_family_key(normalized, late_start, stems) is not None:
            out[normalized] = module
    return out


def _oft_config_from_metadata(metadata: dict[str, str], path: str) -> AdapterOFTConfig:
    """AdapterOFTConfig from ot_config metadata (mirror of oft_align._read_oft_config,
    but from already-read metadata so the adapter file is never mmapped)."""
    cfg = json.loads(metadata.get("ot_config", "{}"))
    if cfg.get("oft_cans"):
        raise NotImplementedError(f"{path}: transfer teacher does not model the CANS path")
    if cfg.get("oft_block_share"):
        raise NotImplementedError(f"{path}: transfer teacher does not model block_share rotations")
    clip = cfg.get("oft_clipped_norm", None)
    if isinstance(clip, bool):  # legacy bool form of the field
        clip = 0.999 if clip else None
    return AdapterOFTConfig(
        block_size=int(cfg.get("oft_block_size", 128)),
        clip=clip,
        scaled=bool(cfg.get("oft_scaled", False)),
    )


def build_teacher_late_tensors(
    base_ckpt_path: str,
    adapter_path: str,
    late_start: int,
    stems: dict[str, tuple[str, ...]] | None,
    expected_shapes: dict[str, torch.Size] | None = None,
) -> dict[str, torch.Tensor]:
    """The transfer teacher's late weights: Base + ORIGINAL adapter education.

    Reads only what it needs (buffered, no mmap -- safe while training holds
    memory), reconstructs each late rotation, and returns fp32
    ``{bare_stem: W_teacher}``. DoRA multipliers, when present, scale output
    rows exactly as ``DoRAOFTModule.apply_to_module`` does. ``expected_shapes``
    (from the live model's late modules) turns silent coverage/shape drift into
    hard failures.

    ``stems=None`` covers EVERY late adapter-covered module -- what the teacher
    pass needs, since the education must land on non-family adapted modules
    (attention.to_out.0) too, not only the trainable families.
    """
    tensors, metadata = read_safetensors_buffered(adapter_path)
    cfg = _oft_config_from_metadata(metadata, adapter_path)
    stem_names = None if stems is None else {s for fam_stems in stems.values() for s in fam_stems}

    vec_keys: dict[str, str] = {}
    dora_keys: dict[str, str] = {}
    for key in tensors:
        for suffix, target in ((".oft_R.weight", vec_keys), (".dora_multiplier", dora_keys)):
            if not key.endswith(suffix):
                continue
            m = _LAYER_KEY_RE.match(key[: -len(suffix)])
            if m and int(m.group(1)) >= late_start and (stem_names is None or m.group(2) in stem_names):
                target[f"layers.{m.group(1)}.{m.group(2)}"] = key

    if expected_shapes is not None:
        uncovered = sorted(set(expected_shapes) - set(vec_keys))
        if uncovered:
            raise ValueError(
                f"teacher adapter {adapter_path} does not cover late modules {uncovered}; "
                f"the transfer families/late_start must match what the adapter was trained on"
            )

    base_weights = read_layer_weights_buffered(base_ckpt_path, set(vec_keys))

    out: dict[str, torch.Tensor] = {}
    for stem, key in vec_keys.items():
        vec = tensors[key].to(torch.float32)
        w_base = base_weights[stem].to(torch.float32)
        merged = merged_full(w_base, reconstruct_oft_rotation(vec, cfg))
        dora_key = dora_keys.get(stem)
        if dora_key is not None:
            merged = merged * tensors[dora_key].to(torch.float32).view(-1, 1)
        if expected_shapes is not None and stem in expected_shapes:
            if tuple(merged.shape) != tuple(expected_shapes[stem]):
                raise ValueError(
                    f"{stem}: teacher shape {tuple(merged.shape)} does not match the model's "
                    f"module shape {tuple(expected_shapes[stem])}"
                )
        out[stem] = merged
    return out


def build_raw_late_tensors(
    ckpt_path: str,
    stems: set[str],
    expected_shapes: dict[str, torch.Size] | None = None,
) -> dict[str, torch.Tensor]:
    """Raw late weights of a checkpoint, with no adapter education applied.

    The redistill Phase A teacher: the OLD distilled variant's own late tensors
    ARE the target behavior. ``stems`` are bare ``layers.N.stem`` keys (the
    student's late adapter-covered modules); coverage or shape drift is a hard
    failure, same contract as ``build_teacher_late_tensors``.
    """
    weights = read_layer_weights_buffered(ckpt_path, set(stems))  # raises KeyError on a missing stem
    out: dict[str, torch.Tensor] = {}
    for stem in stems:
        tensor = weights[stem].to(torch.float32)
        if expected_shapes is not None and stem in expected_shapes:
            if tuple(tensor.shape) != tuple(expected_shapes[stem]):
                raise ValueError(
                    f"{stem}: teacher shape {tuple(tensor.shape)} does not match the model's "
                    f"module shape {tuple(expected_shapes[stem])}"
                )
        out[stem] = tensor
    return out


@contextmanager
def swap_module_weights(modules: dict[str, nn.Module], weights: dict[str, torch.Tensor]):
    """Temporarily point each module's ``.weight.data`` at a replacement tensor.

    Pure pointer swap -- the replacements must already be on the module's
    device/dtype. Restores in ``finally`` (including on partial application),
    so the model is never left corrupted for later steps, samples, or saves.
    """
    saved: dict[str, torch.Tensor] = {}
    try:
        for name, module in modules.items():
            if name not in weights:
                raise KeyError(f"no replacement weight for late module {name}")
            saved[name] = module.weight.data
            module.weight.data = weights[name]
        yield
    finally:
        for name, data in saved.items():
            modules[name].weight.data = data


class _ForkCaptured(Exception):
    def __init__(self, call_args: tuple, call_kwargs: dict):
        super().__init__("transfer fork captured")
        self.call_args = call_args
        self.call_kwargs = call_kwargs


def run_prefix_and_capture(call_forward: Callable[[], object], fork_block: nn.Module) -> tuple[tuple, dict]:
    """Run ``call_forward`` until ``fork_block`` would execute; return its inputs.

    Uses the codebase's forward-replacement idiom (PeftBase.hook_to_module),
    not register_forward_hook: the block's ``forward`` is swapped for a trap
    that raises out of the model forward, and is restored in ``finally`` --
    leaving no residue on the block.
    """

    def trap(*args, **kwargs):
        raise _ForkCaptured(args, kwargs)

    had_override = "forward" in fork_block.__dict__
    previous = fork_block.__dict__.get("forward")
    fork_block.forward = trap
    try:
        call_forward()
    except _ForkCaptured as captured:
        return captured.call_args, captured.call_kwargs
    finally:
        if had_override:
            fork_block.forward = previous
        else:
            del fork_block.__dict__["forward"]
    raise RuntimeError("model forward completed without reaching the fork block")


def run_late_stack(
    blocks: Sequence[nn.Module],
    x: torch.Tensor,
    extra_args: tuple,
    kwargs: dict,
    checkpointing_fn: Callable | None = None,
) -> list[torch.Tensor]:
    """Run the late blocks sequentially (mirrors the transformer's stack loop),
    returning every block's output for per-block matching."""
    outputs = []
    for block in blocks:
        if checkpointing_fn is not None:
            # diffusers convention: _gradient_checkpointing_func(module, *args)
            x = checkpointing_fn(block, x, *extra_args, **kwargs)
        else:
            x = block(x, *extra_args, **kwargs)
        outputs.append(x)
    return outputs


def transfer_activation_loss(
    student_outs: Sequence[torch.Tensor],
    teacher_outs: Sequence[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1e-8,
    block_index_offset: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-block fp32 MSE between student and teacher late-block outputs.

    ``normalize`` divides each block's MSE by the teacher output power so every
    block contributes on the same scale. ``mask`` (batch, tokens) restricts the
    statistics to valid tokens. Returns (scalar loss, metrics) with per-block
    residuals keyed ``block_{block_index_offset + i}``.
    """
    if len(student_outs) != len(teacher_outs):
        raise ValueError(f"{len(student_outs)} student blocks vs {len(teacher_outs)} teacher blocks")

    metrics: dict[str, float] = {}
    per_block = []
    for i, (student, teacher) in enumerate(zip(student_outs, teacher_outs, strict=True)):
        s32 = student.to(torch.float32)
        t32 = teacher.to(torch.float32)
        if mask is not None:
            m = mask.to(dtype=torch.bool, device=s32.device)
            while m.dim() < s32.dim():
                m = m.unsqueeze(-1)
            count = m.expand_as(s32).sum().clamp_min(1).to(torch.float32)
            mse = ((s32 - t32).pow(2) * m).sum() / count
            teacher_power = (t32.pow(2) * m).sum() / count
        else:
            mse = (s32 - t32).pow(2).mean()
            teacher_power = t32.pow(2).mean()
        value = mse / (teacher_power + eps) if normalize else mse
        per_block.append(value)
        metrics[f"block_{block_index_offset + i}"] = float(value.detach())

    loss = torch.stack(per_block).mean()
    metrics["residual"] = float(loss.detach())
    return loss, metrics


class TransferStopCriterion:
    """Convergence tracker for the overnight run.

    Snapshots the best on EVERY numeric improvement (the final adapter is the
    best-seen, never the last), while patience counts validations without a
    MEANINGFUL improvement -- at least ``min_relative_improvement`` below the
    best. ``update`` returns True when training should stop: patience exhausted
    (if patience > 0) or the absolute ``target_residual`` reached (if > 0).
    """

    def __init__(self, patience: int, min_relative_improvement: float, target_residual: float = 0.0):
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.target_residual = target_residual
        self.best_residual = float("inf")
        self.best_tag = None
        self.counter = 0

    def update(self, residual: float, tag=None) -> bool:
        meaningful = residual < self.best_residual * (1.0 - self.min_relative_improvement)
        if residual < self.best_residual:
            self.best_residual = residual
            self.best_tag = tag
        self.counter = 0 if meaningful else self.counter + 1
        if 0.0 < self.target_residual >= residual:
            return True
        return self.patience > 0 and self.counter >= self.patience
