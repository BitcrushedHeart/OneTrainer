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
    base_recovery: dict[str, tuple[torch.Tensor, torch.Tensor | None]] | None = None,
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

    ``base_recovery``, when given, is filled with what ``recover_base_weight``
    needs to turn each teacher tensor back into the raw Base one, so the raw
    Base late stack can be run without holding a second copy of the weights.
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
        rotation = reconstruct_oft_rotation(vec, cfg)
        merged = merged_full(w_base, rotation)
        dora_key = dora_keys.get(stem)
        multiplier = None if dora_key is None else tensors[dora_key].to(torch.float32)
        if multiplier is not None:
            merged = merged * multiplier.view(-1, 1)
        if base_recovery is not None:
            # Inverse rotation minus identity: small entries, so it survives a
            # low-precision cast far better than the near-identity inverse would.
            identity = torch.eye(rotation.shape[-1], dtype=rotation.dtype)
            base_recovery[stem] = (torch.linalg.inv(rotation) - identity, multiplier)
        if expected_shapes is not None and stem in expected_shapes:
            if tuple(merged.shape) != tuple(expected_shapes[stem]):
                raise ValueError(
                    f"{stem}: teacher shape {tuple(merged.shape)} does not match the model's "
                    f"module shape {tuple(expected_shapes[stem])}"
                )
        out[stem] = merged
    return out


def recover_base_weight(
    teacher_weight: torch.Tensor,
    inverse_rotation_delta: torch.Tensor,
    multiplier: torch.Tensor | None,
) -> torch.Tensor:
    """Raw Base weight from the education-baked teacher weight.

    Inverts ``W_t = m * (W_b @ R^T)``: ``W_b = (W_t / m) @ R^-T``, with the
    inverse rotation passed as ``R^-1 - I`` (see ``build_teacher_late_tensors``).
    Computed in fp32 and returned in the teacher tensor's dtype, so it can be
    swapped into the same module. One module at a time this is a few
    milliseconds and no standing memory, where a stored copy of Base's late
    tensors costs as much as the teacher itself.
    """
    w = teacher_weight.to(torch.float32)
    if multiplier is not None:
        w = w / multiplier.to(device=w.device, dtype=torch.float32).view(-1, 1)
    out, inf = w.shape
    delta = inverse_rotation_delta.to(device=w.device, dtype=torch.float32)
    wr = w.reshape(out, delta.shape[0], delta.shape[1])
    wr = wr + torch.einsum("orc,rkc->ork", wr, delta)
    return wr.reshape(out, inf).to(teacher_weight.dtype)


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
    that raises out of the model forward. A Module.compile entry point on the
    fork block is temporarily bypassed so Dynamo never traces the exception.
    Both are restored in ``finally`` before any teacher/student late pass.
    """

    def trap(*args, **kwargs):
        raise _ForkCaptured(args, kwargs)

    had_override = "forward" in fork_block.__dict__
    previous = fork_block.__dict__.get("forward")
    compiled_call = fork_block._compiled_call_impl
    fork_block.forward = trap
    try:
        # Module.compile wraps __call__, so replacing forward alone still
        # sends the trap through Dynamo (and fullgraph rejects the exception).
        # Keep compilation enabled for the prefix and subsequent late passes.
        if compiled_call is not None:
            fork_block._compiled_call_impl = None
        call_forward()
    except _ForkCaptured as captured:
        return captured.call_args, captured.call_kwargs
    finally:
        if had_override:
            fork_block.forward = previous
        else:
            del fork_block.__dict__["forward"]
        if compiled_call is not None:
            fork_block._compiled_call_impl = compiled_call
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


def add_distillation_offset(
    teacher_outs: Sequence[torch.Tensor],
    variant_outs: Sequence[torch.Tensor],
    base_outs: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Distillation-preserving targets: ``teacher + (variant - base)`` per block.

    The plain teacher (Base + education) carries none of the variant's
    distillation, so matching it exactly un-distills the late stack. The raw
    variant and raw Base late stacks, run from the same fork inputs, differ by
    exactly that distillation; adding the difference back makes a zero residual
    mean "educated AND still distilled". fp32, like the loss.
    """
    if not (len(teacher_outs) == len(variant_outs) == len(base_outs)):
        raise ValueError(f"{len(teacher_outs)} teacher vs {len(variant_outs)} variant vs {len(base_outs)} base blocks")
    return [
        t.to(torch.float32) + (v.to(torch.float32) - b.to(torch.float32))
        for t, v, b in zip(teacher_outs, variant_outs, base_outs, strict=True)
    ]


def transfer_education_gaps(
    teacher_outs: Sequence[torch.Tensor],
    base_outs: Sequence[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> list[float]:
    """Per block, the size of the late-stack education's own effect:
    ``|teacher - base|^2 / |teacher|^2``, teacher being Base + education and
    base raw Base, both from the same fork inputs. It is the yardstick for
    ``education_residual``: their ratio is the fraction of that effect the
    student is missing, where the residual alone is relative to the whole
    activation."""
    if len(teacher_outs) != len(base_outs):
        raise ValueError(f"{len(teacher_outs)} teacher blocks vs {len(base_outs)} base blocks")
    gaps = []
    with torch.no_grad():
        for teacher, base in zip(teacher_outs, base_outs, strict=True):
            t32 = teacher.detach().to(torch.float32)
            g = t32 - base.detach().to(torch.float32)
            if mask is not None:
                m = mask.to(dtype=torch.bool, device=g.device)
                while m.dim() < g.dim():
                    m = m.unsqueeze(-1)
                g, t32 = g * m, t32 * m
            gaps.append(float((g * g).sum(dtype=torch.float64) / ((t32 * t32).sum(dtype=torch.float64) + eps)))
    return gaps


def transfer_turbo_metrics(
    student_outs: Sequence[torch.Tensor],
    teacher_outs: Sequence[torch.Tensor],
    variant_outs: Sequence[torch.Tensor],
    base_outs: Sequence[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
    block_index_offset: int = 0,
    education_gaps: Sequence[float] | None = None,
) -> dict[str, float]:
    """How much of the variant's distillation the student still carries.

    Per block, ``d = variant - base`` is the distillation direction and
    ``e = student - teacher`` the student's offset from the plain teacher:

    * ``turbo_retention`` = <e, d> / |d|^2. 1.0: the student is the teacher plus
      the full distillation offset. 0.0: it has collapsed onto the teacher,
      i.e. the late stack is un-distilled.
    * ``education_residual`` = |e - retention * d|^2 / |teacher|^2, the part of
      the teacher mismatch that is NOT distillation -- the education error.
    * ``turbo_gap`` = |d|^2 / |teacher|^2, the size of the distillation offset
      on the residual's scale (a reference level, roughly constant).
    * ``teacher_residual`` = |e|^2 / |teacher|^2, the plain teacher-match
      residual whatever target the loss trained against.

    Each is the mean over blocks; retention is also reported per block as
    ``turbo_retention_block_{n}``. Pure diagnostics: everything is detached.

    With ``education_gaps`` (see ``transfer_education_gaps``) two more:
    ``education_gap``, their mean, and ``education_missing``, the mean per-block
    ``education_residual / education_gap`` -- the fraction of the late-stack
    education's effect the student does not reproduce (1.0: none of it).
    """
    if not (len(student_outs) == len(teacher_outs) == len(variant_outs) == len(base_outs)):
        raise ValueError("student, teacher, variant and base block counts differ")
    if education_gaps is not None and len(education_gaps) != len(student_outs):
        raise ValueError(f"{len(education_gaps)} education gaps for {len(student_outs)} blocks")

    metrics: dict[str, float] = {}
    sums = {"turbo_retention": 0.0, "education_residual": 0.0, "turbo_gap": 0.0, "teacher_residual": 0.0}
    if education_gaps is not None:
        sums.update(education_gap=0.0, education_missing=0.0)
    with torch.no_grad():
        for i, (student, teacher, variant, base) in enumerate(
            zip(student_outs, teacher_outs, variant_outs, base_outs, strict=True)
        ):
            t32 = teacher.detach().to(torch.float32)
            e = student.detach().to(torch.float32) - t32
            d = variant.detach().to(torch.float32) - base.detach().to(torch.float32)
            if mask is not None:
                m = mask.to(dtype=torch.bool, device=e.device)
                while m.dim() < e.dim():
                    m = m.unsqueeze(-1)
                e, d, t32 = e * m, d * m, t32 * m
            # float64 accumulation without float64 copies of the activations;
            # the ratios are scale-free, so no per-count normalization.
            ee = (e * e).sum(dtype=torch.float64)
            dd = (d * d).sum(dtype=torch.float64)
            ed = (e * d).sum(dtype=torch.float64)
            tt = (t32 * t32).sum(dtype=torch.float64)
            retention = float(ed / (dd + eps))
            metrics[f"turbo_retention_block_{block_index_offset + i}"] = retention
            sums["turbo_retention"] += retention
            education_residual = float((ee - ed * ed / (dd + eps)).clamp_min(0.0) / (tt + eps))
            sums["education_residual"] += education_residual
            if education_gaps is not None:
                sums["education_gap"] += education_gaps[i]
                sums["education_missing"] += education_residual / (education_gaps[i] + eps)
            sums["turbo_gap"] += float(dd / (tt + eps))
            sums["teacher_residual"] += float(ee / (tt + eps))
    for key, value in sums.items():
        metrics[key] = value / max(1, len(student_outs))
    return metrics


def high_frequency_ratio(prediction: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> float:
    """High-frequency energy of ``prediction`` relative to ``reference``.

    Both are (batch, channels, H, W) one-step clean-latent predictions from the
    same noisy input. Energy is the mean squared difference from a 3x3 local
    mean. A distilled model predicts a sharp latent from heavy noise where an
    un-distilled one predicts a blurry average, so a ratio well below 1.0
    against the raw variant means the student has lost that sharpness.
    """

    def energy(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().to(torch.float32)
        local_mean = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        return (x - local_mean).pow(2).mean()

    return float(energy(prediction) / (energy(reference) + eps))


class TransferStopCriterion:
    """Convergence tracker for the overnight run.

    Snapshots the best on EVERY numeric improvement (the final adapter is the
    best-seen, never the last), while patience counts validations without a
    MEANINGFUL improvement -- at least ``min_relative_improvement`` below the
    best. ``update`` returns True when training should stop: patience exhausted
    (if patience > 0) or the absolute ``target_residual`` reached (if > 0).

    ``guard_ok=False`` marks a validation that tripped a de-distillation guard:
    it can never become the best, counts against patience whatever its
    residual, and cannot satisfy ``target_residual``.
    """

    def __init__(self, patience: int, min_relative_improvement: float, target_residual: float = 0.0):
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.target_residual = target_residual
        self.best_residual = float("inf")
        self.best_tag = None
        self.counter = 0

    def update(self, residual: float, tag=None, *, guard_ok: bool = True) -> bool:
        if not guard_ok:
            self.counter += 1
            return self.patience > 0 and self.counter >= self.patience
        meaningful = residual < self.best_residual * (1.0 - self.min_relative_improvement)
        if residual < self.best_residual:
            self.best_residual = residual
            self.best_tag = tag
        self.counter = 0 if meaningful else self.counter + 1
        if 0.0 < self.target_residual >= residual:
            return True
        return self.patience > 0 and self.counter >= self.patience
