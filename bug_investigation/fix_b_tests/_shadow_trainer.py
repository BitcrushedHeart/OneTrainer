"""
Shadow trainer for Fix B testing.

Mirrors the relevant slice of `modules/trainer/GenericTrainer.train()` --
specifically the gradient-accumulation accounting, optimizer-step
boundary detection, and loss-logging path (GenericTrainer.py:1037-1102
and TimedActionMixin.py:45) -- without importing OneTrainer's full
model/saver/loader machinery.

Each instance has two modes controlled by `enable_fix`:

  enable_fix=False  (mirrors *current* production behavior)
      save: model weights, optimizer state, global_step only
      load: restore those; `.grad` is None, accumulators reset to 0

  enable_fix=True  (mirrors *Fix B* behavior)
      save: also `accumulated_loss`, `param_grads` by name,
            `scaler.state_dict()`, RNG snapshots, fingerprint
      load: restore all of the above

This lets one test prove both halves: the bug reproduces with
enable_fix=False, and is *bit-identical*-fixed with enable_fix=True.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# ----------------------- model + data ----------------------------

def make_seeded_model(seed: int, *, freeze_first_n: int = 0) -> nn.Module:
    """Tiny deterministic MLP.

    If freeze_first_n > 0, the first N Linear layers have requires_grad=False
    (simulating a frozen base with a trainable head -- the adapter case).
    """
    torch.manual_seed(seed)
    layers = [
        nn.Linear(8, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    ]
    model = nn.Sequential(*layers)
    if freeze_first_n > 0:
        # Walk Linear modules in order; freeze the first N.
        frozen = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if frozen < freeze_first_n:
                    for p in m.parameters():
                        p.requires_grad = False
                    frozen += 1
                else:
                    break
    return model


def make_batches(n: int, *, seed: int, batch_size: int = 4):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n):
        x = torch.randn(batch_size, 8, generator=g)
        y = (x.sum(dim=1, keepdim=True) * 0.3).tanh()
        batches.append((x, y))
    return batches


# ----------------------- fingerprint helper ------------------------

def compute_concept_fingerprint(concepts: list[dict]) -> tuple[str, int]:
    """Mirrors what the production helper does.

    Takes a list of dicts each containing stable identifiers
    (name, path, seed, type, include_subdirectories).  Returns
    (sha256_hex, concept_count).
    """
    payload = sorted(
        [
            (c.get('name', ''), c.get('path', ''), int(c.get('seed', 0)),
             c.get('type', ''), bool(c.get('include_subdirectories', False)))
            for c in concepts
        ],
        key=lambda t: t[1],   # sort by path
    )
    blob = json.dumps(payload, separators=(',', ':'), sort_keys=False).encode()
    return hashlib.sha256(blob).hexdigest(), len(payload)


# ----------------------- shadow trainer ----------------------------

@dataclass
class ShadowState:
    """State the trainer owns across stop/resume."""
    accumulated_loss: float = 0.0
    accumulated_dpo_metrics: dict | None = None
    ema_loss: float | None = None
    ema_loss_steps: int = 0


@dataclass
class TrainResult:
    """Returned by run_epoch."""
    logs: list[tuple[int, float]] = field(default_factory=list)
    final_global_step: int = 0


def _is_update_step(global_step: int, accumulation_steps: int) -> bool:
    """Mirror of TimedActionMixin.py:45 with start_at_zero=False."""
    return (global_step + 1) % accumulation_steps == 0


class ShadowTrainer:
    """Mirrors GenericTrainer.train()'s loop for the non-DPO path."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        accumulation_steps: int,
        enable_fix: bool,
        concepts: list[dict] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_fix = enable_fix
        self.concepts = concepts or []

        # Function-locals in the real trainer; promoted here so save/load can reach them.
        self.state = ShadowState()
        self.global_step = 0
        self.has_gradient = False
        # No real scaler on CPU; keep a None placeholder so save/load paths can
        # exercise the "scaler is None" branch correctly.
        self.scaler = None

        self.loss_fn = nn.MSELoss()

    def run_epoch(
        self,
        batches: list,
        *,
        stop_after_step: int | None = None,
    ) -> TrainResult:
        """Run one epoch (or partial epoch up to stop_after_step).

        Mirrors GenericTrainer.train()'s inner for-batch loop on the
        non-DPO path.  See file-level docstring for line refs.
        """
        logs: list[tuple[int, float]] = []
        accumulated_loss = torch.tensor(
            self.state.accumulated_loss, dtype=torch.float32
        )

        for x, y in batches:
            # Forward + scaled backward
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss = loss / self.accumulation_steps
            loss.backward()

            self.has_gradient = True
            accumulated_loss = accumulated_loss + loss.detach()

            if _is_update_step(self.global_step, self.accumulation_steps):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.has_gradient = False

                logged = float(accumulated_loss.item())
                logs.append((self.global_step, logged))
                accumulated_loss = torch.tensor(0.0, dtype=torch.float32)
                self.state.accumulated_loss = 0.0
            else:
                self.state.accumulated_loss = float(accumulated_loss.item())

            self.global_step += 1

            if stop_after_step is not None and self.global_step > stop_after_step:
                break

        # Persist the final loop-local accumulator value back to state for save.
        if self.has_gradient:
            self.state.accumulated_loss = float(accumulated_loss.item())

        return TrainResult(logs=logs, final_global_step=self.global_step)

    # ------------------- save / load --------------------

    def save_to(self, dest_dir: str) -> None:
        """Write the trainer's snapshot to disk.

        Always written:
          model.pt                     -- model.state_dict()
          optimizer.pt                 -- optimizer.state_dict()
          meta.json                    -- global_step + acc_steps fingerprint

        Only when enable_fix=True:
          accumulator/accumulator.pt   -- per-Fix-B contract
        """
        os.makedirs(dest_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dest_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(dest_dir, "optimizer.pt"))

        meta = {
            "global_step": self.global_step,
            "accumulation_steps": self.accumulation_steps,
        }
        with open(os.path.join(dest_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        if not self.enable_fix:
            return

        # --- Fix B payload ---
        param_grads: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                param_grads[name] = p.grad.detach().to(
                    device="cpu", copy=True
                )

        rng = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,   # CPU tests only
            "python": random.getstate(),
            "numpy": np.random.get_state(legacy=True),
        }

        fp_hash, fp_count = compute_concept_fingerprint(self.concepts)
        payload = {
            "accumulated_loss": self.state.accumulated_loss,
            "accumulated_dpo_metrics": self.state.accumulated_dpo_metrics,
            "param_grads": param_grads,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "rng": rng,
            "fingerprint": {
                "gradient_accumulation_steps": self.accumulation_steps,
                "dataset_hash": fp_hash,
                "concept_count": fp_count,
            },
        }
        os.makedirs(os.path.join(dest_dir, "accumulator"), exist_ok=True)
        torch.save(payload, os.path.join(dest_dir, "accumulator", "accumulator.pt"))

    def load_from(
        self,
        src_dir: str,
        *,
        warn_callback=None,    # invoked as warn_callback(msg) for any mismatch
    ) -> None:
        """Restore state.

        Symmetric to save_to.  If accumulator/accumulator.pt is absent
        (legacy or enable_fix=False save), the partial-accumulator state
        stays at defaults (0.0, no grads) -- exactly today's behavior.
        """
        self.model.load_state_dict(
            torch.load(os.path.join(src_dir, "model.pt"), weights_only=True)
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(src_dir, "optimizer.pt"), weights_only=True)
        )
        with open(os.path.join(src_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        self.global_step = meta["global_step"]

        acc_path = os.path.join(src_dir, "accumulator", "accumulator.pt")
        if not os.path.exists(acc_path):
            # Legacy / unfixed save -- nothing to restore.
            return

        payload = torch.load(acc_path, weights_only=False)

        # Warn-only fingerprint check.
        fp = payload.get("fingerprint", {})
        current_hash, current_count = compute_concept_fingerprint(self.concepts)
        saved_acc = fp.get("gradient_accumulation_steps")
        if saved_acc is not None and saved_acc != self.accumulation_steps:
            msg = (
                f"gradient_accumulation_steps mismatch: saved={saved_acc} "
                f"current={self.accumulation_steps}; restoring partial state anyway"
            )
            (warn_callback or print)(msg)
        if fp.get("dataset_hash") and fp.get("dataset_hash") != current_hash:
            delta = current_count - fp.get("concept_count", current_count)
            msg = (
                f"dataset fingerprint mismatch: saved_concepts={fp.get('concept_count')} "
                f"current_concepts={current_count} (delta={delta}); "
                f"restoring partial state anyway"
            )
            (warn_callback or print)(msg)

        # Restore accumulator state.
        self.state.accumulated_loss = payload.get("accumulated_loss", 0.0)
        self.state.accumulated_dpo_metrics = payload.get("accumulated_dpo_metrics")

        # Restore per-parameter grads.
        saved_grads = payload.get("param_grads", {})
        current_param_names = {n for n, _ in self.model.named_parameters()}
        missing_in_current = [k for k in saved_grads if k not in current_param_names]
        if missing_in_current and len(missing_in_current) / max(len(saved_grads), 1) > 0.10:
            msg = (
                f"{len(missing_in_current)} of {len(saved_grads)} saved grad keys "
                f"have no matching parameter in the current model; skipping"
            )
            (warn_callback or print)(msg)
        applied = 0
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name in saved_grads:
                p.grad = saved_grads[name].to(p.device, p.dtype)
                applied += 1
            else:
                p.grad = None
        self.has_gradient = applied > 0

        # Scaler (None on CPU).
        if self.scaler is not None and payload.get("scaler") is not None:
            self.scaler.load_state_dict(payload["scaler"])

        # RNG -- restore all four.
        rng = payload.get("rng", {})
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        # torch_cuda left for GPU tests
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])


# ----------------------- helpers -----------------------------------

def fresh_optimizer(model: nn.Module, lr: float = 1e-2, momentum: float = 0.9):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(trainable, lr=lr, momentum=momentum)


def snapshot_model_state(model: nn.Module) -> dict:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
