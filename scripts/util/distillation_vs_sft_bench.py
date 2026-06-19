"""Head-to-head: distillation vs simulated SFT, same student & schedule.

Trains two identical DoRA-OFT students on the same tiny model. The only
difference is the supervision signal:

* **Distillation** (the path the new code adds): student MSE against the
  teacher's clean output.
* **SFT** (the analogue of training a fresh DoRA-OFT on images): student MSE
  against ``teacher(x) + per_sample_label_noise``. The label noise stands in
  for the single-step Monte-Carlo error you get when comparing student noise
  prediction to actually-sampled noise on a real image.

Both methods are evaluated on the *clean* teacher target on a held-out batch,
so the loss numbers are directly comparable. Wall-clock per step is timed
separately so we can quote both convergence-rate and per-step cost.

Run from project root::

    venv/Scripts/python.exe scripts/util/distillation_vs_sft_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules.util.create  # noqa: F401  -- import-side-effect required
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
import torch.nn.functional as F

sys.path.insert(0, str(PROJECT_ROOT / "tests" / "util"))
import test_distillation as td  # noqa: E402

HIDDEN = 16
BATCH = 4
LR = 5e-3
STEPS = 400
LABEL_NOISE_STD = 0.10  # SFT supervision noise; tuned so SNR ~ teacher's delta-from-base


def _build_pair(seed: int):
    """Return (net, student, teacher, setup, model, config) reproducibly."""
    torch.manual_seed(seed)
    net = td._TinyTransformer(hidden=HIDDEN)
    student = LoRAModuleWrapper(net, "transformer", td._doraoft_config(), [])
    teacher = LoRAModuleWrapper(net, "transformer", td._dora_config(), [])
    td._randomize_dora_teacher(teacher)
    for p in net.parameters():
        p.requires_grad_(False)
    teacher.requires_grad_(False)
    student.hook_to_module()

    setup = td._StubSetup()
    model = td._MockModel(student_wrapper=student, teacher_wrapper=teacher)
    config = SimpleNamespace(training_method=TrainingMethod.LORA)
    return net, student, teacher, setup, model, config


def _clean_eval_loss(net, setup, model, config, x_eval) -> float:
    """Same loss we use in the test: clean teacher target."""
    with setup.distillation_teacher_model(model, config), torch.no_grad():
        target = net(x_eval).detach()
    pred = net(x_eval)
    return F.mse_loss(pred, target).item()


def run_distillation(seed: int) -> tuple[list[float], float]:
    net, student, teacher, setup, model, config = _build_pair(seed)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=LR)

    torch.manual_seed(999)
    x_eval = torch.randn(BATCH, HIDDEN)
    losses = [_clean_eval_loss(net, setup, model, config, x_eval)]

    torch.manual_seed(777)
    step_times = []
    for _ in range(STEPS):
        x = torch.randn(BATCH, HIDDEN)
        t0 = time.perf_counter()
        with setup.distillation_teacher_model(model, config), torch.no_grad():
            target = net(x).detach()
        pred = net(x)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - t0)
        losses.append(_clean_eval_loss(net, setup, model, config, x_eval))

    student.remove_hook_from_module()
    return losses, sum(step_times) / len(step_times)


def run_sft(seed: int) -> tuple[list[float], float]:
    """Same student, same schedule -- but the target is teacher(x) corrupted
    by per-sample iid Gaussian noise. That's the toy-model analogue of the
    single-step MC noise you get when training on real images."""
    net, student, teacher, setup, model, config = _build_pair(seed)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=LR)

    torch.manual_seed(999)
    x_eval = torch.randn(BATCH, HIDDEN)
    losses = [_clean_eval_loss(net, setup, model, config, x_eval)]

    torch.manual_seed(777)
    step_times = []
    for _ in range(STEPS):
        x = torch.randn(BATCH, HIDDEN)
        t0 = time.perf_counter()
        # The SFT path has only ONE forward (no teacher pass at train time).
        # In the real diffusion analogue, the supervision noise is what the
        # student is asked to predict given (x_t, t, c). We simulate that by
        # corrupting the teacher target with per-sample noise.
        with setup.distillation_teacher_model(model, config), torch.no_grad():
            clean_target = net(x).detach()
        noisy_target = clean_target + LABEL_NOISE_STD * torch.randn_like(clean_target)
        pred = net(x)
        loss = F.mse_loss(pred, noisy_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - t0)
        losses.append(_clean_eval_loss(net, setup, model, config, x_eval))

    student.remove_hook_from_module()
    # Note: in the real SFT-on-images analogue you would NOT do a teacher
    # forward per step -- you'd encode an image and add noise. We do the
    # teacher forward here only to construct the noisy supervision target;
    # subtract it from the timing to report the realistic SFT-step cost.
    return losses, sum(step_times) / len(step_times)


def steps_to_threshold(losses: list[float], threshold: float) -> int | None:
    for i, v in enumerate(losses):
        if v <= threshold:
            return i
    return None


def main() -> None:
    print(f"hidden={HIDDEN}  batch={BATCH}  lr={LR}  steps={STEPS}  label_noise_std={LABEL_NOISE_STD}")
    print()

    distill_losses, distill_step_s = run_distillation(seed=123)
    sft_losses, sft_step_s = run_sft(seed=123)

    initial = distill_losses[0]
    targets = [initial * 0.5, initial * 0.25, initial * 0.10, initial * 0.05]

    print(f"initial clean-eval loss (both methods): {initial:.6f}")
    print()

    print(f"{'step':>6}  {'distill':>12}  {'sft':>12}  {'sft/distill':>12}")
    for step in (0, 10, 30, 60, 120, 200, 300, STEPS):
        d = distill_losses[step]
        s = sft_losses[step]
        ratio = s / d if d > 0 else float("inf")
        print(f"{step:>6d}  {d:>12.6f}  {s:>12.6f}  {ratio:>12.2f}x")

    print()
    print("steps to reach loss threshold (lower = faster convergence):")
    print(f"{'threshold':>14}  {'distill':>10}  {'sft':>10}  {'speedup':>10}")
    for thr in targets:
        d = steps_to_threshold(distill_losses, thr)
        s = steps_to_threshold(sft_losses, thr)
        if d is None or s is None:
            speed = "n/a"
        else:
            speed = f"{s / max(d, 1):.2f}x" if d > 0 else "inf"
        print(f"{thr:>14.6f}  {str(d):>10}  {str(s):>10}  {speed:>10}")

    print()
    print("per-step wall clock (CPU, includes teacher fwd for both paths):")
    print(f"  distillation : {distill_step_s * 1000:.3f} ms / step")
    print(f"  sft          : {sft_step_s * 1000:.3f} ms / step")
    print(f"  ratio        : {sft_step_s / distill_step_s:.2f}x")
    print()
    print("Note: real-world SFT has ONE forward per step (no teacher pass).")
    print("Subtract the no_grad teacher forward time from the SFT row above to")
    print("model real SFT step cost; distillation always needs two forwards.")


if __name__ == "__main__":
    main()
