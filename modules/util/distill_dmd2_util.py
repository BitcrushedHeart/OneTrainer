import torch
from torch import Tensor


def _mse_fp32(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction.float() - target.float()).pow(2).mean(dtype=torch.float32)


def dmd2_fake_score_loss(fake_velocity: Tensor, target_velocity: Tensor) -> Tensor:
    """Denoising score-matching loss for the fake-score adapter.

    ``target_velocity`` is the rectified-flow target (``noise - x0``) for a
    *student-generated* sample that has been re-noised to a random sigma. This
    trains mu_fake to model the student's output distribution (DMD2 6.2), NOT to
    copy the frozen teacher (which would drive the fake delta to zero and
    collapse the KL signal).
    """
    return _mse_fp32(fake_velocity, target_velocity.detach())


def dmd2_kl_pseudo_loss(x_at_k: Tensor, real_velocity: Tensor, fake_velocity: Tensor, weight: float = 1.0) -> Tensor:
    """DMD2 pseudo-KL term.

    The score direction is detached; gradients flow through ``x_at_k`` into the
    student trajectory only.

    Sign note: flow models output velocity ``v = eps - x0``, and the score
    relates to it as ``s = -(1-sigma)/sigma * v``. DMD2 needs the *score*
    difference ``(s_fake - s_real)``, which therefore has the OPPOSITE sign of
    the velocity difference. So we use ``(real - fake)`` here: minimizing this
    moves the student TOWARD the teacher distribution. Using ``(fake - real)``
    inverts the gradient and drives the student away from the teacher.
    """
    score_delta = real_velocity.detach().float() - fake_velocity.detach().float()
    return (score_delta * x_at_k.float()).mean(dtype=torch.float32) * weight


def dmd2_endpoint_loss(generated_latent: Tensor, image_latent: Tensor, weight: float = 1.0) -> Tensor:
    """Regression to the actual image latent at the terminal trajectory point."""
    return _mse_fp32(generated_latent, image_latent) * weight


def build_distill_sigmas(
    target_steps: int,
    teacher_steps: int,
    mode: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if target_steps <= 0:
        raise ValueError("target_steps must be greater than 0")
    if teacher_steps <= 0:
        raise ValueError("teacher_steps must be greater than 0")

    mode = str(mode).upper()
    teacher_grid = torch.linspace(1.0, 0.0, teacher_steps + 1, device=device, dtype=dtype)

    if mode in {"AUTO", "UNIFORM"}:
        indices = torch.linspace(0, teacher_steps, target_steps + 1, device=device)
        indices = indices.round().long()
    elif mode == "TRAILING":
        stride = max(1, teacher_steps // target_steps)
        indices = torch.arange(teacher_steps - (target_steps * stride), teacher_steps + 1, stride, device=device)
        indices = indices.clamp(0, teacher_steps)
        if indices.shape[0] != target_steps + 1:
            indices = torch.linspace(0, teacher_steps, target_steps + 1, device=device).round().long()
    else:
        raise ValueError(f"Unsupported distill timestep grid mode: {mode}")

    indices[0] = 0
    indices[-1] = teacher_steps
    return teacher_grid[indices]


def build_distill_sigma_matrix(
    target_steps: int,
    teacher_steps: Tensor,
    mode: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    teacher_steps = teacher_steps.to(device=device, dtype=torch.long).flatten()
    rows = [
        build_distill_sigmas(target_steps, int(steps.item()), mode, device=device, dtype=dtype)
        for steps in teacher_steps
    ]
    return torch.stack(rows, dim=0)


def _expand_like(value: Tensor | float, target: Tensor) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.tensor(value, device=target.device, dtype=target.dtype)
    tensor = tensor.to(device=target.device, dtype=target.dtype)
    while tensor.dim() < target.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor


def euler_flow_step(latent: Tensor, velocity: Tensor, sigma: Tensor | float, next_sigma: Tensor | float) -> Tensor:
    delta = _expand_like(next_sigma, latent) - _expand_like(sigma, latent)
    return latent + delta * velocity.to(dtype=latent.dtype)


def simulate_flow_trajectory(initial_latent: Tensor, sigmas: Tensor, velocity_fn) -> list[Tensor]:
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a 1D tensor")
    if sigmas.shape[0] < 2:
        raise ValueError("sigmas must contain at least two points")

    trajectory = [initial_latent]
    latent = initial_latent
    for step_index in range(sigmas.shape[0] - 1):
        sigma = sigmas[step_index]
        next_sigma = sigmas[step_index + 1]
        velocity = velocity_fn(latent, sigma, step_index)
        latent = euler_flow_step(latent, velocity, sigma, next_sigma)
        trajectory.append(latent)
    return trajectory


def cfg_baked_velocity(
    conditional_velocity: Tensor, unconditional_velocity: Tensor | None, cfg_scale: Tensor | float
) -> Tensor:
    cfg_tensor = _expand_like(cfg_scale, conditional_velocity)
    if unconditional_velocity is None or bool(torch.all(cfg_tensor == 1)):
        return conditional_velocity
    return unconditional_velocity + cfg_tensor * (conditional_velocity - unconditional_velocity)
