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


def dmd2_kl_pseudo_loss(
    x0: Tensor,
    x_sigma: Tensor,
    sigma: Tensor | float,
    real_velocity: Tensor,
    fake_velocity: Tensor,
    weight: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-3,
) -> Tensor:
    """DMD2 distribution-matching pseudo-loss in clean-sample (``x0``) space.

    Builds the teacher and fake one-step ``x0`` predictions from their velocities
    at the (detached) re-noised point ``x_sigma`` and forms the DMD gradient
    ``grad = pred_fake_x0 - pred_real_x0`` (which equals ``sigma * (v_real -
    v_fake)``). The returned pseudo-loss ``0.5 * mse(x0, (x0 - grad).detach())``
    has autograd gradient ``grad`` w.r.t. ``x0``, so gradient descent moves the
    student's ``x0`` toward the teacher prediction and away from the fake one
    (KL minimization). This matches the official DMD/DMD2 generator update.

    Sign note: flow models output ``v = eps - x0`` and the score relates to it by
    ``s = -(1-sigma)/sigma * v``; in x0-space the one-step prediction is
    ``x0_pred = x_sigma - v * sigma``, so ``pred_fake - pred_real = sigma *
    (v_real - v_fake)``. Moving x0 against this grad (toward ``pred_real``) is the
    teacher direction — the same sign as the previous velocity-space surrogate.

    When ``normalize`` is set, ``grad`` is divided per-sample by ``mean|x0 -
    pred_real_x0|`` (the DMD normalizer, clamped by a scale-aware ``eps``) so the
    signal magnitude is stable across noise levels. Score directions are detached;
    only ``x0`` carries gradient.
    """
    sigma_t = _expand_like(sigma, x_sigma).float()
    x_sigma_d = x_sigma.detach().float()
    pred_real_x0 = x_sigma_d - real_velocity.detach().float() * sigma_t
    pred_fake_x0 = x_sigma_d - fake_velocity.detach().float() * sigma_t
    grad = pred_fake_x0 - pred_real_x0
    if normalize:
        reduce_dims = list(range(1, x0.dim())) or [0]
        normalizer = (x0.detach().float() - pred_real_x0).abs().mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
        grad = grad / normalizer
    grad = torch.nan_to_num(grad)
    target = (x0.float() - grad).detach()
    return 0.5 * (x0.float() - target).pow(2).mean(dtype=torch.float32) * weight


def dmd2_endpoint_loss(generated_latent: Tensor, target_latent: Tensor, weight: float = 1.0) -> Tensor:
    """Mean-squared regression of the student estimate toward a target latent.

    The caller supplies the target: the paired teacher-match uses the frozen
    teacher's one-step denoise at the same trajectory point (shared noise).
    """
    return _mse_fp32(generated_latent, target_latent) * weight


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
