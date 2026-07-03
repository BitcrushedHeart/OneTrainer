from modules.util.distill_dmd2_util import (
    build_distill_sigma_matrix,
    build_distill_sigmas,
    cfg_baked_velocity,
    euler_flow_step,
    shift_distill_sigmas,
    simulate_flow_trajectory,
)

import torch

import pytest


def test_build_distill_sigmas_selects_teacher_scheduler_points():
    sigmas = build_distill_sigmas(target_steps=4, teacher_steps=12, mode="AUTO", device=torch.device("cpu"))

    assert torch.allclose(sigmas, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_euler_flow_trajectory_reaches_endpoint_for_constant_flow():
    noise = torch.tensor([[1.0, 2.0]])
    image = torch.tensor([[0.0, -1.0]])
    velocity = noise - image
    sigmas = build_distill_sigmas(target_steps=4, teacher_steps=12, mode="AUTO", device=torch.device("cpu"))

    trajectory = simulate_flow_trajectory(
        initial_latent=noise,
        sigmas=sigmas,
        velocity_fn=lambda latent, sigma, step_index: velocity,
    )

    assert torch.allclose(trajectory[-1], image, atol=1e-6)


def test_cfg_baked_velocity_supports_per_sample_cfg_and_cfg_one_fast_path():
    cond = torch.tensor([[2.0, 4.0], [10.0, 20.0]])
    uncond = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    cfg = torch.tensor([1.0, 3.0])

    baked = cfg_baked_velocity(cond, uncond, cfg)

    assert torch.allclose(baked[0], cond[0])
    assert torch.allclose(baked[1], uncond[1] + 3.0 * (cond[1] - uncond[1]))


def test_euler_flow_step_broadcasts_per_sample_sigmas():
    latent = torch.ones(2, 3)
    velocity = torch.full((2, 3), 2.0)
    sigma = torch.tensor([1.0, 0.5])
    next_sigma = torch.tensor([0.5, 0.0])

    stepped = euler_flow_step(latent, velocity, sigma, next_sigma)

    assert torch.allclose(stepped, torch.zeros_like(latent))


def test_shift_distill_sigmas_matches_flow_scheduler_map():
    # sigma' = s*t / (1 + (s-1)*t), the FlowMatchEulerDiscreteScheduler /
    # ComfyUI ModelSampling shift. Endpoints are fixed points.
    t = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])

    shifted = shift_distill_sigmas(t, 1.5)

    expected = 1.5 * t / (1.0 + 0.5 * t)
    assert torch.allclose(shifted, expected)
    assert shifted[0] == 1.0
    assert shifted[-1] == 0.0
    # shift > 1 pushes interior points toward high noise
    assert torch.all(shifted[1:-1] > t[1:-1])


def test_shift_distill_sigmas_identity_and_validation():
    t = torch.linspace(1.0, 0.0, 5)
    assert torch.equal(shift_distill_sigmas(t, 1.0), t)
    with pytest.raises(ValueError):
        shift_distill_sigmas(t, 0.0)


def test_build_distill_sigmas_applies_shift_to_teacher_grid():
    unshifted = build_distill_sigmas(target_steps=4, teacher_steps=12, mode="AUTO", device=torch.device("cpu"))
    shifted = build_distill_sigmas(target_steps=4, teacher_steps=12, mode="AUTO", device=torch.device("cpu"), shift=1.5)

    assert torch.allclose(shifted, shift_distill_sigmas(unshifted, 1.5))
    assert shifted[0] == 1.0
    assert shifted[-1] == 0.0
    assert torch.all(shifted[:-1] > shifted[1:])


def test_build_distill_sigma_matrix_passes_shift_through():
    teacher_steps = torch.tensor([12, 8])

    sigmas = build_distill_sigma_matrix(
        target_steps=4,
        teacher_steps=teacher_steps,
        mode="AUTO",
        device=torch.device("cpu"),
        shift=1.5,
    )

    expected = shift_distill_sigmas(torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), 1.5)
    assert torch.allclose(sigmas[0], expected)
    assert torch.allclose(sigmas[1], expected)


def test_build_distill_sigma_matrix_supports_mixed_teacher_steps():
    teacher_steps = torch.tensor([12, 8])

    sigmas = build_distill_sigma_matrix(
        target_steps=4,
        teacher_steps=teacher_steps,
        mode="AUTO",
        device=torch.device("cpu"),
    )

    assert sigmas.shape == (2, 5)
    assert torch.allclose(sigmas[0], torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    assert torch.allclose(sigmas[1], torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
