from modules.util.distill_dmd2_util import dmd2_endpoint_loss, dmd2_fake_score_loss, dmd2_kl_pseudo_loss

import torch
from torch import nn


def _maybe_bfloat16():
    try:
        _ = torch.ones(1, dtype=torch.bfloat16) + torch.ones(1, dtype=torch.bfloat16)
        return torch.bfloat16
    except RuntimeError:
        return torch.float32


def test_dmd2_cpu_smoke_isolates_fake_and_student_gradients():
    torch.manual_seed(123)
    dtype = _maybe_bfloat16()

    base = nn.Linear(4, 4, bias=False)
    student = nn.Linear(4, 4, bias=False)
    fake = nn.Linear(4, 4, bias=False)
    for param in base.parameters():
        param.requires_grad_(False)
    base_before = base.weight.detach().clone()

    noise = torch.randn(3, 4).to(dtype=dtype)
    image_latent = torch.randn(3, 4).to(dtype=dtype)

    fake_opt = torch.optim.SGD(fake.parameters(), lr=0.1)
    student_opt = torch.optim.SGD(student.parameters(), lr=0.1)

    with torch.no_grad():
        student_latent = student(noise.float()).to(dtype=dtype)
        teacher_velocity = base(student_latent.float()).to(dtype=dtype)
    fake_velocity = fake(student_latent.detach().float()).to(dtype=dtype)
    fake_loss = dmd2_fake_score_loss(fake_velocity, teacher_velocity)
    fake_opt.zero_grad()
    student_opt.zero_grad()
    fake_loss.backward()

    assert torch.isfinite(fake_loss)
    assert fake.weight.grad is not None
    assert student.weight.grad is None

    fake_opt.step()
    fake_opt.zero_grad(set_to_none=True)
    student_opt.zero_grad(set_to_none=True)

    x_at_k = student(noise.float()).to(dtype=dtype)
    with torch.no_grad():
        v_real = base(x_at_k.detach().float()).to(dtype=dtype)
        v_fake = fake(x_at_k.detach().float()).to(dtype=dtype)
    student_loss = dmd2_kl_pseudo_loss(x_at_k, x_at_k, 0.5, v_real, v_fake) + dmd2_endpoint_loss(
        x_at_k, image_latent, weight=0.25
    )
    student_loss.backward()

    assert torch.isfinite(student_loss)
    assert student.weight.grad is not None
    assert fake.weight.grad is None
    assert torch.equal(base.weight, base_before)


def test_endpoint_loss_reduces_distance_to_image_latent_in_toy_setup():
    torch.manual_seed(321)
    student_latent = torch.tensor([[1.0, -1.0]], requires_grad=True)
    image_latent = torch.tensor([[0.0, 0.0]])

    before = (student_latent.detach() - image_latent).norm()
    loss = dmd2_endpoint_loss(student_latent, image_latent, weight=1.0)
    loss.backward()

    with torch.no_grad():
        after_step = student_latent - 0.25 * student_latent.grad
    after = (after_step - image_latent).norm()

    assert after < before


def _kl_x0_predictions(x_sigma, sigma, v_real, v_fake):
    sig = sigma.unsqueeze(-1)
    return x_sigma - v_real * sig, x_sigma - v_fake * sig


def test_kl_pseudo_loss_sign_points_toward_teacher():
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, requires_grad=True)
    x_sigma = torch.randn(2, 4)
    sigma = torch.tensor([0.5, 0.5])
    v_real = torch.randn(2, 4)
    v_fake = torch.randn(2, 4)
    pred_real_x0, pred_fake_x0 = _kl_x0_predictions(x_sigma, sigma, v_real, v_fake)

    loss = dmd2_kl_pseudo_loss(x0, x_sigma, sigma, v_real, v_fake, normalize=False)
    loss.backward()

    # dL/dx0 must be parallel to (pred_fake - pred_real): GD then moves x0 toward
    # the teacher prediction and away from the fake one.
    direction = (pred_fake_x0 - pred_real_x0).flatten()
    cos = torch.nn.functional.cosine_similarity(x0.grad.flatten(), direction, dim=0)
    assert cos > 0.99
    step = -x0.grad  # the GD displacement direction
    assert ((step) * (pred_real_x0 - pred_fake_x0)).sum() > 0


def test_kl_pseudo_loss_scale_invariance():
    torch.manual_seed(1)
    x0 = torch.randn(2, 8)
    x_sigma = torch.randn(2, 8)
    sigma = torch.tensor([0.4, 0.6])
    v_real = torch.randn(2, 8)
    v_fake = torch.randn(2, 8)

    def grad_of(scale, normalize):
        xi = (x0 * scale).clone().requires_grad_(True)
        dmd2_kl_pseudo_loss(xi, x_sigma * scale, sigma, v_real * scale, v_fake * scale, normalize=normalize).backward()
        return xi.grad

    # The DMD normalizer cancels a uniform magnitude rescale.
    assert torch.allclose(grad_of(1.0, True), grad_of(10.0, True), rtol=1e-2, atol=1e-4)
    # Without it the gradient scales linearly (proves the normalizer is the cause).
    assert torch.allclose(grad_of(10.0, False), 10.0 * grad_of(1.0, False), rtol=1e-2, atol=1e-4)


def test_kl_pseudo_loss_nan_guard():
    x0 = torch.randn(2, 4, requires_grad=True)
    x_sigma = torch.randn(2, 4)
    sigma = torch.tensor([0.5, 0.5])
    v_real = torch.full((2, 4), float("inf"))
    v_fake = torch.randn(2, 4)

    loss = dmd2_kl_pseudo_loss(x0, x_sigma, sigma, v_real, v_fake)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x0.grad).all()


def test_kl_pseudo_loss_grad_through_x0_only():
    x0 = torch.randn(2, 4, requires_grad=True)
    x_sigma = torch.randn(2, 4, requires_grad=True)
    sigma = torch.tensor([0.5, 0.5])
    v_real = torch.randn(2, 4, requires_grad=True)
    v_fake = torch.randn(2, 4, requires_grad=True)

    dmd2_kl_pseudo_loss(x0, x_sigma, sigma, v_real, v_fake).backward()

    assert x0.grad is not None
    assert x_sigma.grad is None
    assert v_real.grad is None
    assert v_fake.grad is None


def test_kl_pseudo_loss_zero_when_real_equals_fake():
    x0 = torch.randn(2, 4, requires_grad=True)
    x_sigma = torch.randn(2, 4)
    sigma = torch.tensor([0.5, 0.5])
    v = torch.randn(2, 4)

    dmd2_kl_pseudo_loss(x0, x_sigma, sigma, v, v).backward()

    assert x0.grad.abs().max() == 0
