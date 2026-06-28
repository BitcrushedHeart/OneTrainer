import torch
from torch import nn

from modules.util.distill_dmd2_util import dmd2_endpoint_loss, dmd2_fake_score_loss, dmd2_kl_pseudo_loss


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
    student_loss = dmd2_kl_pseudo_loss(x_at_k, v_real, v_fake) + dmd2_endpoint_loss(
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
