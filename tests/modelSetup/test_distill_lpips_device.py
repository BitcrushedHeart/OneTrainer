from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.util.config.TrainConfig import TrainConfig

import torch


class _StubZImageSetup(BaseZImageSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError


def test_lpips_colocated_cuda_unindexed_matches_indexed():
    # The bug: train_device 'cuda' (unindexed) vs a VAE parameter reporting
    # 'cuda:0' must count as co-located, otherwise teacher-match LPIPS is skipped
    # every validation even though the VAE was moved on-device.
    f = BaseZImageSetup._lpips_vae_colocated
    assert f(torch.device("cuda:0"), torch.device("cuda")) is True
    assert f(torch.device("cuda"), torch.device("cuda:0")) is True
    assert f(torch.device("cuda:0"), torch.device("cuda:0")) is True
    assert f(torch.device("cuda"), torch.device("cuda")) is True


def test_lpips_not_colocated_when_offloaded_to_cpu():
    # VAE offloaded to CPU (mgds latent caching) while training on CUDA -> skip.
    f = BaseZImageSetup._lpips_vae_colocated
    assert f(torch.device("cpu"), torch.device("cuda")) is False
    assert f(torch.device("cpu"), torch.device("cuda:0")) is False


def test_lpips_colocated_cpu_training():
    f = BaseZImageSetup._lpips_vae_colocated
    assert f(torch.device("cpu"), torch.device("cpu")) is True


def test_lpips_not_colocated_across_distinct_gpus():
    f = BaseZImageSetup._lpips_vae_colocated
    assert f(torch.device("cuda:1"), torch.device("cuda:0")) is False
    assert f(torch.device("cuda:0"), torch.device("cuda:1")) is False


def test_perceptual_loss_skips_without_decoding_when_offloaded():
    # Wiring: with the VAE on CPU and the compute device CUDA, the perceptual loss
    # must return None at the guard, before importing lpips or decoding. The cuda
    # device object is constructible without a GPU.
    setup = _StubZImageSetup(torch.device("cuda"), torch.device("cpu"), False)

    class _FakeVae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))  # lives on CPU

        def decode(self, *args, **kwargs):
            raise AssertionError("VAE.decode must not run when offloaded")

    class _FakeModel:
        def __init__(self):
            self.vae = _FakeVae()

        def unscale_latents(self, latent):
            raise AssertionError("unscale_latents must not run when offloaded")

    latent = torch.zeros(1, 4, 8, 8)
    result = setup._distill_perceptual_loss(_FakeModel(), latent, latent, TrainConfig.default_values())
    assert result is None
