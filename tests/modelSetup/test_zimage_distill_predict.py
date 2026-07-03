import json
from contextlib import nullcontext

from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

import torch

import pytest


class FakeTransformerOutput:
    def __init__(self, sample):
        self.sample = sample


class FakeTransformer:
    def __init__(self):
        self.calls = []

    def __call__(self, latent_input_list, timestep, text_encoder_output, return_dict=True):
        self.calls.append(
            {
                "latent": torch.stack(latent_input_list, dim=0),
                "timestep": timestep.detach().clone(),
                "text": text_encoder_output.detach().clone(),
            }
        )
        return FakeTransformerOutput(sample=[latent * 2 for latent in latent_input_list])


class FakeZImageModel:
    def __init__(self):
        self.autocast_context = nullcontext()
        self.train_dtype = DataType.FLOAT_32
        self.transformer = FakeTransformer()
        self.encoded_texts = []

    def encode_text(self, train_device, batch_size=1, rand=None, text=None, **kwargs):
        self.encoded_texts.append(text)
        if text is None:
            return torch.full((batch_size, 1, 2), 3.0)
        return torch.full((batch_size, 1, 2), -5.0)

    def scale_latents(self, latent):
        return latent


class TinyZImageSetup(BaseZImageSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError


class FakeListTransformer:
    # The real transformer receives conditioning as a list of per-sample
    # embedding tensors (variable length), unlike the tensor-based fake above.
    def __init__(self):
        self.calls = []

    def __call__(self, latent_input_list, timestep, text_encoder_output, return_dict=True):
        self.calls.append(
            {
                "latent": torch.stack(latent_input_list, dim=0),
                "timestep": timestep.detach().clone(),
                "text_len": len(text_encoder_output),
            }
        )
        return FakeTransformerOutput(sample=[latent * 2 for latent in latent_input_list])


class FakeListZImageModel(FakeZImageModel):
    def __init__(self):
        super().__init__()
        self.transformer = FakeListTransformer()

    def encode_text(self, train_device, batch_size=1, rand=None, text=None, **kwargs):
        self.encoded_texts.append(text)
        value = 3.0 if text is None else -5.0
        return [torch.full((1, 2), value) for _ in range(batch_size)]


def test_zimage_distill_predict_tiles_conditioning_for_fused_batches():
    # CONCURRENT distill fuses [x_k; x_sigma] into one forward: the latent batch
    # is a multiple of the conditioning batch and the embeddings must be tiled.
    setup = TinyZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = FakeListZImageModel()
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    batch = {
        "latent_image": torch.zeros(2, 2),
        "_distill_latent_input": torch.randn(4, 2),
        "_distill_sigma": torch.tensor([1.0, 0.5, 0.25, 0.75]),
        "text_encoder_hidden_state": torch.full((2, 1, 2), 9.0),
    }

    output = setup.predict(model, batch, config, TrainProgress(), deterministic=True)

    call = model.transformer.calls[0]
    assert call["latent"].shape[0] == 4
    assert call["text_len"] == 4
    assert output["predicted"].shape[0] == 4


def test_zimage_distill_predict_rejects_non_multiple_fused_batch():
    setup = TinyZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = FakeListZImageModel()
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    batch = {
        "latent_image": torch.zeros(2, 2),
        "_distill_latent_input": torch.randn(3, 2),
        "_distill_sigma": torch.tensor([1.0, 0.5, 0.25]),
        "text_encoder_hidden_state": torch.full((2, 1, 2), 9.0),
    }

    with pytest.raises(RuntimeError, match="not a multiple"):
        setup.predict(model, batch, config, TrainProgress(), deterministic=True)


def test_zimage_distill_predict_uses_supplied_latent_sigma_and_negative_prompt():
    setup = TinyZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = FakeZImageModel()
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    batch = {
        "latent_image": torch.zeros(2, 2),
        "_distill_latent_input": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "_distill_sigma": torch.tensor([1.0, 0.25]),
        "_distill_conditioning": "unconditional",
        "distill_negative_prompt": ["low quality", "bad anatomy"],
        "text_encoder_hidden_state": torch.full((2, 1, 2), 9.0),
    }

    output = setup.predict(model, batch, config, TrainProgress(), deterministic=True)

    assert torch.allclose(output["predicted"], -batch["_distill_latent_input"] * 2)
    assert torch.allclose(model.transformer.calls[0]["latent"].squeeze(2), batch["_distill_latent_input"])
    assert torch.allclose(model.transformer.calls[0]["timestep"], torch.tensor([0.0, 0.75]))
    assert model.encoded_texts[-1] == ["low quality", "bad anatomy"]
    assert torch.all(model.transformer.calls[0]["text"] == -5.0)


def test_zimage_distill_velocity_uses_sidecar_negative_prompt_for_unconditional_cfg(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not a real image; sidecar lookup only")
    image_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "prompt": "a good image",
                "negative_prompt": "sidecar negative",
                "steps": 12,
                "cfg_scale": 2.0,
            }
        ),
        encoding="utf-8",
    )

    setup = TinyZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = FakeZImageModel()
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    batch = {
        "latent_image": torch.zeros(1, 2),
        "image_path": [str(image_path)],
        "text_encoder_hidden_state": torch.full((1, 1, 2), 9.0),
    }

    setup.predict_distill_velocity(
        model,
        batch,
        config,
        TrainProgress(),
        latent=torch.ones(1, 2),
        sigma=torch.tensor([0.5]),
        conditioning="unconditional",
    )

    assert model.encoded_texts[-1] == ["sidecar negative"]
