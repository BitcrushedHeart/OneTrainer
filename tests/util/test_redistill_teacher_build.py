"""build_raw_late_tensors: the redistill Phase A teacher loader."""

from modules.util.transfer_util import build_raw_late_tensors

import torch

import pytest
from safetensors.torch import save_file

STEMS = {"layers.2.attention.to_q", "layers.2.feed_forward.w1"}


def _save_ckpt(path, *, prefix=""):
    tensors = {
        f"{prefix}layers.2.attention.to_q.weight": torch.full((8, 8), 2.0, dtype=torch.bfloat16),
        f"{prefix}layers.2.feed_forward.w1.weight": torch.full((8, 8), 3.0, dtype=torch.bfloat16),
        f"{prefix}layers.0.attention.to_q.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }
    save_file(tensors, str(path))
    return str(path)


@pytest.mark.parametrize("prefix", ["", "transformer."])
def test_reads_raw_late_tensors_fp32(tmp_path, prefix):
    path = _save_ckpt(tmp_path / "turbo.safetensors", prefix=prefix)
    out = build_raw_late_tensors(path, STEMS)
    assert set(out) == STEMS
    for tensor in out.values():
        assert tensor.dtype == torch.float32
    assert float(out["layers.2.attention.to_q"][0, 0]) == 2.0
    assert float(out["layers.2.feed_forward.w1"][0, 0]) == 3.0


def test_shape_check_hard_fails(tmp_path):
    path = _save_ckpt(tmp_path / "turbo.safetensors")
    with pytest.raises(ValueError, match="does not match"):
        build_raw_late_tensors(path, STEMS, expected_shapes={"layers.2.attention.to_q": torch.Size((4, 4))})


def test_missing_stem_hard_fails(tmp_path):
    path = _save_ckpt(tmp_path / "turbo.safetensors")
    with pytest.raises(KeyError):
        build_raw_late_tensors(path, STEMS | {"layers.3.attention.to_q"})
