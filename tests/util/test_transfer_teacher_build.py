"""Tests for transfer_util.build_teacher_late_tensors.

The transfer-mode teacher is Base's late tensors with the ORIGINAL
Base-trained adapter's education applied in memory -- W_t = m * (W_b @ R^T)
per output row -- built once at setup. No merged checkpoint is ever written
(user decision); correctness is anchored against oft_align's merged_full,
which is the same math OFTModule.apply_to_module bakes with.
"""

from modules.util.adapter_io import read_safetensors_buffered
from modules.util.oft_align import AdapterOFTConfig, merged_full, reconstruct_oft_rotation
from modules.util.transfer_util import build_teacher_late_tensors, parse_transfer_families

import torch

import pytest

LATE_START = 2


def _reference_teacher(adapter_path, base_tensors, stems_map, dora):
    """Independent reference: reconstruct rotation per late stem and bake."""
    tensors, _ = read_safetensors_buffered(adapter_path)
    cfg = AdapterOFTConfig(block_size=6, clip=0.99, scaled=False)
    stem_names = {s for stems in stems_map.values() for s in stems}
    expected = {}
    for stem, wb in base_tensors.items():
        layer = int(stem.split(".")[1])
        mod_stem = stem.split(".", 2)[2]
        if layer < LATE_START or mod_stem not in stem_names:
            continue
        vec = None
        for prefix in ("transformer.", ""):
            key = f"{prefix}{stem}.oft_R.weight"
            if key in tensors:
                vec = tensors[key].to(torch.float32)
                break
        merged = merged_full(wb.to(torch.float32), reconstruct_oft_rotation(vec, cfg))
        if dora:
            mult = tensors[f"transformer.{stem}.dora_multiplier"].to(torch.float32)
            merged = merged * mult.view(-1, 1)
        expected[stem] = merged
    return expected


@pytest.mark.parametrize("dora", [False, True])
@pytest.mark.parametrize("ckpt_prefix", ["", "transformer."])
def test_teacher_matches_reference_bake(
    tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, dora, ckpt_prefix
):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=dora)
    base = make_ckpt_tensors()
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base, prefix=ckpt_prefix)
    fams = parse_transfer_families("qkv,mlp")

    teacher = build_teacher_late_tensors(base_path, adapter_path, LATE_START, fams)

    expected = _reference_teacher(adapter_path, base, fams, dora)
    assert set(teacher) == set(expected)
    for stem, w in teacher.items():
        assert w.dtype == torch.float32
        torch.testing.assert_close(w, expected[stem], rtol=1e-5, atol=1e-6)


def test_scaled_adapter_reads_and_builds(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    """oft_scaled adapters carry a persistent BOOL marker buffer (scaled_oft);
    the buffered readers must handle it (regression: KeyError('BOOL')) and the
    teacher build must apply the scaling factor from ot_config."""
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True, scaled=True)
    tensors, _ = read_safetensors_buffered(adapter_path)
    assert any(k.endswith(".scaled_oft") for k in tensors)  # BOOL marker survived the reader

    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", make_ckpt_tensors())
    teacher = build_teacher_late_tensors(base_path, adapter_path, LATE_START, parse_transfer_families("qkv,mlp"))
    assert teacher
    assert all(torch.isfinite(w).all() for w in teacher.values())


def test_teacher_respects_family_subset(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors")
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", make_ckpt_tensors())
    teacher = build_teacher_late_tensors(base_path, adapter_path, LATE_START, parse_transfer_families("qkv"))
    assert teacher
    assert all(".attention." in stem for stem in teacher)


def test_teacher_missing_base_key_raises(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors")
    base = make_ckpt_tensors()
    base.pop(f"layers.{LATE_START}.attention.to_q")
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    with pytest.raises(KeyError):
        build_teacher_late_tensors(base_path, adapter_path, LATE_START, parse_transfer_families("qkv,mlp"))


def test_teacher_expected_shapes_guard(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors")
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", make_ckpt_tensors())
    fams = parse_transfer_families("qkv,mlp")

    good = build_teacher_late_tensors(base_path, adapter_path, LATE_START, fams)
    shapes = {stem: w.shape for stem, w in good.items()}

    # Matching shapes pass.
    build_teacher_late_tensors(base_path, adapter_path, LATE_START, fams, expected_shapes=shapes)

    # A mismatching shape fails loud.
    bad = dict(shapes)
    first = next(iter(bad))
    bad[first] = torch.Size((1, 1))
    with pytest.raises(ValueError, match="shape"):
        build_teacher_late_tensors(base_path, adapter_path, LATE_START, fams, expected_shapes=bad)

    # A stem the adapter does not cover fails loud (coverage check).
    extra = dict(shapes)
    extra["layers.99.attention.to_q"] = torch.Size((16, 16))
    with pytest.raises(ValueError, match="cover"):
        build_teacher_late_tensors(base_path, adapter_path, LATE_START, fams, expected_shapes=extra)
