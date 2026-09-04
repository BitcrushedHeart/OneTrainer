"""End-to-end tests for scripts/util/_oft_conjugate_retarget.py.

Runs the CLI's main() on tiny synthetic adapter + Base/variant checkpoint
files (variant = exact block transport of Base on the late layers, bit-equal
elsewhere) and checks the written warm-start adapter: verbatim early/mid/marker
keys, conjugated late vectors that actually satisfy the transport equation,
metadata, --force semantics, prefix autodetection, the requested-vs-adjusted
block-size trap (request 6 stores 4-wide blocks for in=16 and 6-wide for
in=24 in the same file), and loadability into a real LoRAModuleWrapper.
"""

import importlib.util
import json
from pathlib import Path

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.adapter_io import read_safetensors_buffered
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType
from modules.util.oft_align import (
    AdapterOFTConfig,
    block_size_from_elements,
    merged_full,
    reconstruct_oft_rotation,
)
from modules.util.oft_conjugate import apply_block_rotations

import torch
from torch import nn

import pytest
from safetensors.torch import save_file

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "util" / "_oft_conjugate_retarget.py"
_spec = importlib.util.spec_from_file_location("oft_conjugate_retarget_script", _SCRIPT_PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

LATE_START = 2  # tiny lineup: layers 0-1 early, 2-3 late


def _transported_variant(base_tensors, make_orthogonal_blocks, tiny_geometry, *, angle=0.1, seed=50):
    """Variant = exact per-tensor block transport on late layers, clone elsewhere."""
    out, rotations = {}, {}
    i = 0
    for stem, wb in base_tensors.items():
        layer = int(stem.split(".")[1])
        if layer < LATE_START:
            out[stem] = wb.clone()
            continue
        out_f, in_f = wb.shape
        br = tiny_geometry["block_for_in"][in_f]  # the ADAPTER'S partition for this tensor
        bl = 4
        i += 1
        r_left = make_orthogonal_blocks(out_f // bl, bl, seed=seed + i, angle=angle)
        r_right = make_orthogonal_blocks(in_f // br, br, seed=seed + 1000 + i, angle=angle)
        out[stem] = apply_block_rotations(wb, r_left=r_left, r_right=r_right)
        rotations[stem] = (r_left, r_right)
    return out, rotations


def _run_script(tmp_path, adapter_path, base_path, variant_path, *extra):
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    rc = script.main(
        [
            "--adapter",
            str(adapter_path),
            "--base",
            str(base_path),
            "--variant",
            f"Toy={variant_path}",
            "--out-dir",
            str(out_dir),
            "--late-start",
            str(LATE_START),
            "--iters",
            "200",
            "--tol",
            "0",
            *extra,
        ]
    )
    return rc, out_dir / f"{Path(adapter_path).stem}_Toy_conj.safetensors"


@pytest.mark.parametrize(
    ("adapter_prefix", "ckpt_prefix"),
    [("transformer.", ""), ("", "transformer.")],
)
def test_script_end_to_end(
    tmp_path,
    make_tiny_adapter,
    make_ckpt_tensors,
    write_tiny_ckpt,
    make_orthogonal_blocks,
    tiny_geometry,
    adapter_prefix,
    ckpt_prefix,
):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True, prefix=adapter_prefix)
    base = make_ckpt_tensors()
    variant, rotations = _transported_variant(base, make_orthogonal_blocks, tiny_geometry)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base, prefix=ckpt_prefix)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant, prefix=ckpt_prefix)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path, "--verify")
    assert rc == 0
    assert out_path.exists()

    source, _ = read_safetensors_buffered(adapter_path)
    result, metadata = read_safetensors_buffered(str(out_path))
    assert set(result) == set(source)

    report = json.loads(metadata["soreal_conjugate"])
    assert report["late_tensors"] == 2 * len(tiny_geometry["stem_shapes"])
    assert report["variant"] == "Toy"
    for fam_stats in report["per_family"].values():
        assert fam_stats["verify_rel_max"] < 1e-3  # exact transport: near machine floor

    cfg = AdapterOFTConfig(block_size=tiny_geometry["block_request"], clip=0.99, scaled=False)
    for key in source:
        layer = int(key.replace("transformer.", "").split(".")[1])
        if layer < LATE_START or not key.endswith(".oft_R.weight"):
            if key.endswith(".dora_multiplier") and layer >= LATE_START:
                continue  # policy-handled, checked in the dora test
            assert torch.equal(result[key], source[key]), f"{key} not copied verbatim"
            continue

        # Conjugated late vector: the merged equation must hold against the
        # GROUND-TRUTH transport (exact transport => fit ~= truth), and it must
        # beat the naive transplant by a wide margin.
        stem = key[: -len(".oft_R.weight")].replace("transformer.", "")
        wb, wv = base[stem], variant[stem]
        r_left, r_right = rotations[stem]
        expected = apply_block_rotations(
            merged_full(wb, reconstruct_oft_rotation(source[key], cfg)), r_left=r_left, r_right=r_right
        )
        conj_err = float((merged_full(wv, reconstruct_oft_rotation(result[key], cfg)) - expected).norm())
        transplant_err = float((merged_full(wv, reconstruct_oft_rotation(source[key], cfg)) - expected).norm())
        assert conj_err < 1e-3
        assert conj_err * 3.0 < transplant_err, f"{key}: conjugation did not beat transplant"


def test_adjusted_block_sizes_derived_from_tensors(tmp_path, make_tiny_adapter, tiny_geometry):
    """Request 6 stores two block widths; derivation must come from the vectors."""
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors")
    tensors, metadata = read_safetensors_buffered(adapter_path)
    assert json.loads(metadata["ot_config"])["oft_block_size"] == tiny_geometry["block_request"]

    widths = set()
    for k, v in tensors.items():
        if k.endswith(".oft_R.weight"):
            widths.add(block_size_from_elements(v.shape[-1]))
    assert widths == {4, 6}


def test_corrupt_vec_width_raises(
    tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, make_orthogonal_blocks, tiny_geometry
):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors")
    tensors, metadata = read_safetensors_buffered(adapter_path)
    bad_key = f"layers.{LATE_START}.attention.to_q.oft_R.weight"
    full_key = "transformer." + bad_key
    tensors[full_key] = torch.zeros(4, 5)  # 5 is not b*(b-1)/2 for any b
    bad_path = tmp_path / "bad.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(bad_path), metadata=metadata)

    base = make_ckpt_tensors()
    variant, _ = _transported_variant(base, make_orthogonal_blocks, tiny_geometry)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    with pytest.raises(ValueError, match="not b\\*\\(b-1\\)/2"):
        _run_script(tmp_path, bad_path, base_path, variant_path)


def test_force_and_skip(
    tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, make_orthogonal_blocks, tiny_geometry
):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors")
    base = make_ckpt_tensors()
    variant, _ = _transported_variant(base, make_orthogonal_blocks, tiny_geometry)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    ref_tensors, ref_metadata = read_safetensors_buffered(str(out_path))

    # Second run without --force: skipped, file untouched (byte-compare is
    # valid here because nothing rewrites the file).
    first_bytes = out_path.read_bytes()
    tampered = first_bytes[:8] + b"\x00" * 8 + first_bytes[16:]
    out_path.write_bytes(tampered)
    rc, _ = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    assert out_path.read_bytes() == tampered

    # With --force: rewritten with the same CONTENT. (Not byte-compared:
    # safetensors serializes its header via a Rust HashMap, so JSON key order
    # in the file is nondeterministic per write.)
    rc, _ = _run_script(tmp_path, adapter_path, base_path, variant_path, "--force")
    assert rc == 0
    new_tensors, new_metadata = read_safetensors_buffered(str(out_path))
    assert set(new_tensors) == set(ref_tensors)
    for key, value in ref_tensors.items():
        assert torch.equal(new_tensors[key], value), key
    assert json.loads(new_metadata["soreal_conjugate"]) == json.loads(ref_metadata["soreal_conjugate"])


def test_plain_oft_no_dora_keys(
    tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, make_orthogonal_blocks, tiny_geometry
):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=False)
    base = make_ckpt_tensors()
    variant, _ = _transported_variant(base, make_orthogonal_blocks, tiny_geometry)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    result, _ = read_safetensors_buffered(str(out_path))
    assert not any(k.endswith(".dora_multiplier") for k in result)


def test_dora_copy_policy_transports_verbatim(
    tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, make_orthogonal_blocks, tiny_geometry
):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant, _ = _transported_variant(base, make_orthogonal_blocks, tiny_geometry)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    source, _ = read_safetensors_buffered(adapter_path)
    result, _ = read_safetensors_buffered(str(out_path))
    for k in source:
        if k.endswith(".dora_multiplier"):
            assert torch.equal(result[k], source[k])


class _TinyAttn(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.to_q = nn.Linear(hidden, hidden, bias=False)
        self.to_k = nn.Linear(hidden, hidden, bias=False)
        self.to_v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.to_q(x) + self.to_k(x) + self.to_v(x)


class _TinyMlp(nn.Module):
    def __init__(self, hidden: int, mlp: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, mlp, bias=False)
        self.w2 = nn.Linear(mlp, hidden, bias=False)
        self.w3 = nn.Linear(hidden, mlp, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int, mlp: int):
        super().__init__()
        self.attention = _TinyAttn(hidden)
        self.feed_forward = _TinyMlp(hidden, mlp)

    def forward(self, x):
        return x + self.attention(x) + self.feed_forward(x)


class _TinyTransformer(nn.Module):
    def __init__(self, hidden: int, mlp: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyBlock(hidden, mlp) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_output_loads_into_wrapper(
    tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, make_orthogonal_blocks, tiny_geometry
):
    """The conjugated file must load into a real LoRAModuleWrapper and survive
    the hook -> forward -> apply -> unhook lifecycle."""
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant, _ = _transported_variant(base, make_orthogonal_blocks, tiny_geometry)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0

    config = TrainConfig.default_values()
    config.peft_type = PeftType.OFT_2
    config.dora_oft = True
    config.oft_block_size = tiny_geometry["block_request"]
    config.oft_scaled = False
    config.oft_clipped_norm = 0.99
    config.oft_block_share = False
    config.oft_cans = False
    config.dropout_probability = 0.0
    transformer = _TinyTransformer(tiny_geometry["hidden"], tiny_geometry["mlp"], layers=4)
    wrapper = LoRAModuleWrapper(transformer, "transformer", config, [])
    assert len(wrapper.lora_modules) == 4 * len(tiny_geometry["stem_shapes"])

    result, _ = read_safetensors_buffered(str(out_path))
    wrapper.load_state_dict(result)
    wrapper.hook_to_module()
    try:
        out = transformer(torch.randn(2, tiny_geometry["hidden"]))
        assert torch.isfinite(out).all()
    finally:
        wrapper.remove_hook_from_module()
    wrapper.hook_to_module()
    wrapper.apply_to_module()
    wrapper.remove_hook_from_module()
