"""Dtype-matrix tests for the conjugation pipeline.

Three independent dtype knobs act at three different places:
  1. file dtype        -- what the adapter file stores (tool preserves it per tensor)
  2. lora_weight_dtype -- cast applied in ZImageLoRASetup via wrapper.to(...);
                          DataType.NONE.torch_dtype() is None => silent no-op
  3. save dtype        -- DtypeModelSaverMixin._convert_state_dict_dtype casts
                          EVERY tensor on save, including scalar marker buffers
                          (bf16 save rounds clipped_oft 0.99 -> 0.98828125 --
                          documented current behavior, not a bug: the clip value
                          difference is immaterial, but it is why transfer-mode
                          startup warns on output_dtype != FLOAT_32)
"""

import importlib.util
from pathlib import Path

from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.adapter_io import read_safetensors_buffered
from modules.util.enum.DataType import DataType

import torch

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "util" / "_oft_conjugate_retarget.py"
_spec = importlib.util.spec_from_file_location("oft_conjugate_retarget_script_dt", _SCRIPT_PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

LATE_START = 2


@pytest.fixture(scope="module")
def conj_outputs(tmp_path_factory):
    """Run the tool once per adapter file dtype; yield {file_dtype: (src, out)}."""
    # Module-scoped, so fixtures from conftest (function-scoped) can't be used;
    # build inputs via the same factories' underlying logic, requested here once.
    import json

    from modules.module.LoRAModule import DoRAOFTModule

    from torch import nn

    from safetensors.torch import save_file

    stem_shapes = {
        "attention.to_q": (16, 16),
        "attention.to_k": (16, 16),
        "attention.to_v": (16, 16),
        "feed_forward.w1": (24, 16),
        "feed_forward.w2": (16, 24),
        "feed_forward.w3": (24, 16),
    }

    def build_adapter(path, file_dtype):
        gen = torch.Generator().manual_seed(0)
        state = {}
        for layer in range(4):
            for stem, (out_f, in_f) in stem_shapes.items():
                module = DoRAOFTModule(
                    prefix="m",
                    orig_module=nn.Linear(in_f, out_f, bias=False),
                    oft_block_size=6,
                    block_share=False,
                    oft_scaled=False,
                    oft_clipped_norm=0.99,
                )
                with torch.no_grad():
                    module.oft_R.weight.copy_(torch.randn(module.oft_R.weight.shape, generator=gen) * 1e-2)
                    module.dora_multiplier.copy_(1.0 + 0.1 * torch.randn(module.dora_multiplier.shape, generator=gen))
                for k, v in module.state_dict().items():
                    state[f"transformer.layers.{layer}.{stem}.{k}"] = v.detach().to(file_dtype).contiguous()
        ot_config = {
            "peft_type": "OFT_2",
            "dora_oft": True,
            "oft_block_size": 6,
            "oft_clipped_norm": 0.99,
            "oft_scaled": False,
            "oft_block_share": False,
            "oft_cans": False,
        }
        save_file(state, str(path), metadata={"ot_config": json.dumps(ot_config)})

    gen = torch.Generator().manual_seed(100)
    base = {
        f"layers.{layer}.{stem}": torch.randn(out_f, in_f, generator=gen)
        for layer in range(4)
        for stem, (out_f, in_f) in stem_shapes.items()
    }
    # A mild but nontrivial transport: rotate late in-blocks by a small skew.
    variant = {}
    rot_gen = torch.Generator().manual_seed(7)
    for stem, wb in base.items():
        if int(stem.split(".")[1]) < LATE_START:
            variant[stem] = wb.clone()
            continue
        in_f = wb.shape[1]
        br = 4 if in_f == 16 else 6
        skew = torch.randn(in_f // br, br, br, generator=rot_gen, dtype=torch.float64)
        skew = skew - skew.transpose(-2, -1)
        skew = skew * (0.1 / torch.linalg.matrix_norm(skew, ord=2)).reshape(-1, 1, 1)
        r_right = torch.matrix_exp(skew).to(torch.float32)
        blocks = wb.reshape(wb.shape[0], in_f // br, br)
        variant[stem] = torch.einsum("ojb,jbc->ojc", blocks, r_right).reshape(wb.shape)

    results = {}
    for file_dtype, tag in ((torch.float32, "f32"), (torch.bfloat16, "bf16")):
        root = tmp_path_factory.mktemp(f"conj_{tag}")
        adapter_path = root / "adapter.safetensors"
        build_adapter(adapter_path, file_dtype)
        base_path = root / "base.safetensors"
        variant_path = root / "variant.safetensors"
        save_file({f"{k}.weight": v.contiguous() for k, v in base.items()}, str(base_path))
        save_file({f"{k}.weight": v.contiguous() for k, v in variant.items()}, str(variant_path))
        rc = script.main(
            [
                "--adapter",
                str(adapter_path),
                "--base",
                str(base_path),
                "--variant",
                f"Toy={variant_path}",
                "--out-dir",
                str(root),
                "--late-start",
                str(LATE_START),
                "--iters",
                "200",
                "--tol",
                "0",
            ]
        )
        assert rc == 0
        results[file_dtype] = (str(adapter_path), str(root / "adapter_Toy_conj.safetensors"))
    return results


@pytest.mark.parametrize("file_dtype", [torch.float32, torch.bfloat16])
def test_output_preserves_file_dtype_per_tensor(conj_outputs, file_dtype):
    src_path, out_path = conj_outputs[file_dtype]
    source, _ = read_safetensors_buffered(src_path)
    result, _ = read_safetensors_buffered(out_path)
    for k, v in result.items():
        assert v.dtype == source[k].dtype == file_dtype, k


@pytest.mark.parametrize("file_dtype", [torch.float32, torch.bfloat16])
def test_markers_and_untouched_keys_bit_exact(conj_outputs, file_dtype):
    src_path, out_path = conj_outputs[file_dtype]
    source, _ = read_safetensors_buffered(src_path)
    result, _ = read_safetensors_buffered(out_path)
    for k in source:
        layer = int(k.replace("transformer.", "").split(".")[1])
        is_late_learned = layer >= LATE_START and k.endswith(".oft_R.weight")
        if not is_late_learned:
            assert torch.equal(result[k], source[k]), k


def test_bf16_conjugation_close_to_f32(conj_outputs):
    """The bf16-file pipeline computes in fp32 internally; its conjugated
    vectors must match the fp32-file run within bf16 resolution."""
    _, out32 = conj_outputs[torch.float32]
    _, out16 = conj_outputs[torch.bfloat16]
    r32, _ = read_safetensors_buffered(out32)
    r16, _ = read_safetensors_buffered(out16)
    for k, v in r32.items():
        if k.endswith(".oft_R.weight"):
            torch.testing.assert_close(r16[k].float(), v.float(), rtol=4e-2, atol=2e-3)


@pytest.mark.parametrize(
    ("lwd", "expected"),
    [
        (DataType.FLOAT_32, torch.float32),
        (DataType.BFLOAT_16, torch.bfloat16),
        (DataType.NONE, None),  # documented no-op: raw file dtype is kept
    ],
)
@pytest.mark.parametrize("file_dtype", [torch.float32, torch.bfloat16])
def test_lora_weight_dtype_cast_semantics(conj_outputs, file_dtype, lwd, expected):
    """The setup-time hop: tensor.to(DataType.torch_dtype()); NONE is a no-op."""
    _, out_path = conj_outputs[file_dtype]
    result, _ = read_safetensors_buffered(out_path)
    key = next(k for k in result if k.endswith(".oft_R.weight"))
    cast = result[key].to(dtype=lwd.torch_dtype())
    assert cast.dtype == (expected if expected is not None else file_dtype)
    if lwd.torch_dtype() is None or lwd.torch_dtype() == file_dtype:
        assert torch.equal(cast, result[key])


class _Saver(DtypeModelSaverMixin):
    pass


@pytest.mark.parametrize(
    ("save_dtype", "rtol"),
    [(torch.float32, 0.0), (torch.bfloat16, 4e-3)],
)
def test_save_cast_and_marker_rounding(conj_outputs, save_dtype, rtol):
    """The save-time hop casts EVERY tensor -- including scalar marker buffers.
    bf16 rounds clipped_oft 0.99 -> 0.98828125: asserted as documented current
    behavior (immaterial to the clip; the transfer-mode startup warning exists
    so overnight runs default to FLOAT_32 saves)."""
    _, out_path = conj_outputs[torch.float32]
    result, _ = read_safetensors_buffered(out_path)
    converted = _Saver()._convert_state_dict_dtype(result, save_dtype)

    marker_key = next(k for k in result if k.endswith(".clipped_oft"))
    assert converted[marker_key].dtype == save_dtype
    if save_dtype == torch.float32:
        assert torch.equal(converted[marker_key], result[marker_key])
        assert float(converted[marker_key]) == pytest.approx(0.99)
    else:
        assert float(converted[marker_key]) == 0.98828125

    vec_key = next(k for k in result if k.endswith(".oft_R.weight"))
    assert converted[vec_key].dtype == save_dtype
    torch.testing.assert_close(converted[vec_key].float(), result[vec_key].float(), rtol=rtol, atol=rtol * 0.1 + 1e-12)
