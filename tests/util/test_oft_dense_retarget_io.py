"""End-to-end tests for scripts/util/_oft_dense_retarget.py.

Runs the CLI's main() on tiny synthetic adapter + Base/variant checkpoints
where the variant is an exact DENSE (not block-diagonal) two-sided rotation of
Base on the late layers -- the regime the dense-transport warm start exists
for. Checks the written ``*_dfit`` adapter: verbatim early/mid/marker keys,
fitted late oft_R vectors and dora multipliers that actually track the
frame-transported teacher target, metadata report, --force semantics, and
loadability into a real LoRAModuleWrapper.
"""

import importlib.util
import json
from pathlib import Path

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.adapter_io import read_safetensors_buffered
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType
from modules.util.oft_align import AdapterOFTConfig, merged_full, reconstruct_oft_rotation

import torch
from torch import nn

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "util" / "_oft_dense_retarget.py"
_spec = importlib.util.spec_from_file_location("oft_dense_retarget_script", _SCRIPT_PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

LATE_START = 2  # tiny lineup: layers 0-1 early, 2-3 late


def _dense_rotation(n: int, angle: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    skew = torch.randn(n, n, generator=gen, dtype=torch.float64)
    skew = skew - skew.T
    skew = skew * (angle / torch.linalg.matrix_norm(skew, ord=2))
    return torch.matrix_exp(skew).to(torch.float32)


def dense_transported_variant(base_tensors, *, angle=0.15, seed=70):
    """Variant = exact per-tensor DENSE transport on late layers, clone elsewhere."""
    out, rotations = {}, {}
    i = 0
    for stem, wb in base_tensors.items():
        layer = int(stem.split(".")[1])
        if layer < LATE_START:
            out[stem] = wb.clone()
            continue
        out_f, in_f = wb.shape
        i += 1
        r_left = _dense_rotation(out_f, angle, seed + i)
        r_right = _dense_rotation(in_f, angle, seed + 1000 + i)
        out[stem] = r_left @ wb @ r_right
        rotations[stem] = (r_left, r_right)
    return out, rotations


def _run_script(tmp_path, adapter_path, base_path, variant_path, *extra, out_name="out"):
    out_dir = tmp_path / out_name
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
            "--device",
            "cpu",
            *extra,
        ]
    )
    # The script nests outputs one folder per adapter revision:
    # {out-dir}/warmstarts/{adapter_stem}/{adapter_stem}_{Variant}_dfit.safetensors
    stem = Path(adapter_path).stem
    return rc, out_dir / "warmstarts" / stem / f"{stem}_Toy_dfit.safetensors"


def test_script_end_to_end(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, tiny_geometry):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant, rotations = dense_transported_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    assert out_path.exists()

    source, _ = read_safetensors_buffered(adapter_path)
    result, metadata = read_safetensors_buffered(str(out_path))
    assert set(result) == set(source)

    report = json.loads(metadata["soreal_dense_fit"])
    assert report["variant"] == "Toy"
    assert report["late_tensors"] == 2 * len(tiny_geometry["stem_shapes"])
    for fam_stats in report["per_family"].values():
        # exact dense transport: the dense fit must reach the fp32 floor
        assert fam_stats["dense_residual_max"] < 1e-3

    cfg = AdapterOFTConfig(block_size=tiny_geometry["block_request"], clip=0.99, scaled=False)
    changed_late_vec = 0
    changed_late_dora = 0
    for key in source:
        layer = int(key.replace("transformer.", "").split(".")[1])
        stem = ".".join(key.replace("transformer.", "").split(".")[2:])
        is_late_family = layer >= LATE_START and ("to_out" not in key)
        if key.endswith(".oft_R.weight") and is_late_family:
            changed_late_vec += not torch.equal(result[key], source[key])
            continue
        if key.endswith(".dora_multiplier") and is_late_family:
            changed_late_dora += not torch.equal(result[key], source[key])
            continue
        assert torch.equal(result[key], source[key]), f"{key} not copied verbatim"
    assert changed_late_vec == 2 * len(tiny_geometry["stem_shapes"])
    assert changed_late_dora == 2 * len(tiny_geometry["stem_shapes"])

    # Deployed-form check on every late tensor: the written (vec', m') must
    # land closer to the frame-transported teacher target T = R_L M_b R_R than
    # the plain transplant does.
    for stem, (r_left, r_right) in rotations.items():
        key = f"transformer.{stem}.oft_R.weight"
        dora_key = f"transformer.{stem}.dora_multiplier"
        wb, wv = base[stem], variant[stem]
        m_b = source[dora_key].view(-1, 1) * merged_full(wb, reconstruct_oft_rotation(source[key], cfg))
        target = r_left @ m_b @ r_right
        d_fit = result[dora_key].view(-1, 1) * merged_full(wv, reconstruct_oft_rotation(result[key], cfg))
        d_transplant = source[dora_key].view(-1, 1) * merged_full(wv, reconstruct_oft_rotation(source[key], cfg))
        err_fit = float((d_fit - target).norm())
        err_transplant = float((d_transplant - target).norm())
        assert err_fit < err_transplant, f"{stem}: dense fit did not beat transplant"


def test_refine_flag_never_regresses_deployed_err(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    """--refine-iters 0 must report identical pre/post deployed errors; with
    refinement on, the written error must never exceed the ALS+encode error
    and never exceed the unrefined run's."""
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant, _ = dense_transported_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, off_path = _run_script(tmp_path, adapter_path, base_path, variant_path, "--refine-iters", "0", out_name="off")
    assert rc == 0
    rc, on_path = _run_script(tmp_path, adapter_path, base_path, variant_path, "--refine-iters", "8", out_name="on")
    assert rc == 0

    _, off_md = read_safetensors_buffered(str(off_path))
    _, on_md = read_safetensors_buffered(str(on_path))
    off_report = json.loads(off_md["soreal_dense_fit"])
    on_report = json.loads(on_md["soreal_dense_fit"])
    assert off_report["refine_iters"] == 0
    assert on_report["refine_iters"] == 8
    for fam, off_stats in off_report["per_family"].items():
        on_stats = on_report["per_family"][fam]
        # refinement disabled: what is written IS the ALS+encode result
        assert off_stats["deployed_err_mean"] == off_stats["deployed_err_pre_refine_mean"]
        # refinement on: keep-best through the true decode path never regresses
        assert on_stats["deployed_err_mean"] <= on_stats["deployed_err_pre_refine_mean"] + 1e-12
        assert on_stats["deployed_err_max"] <= off_stats["deployed_err_max"] + 1e-9
        assert on_stats["refine_projected_blocks"] == 0  # warm-start regime: clip never binds


def test_force_and_skip(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant, _ = dense_transported_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    first_bytes = out_path.read_bytes()
    tampered = first_bytes[:8] + b"\x00" * 8 + first_bytes[16:]
    out_path.write_bytes(tampered)

    rc, _ = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    assert out_path.read_bytes() == tampered  # skipped without --force

    rc, _ = _run_script(tmp_path, adapter_path, base_path, variant_path, "--force")
    assert rc == 0
    assert out_path.read_bytes() != tampered


def test_plain_oft_adapter_works(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=False)
    base = make_ckpt_tensors()
    variant, _ = dense_transported_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    rc, out_path = _run_script(tmp_path, adapter_path, base_path, variant_path)
    assert rc == 0
    result, _ = read_safetensors_buffered(str(out_path))
    assert not any(k.endswith(".dora_multiplier") for k in result)


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


def test_output_loads_into_wrapper(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, tiny_geometry):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant, _ = dense_transported_variant(base)
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
