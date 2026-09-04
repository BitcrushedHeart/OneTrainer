"""End-to-end tests for scripts/util/_dfit_compare.py.

Produces two real conversion outputs on tiny synthetic lineups -- an
old-pipeline build (--refine-iters 0) and a new-pipeline build -- then runs
the comparison CLI on them. Checks the scoring (new never worse than old,
transplant worst, education-norm denominator), the JSON report, and the
provenance guard that refuses to compare dfit files built from a different
adapter revision.
"""

import importlib.util
import json
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retarget = _load("oft_dense_retarget_script", "scripts/util/_oft_dense_retarget.py")
compare = _load("dfit_compare_script", "scripts/util/_dfit_compare.py")

LATE_START = 2  # tiny lineup: layers 0-1 early, 2-3 late


def _dense_rotation(n: int, angle: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    skew = torch.randn(n, n, generator=gen, dtype=torch.float64)
    skew = skew - skew.T
    skew = skew * (angle / torch.linalg.matrix_norm(skew, ord=2))
    return torch.matrix_exp(skew).to(torch.float32)


def _noisy_variant(base_tensors, *, angle=0.15, noise=0.05, seed=70):
    """Variant = dense transport + noise on late layers (the regime where the
    education-anchored target and the old form actually differ)."""
    gen = torch.Generator().manual_seed(seed + 5000)
    out = {}
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
        clean = r_left @ wb @ r_right
        n = torch.randn(out_f, in_f, generator=gen)
        out[stem] = clean + n * (noise * float((clean - wb).norm()) / float(n.norm()))
    return out


def _convert(tmp_path, adapter_path, base_path, variant_path, out_name, refine_iters):
    out_dir = tmp_path / out_name
    out_dir.mkdir(exist_ok=True)
    rc = retarget.main(
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
            "--refine-iters",
            str(refine_iters),
        ]
    )
    assert rc == 0
    return out_dir / "warmstarts" / Path(adapter_path).stem


def _compare(tmp_path, adapter_path, base_path, variant_path, old_dir, new_dir, *extra):
    out_json = tmp_path / "compare.json"
    rc = compare.main(
        [
            "--adapter",
            str(adapter_path),
            "--base",
            str(base_path),
            "--variant",
            f"Toy={variant_path}",
            "--old-dir",
            str(old_dir),
            "--new-dir",
            str(new_dir),
            "--out",
            str(out_json),
            "--late-start",
            str(LATE_START),
            "--device",
            "cpu",
            *extra,
        ]
    )
    return rc, out_json


def test_compare_end_to_end(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt, tiny_geometry):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant = _noisy_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    old_dir = _convert(tmp_path, adapter_path, base_path, variant_path, "old", refine_iters=0)
    new_dir = _convert(tmp_path, adapter_path, base_path, variant_path, "new", refine_iters=8)

    rc, out_json = _compare(tmp_path, adapter_path, base_path, variant_path, old_dir, new_dir)
    assert rc == 0
    report = json.loads(out_json.read_text())

    rows = report["variants"]["Toy"]["tensors"]
    assert len(rows) == 2 * len(tiny_geometry["stem_shapes"])
    for row in rows:
        # education-norm denominator: err_norm is err / ||T - W_v|| exactly
        assert abs(row["err_norm"]["new_dfit"] * row["education_norm"] - row["err"]["new_dfit"]) < 1e-9
        # the refined build is scored from its written tensors: never worse
        assert row["err_norm"]["new_dfit"] <= row["err_norm"]["old_dfit"] + 1e-6
        # both fits must beat the transplant in this dense-transport regime
        assert row["err_norm"]["old_dfit"] < row["err_norm"]["transplant"]
    fams = report["variants"]["Toy"]["families"]
    for stats in fams.values():
        assert stats["new_dfit"]["mean"] <= stats["old_dfit"]["mean"] + 1e-6
        assert stats["vec_rel_change_max"] >= 0.0


def test_provenance_guard_rejects_wrong_adapter(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    other_path = make_tiny_adapter(tmp_path / "other.safetensors", dora=True, seed=9)
    base = make_ckpt_tensors()
    variant = _noisy_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    old_dir = _convert(tmp_path, adapter_path, base_path, variant_path, "old", refine_iters=0)
    new_dir = _convert(tmp_path, adapter_path, base_path, variant_path, "new", refine_iters=0)

    # the dfit files name adapter.safetensors as their source; pointing the
    # comparison at another adapter revision must hard-fail, not mis-score.
    # (the file lookup uses the adapter stem, so give the dirs matching names)
    for d in (old_dir, new_dir):
        src = d / "adapter_Toy_dfit.safetensors"
        src.rename(d / "other_Toy_dfit.safetensors")
    rc, _ = _compare(tmp_path, other_path, base_path, variant_path, old_dir, new_dir)
    assert rc == 1


def test_missing_dfit_file_fails_cleanly(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True)
    base = make_ckpt_tensors()
    variant = _noisy_variant(base)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    old_dir = _convert(tmp_path, adapter_path, base_path, variant_path, "old", refine_iters=0)
    empty = tmp_path / "empty"
    empty.mkdir()
    rc, _ = _compare(tmp_path, adapter_path, base_path, variant_path, old_dir, empty)
    assert rc == 1
