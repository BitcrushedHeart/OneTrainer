"""End-to-end tests for scripts/util/_warmstart_benchmark.py.

The benchmark is the proof instrument for the dense-transport least-squares
warm start: on tiny synthetic files with a DENSE late-layer transport it must
produce the full per-tensor error table (raw + normalized against the
education-in-variant-frame effect ||T - W_v||) for all three candidates and
show the ordering

    err(dense-lsq) <= err(block-conj) <= err(transplant)

on the family means. The block-conj candidate is recomputed via the
modules/util/oft_conjugate.py API inside the benchmark (never read from _conj
output files).
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "util" / "_warmstart_benchmark.py"
_spec = importlib.util.spec_from_file_location("warmstart_benchmark_script", _SCRIPT_PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

# reuse the dense-transport variant builder from the retarget IO test
_RETARGET_TEST = Path(__file__).resolve().parent / "test_oft_dense_retarget_io.py"
_rspec = importlib.util.spec_from_file_location("oft_dense_retarget_io_test", _RETARGET_TEST)
retarget_test = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(retarget_test)

LATE_START = 2


@pytest.fixture
def bench_result(tmp_path, make_tiny_adapter, make_ckpt_tensors, write_tiny_ckpt):
    adapter_path = make_tiny_adapter(tmp_path / "adapter.safetensors", dora=True, theta=2e-2)
    base = make_ckpt_tensors()
    variant, _ = retarget_test.dense_transported_variant(base, angle=0.15)
    base_path = write_tiny_ckpt(tmp_path / "base.safetensors", base)
    variant_path = write_tiny_ckpt(tmp_path / "variant.safetensors", variant)

    out_json = tmp_path / "bench.json"
    rc = script.main(
        [
            "--adapter",
            str(adapter_path),
            "--base",
            str(base_path),
            "--variant",
            f"Toy={variant_path}",
            "--out",
            str(out_json),
            "--late-start",
            str(LATE_START),
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    assert out_json.exists()
    return json.loads(out_json.read_text())


def test_report_structure(bench_result, tiny_geometry):
    assert "Toy" in bench_result["variants"]
    toy = bench_result["variants"]["Toy"]
    tensors = toy["tensors"]
    assert len(tensors) == 2 * len(tiny_geometry["stem_shapes"])
    for row in tensors:
        assert row["family"] in ("qkv", "mlp")
        for cand in ("transplant", "block_conj", "dense_lsq"):
            assert row["err"][cand] > 0.0
            assert row["err_norm"][cand] == pytest.approx(row["err"][cand] / row["denom"], rel=1e-6)
        assert 0.0 <= row["dense_fit_residual"] < 1e-3  # exact dense transport
        assert row["block_fit_residual"] > row["dense_fit_residual"]  # dense transport is not block-expressible
        assert row["denom"] > 0.0
        assert len(row["als_objective_history"]) >= 2
    fams = toy["families"]
    assert set(fams) == {"qkv", "mlp"}
    for stats in fams.values():
        for cand in ("transplant", "block_conj", "dense_lsq"):
            assert stats[cand]["mean"] > 0.0
            assert stats[cand]["max"] >= stats[cand]["mean"]


def test_dense_lsq_beats_both_baselines(bench_result):
    """The proof criterion, at tiny scale: on a dense transport the dense-lsq
    warm start must be strictly best on every family mean, and block-conj must
    not lose to transplant."""
    for fam, stats in bench_result["variants"]["Toy"]["families"].items():
        dense = stats["dense_lsq"]["mean"]
        conj = stats["block_conj"]["mean"]
        transplant = stats["transplant"]["mean"]
        assert dense < conj, f"{fam}: dense-lsq {dense} not below block-conj {conj}"
        assert conj <= transplant * 1.02, f"{fam}: block-conj {conj} worse than transplant {transplant}"


def test_per_tensor_distribution_present_and_floor_reported(bench_result):
    """The dense-lsq residual floor (the fraction of T the parameterization
    cannot reach) is the number that decides whether additive cross-block
    capacity is needed -- it must be reported per tensor and per family."""
    toy = bench_result["variants"]["Toy"]
    for row in toy["tensors"]:
        assert row["err_norm"]["dense_lsq"] == pytest.approx(row["dense_lsq_floor"], rel=1e-9)
    for stats in toy["families"].values():
        assert "dense_lsq" in stats
