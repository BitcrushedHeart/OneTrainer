"""Quantitative benchmark: which warm start best matches the transported teacher?

For every late tensor of every variant, computes the deployed-form error
``||D_x - T||_F`` against the frame-transported teacher target
``T = R_L M_b R_R`` (dense transport of the ORIGINAL adapter's merged teacher
weight, DoRA rows included) for the three candidate warm starts:

    transplant  -- the Base-trained vec + dora_multiplier used verbatim,
    block_conj  -- Q_v = R_R_blk^T Q R_R_blk with R_R_blk from the blockwise
                   Procrustes fit at the ADAPTER partition, recomputed here via
                   the modules/util/oft_conjugate.py API (the *_conj output
                   files are never read -- another process may be writing them),
    dense_lsq   -- the dense-transport least-squares fit of this repo's
                   _oft_dense_retarget.py, evaluated through the full Cayley
                   re-encode + module-decode round trip.

Errors are reported raw and normalized by ``||T - W_v||_F`` (the fraction of
the education-in-variant-frame effect the candidate misses). The dense
transport fit residual is reported per tensor because the target is only as
trustworthy as that fit; the dense_lsq normalized error doubles as the
parameterization FLOOR -- the fraction of T unreachable by block-orthogonal +
per-row-scale form, which decides whether additive cross-block capacity is
needed. Aggregates are per family (qkv/mlp) x variant; the full table lands in
--out as JSON.

Same I/O and threading discipline as _oft_dense_retarget.py: buffered reads,
files closed before any linalg, one CPU thread, TF32 off, one tensor resident
at a time.

Run: venv/Scripts/python.exe scripts/util/_warmstart_benchmark.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # multithreaded Windows MKL segfaults on batched linalg
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.adapter_io import read_layer_weights_buffered, read_safetensors_buffered  # noqa: E402
from modules.util.oft_align import block_size_from_elements, reconstruct_oft_rotation  # noqa: E402
from modules.util.oft_conjugate import (  # noqa: E402
    blockwise_procrustes_rotations,
    conjugate_skew_vec,
    pick_divisor_block,
)
from modules.util.oft_dense_fit import dense_warmstart_for_tensor, student_merged  # noqa: E402
from modules.util.transfer_util import _oft_config_from_metadata  # noqa: E402

LINEUP = r"E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul"
DEFAULT_ADAPTER = r"E:\AI\Data\Packages\OneTrainer\models\v40\SoReal!_V40_OFT.safetensors"
DEFAULT_BASE = rf"{LINEUP}\SoReal!_Base_V1.1.safetensors"
DEFAULT_VARIANTS = (
    rf"Turbo={LINEUP}\SoReal!_Turbo_V1.1.safetensors",
    rf"Lightning={LINEUP}\SoReal!_Lightning_V1.1.safetensors",
)
CANDIDATES = ("transplant", "block_conj", "dense_lsq")


def _late_oft_items(adapter: dict, late_start: int) -> list[tuple[str, int, str, str]]:
    from scripts.util._oft_dense_retarget import _late_oft_items as impl

    return impl(adapter, late_start)


def benchmark_tensor(vec, mult, wb, wv, cfg, args) -> dict:
    """The full three-candidate comparison for one late tensor."""
    block = block_size_from_elements(vec.shape[-1])

    # dense-lsq (also produces the shared target T)
    ws = dense_warmstart_for_tensor(
        vec, mult, wb, wv, cfg, dense_iters=args.dense_iters, dense_tol=args.dense_tol, als_iters=args.als_iters
    )
    target = ws.target
    denom = float((target - wv).norm())
    err_dense = ws.deployed_err

    # transplant: Base-trained parameters verbatim on the variant
    err_transplant = float((student_merged(wv, reconstruct_oft_rotation(vec, cfg), mult) - target).norm())

    # block-conj: recomputed at the adapter partition via the existing API
    blk = blockwise_procrustes_rotations(
        wb, wv, block, pick_divisor_block(wb.shape[0], block), iters=args.conj_iters, tol=args.conj_tol
    )
    vec_conj = conjugate_skew_vec(vec, blk.r_right, cfg)
    err_conj = float((student_merged(wv, reconstruct_oft_rotation(vec_conj, cfg), mult) - target).norm())

    err = {"transplant": err_transplant, "block_conj": err_conj, "dense_lsq": err_dense}
    return {
        "shape": list(wb.shape),
        "block": block,
        "delta_norm": float((wv - wb).norm()),
        "denom": denom,
        "dense_fit_residual": ws.dense.rel_residual,
        "block_fit_residual": blk.rel_residual,
        "err": err,
        "err_norm": {k: v / denom if denom > 0.0 else 0.0 for k, v in err.items()},
        "dense_lsq_floor": err_dense / denom if denom > 0.0 else 0.0,
        "als_objective_history": ws.student.objective_history,
        "neumann_dev": ws.encode.neumann_dev,
        "sigma_max": ws.encode.sigma_max,
        "clipped_blocks": ws.encode.clipped_blocks,
    }


def _family_aggregates(rows: list[dict]) -> dict:
    fams: dict[str, dict] = {}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        fams[fam] = {
            cand: {
                "mean": sum(r["err_norm"][cand] for r in sub) / len(sub),
                "max": max(r["err_norm"][cand] for r in sub),
            }
            for cand in CANDIDATES
        }
        fams[fam]["dense_fit_residual"] = {
            "mean": sum(r["dense_fit_residual"] for r in sub) / len(sub),
            "max": max(r["dense_fit_residual"] for r in sub),
        }
        fams[fam]["block_fit_residual"] = {
            "mean": sum(r["block_fit_residual"] for r in sub) / len(sub),
            "max": max(r["block_fit_residual"] for r in sub),
        }
    return fams


def benchmark_variant(adapter, cfg, w_base, w_variant, args) -> dict:
    device = torch.device(args.device)
    rows = []
    for key, _layer, stem, fam in _late_oft_items(adapter, args.late_start):
        vec = adapter[key].to(device=device, dtype=torch.float32)
        dora_key = key[: -len("oft_R.weight")] + "dora_multiplier"
        mult = adapter[dora_key].to(device=device, dtype=torch.float32) if dora_key in adapter else None
        wb = w_base[stem].to(device=device, dtype=torch.float32)
        wv = w_variant[stem].to(device=device, dtype=torch.float32)

        row = {"stem": stem, "family": fam, **benchmark_tensor(vec, mult, wb, wv, cfg, args)}
        rows.append(row)
        print(
            f"    {stem}: dense-resid {row['dense_fit_residual']:.4f} block-resid {row['block_fit_residual']:.4f}  "
            f"err/|T-Wv| transplant {row['err_norm']['transplant']:.4f} "
            f"conj {row['err_norm']['block_conj']:.4f} dense {row['err_norm']['dense_lsq']:.4f}",
            flush=True,
        )
        del vec, mult, wb, wv
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"tensors": rows, "families": _family_aggregates(rows)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--variant", action="append", metavar="NAME=PATH")
    ap.add_argument(
        "--out", default=None, help="JSON output path (default: out-dir/warmstart_benchmark_{adapter}.json)"
    )
    ap.add_argument("--out-dir", default=r"E:\AI\Data\Models\Lora")
    ap.add_argument("--late-start", type=int, default=22)
    ap.add_argument("--dense-iters", type=int, default=30)
    ap.add_argument("--dense-tol", type=float, default=1e-9)
    ap.add_argument("--als-iters", type=int, default=4)
    ap.add_argument("--conj-iters", type=int, default=100, help="blockwise Procrustes sweeps (matches the conj CLI)")
    ap.add_argument("--conj-tol", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    variants = args.variant or list(DEFAULT_VARIANTS)
    parsed = []
    for spec in variants:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            print(f"ERROR: --variant expects NAME=PATH, got {spec!r}")
            return 1
        parsed.append((name, path))

    adapter, metadata = read_safetensors_buffered(args.adapter)
    cfg = _oft_config_from_metadata(metadata, args.adapter)
    items = _late_oft_items(adapter, args.late_start)
    if not items:
        print(f"ERROR: no late-family oft_R tensors at layers >= {args.late_start} in {args.adapter}")
        return 1
    stems = {stem for _, _, stem, _ in items}
    w_base = read_layer_weights_buffered(args.base, stems)

    report = {
        "adapter": args.adapter,
        "base": args.base,
        "config": {
            "late_start": args.late_start,
            "dense_iters": args.dense_iters,
            "dense_tol": args.dense_tol,
            "als_iters": args.als_iters,
            "conj_iters": args.conj_iters,
            "conj_tol": args.conj_tol,
            "device": args.device,
        },
        "variants": {},
    }
    for name, ckpt in parsed:
        print(f"{name}: benchmarking {len(items)} late tensors against {Path(ckpt).name}")
        w_variant = read_layer_weights_buffered(ckpt, stems)
        report["variants"][name] = benchmark_variant(adapter, cfg, w_base, w_variant, args)
        del w_variant

        print(f"  {name} family aggregates (err / ||T - W_v||):")
        for fam, stats in report["variants"][name]["families"].items():
            line = f"    {fam}:"
            for cand in CANDIDATES:
                line += f"  {cand} mean {stats[cand]['mean']:.4f} max {stats[cand]['max']:.4f}"
            line += (
                f"  | dense-fit-resid mean {stats['dense_fit_residual']['mean']:.4f}"
                f"  block-fit-resid mean {stats['block_fit_residual']['mean']:.4f}"
            )
            print(line)

    out_path = (
        Path(args.out) if args.out else Path(args.out_dir) / f"warmstart_benchmark_{Path(args.adapter).stem}.json"
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
