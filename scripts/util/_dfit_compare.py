"""Old-pipeline vs new-pipeline dfit comparison for one adapter revision.

Reads the ORIGINAL Base-trained adapter plus two finished conversion outputs
per variant -- an old-pipeline ``*_dfit`` file (``--refine-iters 0`` semantics
or a pre-refinement build) and a new-pipeline one -- and scores what each file
ACTUALLY wrote. Per late tensor:

1. rebuild the dense transport R_L/R_R from Base -> variant (same damped
   Gauss-Newton as the conversion) and form the education-anchored target
   ``T = W_v + R_L (M_b - W_b) R_R``; ``||T - W_v||`` IS the education norm,
   so every normalized error below reads as "fraction of the education
   missed",
2. decode each candidate's stored (oft_R vec, dora_multiplier) through the
   module's reload path (clip + Neumann-5) and report ``||D - T|| / ||T -
   W_v||`` for: transplant (original adapter verbatim), old_dfit, new_dfit,
3. report how far the written parameters moved between pipelines
   (``||vec_new - vec_old|| / ||vec_old||``).

Provenance is enforced, not assumed: each dfit file's ``soreal_dense_fit``
metadata must name the SAME source adapter passed via --adapter and the
matching variant, or the script exits 1 -- with several live revisions on disk
(V40_3/_5/_6/_9_1, V41_*) a silent cross-revision comparison would be worse
than no comparison.

Note the target is the NEW formulation for both candidates. That is the point:
it is the correct target (the old form differs from it by distillation residue
the student should never have chased), so the old pipeline is scored against
what it SHOULD have fit, and the "old_dfit" column measures exactly what
shipping the old converter cost.

Same I/O and threading discipline as _oft_dense_retarget.py: buffered reads,
files closed before any linalg, one CPU thread, TF32 off.

Run: venv/Scripts/python.exe scripts/util/_dfit_compare.py --adapter <final>
     --old-dir <folder with old-pipeline dfit files> --new-dir <folder with new>
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
from modules.util.oft_align import reconstruct_oft_rotation  # noqa: E402
from modules.util.oft_conjugate import clipped_skew_from_vec  # noqa: E402
from modules.util.oft_dense_fit import dense_transport_fit, neumann5_rotation, student_merged  # noqa: E402
from modules.util.transfer_util import _oft_config_from_metadata  # noqa: E402

LINEUP = r"E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul"
DEFAULT_BASE = rf"{LINEUP}\SoReal!_Base_V1.1.safetensors"
DEFAULT_VARIANTS = (
    rf"Turbo={LINEUP}\SoReal!_Turbo_V1.1.safetensors",
    rf"Lightning={LINEUP}\SoReal!_Lightning_V1.1.safetensors",
)
CANDIDATES = ("transplant", "old_dfit", "new_dfit")


def _late_oft_items(adapter: dict, late_start: int) -> list[tuple[str, int, str, str]]:
    from scripts.util._oft_dense_retarget import _late_oft_items as impl

    return impl(adapter, late_start)


def _check_provenance(md: dict, label: str, adapter_name: str, variant: str) -> str | None:
    """None if the dfit file's metadata matches this comparison, else the error."""
    raw = md.get("soreal_dense_fit")
    if raw is None:
        return f"{label}: no soreal_dense_fit metadata -- not a dfit conversion output"
    rep = json.loads(raw)
    if rep.get("source_adapter") != adapter_name:
        return f"{label}: built from {rep.get('source_adapter')!r}, not the --adapter {adapter_name!r}"
    if rep.get("variant") != variant:
        return f"{label}: built for variant {rep.get('variant')!r}, expected {variant!r}"
    return None


def _deployed_err(vec, mult, wv, cfg, target) -> float:
    return float((student_merged(wv, reconstruct_oft_rotation(vec, cfg), mult) - target).norm())


def compare_tensor(key, dora_key, adapter, old_fit, new_fit, wb, wv, cfg, args, device) -> dict:
    """Target build + three-candidate deployed errors for one late tensor."""

    def _pair(src):
        vec = src[key].to(device=device, dtype=torch.float32)
        mult = src[dora_key].to(device=device, dtype=torch.float32) if dora_key in src else None
        return vec, mult

    vec, mult = _pair(adapter)
    vec_old, mult_old = _pair(old_fit)
    vec_new, mult_new = _pair(new_fit)

    q_b, _ = clipped_skew_from_vec(vec, cfg)
    m_b = student_merged(wb, neumann5_rotation(q_b), mult)
    dense = dense_transport_fit(wb, wv, iters=args.dense_iters, tol=args.dense_tol)
    target = wv + dense.r_left @ (m_b - wb) @ dense.r_right
    denom = float((target - wv).norm())  # the education norm, by orthogonal invariance

    err = {
        "transplant": _deployed_err(vec, mult, wv, cfg, target),
        "old_dfit": _deployed_err(vec_old, mult_old, wv, cfg, target),
        "new_dfit": _deployed_err(vec_new, mult_new, wv, cfg, target),
    }
    vec_move = float((vec_new - vec_old).norm()) / max(float(vec_old.norm()), torch.finfo(torch.float32).tiny)
    return {
        "shape": list(wb.shape),
        "dense_fit_residual": dense.rel_residual,
        "education_norm": denom,
        "err": err,
        "err_norm": {k: v / denom if denom > 0.0 else 0.0 for k, v in err.items()},
        "vec_rel_change": vec_move,
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
        fams[fam]["dense_fit_residual_mean"] = sum(r["dense_fit_residual"] for r in sub) / len(sub)
        fams[fam]["vec_rel_change_mean"] = sum(r["vec_rel_change"] for r in sub) / len(sub)
        fams[fam]["vec_rel_change_max"] = max(r["vec_rel_change"] for r in sub)
    return fams


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="the ORIGINAL Base-trained adapter both dfit files came from")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--variant", action="append", metavar="NAME=PATH")
    ap.add_argument("--old-dir", required=True, help="folder holding {adapter_stem}_{Variant}_dfit.safetensors (old)")
    ap.add_argument("--new-dir", required=True, help="folder holding {adapter_stem}_{Variant}_dfit.safetensors (new)")
    ap.add_argument("--out", default=None, help="JSON output path (default: new-dir/dfit_compare_{adapter}.json)")
    ap.add_argument("--late-start", type=int, default=22)
    ap.add_argument("--dense-iters", type=int, default=30)
    ap.add_argument("--dense-tol", type=float, default=1e-9)
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
    adapter_name = Path(args.adapter).name
    adapter_stem = Path(args.adapter).stem

    w_base = read_layer_weights_buffered(args.base, stems)
    device = torch.device(args.device)
    report = {
        "adapter": args.adapter,
        "base": args.base,
        "old_dir": args.old_dir,
        "new_dir": args.new_dir,
        "config": {
            "late_start": args.late_start,
            "dense_iters": args.dense_iters,
            "dense_tol": args.dense_tol,
            "device": args.device,
        },
        "variants": {},
    }

    for name, ckpt in parsed:
        fits = {}
        for label, folder in (("old_dfit", args.old_dir), ("new_dfit", args.new_dir)):
            path = Path(folder) / f"{adapter_stem}_{name}_dfit.safetensors"
            if not path.exists():
                print(f"ERROR: {label} file missing: {path}")
                return 1
            tensors, md = read_safetensors_buffered(str(path))
            problem = _check_provenance(md, f"{label} ({path.name})", adapter_name, name)
            if problem:
                print(f"ERROR: {problem}")
                return 1
            fits[label] = tensors
            rep = json.loads(md["soreal_dense_fit"])
            print(f"{name} {label}: {path.name} (refine_iters {rep.get('refine_iters', 'pre-refine build')})")

        print(f"{name}: comparing {len(items)} late tensors against {Path(ckpt).name}")
        w_variant = read_layer_weights_buffered(ckpt, stems)
        rows = []
        for key, _layer, stem, fam in items:
            dora_key = key[: -len("oft_R.weight")] + "dora_multiplier"
            wb = w_base[stem].to(device=device, dtype=torch.float32)
            wv = w_variant[stem].to(device=device, dtype=torch.float32)
            row = {
                "stem": stem,
                "family": fam,
                **compare_tensor(key, dora_key, adapter, fits["old_dfit"], fits["new_dfit"], wb, wv, cfg, args, device),
            }
            rows.append(row)
            print(
                f"    {stem}: err/edu transplant {row['err_norm']['transplant']:.4f} "
                f"old {row['err_norm']['old_dfit']:.4f} new {row['err_norm']['new_dfit']:.4f}  "
                f"vec-moved {row['vec_rel_change']:.4f}",
                flush=True,
            )
            del wb, wv
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del w_variant, fits
        report["variants"][name] = {"tensors": rows, "families": _family_aggregates(rows)}

        print(f"  {name} family aggregates (err / education norm):")
        for fam, stats in report["variants"][name]["families"].items():
            line = f"    {fam}:"
            for cand in CANDIDATES:
                line += f"  {cand} mean {stats[cand]['mean']:.4f} max {stats[cand]['max']:.4f}"
            line += f"  | vec-moved mean {stats['vec_rel_change_mean']:.4f}"
            print(line)

    out_path = Path(args.out) if args.out else Path(args.new_dir) / f"dfit_compare_{adapter_stem}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
