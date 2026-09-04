"""Conjugate a Base-trained OFT / DoRA-OFT adapter into per-variant warm-start adapters.

The distillation delta on the late qkv/mlp tensors (layers >= 22) is ~99.95% a
two-sided block rotation ``W_v ~= R_L W_b R_R`` (_distill_rotation_diag.py /
_distill_rotation_blockfit.py). A Base-trained adapter's skew generator lives in
Base's input frame; transplanted verbatim it rotates the variant's late blocks
off-axis. This tool recovers the per-block input-side rotation ``R_R`` for each
late tensor (alternating blockwise Procrustes, at the ADAPTER'S actual block
partition) and writes the frame-correct generator ``Q_v = R_R^T Q_b R_R`` --
exact in the Cayley-Neumann parameterization; R_L cancels; the spectral clip is
preserved (see modules/util/oft_conjugate.py). Everything outside the late
families (early/mid layers, attention.to_out.0, marker buffers) is copied
bit-exact.

DoRA multipliers are per-output-row and do not conjugate. Default policy
``copy`` transports them unchanged (correct when R_L ~= I; the fit's
left-identity deviation is checked against --dora-left-tol and warned on
breach). ``scale`` applies the retarget-style map ``m -> 1 + s(m-1)`` with
--dora-scale. Plain OFT adapters (no dora_multiplier keys) work unchanged.

Outputs ``{adapter_stem}_{Variant}_conj.safetensors`` next to --out-dir, meant
to be loaded via ``lora_model_name`` as the warm start of a transfer-mode
training run (and usable directly as the fallback deliverable).

Reads never mmap the big checkpoints (modules/util/adapter_io.py buffered
readers); math runs only after every file is closed. Single-threaded by
default per the Windows-MKL convention -- the fit is a loop of non-batched
SVDs, ~1-3 h per variant for the full 48-tensor set at default settings.
--threads N lifts the limit at your own risk (the known segfault is batched
linalg; per-matrix SVD loops have been stable multithreaded).

Run: venv/Scripts/python.exe scripts/util/_oft_conjugate_retarget.py [--verify]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # multithreaded Windows MKL segfaults on batched linalg

from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.adapter_io import read_layer_weights_buffered, read_safetensors_buffered  # noqa: E402
from modules.util.oft_align import (  # noqa: E402
    block_size_from_elements,
    merged_full,
    reconstruct_oft_rotation,
)
from modules.util.oft_conjugate import (  # noqa: E402
    apply_block_rotations,
    blockwise_procrustes_rotations,
    conjugate_skew_vec,
    pick_divisor_block,
    rotation_agreement,
    transport_dora_multiplier,
)
from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS  # noqa: E402
from modules.util.transfer_util import _oft_config_from_metadata  # noqa: E402

LINEUP = r"E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul"
DEFAULT_ADAPTER = r"E:\AI\Data\Models\Lora\SoReal!_V40_OFT_9.safetensors"
DEFAULT_BASE = rf"{LINEUP}\SoReal!_Base_V1.1.safetensors"
DEFAULT_VARIANTS = (
    rf"Turbo={LINEUP}\SoReal!_Turbo_V1.1.safetensors",
    rf"Lightning={LINEUP}\SoReal!_Lightning_V1.1.safetensors",
)

_STEM_TO_FAM = {s: f for f, ss in DEFAULT_LATE_FAMILY_STEMS.items() for s in ss}
_OFT_KEY = re.compile(r"^(?:transformer\.)?layers\.(\d+)\.(.+)\.oft_R\.weight$")
_DORA_KEY = re.compile(r"^(?:transformer\.)?layers\.(\d+)\.(.+)\.dora_multiplier$")
_QKV_STEMS = ("attention.to_q", "attention.to_k", "attention.to_v")


def _late_oft_items(adapter: dict[str, torch.Tensor], late_start: int) -> list[tuple[str, int, str, str]]:
    """[(adapter_key, layer, module_stem, family)] for late-family oft_R vectors."""
    items = []
    for k in adapter:
        m = _OFT_KEY.match(k)
        if not m:
            continue
        layer, stem = int(m.group(1)), m.group(2)
        fam = _STEM_TO_FAM.get(stem)
        if fam is not None and layer >= late_start:
            items.append((k, layer, f"layers.{layer}.{stem}", fam))
    return sorted(items, key=lambda t: (t[1], t[2]))


class _FamilyStats:
    def __init__(self):
        self.resid = []
        self.left_dev = 0.0
        self.right_dev = 0.0
        self.verify = 0.0
        self.vec_change = []
        self.conj_vs_transplant = []  # conj_err / transplant_err vs the fitted-transport reference

    def as_dict(self) -> dict:
        n = max(len(self.resid), 1)
        d = {
            "rel_residual_mean": sum(self.resid) / n,
            "rel_residual_max": max(self.resid, default=0.0),
            "left_identity_dev_max": self.left_dev,
            "right_identity_dev_max": self.right_dev,
            "verify_rel_max": self.verify,
        }
        if self.vec_change:
            d["vec_rel_change_mean"] = sum(self.vec_change) / len(self.vec_change)
            d["vec_rel_change_max"] = max(self.vec_change)
        if self.conj_vs_transplant:
            d["conj_vs_transplant_ratio_max"] = max(self.conj_vs_transplant)
        return d


def conjugate_variant(
    adapter: dict[str, torch.Tensor],
    cfg,
    w_base: dict[str, torch.Tensor],
    w_variant: dict[str, torch.Tensor],
    args,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Conjugated adapter tensors + report dict for one variant."""
    compute_dtype = torch.float64 if args.fp64 else torch.float32
    items = _late_oft_items(adapter, args.late_start)

    out = {k: v.contiguous() for k, v in adapter.items()}
    stats: dict[str, _FamilyStats] = {f: _FamilyStats() for f in DEFAULT_LATE_FAMILY_STEMS}
    left_dev_by_stem: dict[str, float] = {}
    qkv_rights: dict[int, dict[str, torch.Tensor]] = {}
    qkv_agreement_max = 0.0

    for key, layer, stem, fam in items:
        vec = adapter[key].to(compute_dtype)
        wb = w_base[stem].to(compute_dtype)
        wv = w_variant[stem].to(compute_dtype)

        block_right = block_size_from_elements(vec.shape[-1])
        block_left = pick_divisor_block(wb.shape[0], block_right)
        fit = blockwise_procrustes_rotations(wb, wv, block_right, block_left, iters=args.iters, tol=args.tol)

        vec_v = conjugate_skew_vec(vec, fit.r_right, cfg)
        out[key] = vec_v.to(adapter[key].dtype).contiguous()

        st = stats[fam]
        st.resid.append(fit.rel_residual)
        st.left_dev = max(st.left_dev, fit.left_identity_dev)
        st.right_dev = max(st.right_dev, fit.right_identity_dev)
        left_dev_by_stem[stem] = fit.left_identity_dev
        vec_norm = float(vec.norm())
        if vec_norm > 0.0:
            st.vec_change.append(float((vec_v - vec).norm()) / vec_norm)

        if args.verify:
            # NOTE on semantics: the merged equation M_v_conj == R_L M_b R_R is
            # exact only when the transport itself is exactly block-diagonal;
            # the fit error passes straight through, so this residual tracks
            # the (possibly large) non-block-diagonal fraction of the delta --
            # it is NOT an internal-consistency check of the conjugation. The
            # actionable check is incremental: against the same fitted-transport
            # reference, the conjugated adapter must not be WORSE than the
            # plain transplant. On real lineups with a mostly-dense transport
            # both errors are dominated by the shared fit error, so the ratio
            # sits just under 1; on block-expressible transports it drops far
            # below 1 (see test_oft_conjugate_io.py).
            lhs = merged_full(wv, reconstruct_oft_rotation(vec_v, cfg))
            rhs = apply_block_rotations(
                merged_full(wb, reconstruct_oft_rotation(vec, cfg)),
                r_left=fit.r_left,
                r_right=fit.r_right,
            )
            transplant_lhs = merged_full(wv, reconstruct_oft_rotation(vec, cfg))
            delta = float((wv - wb).norm())
            conj_err = float((lhs - rhs).norm())
            transplant_err = float((transplant_lhs - rhs).norm())
            if delta > 0.0:
                st.verify = max(st.verify, conj_err / delta)
            if transplant_err > 0.0:
                st.conj_vs_transplant.append(conj_err / transplant_err)

        short = stem.rsplit(".", 2)[-1] if stem.endswith(tuple(_QKV_STEMS)) else None
        if short is not None:
            qkv_rights.setdefault(layer, {})[short] = fit.r_right
            group = qkv_rights[layer]
            if len(group) == 3:
                for a, b in (("to_q", "to_k"), ("to_q", "to_v"), ("to_k", "to_v")):
                    qkv_agreement_max = max(qkv_agreement_max, rotation_agreement(group[a], group[b]))
                del qkv_rights[layer]
        print(f"    {stem}: resid {fit.rel_residual:.4f}  L-dev {fit.left_identity_dev:.4f}", flush=True)

    # DoRA multipliers: transport per policy; warn where R_L is not ~identity.
    warned_fams = set()
    for k in adapter:
        m = _DORA_KEY.match(k)
        if not m:
            continue
        layer, mod_stem = int(m.group(1)), m.group(2)
        fam = _STEM_TO_FAM.get(mod_stem)
        if fam is None or layer < args.late_start:
            continue
        stem = f"layers.{layer}.{mod_stem}"
        left_dev = left_dev_by_stem.get(stem, 0.0)
        if args.dora_fallback == "copy" and left_dev > args.dora_left_tol and fam not in warned_fams:
            warned_fams.add(fam)
            print(
                f"  WARNING: {fam} left-rotation deviation {left_dev:.3f} > {args.dora_left_tol} -- "
                f"dora_multiplier copy is only exact for R_L ~= I; the polish run must absorb the rest."
            )
        mult = adapter[k].to(torch.float32)
        moved = transport_dora_multiplier(
            mult, args.dora_fallback, s=args.dora_scale if args.dora_fallback == "scale" else None
        )
        out[k] = moved.to(adapter[k].dtype).contiguous()

    report = {
        "per_family": {f: s.as_dict() for f, s in stats.items() if s.resid},
        "qkv_agreement_max": qkv_agreement_max,
        "late_tensors": len(items),
        "dora_policy": args.dora_fallback,
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE, help="Base checkpoint (the frame the adapter was trained in)")
    ap.add_argument(
        "--variant",
        action="append",
        metavar="NAME=PATH",
        help="variant checkpoint; repeatable (default: Turbo and Lightning V1.1)",
    )
    ap.add_argument("--out-dir", default=r"E:\AI\Data\Models\Lora")
    ap.add_argument("--late-start", type=int, default=22)
    ap.add_argument("--iters", type=int, default=100, help="max alternating Procrustes sweeps per tensor")
    ap.add_argument("--tol", type=float, default=1e-6, help="early-exit when a sweep improves < tol * ||delta||")
    ap.add_argument("--fp64", action="store_true", help="fit and conjugate in float64")
    ap.add_argument("--dora-fallback", choices=("copy", "scale"), default="copy")
    ap.add_argument("--dora-scale", type=float, default=1.0, help="s for --dora-fallback scale: m -> 1 + s(m-1)")
    ap.add_argument("--dora-left-tol", type=float, default=0.1, help="warn when R_L deviates more than this")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (see docstring before raising)")
    ap.add_argument("--verify", action="store_true", help="check M_v == R_L M_b R_R per tensor (slower)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))

    variants = args.variant or list(DEFAULT_VARIANTS)
    parsed = []
    for spec in variants:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            print(f"ERROR: --variant expects NAME=PATH, got {spec!r}")
            return 1
        parsed.append((name, path))

    adapter, metadata = read_safetensors_buffered(args.adapter)
    # Parse ot_config from the already-read metadata: no mmap ever touches the
    # adapter (the repo's no-mmap discipline while training holds memory).
    cfg = _oft_config_from_metadata(metadata, args.adapter)
    items = _late_oft_items(adapter, args.late_start)
    if not items:
        print(f"ERROR: no late-family oft_R tensors at layers >= {args.late_start} in {args.adapter}")
        return 1
    stems = {stem for _, _, stem, _ in items}
    w_base = read_layer_weights_buffered(args.base, stems)

    adapter_stem = Path(args.adapter).stem
    failed = False
    for name, ckpt in parsed:
        out_path = Path(args.out_dir) / f"{adapter_stem}_{name}_conj.safetensors"
        if out_path.exists() and not args.force:
            print(f"SKIP {out_path.name}: exists (use --force)")
            continue
        print(f"{name}: fitting {len(items)} late tensors against {Path(ckpt).name}")
        w_variant = read_layer_weights_buffered(ckpt, stems)

        out, report = conjugate_variant(adapter, cfg, w_base, w_variant, args)
        del w_variant

        for fam, s in report["per_family"].items():
            line = (
                f"  {fam}: resid mean {s['rel_residual_mean']:.4f} max {s['rel_residual_max']:.4f}  "
                f"L-dev {s['left_identity_dev_max']:.4f}"
            )
            if "vec_rel_change_mean" in s:
                line += f"  vec-change mean {s['vec_rel_change_mean']:.4f} max {s['vec_rel_change_max']:.4f}"
            if args.verify:
                line += f"  verify {s['verify_rel_max']:.2e}"
                if "conj_vs_transplant_ratio_max" in s:
                    line += f"  conj/transplant {s['conj_vs_transplant_ratio_max']:.4f}"
            print(line)
        print(f"  qkv R_R agreement max {report['qkv_agreement_max']:.4f}")
        if args.verify:
            # Gate on the incremental claim only: the conjugated adapter must not
            # be worse than the plain transplant against the fitted-transport
            # reference. A large verify residual by itself just means the real
            # transport is mostly non-block-diagonal (reported, not fatal).
            worst_ratio = max(
                (s.get("conj_vs_transplant_ratio_max", 0.0) for s in report["per_family"].values()),
                default=0.0,
            )
            if worst_ratio > 1.001:  # margin: both errors share the fit term, fp jitter sits at ~1e-6
                print(
                    f"  ERROR: conjugation is WORSE than transplant on some tensor "
                    f"(ratio {worst_ratio:.4f} > 1) -- not writing."
                )
                failed = True
                continue

        out_md = dict(metadata)
        out_md["soreal_conjugate"] = json.dumps(
            {
                "tool": Path(__file__).name,
                "source_adapter": Path(args.adapter).name,
                "base_checkpoint": Path(args.base).name,
                "variant": name,
                "variant_checkpoint": Path(ckpt).name,
                "late_start": args.late_start,
                "iters": args.iters,
                "tol": args.tol,
                "fp64": args.fp64,
                **report,
            }
        )
        save_file(out, str(out_path), metadata=out_md)
        print(f"  wrote {out_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
