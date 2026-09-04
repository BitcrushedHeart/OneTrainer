"""Dense-transport least-squares warm starts for a Base-trained adapter.

Auto-detects the adapter family from its tensor keys and supports OFT,
DoRA-OFT, LoRA, DoRA (LoRA + magnitude), LoKr, and DoRA-LoKr. LoHa is
rejected explicitly.

The distillation delta on the late qkv/mlp tensors (layers >= 22) is ~99.95% a
DENSE two-sided rotation ``W_v ~= R_L W_b R_R``, but only ~10-20% of the delta
energy is block-expressible at the adapter partition -- so conjugating the skew
generator by the blockwise-fitted rotation (_oft_conjugate_retarget.py) barely
moves the adapter. This tool goes through the dense transport instead, per
late tensor:

1. recover dense R_L/R_R (damped Gauss-Newton in W_b's singular frame),
2. form the education-anchored teacher target T = W_v + R_L (M_b - W_b) R_R
   with M_b the ORIGINAL adapter's merged teacher weight (DoRA rows included)
   -- only the education delta rides the transport, so the dense fit's
   residual (distillation residue, not education) never leaks into T,
3. fit the adapter's DEPLOYED form to T on the variant weights:
   - OFT / DoRA-OFT (modules/util/oft_dense_fit.py): blockwise-Procrustes ALS,
     exact-Cayley re-encode, then LBFGS through the module's actual decode
     (Neumann-5, clip enforced by projection; --refine-iters, keep-best),
   - LoRA (modules/util/peft_dense_fit.py): one truncated SVD -- globally
     optimal, refinement skipped,
   - DoRA: axis-scale/low-rank ALS with the stored dora_scale recovered as
     c * (norm(WP) + eps), plus keep-best LBFGS refinement,
   - LoKr / DoRA-LoKr: Van Loan nearest-Kronecker fit at the stored factor
     layout, plus keep-best LBFGS refinement;
4. write the fitted tensors; everything outside the late families is copied
   bit-exact.

Outputs ``{adapter_stem}_{Variant}_dfit.safetensors`` in --out-dir (the suffix
never collides with the conjugation tool's ``_conj`` outputs), meant to be
loaded via ``lora_model_name`` as the warm start of a transfer-mode run.

Reads never mmap the big checkpoints (modules/util/adapter_io.py buffered
readers); math runs only after every file is closed, single CPU thread per the
Windows-MKL convention. Heavy linalg runs on --device (default cuda when
available) with TF32 forced OFF -- the ~1e-3 TF32 matmul error is not below
the signal these fits chase.

Run: venv/Scripts/python.exe scripts/util/_oft_dense_retarget.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # multithreaded Windows MKL segfaults on batched linalg
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.adapter_io import read_layer_weights_buffered, read_safetensors_buffered  # noqa: E402
from modules.util.oft_dense_fit import dense_warmstart_for_tensor  # noqa: E402
from modules.util.peft_dense_fit import (  # noqa: E402
    dense_warmstart_for_peft_tensor,
    detect_adapter_kind,
    lokr_params_from_tensors,
    lora_params_from_tensors,
    marker_suffixes,
)
from modules.util.transfer_util import _oft_config_from_metadata  # noqa: E402

LINEUP = r"E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul"
DEFAULT_ADAPTER = r"E:\AI\Data\Packages\OneTrainer\models\v40\SoReal!_V40_OFT.safetensors"
DEFAULT_BASE = rf"{LINEUP}\SoReal!_Base_V1.1.safetensors"
DEFAULT_VARIANTS = (
    rf"Turbo={LINEUP}\SoReal!_Turbo_V1.1.safetensors",
    rf"Lightning={LINEUP}\SoReal!_Lightning_V1.1.safetensors",
)


_OFT_KEY = re.compile(r"^(?:transformer\.)?layers\.(\d+)\.(.+)\.oft_R\.weight$")


def _late_oft_items(adapter: dict, late_start: int) -> list[tuple[str, int, str, str]]:
    """[(adapter_key, layer, module_stem, family)] for late-family oft_R vectors
    (same selection as _oft_conjugate_retarget.py)."""
    from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS

    stem_to_fam = {s: f for f, ss in DEFAULT_LATE_FAMILY_STEMS.items() for s in ss}
    items = []
    for k in adapter:
        m = _OFT_KEY.match(k)
        if not m:
            continue
        layer, stem = int(m.group(1)), m.group(2)
        fam = stem_to_fam.get(stem)
        if fam is not None and layer >= late_start:
            items.append((k, layer, f"layers.{layer}.{stem}", fam))
    return sorted(items, key=lambda t: (t[1], t[2]))


def _late_peft_items(adapter: dict, late_start: int, kind: str) -> list[tuple[str, int, str, str]]:
    """[(stem_key_prefix, layer, module_stem, family)] for late-family adapted
    modules of a LoRA / DoRA / LoKr adapter. ``stem_key_prefix`` includes the
    trailing dot (``transformer.layers.N.attention.to_q.``)."""
    from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS

    stem_to_fam = {s: f for f, ss in DEFAULT_LATE_FAMILY_STEMS.items() for s in ss}
    key_re = re.compile(
        r"^((?:transformer\.)?layers\.(\d+)\.(.+)\.)(?:"
        + "|".join(re.escape(s.lstrip(".")) for s in marker_suffixes(kind))
        + r")$"
    )
    items = []
    for k in adapter:
        m = key_re.match(k)
        if not m:
            continue
        layer, stem = int(m.group(2)), m.group(3)
        fam = stem_to_fam.get(stem)
        if fam is not None and layer >= late_start:
            items.append((m.group(1), layer, f"layers.{layer}.{stem}", fam))
    return sorted(items, key=lambda t: (t[1], t[2]))


class _PeftFamilyStats:
    def __init__(self):
        self.dense_resid = []
        self.floor = []  # ||D - T|| / ||T - W_v||: the parameterization floor
        self.fit_first = []
        self.fit_last = []
        self.pre_refine = []
        self.refined = []

    def as_dict(self) -> dict:
        n = max(len(self.dense_resid), 1)
        return {
            "dense_residual_mean": sum(self.dense_resid) / n,
            "dense_residual_max": max(self.dense_resid, default=0.0),
            "dense_lsq_floor_mean": sum(self.floor) / max(len(self.floor), 1),
            "dense_lsq_floor_max": max(self.floor, default=0.0),
            "fit_objective_first_mean": sum(self.fit_first) / max(len(self.fit_first), 1),
            "fit_objective_last_mean": sum(self.fit_last) / max(len(self.fit_last), 1),
            "deployed_err_pre_refine_mean": sum(self.pre_refine) / max(len(self.pre_refine), 1),
            "deployed_err_pre_refine_max": max(self.pre_refine, default=0.0),
            "deployed_err_mean": sum(self.refined) / max(len(self.refined), 1),
            "deployed_err_max": max(self.refined, default=0.0),
        }


def peft_retarget_variant(adapter, kind, metadata, w_base, w_variant, args):
    """Fitted adapter tensors + report dict for one variant (LoRA/DoRA/LoKr)."""
    from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS

    device = torch.device(args.device)
    ot_cfg = json.loads(metadata.get("ot_config", "{}"))
    norm_eps = bool(ot_cfg.get("lora_decompose_norm_epsilon", True))
    items = _late_peft_items(adapter, args.late_start, kind)
    out = {k: v.contiguous() for k, v in adapter.items()}
    stats = {f: _PeftFamilyStats() for f in DEFAULT_LATE_FAMILY_STEMS}

    for key_prefix, _layer, stem, fam in items:
        if kind in ("lora", "dora"):
            p = lora_params_from_tensors(adapter, key_prefix, norm_eps=norm_eps, device=device)
        else:
            p = lokr_params_from_tensors(adapter, key_prefix, device=device)
        wb = w_base[stem].to(device=device, dtype=torch.float32)
        wv = w_variant[stem].to(device=device, dtype=torch.float32)

        ws = dense_warmstart_for_peft_tensor(
            kind,
            p,
            wb,
            wv,
            dense_iters=args.dense_iters,
            dense_tol=args.dense_tol,
            als_iters=args.als_iters,
            refine_iters=args.refine_iters,
        )
        for suffix, tensor in ws.tensors_new.items():
            full_key = key_prefix + suffix
            out[full_key] = tensor.to(device="cpu", dtype=adapter[full_key].dtype).contiguous()

        denom = float((ws.target - wv).norm())
        floor = ws.deployed_err / denom if denom > 0.0 else 0.0
        st = stats[fam]
        st.dense_resid.append(ws.dense.rel_residual)
        st.floor.append(floor)
        st.fit_first.append(ws.fit_history[0])
        st.fit_last.append(ws.fit_history[-1])
        st.pre_refine.append(ws.deployed_err_pre_refine)
        st.refined.append(ws.deployed_err)
        refine_note = (
            f"  refine {ws.deployed_err_pre_refine:.4g}->{ws.deployed_err:.4g} ({ws.refine.steps} steps)"
            if ws.refine is not None
            else ""
        )
        print(
            f"    {stem}: dense-resid {ws.dense.rel_residual:.4f}  floor {floor:.4f}  "
            f"fit {ws.fit_history[0]:.4g}->{ws.fit_history[-1]:.4g}{refine_note}",
            flush=True,
        )
        del ws, p, wb, wv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out, {
        "per_family": {f: s.as_dict() for f, s in stats.items() if s.dense_resid},
        "late_tensors": len(items),
    }


class _FamilyStats:
    def __init__(self):
        self.dense_resid = []
        self.floor = []  # ||D - T|| / ||T - W_v||: the parameterization floor
        self.neumann_dev = 0.0
        self.sigma_max = 0.0
        self.clipped_blocks = 0
        self.als_first = []
        self.als_last = []
        self.pre_refine = []  # deployed err of the ALS+encode result
        self.refined = []  # deployed err of what is actually written
        self.refine_projected = 0  # blocks clip-projected in written tensors

    def as_dict(self) -> dict:
        n = max(len(self.dense_resid), 1)
        return {
            "dense_residual_mean": sum(self.dense_resid) / n,
            "dense_residual_max": max(self.dense_resid, default=0.0),
            "dense_lsq_floor_mean": sum(self.floor) / max(len(self.floor), 1),
            "dense_lsq_floor_max": max(self.floor, default=0.0),
            "als_objective_first_mean": sum(self.als_first) / max(len(self.als_first), 1),
            "als_objective_last_mean": sum(self.als_last) / max(len(self.als_last), 1),
            "deployed_err_pre_refine_mean": sum(self.pre_refine) / max(len(self.pre_refine), 1),
            "deployed_err_pre_refine_max": max(self.pre_refine, default=0.0),
            "deployed_err_mean": sum(self.refined) / max(len(self.refined), 1),
            "deployed_err_max": max(self.refined, default=0.0),
            "refine_projected_blocks": self.refine_projected,
            "neumann_dev_max": self.neumann_dev,
            "sigma_max": self.sigma_max,
            "clipped_blocks": self.clipped_blocks,
        }


def dense_retarget_variant(adapter, cfg, w_base, w_variant, args):
    """Fitted adapter tensors + report dict for one variant."""
    from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS

    device = torch.device(args.device)
    items = _late_oft_items(adapter, args.late_start)
    out = {k: v.contiguous() for k, v in adapter.items()}
    stats = {f: _FamilyStats() for f in DEFAULT_LATE_FAMILY_STEMS}

    for key, _layer, stem, fam in items:
        vec = adapter[key].to(device=device, dtype=torch.float32)
        dora_key = key[: -len("oft_R.weight")] + "dora_multiplier"
        mult = adapter[dora_key].to(device=device, dtype=torch.float32) if dora_key in adapter else None
        wb = w_base[stem].to(device=device, dtype=torch.float32)
        wv = w_variant[stem].to(device=device, dtype=torch.float32)

        ws = dense_warmstart_for_tensor(
            vec,
            mult,
            wb,
            wv,
            cfg,
            dense_iters=args.dense_iters,
            dense_tol=args.dense_tol,
            als_iters=args.als_iters,
            refine_iters=args.refine_iters,
        )
        out[key] = ws.vec_new.to(device="cpu", dtype=adapter[key].dtype).contiguous()
        if ws.mult_new is not None:
            out[dora_key] = ws.mult_new.to(device="cpu", dtype=adapter[dora_key].dtype).contiguous()

        denom = float((ws.target - wv).norm())
        floor = ws.deployed_err / denom if denom > 0.0 else 0.0
        st = stats[fam]
        st.dense_resid.append(ws.dense.rel_residual)
        st.floor.append(floor)
        st.neumann_dev = max(st.neumann_dev, ws.encode.neumann_dev)
        st.sigma_max = max(st.sigma_max, ws.encode.sigma_max)
        st.clipped_blocks += ws.encode.clipped_blocks
        st.als_first.append(ws.student.objective_history[0])
        st.als_last.append(ws.student.objective_history[-1])
        st.pre_refine.append(ws.deployed_err_pre_refine)
        st.refined.append(ws.deployed_err)
        if ws.refine is not None:
            st.refine_projected += ws.refine.projected_blocks
        refine_note = (
            f"  refine {ws.deployed_err_pre_refine:.4g}->{ws.deployed_err:.4g} ({ws.refine.steps} steps)"
            if ws.refine is not None
            else ""
        )
        print(
            f"    {stem}: dense-resid {ws.dense.rel_residual:.4f}  floor {floor:.4f}  "
            f"als {ws.student.objective_history[0]:.4g}->{ws.student.objective_history[-1]:.4g}{refine_note}",
            flush=True,
        )
        del ws, vec, mult, wb, wv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out, {
        "per_family": {f: s.as_dict() for f, s in stats.items() if s.dense_resid},
        "late_tensors": len(items),
    }


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
    ap.add_argument("--dense-iters", type=int, default=30, help="max Gauss-Newton iterations per tensor")
    ap.add_argument("--dense-tol", type=float, default=1e-9, help="early-exit when a step improves < tol * ||delta||")
    ap.add_argument("--als-iters", type=int, default=4, help="alternating least-squares iterations for the student fit")
    ap.add_argument(
        "--refine-iters",
        type=int,
        default=60,
        help="LBFGS refinement steps through the module's actual decode path, keep-best (0 disables)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
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
    try:
        kind = detect_adapter_kind(adapter.keys())
    except (ValueError, NotImplementedError) as e:
        print(f"ERROR: {args.adapter}: {e}")
        return 1
    is_oft = kind in ("oft", "dora_oft")
    if is_oft:
        cfg = _oft_config_from_metadata(metadata, args.adapter)
        items = _late_oft_items(adapter, args.late_start)
    else:
        cfg = None
        items = _late_peft_items(adapter, args.late_start, kind)
    print(f"adapter kind: {kind}")
    if not items:
        print(f"ERROR: no late-family {kind} tensors at layers >= {args.late_start} in {args.adapter}")
        return 1
    stems = {stem for _, _, stem, _ in items}
    w_base = read_layer_weights_buffered(args.base, stems)

    adapter_stem = Path(args.adapter).stem
    # One folder per adapter revision keeps the Lora root uncluttered across
    # the lineup's many train->merge->retrain iterations.
    out_folder = Path(args.out_dir) / "warmstarts" / adapter_stem
    out_folder.mkdir(parents=True, exist_ok=True)
    for name, ckpt in parsed:
        out_path = out_folder / f"{adapter_stem}_{name}_dfit.safetensors"
        if out_path.exists() and not args.force:
            print(f"SKIP {out_path.name}: exists (use --force)")
            continue
        print(f"{name}: dense-fitting {len(items)} late tensors against {Path(ckpt).name}")
        w_variant = read_layer_weights_buffered(ckpt, stems)

        if is_oft:
            out, report = dense_retarget_variant(adapter, cfg, w_base, w_variant, args)
        else:
            out, report = peft_retarget_variant(adapter, kind, metadata, w_base, w_variant, args)
        del w_variant

        for fam, s in report["per_family"].items():
            line = (
                f"  {fam}: dense-resid mean {s['dense_residual_mean']:.4f} max {s['dense_residual_max']:.4f}  "
                f"floor mean {s['dense_lsq_floor_mean']:.4f} max {s['dense_lsq_floor_max']:.4f}  "
                f"refine {s['deployed_err_pre_refine_mean']:.4g}->{s['deployed_err_mean']:.4g}"
            )
            if is_oft:
                line += f"  clipped {s['clipped_blocks']}  projected {s['refine_projected_blocks']}"
            print(line)

        out_md = dict(metadata)
        out_md["soreal_dense_fit"] = json.dumps(
            {
                "tool": Path(__file__).name,
                "adapter_kind": kind,
                "source_adapter": Path(args.adapter).name,
                "base_checkpoint": Path(args.base).name,
                "variant": name,
                "variant_checkpoint": Path(ckpt).name,
                "late_start": args.late_start,
                "dense_iters": args.dense_iters,
                "dense_tol": args.dense_tol,
                "als_iters": args.als_iters,
                "refine_iters": args.refine_iters,
                **report,
            }
        )
        save_file(out, str(out_path), metadata=out_md)
        print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
