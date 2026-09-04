"""Bake an OFT adapter into the SoReal! Base / Turbo / Lightning lineup, holding
every block at strength 1.0 EXCEPT the late qkv/mlp blocks (layers >= 22), whose
strength is AUTO-DERIVED to align each model to the Turbo sentinel.

Standing policy (modules/util/oft_align.py): Turbo is the sentinel -- the DPO /
preference data is developed against Turbo's outputs, and Base/Turbo/Lightning
differ only in their late qkv_late / mlp_late rotations. So every adapter baked
across the lineup aligns its late blocks to Turbo's as closely as a scalar merge
strength allows; the shared 90% of the network is baked identically at 1.0. The
per-model late strengths are computed at merge time by least-squares against Turbo
(no hand-tuned numbers), and a family whose strength leaves [0.5, 1.5] raises a
warning -- a scalar can't sensibly align that block and needs a human look.

USAGE
  1. DRY-RUN by default (READY = False): computes + prints the auto-derived per-model
     strengths and any warnings, and bakes nothing. Run it now to preview.
  2. When the DPO run finishes: point FINAL_ADAPTER at the final checkpoint, set
     READY = True, and run again to bake. Strengths are re-derived from the final
     checkpoint automatically -- nothing to copy by hand.

The merge routes through modules.util.oft_merge.merge_oft_adapter (clamped, NaN/dead-row
gated), so each result is bit-equivalent to training at the given per-module strengths.
"""

from __future__ import annotations

import os
from datetime import datetime

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.oft_align import family_match_strengths
from modules.util.oft_merge import make_late_family_strength, merge_oft_adapter
from modules.util.oft_verify import MergeVerificationError

from safetensors import safe_open

# ======================= CONFIG (edit before baking) =======================
READY = True  # keep False = dry-run preview only; flip True to actually bake.

# Adapter to bake across the lineup.
FINAL_ADAPTER = r"E:\AI\Data\Models\Lora\SoReal!_V39_OFT_9.safetensors"

LATE_START = 22  # first "late" layer; qkv_late/mlp_late = layers >= this
WARN_LOW, WARN_HIGH = 0.5, 1.5  # warn if an aligned family strength leaves this band

ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
SRC_DIR = os.path.join(ORIGIN, "10Jul")  # source checkpoints to bake onto
SRC_VER = "V1.0"
NEW_VER = "V1.1"
SENTINEL = "Turbo"  # the reference model every other variant aligns to
LINEUP = ("Base", "Turbo", "Lightning")
# ===========================================================================


def _src(model: str) -> str:
    return os.path.join(SRC_DIR, f"SoReal!_{model}_{SRC_VER}.safetensors")


def derive_strengths() -> dict[str, dict[str, float]]:
    """Auto-derive per-model late qkv/mlp strengths by aligning to the Turbo sentinel.

    The sentinel aligns to itself at exactly 1.0 (skipped). Warnings surface any
    family whose strength leaves [WARN_LOW, WARN_HIGH].
    """
    sentinel_src = _src(SENTINEL)
    out: dict[str, dict[str, float]] = {}
    for model in LINEUP:
        if model == SENTINEL:
            out[model] = {"qkv": 1.0, "mlp": 1.0}
            continue
        target_src = _src(model)
        if not (os.path.exists(target_src) and os.path.exists(sentinel_src)):
            out[model] = {"qkv": 1.0, "mlp": 1.0}
            continue
        strengths, _ = family_match_strengths(
            FINAL_ADAPTER,
            sentinel_src,
            target_src,
            late_start=LATE_START,
            warn_low=WARN_LOW,
            warn_high=WARN_HIGH,
            label=model,
        )
        out[model] = strengths
    return out


def adapter_module_keys(path: str) -> list[str]:
    """Every OFT module prefix in the adapter (key stripped of '.oft_R.weight')."""
    suffix = ".oft_R.weight"
    keys = set()
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 -- safe_open is not a dict
            if k.endswith(suffix):
                keys.add(k[: -len(suffix)])
    return sorted(keys)


def preview(module_keys: list[str], strengths: dict[str, dict[str, float]]) -> None:
    """Print, per model, how many modules land at each strength and list the non-1.0 ones."""
    print(f"\nadapter: {os.path.basename(FINAL_ADAPTER)}  ({len(module_keys)} OFT modules)")
    print(f"sentinel = {SENTINEL}, late_start = layer {LATE_START}\n")
    for model, fam_strength in strengths.items():
        resolver = make_late_family_strength(LATE_START, fam_strength)
        by_strength: dict[float, list[str]] = {}
        for key in module_keys:
            by_strength.setdefault(round(resolver(key), 6), []).append(key)
        parts = [f"{len(by_strength[s]):>3} @ {s:.3f}" for s in sorted(by_strength, reverse=True)]
        print(f"{model:>10}: " + " | ".join(parts))
        for s in sorted(by_strength):
            if s == 1.0:
                continue
            sample = by_strength[s]
            lyrs = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in sample})
            fams = sorted({".".join(k.split(".")[-2:]).rstrip("0").rstrip(".") for k in sample})
            print(f"             -> {len(sample)} modules @ {s:.3f}: layers {lyrs[0]}-{lyrs[-1]}, {fams}")
    print()


def bake(strengths: dict[str, dict[str, float]]) -> None:
    out_dir = os.path.join(ORIGIN, datetime.now().strftime("%d%b"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[merge] output dir: {out_dir}")

    for model, fam_strength in strengths.items():
        src = _src(model)
        if not os.path.exists(src):
            print(f"\n[merge] SKIP {model}: source checkpoint missing ({src})")
            continue
        final_output = os.path.join(out_dir, f"SoReal!_{model}_{NEW_VER}.safetensors")
        output = final_output.replace(".safetensors", ".NEW.safetensors")

        resolver = make_late_family_strength(LATE_START, fam_strength)
        print(
            f"\n========== merging {model}: {os.path.basename(FINAL_ADAPTER)} onto {os.path.basename(src)} =========="
        )
        print(f"           late qkv={fam_strength['qkv']:.3f} mlp={fam_strength['mlp']:.3f}, everything else 1.0")

        ui = TrainConfig.default_values()
        ui.base_model_name = "Tongyi-MAI/Z-Image"
        ui.transformer.model_name = src
        ui.vae.model_name = ""

        try:
            report = merge_oft_adapter(
                oft_adapter_path=FINAL_ADAPTER,
                output_path=output,
                output_dtype=DataType.BFLOAT_16,
                output_format=ModelFormat.SAFETENSORS,
                ui_train_config=ui,
                key_strength=resolver,
                compute_device="cuda",
            )
        except MergeVerificationError as exc:
            print(f"\n=== MERGE GATE FAILED ({model}) ===\n{exc}")
            if os.path.exists(output):
                os.remove(output)
            continue

        os.replace(output, final_output)
        new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
        print(f"\n=== MERGE REPORT ({model}) === modules={len(report.post_zero_rows)} new_dead_rows={new_dead}")
        print("output:", final_output)


def main() -> None:
    module_keys = adapter_module_keys(FINAL_ADAPTER)
    print("deriving late qkv/mlp strengths by aligning each model to the Turbo sentinel...")
    strengths = derive_strengths()
    preview(module_keys, strengths)
    if not READY:
        print("READY = False -> dry-run only, nothing baked.")
        print("To bake: point FINAL_ADAPTER at the finished checkpoint, set READY = True, re-run.")
        print("Strengths are re-derived from that checkpoint automatically.")
        return
    bake(strengths)


if __name__ == "__main__":
    main()
