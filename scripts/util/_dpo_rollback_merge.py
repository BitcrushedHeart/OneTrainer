"""Two-step rollback bake for the SoReal! lineup (V0.95 -> V0.96 -> V1.0).

A bug crept into the latest OFT, so V1.0 is being re-derived from the V0.95
lineup instead of the poisoned intermediate:

  STEP 1 (V0.95 -> V0.96):  bake V31 (SoReal!_V31_OFT_4) onto the 03Jul V0.95
    lineup at the standing full-strength align policy: Turbo takes the whole
    adapter at 1.0, Base/Lightning share 1.0 on the ~90% of blocks baked
    identically across the lineup, and their late qkv/mlp blocks (layers >= 22)
    are least-squares aligned to the Turbo sentinel at sentinel_strength = 1.0.

  STEP 2 (V0.96 -> V1.0):  bake V32 (SoReal!_V32_OFT_9_1) onto the V0.96 lineup
    at the standing full-strength policy: shared blocks at 1.0, late qkv/mlp
    aligned to the (new) V0.96 Turbo sentinel at sentinel_strength = 1.0.

The existing V1.0 checkpoints are already renamed *_V1.0_bak in 08Jul/, so this
only writes new files. Both V0.96 and V1.0 land in a fresh dated output dir.

Each bake routes through modules.util.oft_merge.merge_oft_adapter (clamped,
orthogonality/NaN/dead-row gated), so every result is bit-equivalent to training
at the given per-module strengths.

USAGE
  READY = False -> dry-run: derive + print every per-model strength and any
    out-of-band warnings, bake nothing. Run first to preview.
  READY = True  -> actually bake both steps.
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

# ============================ CONFIG ============================
READY = True  # False = dry-run preview only; flip True to bake.

ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
LINEUP = ("Base", "Turbo", "Lightning")
SENTINEL = "Turbo"
LATE_START = 22
WARN_LOW, WARN_HIGH = 0.5, 1.5
BASE_MODEL = "Tongyi-MAI/Z-Image"

# Step 1: V31 onto the 03Jul V0.95 lineup -> V0.96, standing full-strength align policy.
V31 = r"E:\AI\Data\Models\Lora\DPO\SoReal!_V31_OFT_4.safetensors"
STEP1_SRC_DIR = os.path.join(ORIGIN, "03Jul")
STEP1_SRC_VER = "V0.95"
STEP1_NEW_VER = "V0.96"
STEP1_TURBO_STRENGTH = 1.0  # Turbo whole adapter @ 1.0; late blocks aligned relative as usual

# Step 2: V32 onto the V0.96 lineup -> V1.0, standing full-strength align policy.
V32 = r"E:\AI\Data\Models\Lora\DPO\Tests4\SoReal!_V32_OFT_4.safetensors"
STEP2_NEW_VER = "V1.0"
STEP2_TURBO_STRENGTH = 1.0

# Reuse an already-baked V0.96 lineup in the output dir instead of re-running step 1.
# Set True when only step 2's adapter changed and step 1's V0.96 files are valid.
REUSE_V096 = True
# ================================================================


def src_path(src_dir: str, ver: str, model: str) -> str:
    return os.path.join(src_dir, f"SoReal!_{model}_{ver}.safetensors")


def module_keys(path: str) -> list[str]:
    suffix = ".oft_R.weight"
    keys = set()
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 -- safe_open is not a dict
            if k.endswith(suffix):
                keys.add(k[: -len(suffix)])
    return sorted(keys)


def derive_strengths(adapter: str, src_dir: str, ver: str, turbo_strength: float) -> dict[str, dict[str, float]]:
    """Per-model late qkv/mlp strengths.

    Turbo (sentinel) is uniform at ``turbo_strength``. Base/Lightning late blocks are
    least-squares aligned to Turbo baked at ``turbo_strength`` (sentinel_strength).
    """
    sentinel_src = src_path(src_dir, ver, SENTINEL)
    out: dict[str, dict[str, float]] = {}
    for model in LINEUP:
        if model == SENTINEL:
            out[model] = {"qkv": turbo_strength, "mlp": turbo_strength}
            continue
        target_src = src_path(src_dir, ver, model)
        if not (os.path.exists(target_src) and os.path.exists(sentinel_src)):
            raise FileNotFoundError(f"missing source for {model}: {target_src}")
        strengths, _ = family_match_strengths(
            adapter,
            sentinel_src,
            target_src,
            late_start=LATE_START,
            sentinel_strength=turbo_strength,
            warn_low=WARN_LOW,
            warn_high=WARN_HIGH,
            label=f"{ver}->{model}",
        )
        out[model] = strengths
    return out


def preview(adapter: str, strengths: dict[str, dict[str, float]], turbo_strength: float) -> None:
    keys = module_keys(adapter)
    print(f"\nadapter: {os.path.basename(adapter)}  ({len(keys)} OFT modules)")
    print(
        f"sentinel = {SENTINEL} @ {turbo_strength}, late_start = layer {LATE_START}, shared base_strength = {turbo_strength}\n"
    )
    for model, fam in strengths.items():
        resolver = make_late_family_strength(LATE_START, fam, base_strength=turbo_strength)
        by_strength: dict[float, list[str]] = {}
        for key in keys:
            by_strength.setdefault(round(resolver(key), 6), []).append(key)
        parts = [f"{len(by_strength[s]):>3} @ {s:.3f}" for s in sorted(by_strength, reverse=True)]
        print(f"{model:>10}: qkv={fam['qkv']:.4f} mlp={fam['mlp']:.4f}  ->  " + " | ".join(parts))


def bake_step(
    adapter: str,
    src_dir: str,
    src_ver: str,
    new_ver: str,
    out_dir: str,
    strengths: dict[str, dict[str, float]],
    turbo_strength: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n########## BAKE {os.path.basename(adapter)}: {src_ver} -> {new_ver}  (out: {out_dir}) ##########")
    for model, fam in strengths.items():
        src = src_path(src_dir, src_ver, model)
        if not os.path.exists(src):
            raise FileNotFoundError(f"source checkpoint missing for {model}: {src}")
        final_output = os.path.join(out_dir, f"SoReal!_{model}_{new_ver}.safetensors")
        tmp_output = final_output.replace(".safetensors", ".NEW.safetensors")
        resolver = make_late_family_strength(LATE_START, fam, base_strength=turbo_strength)

        print(f"\n===== merging {model}: {os.path.basename(adapter)} onto {os.path.basename(src)} =====")
        print(f"      shared base={turbo_strength:.3f}, late qkv={fam['qkv']:.4f} mlp={fam['mlp']:.4f}")

        ui = TrainConfig.default_values()
        ui.base_model_name = BASE_MODEL
        ui.transformer.model_name = src
        ui.vae.model_name = ""

        try:
            report = merge_oft_adapter(
                oft_adapter_path=adapter,
                output_path=tmp_output,
                output_dtype=DataType.BFLOAT_16,
                output_format=ModelFormat.SAFETENSORS,
                ui_train_config=ui,
                key_strength=resolver,
                compute_device="cuda",
            )
        except MergeVerificationError as exc:
            print(f"\n=== MERGE GATE FAILED ({model} {new_ver}) ===\n{exc}")
            if os.path.exists(tmp_output):
                os.remove(tmp_output)
            raise

        os.replace(tmp_output, final_output)
        new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
        print(f"=== MERGE REPORT ({model} {new_ver}) === modules={len(report.post_zero_rows)} new_dead_rows={new_dead}")
        print("output:", final_output)


def main() -> None:
    out_dir = os.path.join(ORIGIN, datetime.now().strftime("%d%b"))

    print("=" * 78)
    print("STEP 1: V31 -> V0.96  (standing full-strength policy, late aligned to V0.95 Turbo@1.0)")
    print("=" * 78)
    s1 = derive_strengths(V31, STEP1_SRC_DIR, STEP1_SRC_VER, STEP1_TURBO_STRENGTH)
    preview(V31, s1, STEP1_TURBO_STRENGTH)

    print("\n" + "=" * 78)
    print("STEP 2: V32 -> V1.0  (standing full-strength policy, late aligned to V0.96 Turbo@1.0)")
    print("=" * 78)
    # Step 2 aligns against the V0.96 lineup, which only exists after step 1 bakes.
    if READY:
        pass  # strengths derived after step 1 writes V0.96 below
    else:
        v096_exists = all(os.path.exists(src_path(out_dir, STEP1_NEW_VER, m)) for m in LINEUP)
        if v096_exists:
            s2 = derive_strengths(V32, out_dir, STEP1_NEW_VER, STEP2_TURBO_STRENGTH)
            preview(V32, s2, STEP2_TURBO_STRENGTH)
        else:
            print(f"(V0.96 not yet baked in {out_dir}; step-2 strengths derive at bake time.)")

    if not READY:
        print("\nREADY = False -> dry-run only, nothing baked. Set READY = True to bake both steps.")
        return

    v096_ready = all(os.path.exists(src_path(out_dir, STEP1_NEW_VER, m)) for m in LINEUP)
    if REUSE_V096 and v096_ready:
        print(f"\nREUSE_V096 = True and V0.96 lineup present in {out_dir}; skipping step-1 bake.")
    else:
        bake_step(V31, STEP1_SRC_DIR, STEP1_SRC_VER, STEP1_NEW_VER, out_dir, s1, STEP1_TURBO_STRENGTH)

    print("\nderiving step-2 strengths against the freshly-baked V0.96 lineup...")
    s2 = derive_strengths(V32, out_dir, STEP1_NEW_VER, STEP2_TURBO_STRENGTH)
    preview(V32, s2, STEP2_TURBO_STRENGTH)
    bake_step(V32, out_dir, STEP1_NEW_VER, STEP2_NEW_VER, out_dir, s2, STEP2_TURBO_STRENGTH)

    print("\nDONE. V0.96 and V1.0 lineups written to", out_dir)


if __name__ == "__main__":
    main()
