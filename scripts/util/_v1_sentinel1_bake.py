"""Bake SoReal!_V33_OFT_9 onto the V1.0_bak lineup (Base/Turbo/Lightning), aligning
each model's late qkv/mlp blocks to the Turbo sentinel per the standing policy in
modules/util/oft_align.py. The sentinel itself is baked at sentinel_strength=1.0 (the
default full-strength policy); Base/Lightning strengths are auto-derived by
least-squares against Turbo's merged late blocks -- nothing hand-tuned.

Sources: 08Jul/SoReal!_{model}_V1.0_bak.safetensors
Output:  10Jul/SoReal!_{model}_V1.0.safetensors (overwrites the current V1.0 files)
"""

from __future__ import annotations

import os

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.oft_align import family_match_strengths
from modules.util.oft_merge import make_late_family_strength, merge_oft_adapter
from modules.util.oft_verify import MergeVerificationError

# ======================= CONFIG =======================
READY = True

ADAPTER = r"E:\AI\Data\Models\Lora\DPO\Tests5\SoReal!_V33_OFT_9.safetensors"

ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
SRC_DIR = os.path.join(ORIGIN, "08Jul")
SRC_SUFFIX = "V1.0_bak"

OUT_DIR = os.path.join(ORIGIN, "10Jul")
OUT_SUFFIX = "V1.0"

LINEUP = ("Base", "Turbo", "Lightning")
SENTINEL = "Turbo"
LATE_START = 22
WARN_LOW, WARN_HIGH = 0.5, 1.5
# =======================================================


def _src(model: str) -> str:
    return os.path.join(SRC_DIR, f"SoReal!_{model}_{SRC_SUFFIX}.safetensors")


def _out(model: str) -> str:
    return os.path.join(OUT_DIR, f"SoReal!_{model}_{OUT_SUFFIX}.safetensors")


def derive_strengths() -> dict[str, dict[str, float]]:
    sentinel_src = _src(SENTINEL)
    out: dict[str, dict[str, float]] = {}
    for model in LINEUP:
        if model == SENTINEL:
            out[model] = {"qkv": 1.0, "mlp": 1.0}
            continue
        strengths, _ = family_match_strengths(
            ADAPTER,
            sentinel_src,
            _src(model),
            late_start=LATE_START,
            warn_low=WARN_LOW,
            warn_high=WARN_HIGH,
            label=model,
        )
        out[model] = strengths
    return out


def bake(strengths: dict[str, dict[str, float]]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[merge] adapter: {os.path.basename(ADAPTER)}")
    print(f"[merge] sentinel = {SENTINEL} @ 1.0, late_start = layer {LATE_START}")
    print(f"[merge] output dir: {OUT_DIR}\n")

    for model in LINEUP:
        src = _src(model)
        if not os.path.exists(src):
            print(f"[merge] SKIP {model}: source checkpoint missing ({src})")
            continue

        final_output = _out(model)
        output = final_output.replace(".safetensors", ".NEW.safetensors")

        fam_strength = strengths[model]
        resolver = make_late_family_strength(LATE_START, fam_strength)
        print(f"========== merging {model}: {os.path.basename(src)} -> {os.path.basename(final_output)} ==========")
        print(f"           late qkv={fam_strength['qkv']:.3f} mlp={fam_strength['mlp']:.3f}, everything else 1.0")

        ui = TrainConfig.default_values()
        ui.base_model_name = "Tongyi-MAI/Z-Image"
        ui.transformer.model_name = src
        ui.vae.model_name = ""

        try:
            report = merge_oft_adapter(
                oft_adapter_path=ADAPTER,
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
        print(f"=== MERGE REPORT ({model}) === modules={len(report.post_zero_rows)} new_dead_rows={new_dead}")
        print("output:", final_output, "\n")


def main() -> None:
    strengths = derive_strengths()
    if not READY:
        print("READY = False -> dry-run only, nothing baked.")
        return
    bake(strengths)


if __name__ == "__main__":
    main()
