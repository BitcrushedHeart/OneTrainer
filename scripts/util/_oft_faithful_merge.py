"""Bake an OFT / DoRA-OFT adapter into one or more target checkpoints via
OneTrainer's clamped, NaN-gated merge path (modules.util.oft_merge.merge_oft_adapter),
so the result is bit-equivalent to training at strength=1.0.

Outputs land in a date-stamped subfolder (e.g. .../SoReal!/31May) of the origin
folder, one merged file per target. Each target is merged independently so a
verification-gate failure on one (e.g. a cross-base DoRA magnitude mismatch)
does not abort the others.
"""

from __future__ import annotations

import os
from datetime import datetime

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.oft_merge import merge_oft_adapter
from modules.util.oft_verify import MergeVerificationError

ADAPTER = r"E:\AI\Data\Models\Lora\SoReal!_V26_OFT_7.safetensors"
ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
SRC = os.path.join(ORIGIN, "08Jun")

# (label, transformer checkpoint, output filename)
# NEW dora_multiplier format is RELATIVE (no baked initial_norm), so the learned
# magnitude applies faithfully to ANY base -> both the Base (the literal training
# base) and Distilled targets are clean merges, with none of the cross-base
# magnitude approximation the old absolute dora_scale/initial_norm format had.
#
# V0.91 bakes V26 OFT ckpt 7 (the clean, pre-CANS converged checkpoint; scaled_oft
# + relative dora_multiplier, no CANS/clip markers) onto the V0.90 base+distilled.
# V26 was trained directly on SoReal!_Base_V0.90 (confirmed via the adapter's
# embedded transformer.model_name), so this is the natural V0.90 -> V0.91 lineage
# and a clean fresh training state.
TARGETS = [
    (
        "base (training base)",
        os.path.join(SRC, "SoReal!_Base_V0.90.safetensors"),
        "SoReal!_Base_V0.91.safetensors",
    ),
    (
        "distilled",
        os.path.join(SRC, "SoReal!_Distilled_V0.90.safetensors"),
        "SoReal!_Distilled_V0.91.safetensors",
    ),
]

out_dir = os.path.join(ORIGIN, datetime.now().strftime("%d%b"))
os.makedirs(out_dir, exist_ok=True)
print(f"[merge] adapter: {ADAPTER}")
print(f"[merge] output dir: {out_dir}")

for label, transformer, out_name in TARGETS:
    output = os.path.join(out_dir, out_name)
    # Idempotent re-runs: skip targets whose output already exists and is plausibly
    # complete (>1 GB guards against counting a truncated failed-save stub as done).
    if os.path.exists(output) and os.path.getsize(output) > 1_000_000_000:
        print(f"\n[merge] SKIP {label}: output already exists ({output})")
        continue
    print(f"\n========== merging into {label}: {transformer} ==========")

    ui = TrainConfig.default_values()
    ui.base_model_name = "Tongyi-MAI/Z-Image"
    ui.transformer.model_name = transformer
    ui.vae.model_name = ""

    try:
        report = merge_oft_adapter(
            oft_adapter_path=ADAPTER,
            output_path=output,
            output_dtype=DataType.BFLOAT_16,
            output_format=ModelFormat.SAFETENSORS,
            ui_train_config=ui,
            strength=1.0,
            compute_device="cuda",
        )
    except MergeVerificationError as exc:
        print(f"\n=== MERGE GATE FAILED ({label}) ===\n{exc}")
        continue

    print(f"\n=== MERGE REPORT ({label}) ===")
    print("modules merged:", len(report.post_zero_rows))
    new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
    print("total NEW dead rows created by merge:", new_dead)
    print("output:", output)
