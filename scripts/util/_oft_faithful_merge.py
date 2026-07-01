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

ADAPTER = r"E:\AI\Data\Models\Lora\DPO\SoReal!_DPO_6.safetensors"
ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
SRC = os.path.join(ORIGIN, "26Jun")

# (label, transformer checkpoint, output filename)
# NEW dora_multiplier format is RELATIVE (no baked initial_norm), so the learned
# magnitude applies faithfully to ANY base -> both the Base (the literal training
# base) and Distilled targets are clean merges, with none of the cross-base
# magnitude approximation the old absolute dora_scale/initial_norm format had.
#
# V0.94 bakes the DPO_6 OFT-ONLY adapter (block 128, scaled_oft OFF, ||R-I|| ~0.022,
# to_out.0 hotspot ~0.055) onto the V0.93 base+distilled AT STRENGTH 0.9 -- partial
# bake: 0.9*baked + 0.1*base per layer (see strength= below). DoRA gate no-ops
# (OFT-only) and is skipped at strength!=1 anyway; orthogonality/NaN/dead-row still run.
TARGETS = [
    (
        "base (training base)",
        os.path.join(SRC, "SoReal!_Base_V0.93.safetensors"),
        "SoReal!_Base_V0.94.safetensors",
    ),
    (
        "distilled",
        os.path.join(SRC, "SoReal!_Distilled_V0.93.safetensors"),
        "SoReal!_Distilled_V0.94.safetensors",
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
            strength=0.9,
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
