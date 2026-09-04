"""Bake OFT / DoRA-OFT adapters into the SoReal! release checkpoints via
OneTrainer's clamped, NaN-gated merge path (modules.util.oft_merge.merge_oft_adapter),
so each result is bit-equivalent to training at the given strength.

Produces the three release variants per run:
  * Base       - style/preference OFT partial-baked onto the previous Base.
  * Distilled  - same style OFT partial-baked onto the previous Distilled.
  * Lightning  - the few-step OFT full-baked onto THIS run's Distilled output
                 (so it inherits the style bake), matching how it was trained.

Outputs land in a date-stamped subfolder (e.g. .../SoReal!/02Jul) of the origin
folder, one merged file per target. Each target merges independently so a
verification-gate failure on one does not abort the others.

The NEW dora_multiplier / OFT format is RELATIVE (no baked initial_norm), so a
learned adapter applies faithfully to ANY base -- clean cross-base merges.
"""

from __future__ import annotations

import os
from datetime import datetime

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.oft_merge import merge_oft_adapter
from modules.util.oft_verify import MergeVerificationError

# Style/preference OFT (V30) full-baked onto every V0.94 release variant to produce V0.95.
STYLE_ADAPTER = r"E:\AI\Data\Models\Lora\SoReal!_V30_OFT.safetensors"

ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"

# Source (previous) and target (new) release versions.
PREV_VER = "V0.94"
NEW_VER = "V0.95"

out_dir = os.path.join(ORIGIN, datetime.now().strftime("%d%b"))

# (label, adapter, transformer source, output filename, strength)
# V0.95 BAKE (2026-07-03): the V30 style/preference OFT full-baked (strength=1.0) onto each
# of the four V0.94 checkpoints independently -- Base/Turbo/Distilled/Lightning each get the
# same adapter applied directly to their own V0.94 source.
PREV_DIR = os.path.join(ORIGIN, "28Jun")
TARGETS = [
    (
        variant.lower(),
        STYLE_ADAPTER,
        os.path.join(PREV_DIR, f"SoReal!_{variant}_{PREV_VER}.safetensors"),
        f"SoReal!_{variant}_{NEW_VER}.safetensors",
        1.0,
    )
    for variant in ("Base", "Turbo", "Distilled", "Lightning")
]

os.makedirs(out_dir, exist_ok=True)
print(f"[merge] output dir: {out_dir}")

for label, adapter, transformer, out_name, strength in TARGETS:
    final_output = os.path.join(out_dir, out_name)
    # Bake to a temp sibling and atomically replace the final only on success, so an
    # intentional overwrite (this is a re-bake) never destroys the existing good file
    # if the merge or its verification gate fails partway.
    output = final_output.replace(".safetensors", ".NEW.safetensors")
    # Skip cleanly if the source checkpoint is missing.
    if not os.path.exists(transformer):
        print(f"\n[merge] SKIP {label}: source checkpoint missing ({transformer})")
        continue
    print(f"\n========== merging {label}: adapter={os.path.basename(adapter)} strength={strength} ==========")
    print(f"           into {transformer}")

    ui = TrainConfig.default_values()
    ui.base_model_name = "Tongyi-MAI/Z-Image"
    ui.transformer.model_name = transformer
    ui.vae.model_name = ""

    try:
        report = merge_oft_adapter(
            oft_adapter_path=adapter,
            output_path=output,
            output_dtype=DataType.BFLOAT_16,
            output_format=ModelFormat.SAFETENSORS,
            ui_train_config=ui,
            strength=strength,
            compute_device="cuda",
        )
    except MergeVerificationError as exc:
        print(f"\n=== MERGE GATE FAILED ({label}) ===\n{exc}")
        if os.path.exists(output):
            os.remove(output)
        continue

    os.replace(output, final_output)  # atomically publish the V0.95 output only on success
    print(f"\n=== MERGE REPORT ({label}) ===")
    print("modules merged:", len(report.post_zero_rows))
    new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
    print("total NEW dead rows created by merge:", new_dead)
    print("output:", final_output)
