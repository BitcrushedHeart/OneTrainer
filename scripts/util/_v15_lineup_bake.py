r"""Bake the V1.5 SoReal! lineup: Base + Turbo from the V49 / V50 adapters.

First release whose Turbo adapter came out of transfer mode instead of a
dense-transport (dfit) conversion:

  * Base  takes ``Lora\SoReal!_V49_OFT_7.safetensors``, the DPO DoRA-OFT run
    trained ON Base V1.4 per its own ``transformer.model_name``.
  * Turbo takes ``Lora\SoReal!_V50_OFT.safetensors``, the transfer-mode run
    trained ON Turbo V1.4 (teacher = Base V1.4 + V49_OFT_7, warm-started from
    the V49_OFT_7 Turbo dfit, ``transfer_preserve_distillation`` on).

Frame: each adapter trained on its own 12Sep V1.4 checkpoint, so each bakes
onto exactly that checkpoint. The pre-flight aborts if either adapter's
recorded ``transformer.model_name`` disagrees -- a different frame would
silently bake off-axis education. The 12Sep lineup has no Lightning, so none
is produced.

Same clamped, NaN-gated merge path as _v14_lineup_bake.py
(modules.util.oft_merge.merge_oft_adapter -- bit-equivalent to training), full
strength everywhere (no late-family scaling), bf16 out, atomic .NEW -> replace
on success only.

Outputs land in a fresh dated folder alongside 12Sep. Each target merges
independently, so a gate failure on one does not abort the other.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.DataType import DataType  # noqa: E402
from modules.util.enum.ModelFormat import ModelFormat  # noqa: E402
from modules.util.oft_merge import merge_oft_adapter  # noqa: E402
from modules.util.oft_verify import MergeVerificationError  # noqa: E402

ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
LINEUP_V14 = os.path.join(ORIGIN, "12Sep")
LINEUP_V15 = os.path.join(ORIGIN, datetime.now().strftime("%d%b"))

LORA_DIR = r"E:\AI\Data\Models\Lora"

# (label, adapter, transformer source, output filename)
TARGETS = [
    (
        "Base",
        os.path.join(LORA_DIR, "SoReal!_V49_OFT_7.safetensors"),
        os.path.join(LINEUP_V14, "SoReal!_Base_V1.4.safetensors"),
        "SoReal!_Base_V1.5.safetensors",
    ),
    (
        "Turbo",
        os.path.join(LORA_DIR, "SoReal!_V50_OFT.safetensors"),
        os.path.join(LINEUP_V14, "SoReal!_Turbo_V1.4.safetensors"),
        "SoReal!_Turbo_V1.5.safetensors",
    ),
]


def read_metadata(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


# ---- pre-flight: every path present, and each adapter provably trained on the
# ---- exact checkpoint it is about to be baked onto.
for label, adapter, transformer, _ in TARGETS:
    for path in (adapter, transformer):
        if not os.path.exists(path):
            raise SystemExit(f"ABORT {label}: missing {path}")
    trained_on = json.loads(read_metadata(adapter)["ot_config"])["transformer"]["model_name"]
    if Path(trained_on).name != Path(transformer).name:
        raise SystemExit(f"ABORT {label}: adapter trained on {trained_on!r}, not {Path(transformer).name}")
    print(f"[preflight] {label} frame OK ({Path(trained_on).name})")

os.makedirs(LINEUP_V15, exist_ok=True)
print(f"[merge] output dir: {LINEUP_V15}")

for label, adapter, transformer, out_name in TARGETS:
    final_output = os.path.join(LINEUP_V15, out_name)
    if os.path.exists(final_output):
        print(f"\n[merge] SKIP {label}: output already exists ({final_output})")
        continue

    output = final_output.replace(".safetensors", ".NEW.safetensors")
    print(f"\n========== merging {label} V1.5: adapter={os.path.basename(adapter)} ==========")
    print(f"           onto {transformer}")

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
            strength=1.0,
            compute_device="cuda",
        )
    except MergeVerificationError as exc:
        print(f"\n=== MERGE GATE FAILED ({label}) -- nothing published ===\n{exc}")
        if os.path.exists(output):
            os.remove(output)
        continue

    os.replace(output, final_output)  # publish only on success

    print(f"\n=== MERGE REPORT ({label}) ===")
    print("modules merged:", len(report.post_zero_rows))
    new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
    print("total NEW dead rows created by merge:", new_dead)
    print("output:", final_output)
