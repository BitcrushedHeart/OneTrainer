r"""Bake the V1.4 SoReal! lineup: Base + Turbo + Lightning from the V45 adapter.

2026-09-12 rebuild. Replaces the 26Aug V1.4, which was baked from the SFT+DPO
adapter mislabelled ``models\SoReal!_V41_OFT.safetensors``.

Source adapter is ``Lora\SoReal!_V45_OFT.safetensors``: the SFT+DPO DoRA-OFT
add-on run continued from V43 with the on-image text reward enabled
(``rlhf_mode=SFT_DPO``, ``dora_oft=True``, ``oft_block_size=833``), trained ON
Base V1.3 per its own ``transformer.model_name``. Late-family ``oft_R`` screen:
0.0769 qkv / 0.0902 mlp, in line with V44_10 (0.0652 / 0.0787) and V45_6
(0.0741 / 0.0875) -- no V41-style blow-up.

Frame: the adapter trained on Base V1.3, so every target is its own 09Aug V1.3
checkpoint. Base takes the ORIGINAL adapter; Turbo AND Lightning both take the
Turbo dense-transport (dfit) conversion from _oft_dense_retarget.py -- the two
variants are near-identical, and the Lightning fit costs another ~25 min.

Same clamped, NaN-gated merge path as _v13_lineup_bake.py
(modules.util.oft_merge.merge_oft_adapter -- bit-equivalent to training), full
strength everywhere (no late-family scaling), bf16 out, atomic .NEW -> replace
on success only.

Outputs land in a fresh dated folder alongside 09Aug. Each target merges
independently, so a gate failure on one does not abort the others.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.DataType import DataType  # noqa: E402
from modules.util.enum.ModelFormat import ModelFormat  # noqa: E402
from modules.util.oft_merge import merge_oft_adapter  # noqa: E402
from modules.util.oft_verify import MergeVerificationError  # noqa: E402

LINEUP_V13 = r"E:\AI\Data\Models\StableDiffusion\SoReal!\09Aug"
LINEUP_V14 = r"E:\AI\Data\Models\StableDiffusion\SoReal!\12Sep"

ADAPTER_STEM = "SoReal!_V45_OFT"
ADAPTER = rf"E:\AI\Data\Models\Lora\{ADAPTER_STEM}.safetensors"
DFIT_DIR = rf"E:\AI\Data\Models\Lora\warmstarts\{ADAPTER_STEM}"

# (label, adapter, transformer source, output filename)
TARGETS = [
    (
        variant,
        ADAPTER if variant == "Base" else os.path.join(DFIT_DIR, f"{ADAPTER_STEM}_Turbo_dfit.safetensors"),
        os.path.join(LINEUP_V13, f"SoReal!_{variant}_V1.3.safetensors"),
        f"SoReal!_{variant}_V1.4.safetensors",
    )
    for variant in ("Base", "Turbo", "Lightning")
]


def read_metadata(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


# ---- pre-flight: every path present, and the adapter provably trained in the
# ---- V1.3 frame -- a different frame would silently bake off-axis education.
for label, adapter, transformer, _ in TARGETS:
    for path in (adapter, transformer):
        if not os.path.exists(path):
            raise SystemExit(f"ABORT {label}: missing {path}")

trained_on = json.loads(read_metadata(ADAPTER)["ot_config"])["transformer"]["model_name"]
if Path(trained_on).name != "SoReal!_Base_V1.3.safetensors":
    raise SystemExit(f"ABORT: adapter trained on {trained_on!r}, not Base V1.3")
print(f"[preflight] adapter frame OK ({Path(trained_on).name})")

os.makedirs(LINEUP_V14, exist_ok=True)
print(f"[merge] output dir: {LINEUP_V14}")

for label, adapter, transformer, out_name in TARGETS:
    final_output = os.path.join(LINEUP_V14, out_name)
    if os.path.exists(final_output):
        print(f"\n[merge] SKIP {label}: output already exists ({final_output})")
        continue

    output = final_output.replace(".safetensors", ".NEW.safetensors")
    print(f"\n========== merging {label} V1.4: adapter={os.path.basename(adapter)} ==========")
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
