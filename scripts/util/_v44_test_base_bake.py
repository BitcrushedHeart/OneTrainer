r"""Bake an experimental Base-only test checkpoint from SoReal!_V44_OFT_5.

2026-09-05. One-off: the user wants a throwaway Base to generate against, no
Turbo/Lightning, no dfit conversions, no preflight/health reporting.

Frame check is the only thing kept, because it is free and a wrong frame would
make the output meaningless rather than merely experimental: the adapter's own
``ot_config.transformer.model_name`` must name the Base it is being merged onto
(V44_OFT_5 records 09Aug\SoReal!_Base_V1.3.safetensors).

Merge goes through ``modules.util.oft_merge.merge_oft_adapter`` at full strength
(CLAUDE.md: bit-equivalent to training). No sentinel-align / late-family scaling
-- those belong to the Turbo/Lightning transplant flow. bf16 out, atomic
.NEW -> replace on success.
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

ADAPTER = r"E:\AI\Data\Models\Lora\SoReal!_V44_OFT_5.safetensors"
BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\09Aug\SoReal!_Base_V1.3.safetensors"
OUT_DIR = r"E:\AI\Data\Models\StableDiffusion\SoReal!\05Sep_test"
OUT_NAME = "SoReal!_Base_V44_5_TEST.safetensors"


def read_metadata(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


for path in (ADAPTER, BASE):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: missing {path}")

trained_on = json.loads(read_metadata(ADAPTER)["ot_config"])["transformer"]["model_name"]
if os.path.normcase(os.path.abspath(trained_on)) != os.path.normcase(os.path.abspath(BASE)):
    raise SystemExit(f"ABORT: adapter trained on {trained_on!r}, merging onto {BASE!r}")
print(f"[preflight] frame OK: adapter trained on {os.path.basename(BASE)}")

os.makedirs(OUT_DIR, exist_ok=True)
final_output = os.path.join(OUT_DIR, OUT_NAME)
if os.path.exists(final_output):
    raise SystemExit(f"ABORT: output already exists ({final_output})")
output = final_output.replace(".safetensors", ".NEW.safetensors")

print(f"========== merging {os.path.basename(ADAPTER)} ==========")
print(f"           onto {BASE}")

ui = TrainConfig.default_values()
ui.base_model_name = "Tongyi-MAI/Z-Image"
ui.transformer.model_name = BASE
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
    print(f"\n=== MERGE GATE FAILED -- nothing published ===\n{exc}")
    if os.path.exists(output):
        os.remove(output)
    raise SystemExit(1) from exc

os.replace(output, final_output)  # publish only on success

print("\n=== MERGE REPORT ===")
print("modules merged:", len(report.post_zero_rows))
new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
print("total NEW dead rows created by merge:", new_dead)
print("output:", final_output)
