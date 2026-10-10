r"""Bake the V1.6 SoReal! lineup: Base + Turbo from the V51 / V52 adapters.

  * Base  takes ``models\SoReal!_V51_OFT.safetensors``, the DPO DoRA-OFT run
    trained ON Base V1.5 (06Oct) -- the pre-transfer adapter.
  * Turbo takes ``models\SoReal!_V52_OFT.safetensors``, the transfer-mode run
    trained ON Turbo V1.5 (06Oct), teacher = Base V1.5 + V51, warm-started from
    the V51 Turbo dfit, ``transfer_preserve_distillation`` on -- the
    post-transfer adapter, saved from its best-residual checkpoint.

Frame: each adapter bakes onto exactly the checkpoint its recorded
``transformer.model_name`` names; the pre-flight aborts otherwise. Output folder
is pinned to ``08Oct`` (the character queue's config already points there), so
running past midnight does not move it.

Same clamped, NaN-gated merge path as _v15_lineup_bake.py, full strength,
bf16 out, atomic .NEW -> replace on success only.
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

ORIGIN = r"E:\AI\Data\Models\StableDiffusion\SoReal!"
LINEUP_V15 = os.path.join(ORIGIN, "06Oct")
LINEUP_V16 = os.path.join(ORIGIN, "08Oct")

ADAPTER_DIR = r"E:\AI\Data\Packages\OneTrainer\models"

# (label, adapter, transformer source, output filename)
TARGETS = [
    (
        "Base",
        os.path.join(ADAPTER_DIR, "SoReal!_V51_OFT.safetensors"),
        os.path.join(LINEUP_V15, "SoReal!_Base_V1.5.safetensors"),
        "SoReal!_Base_V1.6.safetensors",
    ),
    (
        "Turbo",
        os.path.join(ADAPTER_DIR, "SoReal!_V52_OFT.safetensors"),
        os.path.join(LINEUP_V15, "SoReal!_Turbo_V1.5.safetensors"),
        "SoReal!_Turbo_V1.6.safetensors",
    ),
]


def read_metadata(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


def main() -> int:
    only = set(sys.argv[1:])  # optional: restrict to "Base" / "Turbo"
    targets = [t for t in TARGETS if not only or t[0] in only]

    for label, adapter, transformer, _ in targets:
        for path in (adapter, transformer):
            if not os.path.exists(path):
                raise SystemExit(f"ABORT {label}: missing {path}")
        trained_on = json.loads(read_metadata(adapter)["ot_config"])["transformer"]["model_name"]
        if Path(trained_on).name != Path(transformer).name:
            raise SystemExit(f"ABORT {label}: adapter trained on {trained_on!r}, not {Path(transformer).name}")
        print(f"[preflight] {label} frame OK ({Path(trained_on).name})")

    os.makedirs(LINEUP_V16, exist_ok=True)
    print(f"[merge] output dir: {LINEUP_V16}")

    failed = []
    for label, adapter, transformer, out_name in targets:
        final_output = os.path.join(LINEUP_V16, out_name)
        if os.path.exists(final_output):
            print(f"\n[merge] SKIP {label}: output already exists ({final_output})")
            continue

        output = final_output.replace(".safetensors", ".NEW.safetensors")
        print(f"\n========== merging {label} V1.6: adapter={os.path.basename(adapter)} ==========")
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
            failed.append(label)
            continue

        os.replace(output, final_output)  # publish only on success

        print(f"\n=== MERGE REPORT ({label}) ===")
        print("modules merged:", len(report.post_zero_rows))
        new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
        print("total NEW dead rows created by merge:", new_dead)
        print("output:", final_output)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
