r"""Re-bake the V1.6 SoReal! lineup (Base + Turbo) from the V54 adapter.

The 08Oct V1.6 (from V51 / V52) was a poor model, so training went back to
Base V1.5 and V54 trained on it. This replaces V1.6 with:

  * Base   takes ``models\SoReal!_V54_OFT.safetensors``, the DPO DoRA-OFT run
    trained ON Base V1.5 (06Oct).
  * Turbo  takes the V54 Turbo dfit conversion (``_oft_dense_retarget.py``,
    fitted against Base V1.5 06Oct and Turbo V1.5 **10Oct**, the user's pick),
    baked onto that same 10Oct Turbo V1.5. No transfer run, as in V1.3.

Frame: Base checks the adapter's recorded ``transformer.model_name``; Turbo checks
the dfit file's ``soreal_dense_fit`` metadata (source adapter, base, variant) and
the off-regime guardrails. The dfit metadata stores file names only and both
Turbo V1.5 files share one, so the 10Oct folder is fixed here, not inferred.

Same clamped, NaN-gated merge path as _v16_lineup_bake.py, full strength, bf16
out, atomic .NEW -> replace on success only. Nothing is overwritten: the output
folder is a required argument and existing outputs are skipped.

  venv\Scripts\python.exe scripts\util\_v16r_lineup_bake.py <out folder name, e.g. 17Oct> [Base] [Turbo]

The dfit step first (writes the Turbo input this script expects):
  venv\Scripts\python.exe scripts\util\_oft_dense_retarget.py
      --adapter "E:\AI\Data\Packages\OneTrainer\models\SoReal!_V54_OFT.safetensors"
      --base "E:\AI\Data\Models\StableDiffusion\SoReal!\06Oct\SoReal!_Base_V1.5.safetensors"
      --variant "Turbo=E:\AI\Data\Models\StableDiffusion\SoReal!\10Oct\SoReal!_Turbo_V1.5.safetensors"
      --out-dir "E:\AI\Data\Models\Lora\warmstarts\SoReal!_V54_OFT"
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
BASE_V15 = os.path.join(ORIGIN, "06Oct", "SoReal!_Base_V1.5.safetensors")
TURBO_V15 = os.path.join(ORIGIN, "10Oct", "SoReal!_Turbo_V1.5.safetensors")

ADAPTER = r"E:\AI\Data\Packages\OneTrainer\models\SoReal!_V54_OFT.safetensors"
DFIT = r"E:\AI\Data\Models\Lora\warmstarts\SoReal!_V54_OFT\SoReal!_V54_OFT_Turbo_dfit.safetensors"

# (label, adapter, transformer source, output filename)
TARGETS = [
    ("Base", ADAPTER, BASE_V15, "SoReal!_Base_V1.6.safetensors"),
    ("Turbo", DFIT, TURBO_V15, "SoReal!_Turbo_V1.6.safetensors"),
]


def read_metadata(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


def preflight(label: str, adapter: str, transformer: str) -> None:
    for path in (adapter, transformer):
        if not os.path.exists(path):
            raise SystemExit(f"ABORT {label}: missing {path}")
    if label == "Base":
        trained_on = json.loads(read_metadata(adapter)["ot_config"])["transformer"]["model_name"]
        if Path(trained_on).name != Path(transformer).name:
            raise SystemExit(f"ABORT {label}: adapter trained on {trained_on!r}, not {Path(transformer).name}")
        print(f"[preflight] {label} frame OK ({Path(trained_on).name})")
        return
    fit = json.loads(read_metadata(adapter).get("soreal_dense_fit", "{}"))
    expected = {
        "source_adapter": Path(ADAPTER).name,
        "base_checkpoint": Path(BASE_V15).name,
        "variant": label,
        "variant_checkpoint": Path(TURBO_V15).name,
    }
    for key, want in expected.items():
        if fit.get(key) != want:
            raise SystemExit(f"ABORT {label}: dfit {key}={fit.get(key)!r}, expected {want!r}")
    # Off-regime conversions are a hard stop (V12_DFIT_RUNBOOK guardrails).
    for family, stats in fit["per_family"].items():
        if stats["clipped_blocks"] or stats["refine_projected_blocks"] or stats["dense_residual_max"] > 0.06:
            raise SystemExit(f"ABORT {label}: {family} off-regime {stats}")
    print(f"[preflight] {label}: dfit frame + regime OK")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("Base", "Turbo"):
        raise SystemExit("usage: _v16r_lineup_bake.py <out folder name, e.g. 17Oct> [Base] [Turbo]")
    out_dir = os.path.join(ORIGIN, sys.argv[1])
    only = set(sys.argv[2:])
    targets = [t for t in TARGETS if not only or t[0] in only]

    for label, adapter, transformer, _ in targets:
        preflight(label, adapter, transformer)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[merge] output dir: {out_dir}")

    failed = []
    for label, adapter, transformer, out_name in targets:
        final_output = os.path.join(out_dir, out_name)
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
