r"""Bake the V1.4 SoReal! lineup: Base + Turbo + Lightning from the SFT+DPO adapter.

2026-08-26. Source adapter is ``models\SoReal!_V41_OFT.safetensors`` -- the name
is STALE (it is the destination path an older run left in the config, not a V41
revision). It is really the SFT+DPO DoRA-OFT add-on run: ``rlhf_mode=SFT_DPO``,
``dora_oft=True``, ``oft_block_size=833`` (snaps to 768/1024 -- the largest
SoReal! OFT to date, 1.69 GB), trained ON Base V1.3 per its own
``transformer.model_name``.

Do NOT confuse it with ``Lora\SoReal!_V41_OFT.safetensors`` (the 08Aug V41
final, whose late-family ``oft_R`` blew up ~16x and which generates badly).
This adapter was screened against that failure: late-family ``oft_R`` is
0.0607 qkv / 0.0738 mlp, matching the good V41 save's 0.0625 / 0.0955 regime.

Frame: the adapter trained on Base V1.3, so every target is its own 09Aug V1.3
checkpoint. Base takes the ORIGINAL adapter; Turbo/Lightning take their
dense-transport (dfit) conversions, whose ``soreal_dense_fit`` metadata is
verified below to name the matching V1.3 base/variant pair.

Same clamped, NaN-gated merge path as _v13_lineup_bake.py
(modules.util.oft_merge.merge_oft_adapter -- bit-equivalent to training), full
strength, bf16 out, atomic .NEW -> replace on success only. The dfit adapters
are per-variant and frame-corrected already: NO sentinel-align push and NO
late-family strength scaling (both belong to the transplant flow; applying them
on top of a dense-transported adapter would double-count the distill direction
-- CLAUDE.md).

Nothing is overwritten: outputs land in a fresh dated folder alongside 09Aug.
Each target merges independently, so a gate failure on one does not abort the
others.
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
LINEUP_V14 = r"E:\AI\Data\Models\StableDiffusion\SoReal!\26Aug"

ADAPTER_STEM = "SoReal!_V41_OFT"
ADAPTER = rf"E:\AI\Data\Packages\OneTrainer\models\{ADAPTER_STEM}.safetensors"
DFIT_DIR = rf"E:\AI\Data\Models\Lora\warmstarts\{ADAPTER_STEM}_26Aug"

# (label, adapter, transformer source, output filename)
TARGETS = [
    (
        "Base",
        ADAPTER,
        os.path.join(LINEUP_V13, "SoReal!_Base_V1.3.safetensors"),
        "SoReal!_Base_V1.4.safetensors",
    ),
    *[
        (
            variant,
            os.path.join(DFIT_DIR, f"{ADAPTER_STEM}_{variant}_dfit.safetensors"),
            os.path.join(LINEUP_V13, f"SoReal!_{variant}_V1.3.safetensors"),
            f"SoReal!_{variant}_V1.4.safetensors",
        )
        for variant in ("Turbo", "Lightning")
    ],
]


def read_metadata(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


# ---- pre-flight: every path present, and each dfit file provably fitted for
# ---- THIS adapter against THIS variant's own V1.3 pair. A dfit file carrying a
# ---- different frame would silently bake off-axis education. The stem
# ---- "SoReal!_V41_OFT" also names a DIFFERENT adapter under Lora\ with its own
# ---- warmstarts\ folder, so the frame check is what separates them.
for label, adapter, transformer, _ in TARGETS:
    for path in (adapter, transformer):
        if not os.path.exists(path):
            raise SystemExit(f"ABORT {label}: missing {path}")

    if adapter == ADAPTER:
        continue

    fit = json.loads(read_metadata(adapter).get("soreal_dense_fit", "{}"))
    expected = {
        "source_adapter": f"{ADAPTER_STEM}.safetensors",
        "base_checkpoint": "SoReal!_Base_V1.3.safetensors",
        "variant": label,
        "variant_checkpoint": f"SoReal!_{label}_V1.3.safetensors",
    }
    for key, want in expected.items():
        if fit.get(key) != want:
            raise SystemExit(f"ABORT {label}: dfit {key}={fit.get(key)!r}, expected {want!r}")

    # Off-regime conversions are a hard stop (V12_DFIT_RUNBOOK guardrails).
    for family, stats in fit["per_family"].items():
        if stats["clipped_blocks"] or stats["refine_projected_blocks"] or stats["dense_residual_max"] > 0.06:
            raise SystemExit(f"ABORT {label}: {family} off-regime {stats}")
    print(f"[preflight] {label}: dfit frame + regime OK")

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
