"""Re-bake Turbo/Lightning V1.2 from the NEW-pipeline dfit adapters.

2026-08-05: the shipped 02Aug Turbo/Lightning V1.2 were merged from
OLD-pipeline dfit conversions of V40_9_1. This re-bakes both variants from the
new-pipeline conversions (education-anchored target + decode-path refinement,
E:\\AI\\Data\\Models\\Lora\\dfit_newpipe\\...), replacing the 02Aug files in
place. Base V1.2 is NOT touched (no conversion involved in its bake).

Same clamped, NaN-gated merge path as scripts/util/_oft_faithful_merge.py
(modules.util.oft_merge.merge_oft_adapter -- bit-equivalent to training),
full strength, bf16 out, atomic .NEW -> replace on success only. The dfit
adapters are per-variant and frame-corrected already: NO sentinel-align push
and NO late-family strength scaling here (both belong to the transplant flow;
applying them on top of a dense-transported adapter would double-count the
distill direction -- CLAUDE.md).

Before replacing, the shipped file is renamed once to
``SoReal!_{V}_V1.2.oldpipe.safetensors`` (kept for the generation A/B; never
overwritten if it already exists).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.DataType import DataType  # noqa: E402
from modules.util.enum.ModelFormat import ModelFormat  # noqa: E402
from modules.util.oft_merge import merge_oft_adapter  # noqa: E402
from modules.util.oft_verify import MergeVerificationError  # noqa: E402

LINEUP_V11 = r"E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul"
LINEUP_V12 = r"E:\AI\Data\Models\StableDiffusion\SoReal!\02Aug"
DFIT_DIR = r"E:\AI\Data\Models\Lora\dfit_newpipe\warmstarts\SoReal!_V40_OFT_9_1"

TARGETS = [
    (
        variant,
        os.path.join(DFIT_DIR, f"SoReal!_V40_OFT_9_1_{variant}_dfit.safetensors"),
        os.path.join(LINEUP_V11, f"SoReal!_{variant}_V1.1.safetensors"),
        os.path.join(LINEUP_V12, f"SoReal!_{variant}_V1.2.safetensors"),
    )
    for variant in ("Turbo", "Lightning")
]

for label, adapter, transformer, final_output in TARGETS:
    for path in (adapter, transformer):
        if not os.path.exists(path):
            raise SystemExit(f"ABORT {label}: missing {path}")

    output = final_output.replace(".safetensors", ".NEW.safetensors")
    print(f"\n========== re-baking {label} V1.2: adapter={os.path.basename(adapter)} ==========")
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
        print(f"\n=== MERGE GATE FAILED ({label}) -- shipped V1.2 left untouched ===\n{exc}")
        if os.path.exists(output):
            os.remove(output)
        continue

    backup = final_output.replace(".safetensors", ".oldpipe.safetensors")
    if os.path.exists(final_output) and not os.path.exists(backup):
        os.rename(final_output, backup)
        print(f"backed up shipped file -> {os.path.basename(backup)}")
    os.replace(output, final_output)  # publish only on success

    print(f"\n=== MERGE REPORT ({label}) ===")
    print("modules merged:", len(report.post_zero_rows))
    new_dead = sum(max(0, report.post_zero_rows[k] - report.pre_zero_rows.get(k, 0)) for k in report.post_zero_rows)
    print("total NEW dead rows created by merge:", new_dead)
    print("output:", final_output)
