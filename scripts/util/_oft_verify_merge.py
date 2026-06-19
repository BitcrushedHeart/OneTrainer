"""Verify the 31May merges: NaN/zero-row scan + DoRA faithfulness
(||baked_row|| should equal |dora_scale|). Writes a JSON report so results
are read from disk rather than a glitchy console."""

from __future__ import annotations

import json

import torch

from safetensors import safe_open

ADP = r"F:\workspace\SoReal!\Base\backup\2026-05-31_00-35-36-backup-30144-0-30144\lora\lora.safetensors"
FILES = {
    "base_FAITHFUL": r"E:\AI\Data\Models\StableDiffusion\SoReal!\31May\SoReal!_Base_PreRelease+V21+V24_OTmerge.safetensors",
    "distilled_crossbase": r"E:\AI\Data\Models\StableDiffusion\SoReal!\31May\SoReal!_Distilled_PreRelease+V21+V24_OTmerge.safetensors",
}

# adapter dora_scale per module (the faithful target row-norm)
ds = {}
with safe_open(ADP, framework="pt", device="cpu") as f:
    for k in f.keys():  # noqa: SIM118 -- safe_open is not a dict
        if k.endswith(".dora_scale"):
            stem = k[: -len(".dora_scale")].replace("transformer.", "")
            ds[stem] = f.get_tensor(k).float().flatten()

report = {}
for label, path in FILES.items():
    nan_tensors = 0
    dead_rows = 0
    checked2d = 0
    rel_errs = []  # ||row|| vs |dora_scale| over matchable modules
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 -- safe_open is not a dict
            if not k.endswith(".weight"):
                continue
            t = f.get_tensor(k).float()
            if torch.isnan(t).any() or torch.isinf(t).any():
                nan_tensors += 1
            if t.ndim == 2:
                checked2d += 1
                rn = torch.norm(t, dim=1)
                dead_rows += int((rn == 0).sum())
                stem = k[: -len(".weight")]
                if stem in ds and ds[stem].shape[0] == rn.shape[0]:
                    tgt = ds[stem].abs()
                    m = tgt > 1e-6
                    if m.any():
                        rel_errs.append(((rn[m] - tgt[m]).abs() / tgt[m]).median().item())
    rel = torch.tensor(rel_errs) if rel_errs else torch.tensor([float("nan")])
    report[label] = {
        "path": path,
        "nan_inf_tensors": nan_tensors,
        "dead_rows_total": dead_rows,
        "checked_2d_weights": checked2d,
        "dora_modules_matched": len(rel_errs),
        "faithfulness_median_relerr": float(rel.median()),
        "faithfulness_p95_relerr": float(rel.quantile(0.95)) if rel_errs else float("nan"),
    }

print(json.dumps(report, indent=2))
out = r"E:\AI\Data\Models\StableDiffusion\SoReal!\31May\_merge_verify.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print("WROTE", out)
