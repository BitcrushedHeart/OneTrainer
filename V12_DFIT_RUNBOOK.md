# V1.2 lineup + new dfit pipeline runbook

Prepared 2026-08-05, waiting on GPU all-clear. State of the world (verified on
disk):

- **V1.2 already shipped** (`E:\AI\Data\Models\StableDiffusion\SoReal!\02Aug\`,
  built 2026-08-02 13:07–13:13): Base V1.2 = Base V1.1 + V40_9_1 (unaffected by
  any of this); Turbo/Lightning V1.2 were merged from **old-pipeline** dfit
  conversions of V40_9_1 (files at
  `E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT_9_1\`, finished 12:55 /
  13:04 the same day — the original V40_9_1 adapter sits at
  `E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT\SoReal!_V40_OFT_9_1.safetensors`).
- **V41 is training ON Base V1.2** (preset `transformer.model_name` = the 02Aug
  Base; output `models\SoReal!_V41_OFT.safetensors`; block 768 DoRA-OFT).
  Its eventual conversion frame is the **02Aug** lineup. The scripts'
  `DEFAULT_BASE`/`DEFAULT_VARIANTS`/`DEFAULT_ADAPTER` constants still point at
  V1.1/old-V40 — ALWAYS pass `--adapter`, `--base`, `--variant` explicitly.

Two tracks. A measures (and optionally repairs) the shipped V1.2 variants; B is
the V41 conversion when training ends. If A ends in a re-merge, do it BEFORE B,
so V41's education lands on the corrected variants — otherwise the old
conversion error is baked into the substrate V41 transports onto.

## Track A — measure the shipped V1.2 variants (V40_9_1, V1.1 frame)

The old side already exists on disk; only the new conversion + compare run.
Frame: V40_9_1 was trained on Base V1.1, targets are the V1.1 variants (27Jul).

```
venv\Scripts\python.exe scripts\util\_oft_dense_retarget.py --adapter "E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT\SoReal!_V40_OFT_9_1.safetensors" --out-dir "E:\AI\Data\Models\Lora\dfit_newpipe"

venv\Scripts\python.exe scripts\util\_dfit_compare.py --adapter "E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT\SoReal!_V40_OFT_9_1.safetensors" --old-dir "E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT_9_1" --new-dir "E:\AI\Data\Models\Lora\dfit_newpipe\warmstarts\SoReal!_V40_OFT_9_1"
```

(Base/variant defaults are correct for this track — V1.1/27Jul.) The compare is
provenance-guarded against the adapter file. Its `old_dfit` column is the error
of exactly what V1.2 shipped with, as fraction-of-education-missed; `new_dfit`
is what the fixed pipeline achieves; the gap decides whether a re-merge is
worth it. If it is (confirm with generation A/B at Turbo 8–12 / Lightning 4
steps): re-bake Turbo/Lightning V1.2 from the new dfit files via the usual
merge flow, replacing the 02Aug variants. Base V1.2 stays untouched either way.

## Track B — V41 conversion (V1.2 frame) when training completes

Confirm the final adapter with the user (periodic saves land in
`E:\AI\Data\Models\Lora\warmstarts\SoReal!_V41_OFT_N.safetensors`; preset
output is `models\SoReal!_V41_OFT.safetensors`). Sanity: 630 tensors,
`ot_config` OFT_2 / dora_oft true / block 768 / clip 0.99 / scaled false.
`V12LINEUP` below = `E:\AI\Data\Models\StableDiffusion\SoReal!\02Aug`; if Track
A re-merged the variants, point at the re-merged files instead.

```
venv\Scripts\python.exe scripts\util\_oft_dense_retarget.py --adapter "ADAPTER" --base "V12LINEUP\SoReal!_Base_V1.2.safetensors" --variant "Turbo=V12LINEUP\SoReal!_Turbo_V1.2.safetensors" --variant "Lightning=V12LINEUP\SoReal!_Lightning_V1.2.safetensors"
```

There is no old-pipeline baseline to run: the `--legacy-target` flag and the
pre-2026-08 target `R_L M_b R_R` were removed from the code on 2026-08-08. Track
A already quantified the pipeline change once and for all (old→new is +0.63%
qkv / +0.34% mlp of education error, against +7.2% / +5.5% for dfit over plain
transplant), and the old-pipeline artifacts it compared are still on disk —
`…\SoReal!\02Aug\*.oldpipe.safetensors` and the V40_9_1 old-dfit files — so the
historical comparison stays reproducible without a code path that can silently
produce a bad-target file under a good-looking filename.

## Shared guardrails

- One GPU consumer at a time; nothing starts while V41 trains.
- Expected per-family console stats: dense-resid ~0.02–0.03, `refine pre->post`
  strictly improving, clipped 0, projected 0. **Hard-stop** on clipped > 0,
  projected > 0, or dense-resid max > 0.06 (off-regime = the revision behaves
  differently; surface, don't push through). Floors landing BELOW the old
  0.35–0.37 expectation is the fix working, not an anomaly.
- Frobenius numbers decide nothing (standing policy): any bake/re-merge is
  confirmed by generation A/B at the variant's own step count.
- If the variants still visibly trail Base's education after all this, that is
  the ~0.36 parameterization floor — the next lever is the transfer training
  run (dfit outputs are its warm starts; see `TRANSFER_RUN_HANDOVER.md`).
