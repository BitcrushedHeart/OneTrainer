# Handover: V40 conversion + unattended transfer runs (Turbo, Lightning)

For a fresh agent. The user is REMOTE (no GUI access). Your job, in order: (0) wait for
the final V40 save, (1) convert it to dense-lsq warm starts, (2) build two headless
training configs from the user's preset, (3) launch the Turbo then Lightning transfer
runs, (4) monitor them and push periodic progress + ETA reports. Everything below is
verified fact as of 2026-07-31 morning unless marked otherwise.

## Background (2 paragraphs, everything else is operational)

SoReal! is a Z-Image DiT lineup: `SoReal!_Base_V1.1.safetensors` plus two self-distilled
variants (Turbo, Lightning) in `E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul\`. The
variants differ from Base in exactly 48 tensors (layers 22-29 x qkv/mlp stems). OFT /
DoRA-OFT adapters are trained on Base; transplanted onto variants they miss ~37-40% of
the frame-correct education (measured). The fix is "transfer mode" (`transfer_enabled`
in TrainConfig): student = variant + a dense-lsq warm-start adapter, teacher = Base's
late tensors + the ORIGINAL adapter's education built in memory, forward forks at layer
22, per-block activation MSE, only late-family adapter params train, auto-stop with
best-checkpoint restore. All of this is implemented and tested (suite: 279 passed,
1 skipped — run `venv\Scripts\python.exe -m pytest tests\ -q` from the repo root if you
need to confirm nothing drifted; do NOT modify code).

The warm starts are produced by `scripts/util/_oft_dense_retarget.py` (GPU, ~15-20 min
for both variants) which writes `{out}\warmstarts\{adapter_stem}\{stem}_{Variant}_dfit.safetensors`.
Benchmarked on real data: dense-lsq beats transplant on 96/96 tensors (mean 6% relative);
the weight-space floor is ~0.36 — the first `transfer_val/residual` reading vs its final
floor is the run's key scientific output.

## Step 0 — WAIT for the final V40 (do not skip)

The user is ending a training run remotely and saving the final model. The final adapter
will land at (or newer than): `E:\AI\Data\Packages\OneTrainer\models\SoReal!_V40_OFT.safetensors`.
The copy there right now (mtime 2026-07-31 07:28:05) is an INTERMEDIATE save — do not use it.
(`models\v40\SoReal!_V40_OFT.safetensors`, mtime 07-30, is yesterday's revision. The path
the user mentioned, `E:\AI\Packages\OneTrainer\Models`, does not exist.)

Proceed only when ALL of: (a) that file's mtime is strictly later than 2026-07-31
07:28:05, (b) the mtime has been stable for >= 5 minutes, (c) `nvidia-smi` shows the GPU
essentially idle (low util AND well under ~20 GB used — training holds ~23 GB). Poll
every few minutes. If nothing new appears after ~4 hours, SendMessage to `main` asking
for guidance instead of acting on the stale file. Once found, sanity-check it:
`modules/util/adapter_io.read_safetensors_buffered` → 630 tensors, ot_config metadata
has `peft_type OFT_2, dora_oft true, oft_block_size 512, oft_clipped_norm 0.99,
oft_scaled false`.

## Step 1 — conversion

Old outputs collide by name (same adapter stem as yesterday). First rename the stale
folder: `E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT` →
`...\SoReal!_V40_OFT.prev-2026-07-30`. Then:

```
venv\Scripts\python.exe scripts\util\_oft_dense_retarget.py --adapter "E:\AI\Data\Packages\OneTrainer\models\SoReal!_V40_OFT.safetensors"
```

Defaults handle Base/variant paths and write to
`E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT\`. Uses CUDA by default (GPU is free
by now). Expect per-family console stats: dense-resid ~0.02-0.03, floor ~0.35-0.37,
clipped 0. Verify both output files: same key set as the source adapter, exactly 48
changed `oft_R.weight` + 48 changed `dora_multiplier` (layers >= 22), remaining 534 keys
bit-identical, all finite, `soreal_dense_fit` metadata parses. Hard-stop and report if
clipped blocks > 0 or dense-resid max > 0.06 (numbers far off the benchmarked regime
mean the new revision behaves differently — surface, don't push through).

## Step 2 — configs

The user's live GUI state is `training_presets\#.json` (last written 2026-07-31 07:43 —
their current V40 training setup; concepts/resolution/optimizer in there are correct and
their latent/text caches match it — do not change concepts or resolutions). Copy it
twice: `training_presets\transfer_turbo.json` and `transfer_lightning.json`, then edit
via python json load/update/dump (never hand-edit), setting per variant:

| key | Turbo | Lightning |
|---|---|---|
| transformer.model_name | `E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul\SoReal!_Turbo_V1.1.safetensors` | same but `Lightning` |
| lora_model_name | `E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT\SoReal!_V40_OFT_Turbo_dfit.safetensors` | same but `Lightning` |
| output_model_destination | `E:\AI\Data\Models\Lora\warmstarts\SoReal!_V40_OFT\SoReal!_V40_OFT_Turbo_transferred.safetensors` | same but `Lightning` |
| workspace_dir | `<preset workspace parent>\transfer_turbo` | `...\transfer_lightning` |

NOTE: `base_model_name` stays as the preset's hub repo ("Tongyi-MAI/Z-Image" — supplies
tokenizer/TE/VAE/scheduler). ZImageModelLoader rejects single-file base models; the
variant DiT goes in the `transformer.model_name` single-file override, exactly as the
user's preset already does for Base. (Corrected 2026-07-31 after the operator agent
flagged the original table's error.)

Both configs additionally:
- `transfer_enabled: true`, `transfer_base_model_name`: the Base V1.1 path above,
  `transfer_teacher_adapter_name`: the NEW adapter (`models\SoReal!_V40_OFT.safetensors`
  — the ORIGINAL, never a _dfit file). Leave other `transfer_*` at defaults (late_start
  22, families "qkv,mlp", validation+patience on).
- `dropout_probability: 0` (hard startup requirement), `output_dtype: "FLOAT_32"`,
  `continue_last_backup: false`, `distill_enabled: false`, `rlhf_enabled: false`,
  `distillation_teacher_lora_model_name: ""`.
- `epochs`: min(preset value, 30) — auto-stop usually ends the run earlier.
- Disable periodic image sampling (`sample_after_unit: "NEVER"`) — validation drives
  everything; samples just cost time.
- Keep training_method LORA / peft OFT_2 / dora_oft true / oft_block_size 512 from the
  preset (verify they're there).

VALIDATE each config before launching: load with
`TrainConfig.default_values().from_dict(json)`, assert
`validate_transfer_startup() is None`, and check every path field exists on disk. Fix or
report before proceeding.

## Step 3 — launch (sequential, headless)

```
venv\Scripts\python.exe scripts\train.py --config-path training_presets\transfer_turbo.json
```
as a background command with output captured to a log file; when it exits, verify the
output adapter file exists and is ~936 MB, then launch Lightning the same way. Never run
both at once (one GPU). Expected startup sequence in the log: teacher-tensor build,
cache validation, then training steps. First validation appears at the preset's
validate_after cadence.

## Step 4 — monitor + report

While a run is live, check roughly every 20-30 minutes (Monitor tool if available, else
a bounded sleep-then-read loop) and push a SendMessage to `main` with: global step /
epoch, smoothed train loss (`loss/transfer` trend), latest `transfer_val/residual` and
best-so-far (log lines: "Saved transfer best checkpoint ..."), patience counter, and an
ETA computed from steps/sec against the epochs cap — state it as an UPPER bound since
patience usually stops earlier. TensorBoard event files live under
`{workspace_dir}\tensorboard\` if the stdout log is awkward to parse
(`tensorboard.backend.event_processing.event_accumulator` is available in the venv).

Escalate IMMEDIATELY via SendMessage (don't wait for the next interval) on: process
exit before any "Saved transfer best" line, a traceback in the log, the warning
"transfer validation has failed 3 times in a row", or `transfer_val/residual` failing to
drop below its FIRST reading after 3+ validations (means training isn't beating the
warm start — the run is wasting GPU).

Final report when both runs finish: per variant — first residual vs best residual vs the
0.36 weight-space floor, stop reason (patience/target/epochs), best step, wall time,
output file path. That residual comparison is the answer to "does function-space
training beat the weight-space ceiling"; report it plainly either way.

## Hard rails

- Repo conventions (CLAUDE.md): venv python only; never read dataset/queue/cache
  contents; no commits.
- Do NOT modify any code. Config JSONs, folder renames, and launching/monitoring only.
- Do NOT delete anything. The `.prev-2026-07-30` folder and the old flat `_conj/_init`
  files in `E:\AI\Data\Models\Lora` stay.
- One GPU consumer at a time: no conversions or benchmarks while training runs.
- If anything ambiguous or off-script happens, SendMessage to `main` and wait rather
  than improvising.
