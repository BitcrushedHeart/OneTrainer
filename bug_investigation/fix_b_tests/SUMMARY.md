# Fix B implementation summary

Persist gradient-accumulation state + per-parameter `.grad` tensors so
stop/restart is byte-identical for the in-flight window. Stop semantics
unchanged — no Fix-A-style deferral.

## Files modified

Production code (6 files, ~338 lines added):

| File | Change |
|------|--------|
| `modules/model/BaseModel.py` | Added `accumulator_state: dict \| None = None` (matches existing `optimizer_state_dict` / `ema_state_dict` pattern) |
| `modules/modelSaver/mixin/InternalModelSaverMixin.py` | Writes `accumulator/accumulator.pt` when `model.accumulator_state` is set (no-op for non-training save paths) |
| `modules/modelLoader/mixin/InternalModelLoaderMixin.py` | Reads `accumulator/accumulator.pt` under `contextlib.suppress(FileNotFoundError)` (legacy backups silently fall through) |
| `modules/trainer/GenericTrainer.py` | Promotes loop locals to `self._loop_*` mirrors; new `_stage_accumulator_state_for_save` + `_restore_accumulator_state` methods; `__backup`/`__save` stage before save and clear after; `train()` entry restores if present |
| `modules/util/NamedParameterGroup.py` | New `iter_named_parameters()` yielding `(stable_key, parameter)` pairs (key = `{group.unique_name}.{idx}`) |
| `modules/util/dataset_fingerprint.py` | New helper `compute_concept_fingerprint(concepts) -> (sha256_hex, count)` |

Test code (10 files in `bug_investigation/fix_b_tests/`, ~1370 lines):

- `_shadow_trainer.py` — shared shadow-trainer infrastructure (mirrors
  `GenericTrainer.train()` accumulation logic, with `enable_fix` flag
  for unfixed-vs-fixed contrast)
- `test_regression_bit_identical.py` — the load-bearing test:
  unfixed-diverges + fixed-bit-identical, parametrised over
  `acc ∈ {5, 50, 150}`
- `test_adapter_requires_grad.py` — frozen base + trainable head:
  saved payload contains only `requires_grad=True` params
- `test_dataset_mismatch.py` — warn-and-continue on dataset fingerprint diff
- `test_acc_steps_mismatch.py` — warn-and-continue on `accumulation_steps` change
- `test_param_mismatch.py` — silent skip for <10% missing keys, warn for >10%
- `test_rng_determinism.py` — RNG round-trip determinism
- `test_edge_of_window.py` — bit-identical at k_lost=acc-1 AND at clean boundaries
- `test_backwards_compat.py` — legacy backups (no accumulator file) load silently
- `test_integration_real_save_load.py` — uses the REAL
  `InternalModelSaverMixin._save_internal_data` and
  `InternalModelLoaderMixin._load_internal_data` against a duck-typed
  `BaseModel`, proving the production save/load wiring actually
  round-trips the payload

## Test results

```
20 passed in 11.21s
```

Run with:
```
cd OneTrainer-fixb
./venv/Scripts/python.exe -m pytest bug_investigation/fix_b_tests/ -v
```

The `test_unfixed_save_diverges_from_baseline` cases (acc=5, 50, 150) prove
the bug exists in pre-fix mode; the `test_fixed_save_matches_baseline_bit_identical`
cases prove Fix B is bit-identical. Both are inside the same suite, so the
"sanity-check the test would catch the bug" requirement is met
permanently as long as the suite is run.

## Checkpoint size delta

Measured on a realistic adapter-shape model (LoRA-like rank-16, 12
adapter blocks on a 768-dim base; 295K trainable params, 591K frozen):

| | Bytes | MiB |
|---|---:|---:|
| Legacy checkpoint (model.pt + optimizer.pt + meta.json) | 3,551,418 | 3.39 |
| Fixed checkpoint (legacy + accumulator/accumulator.pt) | 4,752,398 | 4.53 |
| **Added by Fix B (`accumulator/accumulator.pt`)** | **1,200,980** | **1.15** |
| Increase | | **+33.8 %** |

Matches the plan's 30–40% estimate. The accumulator file is dominated
by per-parameter `.grad` CPU copies of the 295K trainable tensors.
With `requires_grad=False` parameters skipped, frozen-base costs are
zero — important for full-finetune workloads where this number would
scale with model size instead of adapter size.

## Serialisation choice

`torch.save` / `torch.load` throughout, matching
`InternalModelSaverMixin.py:26` and `InternalModelLoaderMixin.py:48`.
Loader uses `weights_only=False` for the accumulator file (payload
mixes tensors with Python dicts/tuples for RNG state); same trust
boundary as `optimizer.pt` in the user's own backup tree.

## Residual non-determinism

MGDS supports mid-epoch resume via `initial_epoch_sample=train_progress.epoch_sample`
(see `modules/dataLoader/mixin/DataLoaderMgdsMixin.py:97`), but concept-balancing
can interleave samples differently across resumes. Fix B restores the
accumulator + grads + RNG bit-perfectly, but the post-resume
micro-batches in the in-flight window may still arrive in a slightly
different order than they would in an unbroken run. This is documented
honestly rather than "solved" — fixing the dataloader-side ordering
would require upstream MGDS changes outside this fix's scope.

## What stayed off the table (deliberate)

- Stop check at `GenericTrainer.py:1127` (now line ~1310 after edits) —
  **untouched**. Immediate-stop semantics preserved, which was the
  motivation for choosing Fix B over Fix A.
- `ema_loss` / `ema_loss_steps` — display-only, recover cosmetically.
- `lr_scheduler` — schedule is a deterministic function of `global_step`.
- `has_gradient` — derivable on load from "any saved param_grad present?".

## Deliverables in this directory

- `PATCH.diff` — unified diff of production-code changes (`modules/` only)
- `SUMMARY.md` — this file
- 10 test files (8 from the original brief + `_shadow_trainer.py` shared
  infra + `test_integration_real_save_load.py` for the real save/load
  helper round-trip)
