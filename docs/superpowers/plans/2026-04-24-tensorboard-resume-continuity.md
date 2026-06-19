# TensorBoard Resume Continuity + Validation-on-Resume Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a training run resumes via "Continue from last backup" in upstream `Nerogar/OneTrainer`, continue logging to the same TensorBoard run instead of a new timestamped subdirectory, and fix the validation event that is silently dropped when the backup was taken mid-epoch.

**Architecture:** Persist two new fields in the backup's `meta.json` — `tensorboard_subdir` (the TB log-dir basename used by this training lineage) and `last_action_epoch` (a dict keyed by action name holding the last epoch each epoch-scheduled action fired on). On resume, reuse the persisted TB subdir with `SummaryWriter(..., purge_step=global_step)`; gate epoch-scheduled actions on `epoch > last_action_epoch[name]` instead of the brittle `epoch_step == 0` check. The TB writer construction is deferred from `GenericTrainer.__init__` to `GenericTrainer.start()` post-model-load so the loaded backup's metadata is available first.

**Tech Stack:** Python 3.10+, PyTorch (`torch.utils.tensorboard.SummaryWriter`), pytest. CPU-only — no GPU required for any test.

**Target repo:** Upstream `Nerogar/OneTrainer` master (checked out in worktree at `.worktrees/upstream-investigate/`, currently at commit `b734d60f`). The eventual PR commit(s) contain only the upstream code changes — tests and this plan document stay local.

**Scope (explicit non-goals):**
- Dataloader position / RNG state is NOT restored on resume. That is a separate, bigger fix and is out of scope here. Pre-existing behavior retained.
- No UI / web / frontend changes. All edits are in `modules/`.

---

## Test Environment

All tests run against the worktree using the existing venv at the repo root:

- Python interpreter: `E:\AI\Data\Packages\OneTrainer\venv\Scripts\python.exe`
- Pytest: `E:\AI\Data\Packages\OneTrainer\venv\Scripts\pytest.exe`
- Upstream modules directory (for `sys.path` / `PYTHONPATH`): `E:\AI\Data\Packages\OneTrainer\.worktrees\upstream-investigate`

**Test files live OUTSIDE the worktree** so they cannot be accidentally committed to upstream:

- Test dir: `E:\AI\Data\Packages\OneTrainer\.diagnostics\upstream-resume-tests\`
- Already-gitignored: `.diagnostics/` is listed in the fork's `.gitignore` as a scratch location.

Each test module uses a `conftest.py` at the test dir that injects the worktree path onto `sys.path` so `from modules.util.TrainProgress import TrainProgress` resolves to the pristine upstream copy, not our fork's copy.

---

## File Structure

### Upstream files that will be modified (the PR)
| Path (inside worktree) | Responsibility | Change |
|---|---|---|
| `modules/util/TrainProgress.py` | Holds mutable training counters | Add `last_action_epoch: dict[str, int]` with default `{}` |
| `modules/util/TimedActionMixin.py` | Evaluates "is it time for action X" | Change EPOCH + `start_at_zero=True` branch to gate on `train_progress.last_action_epoch.get(name, -1)` and update the marker when firing |
| `modules/modelSaver/mixin/InternalModelSaverMixin.py` | Writes `meta.json` during backup | Add `last_action_epoch` and `tensorboard_subdir` fields to the saved dict |
| `modules/modelLoader/BaseModelLoader.py` | Reads `meta.json` during resume | Read the two new fields; populate `train_progress.last_action_epoch` and `model.resumed_tensorboard_subdir` |
| `modules/model/BaseModel.py` | Model container holding training state | Add `resumed_tensorboard_subdir: str \| None = None` attribute (populated by loader) |
| `modules/trainer/GenericTrainer.py` | Main trainer; builds the SummaryWriter | Defer `SummaryWriter` creation from `__init__` to end of `start()` (after `model_loader.load`); reuse `model.resumed_tensorboard_subdir` if present; pass `purge_step=train_progress.global_step`; stash the resolved subdir back onto the model so subsequent backups persist the same value |

### Local test files (NOT in the PR)
| Path | Responsibility |
|---|---|
| `.diagnostics/upstream-resume-tests/conftest.py` | Adds worktree path to `sys.path` |
| `.diagnostics/upstream-resume-tests/test_timed_action_mixin.py` | Unit tests for the validation trigger regression |
| `.diagnostics/upstream-resume-tests/test_meta_roundtrip.py` | Unit tests for saver/loader metadata roundtrip |
| `.diagnostics/upstream-resume-tests/test_resume_integration.py` | Lightweight end-to-end: simulate stop/resume and assert no fire point is skipped |

### Upstream files INTENTIONALLY NOT touched
- Web/GUI layer (we're a PR to upstream only)
- `scripts/`, `resources/`, `training_presets/`
- Any dataloader — we do not touch RNG/position restoration.

---

## Task 1: Scaffold local test directory and confirm baseline

**Files:**
- Create: `E:\AI\Data\Packages\OneTrainer\.diagnostics\upstream-resume-tests\conftest.py`
- Create: `E:\AI\Data\Packages\OneTrainer\.diagnostics\upstream-resume-tests\__init__.py` (empty)

- [ ] **Step 1: Create the test directory**

```bash
mkdir -p .diagnostics/upstream-resume-tests
```

- [ ] **Step 2: Write the conftest that injects the worktree onto sys.path**

`.diagnostics/upstream-resume-tests/conftest.py`:
```python
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2] / ".worktrees" / "upstream-investigate"
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
```

- [ ] **Step 3: Create an empty __init__.py**

`.diagnostics/upstream-resume-tests/__init__.py`: (empty file)

- [ ] **Step 4: Verify pytest can discover and import from the worktree**

Create a smoke test `.diagnostics/upstream-resume-tests/test_smoke.py`:
```python
def test_can_import_train_progress():
    from modules.util.TrainProgress import TrainProgress
    tp = TrainProgress()
    assert tp.epoch == 0
    assert tp.global_step == 0
```

Run:
```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/test_smoke.py -v
```
Expected: PASS.

- [ ] **Step 5: Delete the smoke test**

```bash
rm .diagnostics/upstream-resume-tests/test_smoke.py
```

(No commit — these files are all local, not staged for the upstream PR.)

---

## Task 2: Write the failing test for the validation trigger bug (TDD)

**Files:**
- Create: `.diagnostics/upstream-resume-tests/test_timed_action_mixin.py`

**Context:** The bug lives at `modules/util/TimedActionMixin.py:28-33`. The `EPOCH` + `start_at_zero=True` branch gates on `epoch_step == 0`. When resuming mid-epoch (epoch_step > 0), the check for the resumed epoch never returns True, silently dropping that epoch's scheduled action.

- [ ] **Step 1: Write the failing test**

`.diagnostics/upstream-resume-tests/test_timed_action_mixin.py`:
```python
"""Reproduces the mid-epoch-resume validation skip bug.

Upstream behavior (pre-fix):
- repeating_action_needed with EPOCH unit gates on `epoch_step == 0`.
- Resume at epoch=4, epoch_step=50 => never fires during the rest of epoch 4.
- That epoch's validation is silently dropped.

Post-fix behavior:
- repeating_action_needed uses train_progress.last_action_epoch[name]
  to track which epochs have already fired.
- Resume at epoch=4, epoch_step=50 with last_action_epoch={'validate': 3}
  => fires True on first call of that epoch (since 4 > 3).
"""
import pytest

from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.TimeUnit import TimeUnit


class _TestHarness(TimedActionMixin):
    """Minimal TimedActionMixin concrete subclass for testing."""


@pytest.fixture
def harness():
    return _TestHarness()


def _make_progress(epoch, epoch_step, last_validated=-1):
    tp = TrainProgress(epoch=epoch, epoch_step=epoch_step)
    tp.last_action_epoch = {"validate": last_validated} if last_validated >= 0 else {}
    return tp


def test_fires_once_per_epoch_fresh_run(harness):
    """First batch of epoch 0: fires. Subsequent batches in epoch 0: do not fire."""
    tp = _make_progress(epoch=0, epoch_step=0)
    assert harness.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp) is True
    # same epoch, step > 0 after firing once -> must not re-fire
    tp.epoch_step = 1
    assert harness.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp) is False


def test_fires_at_start_of_next_epoch_after_firing(harness):
    """Epoch 1 start: fires True again."""
    tp = _make_progress(epoch=0, epoch_step=0)
    harness.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp)
    tp.epoch, tp.epoch_step = 1, 0
    assert harness.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp) is True


def test_resume_mid_epoch_fires_for_that_epoch(harness):
    """THE BUG — resume at epoch=4, epoch_step=50 with last_validated=3.

    Pre-fix: returns False because epoch_step != 0 -> validation skipped.
    Post-fix: returns True because epoch (4) > last_validated (3).
    """
    tp = _make_progress(epoch=4, epoch_step=50, last_validated=3)
    assert harness.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp) is True


def test_resume_mid_epoch_does_not_refire_if_already_validated_this_epoch(harness):
    """Resume at epoch=4, step=50 with last_validated=4 (we already validated this epoch pre-stop)."""
    tp = _make_progress(epoch=4, epoch_step=50, last_validated=4)
    assert harness.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp) is False


def test_respects_interval(harness):
    """validate_after=2: fires on even epochs only."""
    tp = _make_progress(epoch=1, epoch_step=0)
    assert harness.repeating_action_needed("validate", 2, TimeUnit.EPOCH, tp) is False
    tp.epoch, tp.epoch_step = 2, 0
    assert harness.repeating_action_needed("validate", 2, TimeUnit.EPOCH, tp) is True


def test_interval_zero_never_fires(harness):
    tp = _make_progress(epoch=0, epoch_step=0)
    assert harness.repeating_action_needed("validate", 0, TimeUnit.EPOCH, tp) is False
```

- [ ] **Step 2: Run the tests and verify they fail in the expected way**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/test_timed_action_mixin.py -v
```

Expected failures (pre-fix):
- `test_resume_mid_epoch_fires_for_that_epoch` → FAIL (returns False instead of True)
- `test_resume_mid_epoch_does_not_refire_if_already_validated_this_epoch` → FAIL
  (returns False which happens to match expected False — but for the wrong reason.
  This test will become a real regression check post-fix when `epoch_step == 0`
  ceases to be the guard.)
- The other tests may pass against current code (they don't exercise the bug path). This is OK.

Do NOT edit the tests to make the currently-passing tests fail; we only require that the bug-reproducing tests (`test_resume_mid_epoch_fires_for_that_epoch`) fail pre-fix and pass post-fix.

- [ ] **Step 3: Record the exact failure output**

Save the failure output to `.diagnostics/upstream-resume-tests/BASELINE_FAILURE.txt` for the validation reviewer to compare against later. (Local only.)

---

## Task 3: Write the failing test for meta.json roundtrip

**Files:**
- Create: `.diagnostics/upstream-resume-tests/test_meta_roundtrip.py`

**Context:** `InternalModelSaverMixin._save_internal_data` at `modules/modelSaver/mixin/InternalModelSaverMixin.py:14-42` writes `meta.json` with only `train_progress` (epoch/epoch_step/epoch_sample/global_step). `BaseModelLoader._load_internal_state` at `modules/modelLoader/BaseModelLoader.py:18-42` reads only those four. We need to add `tensorboard_subdir` and `last_action_epoch` to the roundtrip.

- [ ] **Step 1: Write the failing test**

`.diagnostics/upstream-resume-tests/test_meta_roundtrip.py`:
```python
"""Verifies meta.json carries the new fields through save/load roundtrip."""
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.util.TrainProgress import TrainProgress


class _FakeSaver(InternalModelSaverMixin):
    pass


class _FakeLoader(BaseModelLoader):
    def load(self, *args, **kwargs):
        raise NotImplementedError


class _StubModel:
    """Minimal stand-in for BaseModel with the attributes save uses."""
    def __init__(self, tp, tb_subdir=None):
        self.optimizer = SimpleNamespace(state_dict=lambda: {})
        self.param_group_mapping = []
        self.train_config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="fake"))
        self.train_progress = tp
        self.ema = None
        self.resumed_tensorboard_subdir = tb_subdir
        # what the trainer will set when it resolves the TB subdir
        self.tensorboard_subdir = tb_subdir


def _make_backup(tmp_path, tp, tb_subdir):
    saver = _FakeSaver()
    m = _StubModel(tp, tb_subdir=tb_subdir)
    dest = str(tmp_path)
    saver._save_internal_data(m, dest)
    return dest


def test_meta_includes_last_action_epoch(tmp_path):
    tp = TrainProgress(epoch=4, epoch_step=50, epoch_sample=200, global_step=1234)
    tp.last_action_epoch = {"validate": 3, "sample": 2}
    dest = _make_backup(tmp_path, tp, tb_subdir="myrun_2026-04-24_10-00-00")
    with open(os.path.join(dest, "meta.json")) as f:
        meta = json.load(f)
    assert meta["last_action_epoch"] == {"validate": 3, "sample": 2}


def test_meta_includes_tensorboard_subdir(tmp_path):
    tp = TrainProgress(epoch=0, epoch_step=0, epoch_sample=0, global_step=0)
    tp.last_action_epoch = {}
    dest = _make_backup(tmp_path, tp, tb_subdir="myrun_2026-04-24_10-00-00")
    with open(os.path.join(dest, "meta.json")) as f:
        meta = json.load(f)
    assert meta["tensorboard_subdir"] == "myrun_2026-04-24_10-00-00"


def test_loader_restores_last_action_epoch(tmp_path):
    tp = TrainProgress(epoch=4, epoch_step=50, epoch_sample=200, global_step=1234)
    tp.last_action_epoch = {"validate": 3}
    dest = _make_backup(tmp_path, tp, tb_subdir="myrun")

    loader = _FakeLoader()
    m = _StubModel(TrainProgress())  # fresh model
    loader._load_internal_state(m, dest)

    assert m.train_progress.epoch == 4
    assert m.train_progress.last_action_epoch == {"validate": 3}


def test_loader_restores_tensorboard_subdir(tmp_path):
    tp = TrainProgress(epoch=0, epoch_step=0, epoch_sample=0, global_step=0)
    tp.last_action_epoch = {}
    dest = _make_backup(tmp_path, tp, tb_subdir="myrun_2026-04-24_10-00-00")

    loader = _FakeLoader()
    m = _StubModel(TrainProgress())
    loader._load_internal_state(m, dest)

    assert m.resumed_tensorboard_subdir == "myrun_2026-04-24_10-00-00"


def test_loader_tolerates_legacy_backup_without_new_fields(tmp_path):
    """Old backups (before this fix) should still load — missing fields default sensibly."""
    # Hand-craft a legacy meta.json with only train_progress.
    os.makedirs(os.path.join(tmp_path, "optimizer"), exist_ok=True)
    torch.save({}, os.path.join(tmp_path, "optimizer", "optimizer.pt"))
    with open(os.path.join(tmp_path, "meta.json"), "w") as f:
        json.dump({"train_progress": {"epoch": 2, "epoch_step": 0, "epoch_sample": 0, "global_step": 100}}, f)

    loader = _FakeLoader()
    m = _StubModel(TrainProgress())
    loader._load_internal_state(m, str(tmp_path))

    assert m.train_progress.epoch == 2
    assert m.train_progress.last_action_epoch == {}
    assert m.resumed_tensorboard_subdir is None
```

- [ ] **Step 2: Run the tests and verify they fail**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/test_meta_roundtrip.py -v
```

Expected failures (pre-fix):
- `test_meta_includes_last_action_epoch` → FAIL (`KeyError: 'last_action_epoch'`)
- `test_meta_includes_tensorboard_subdir` → FAIL
- `test_loader_restores_last_action_epoch` → FAIL (`AttributeError` — attribute doesn't exist)
- `test_loader_restores_tensorboard_subdir` → FAIL
- `test_loader_tolerates_legacy_backup_without_new_fields` → FAIL (assertion on `m.train_progress.last_action_epoch` since the attribute doesn't exist yet)

---

## Task 4: Write the failing integration test

**Files:**
- Create: `.diagnostics/upstream-resume-tests/test_resume_integration.py`

**Context:** We want confidence that the fix's persistent-marker semantics hold through a stop/resume cycle, not just in isolated unit calls. Each assertion below targets state (`tp.last_action_epoch`) that the mixin *only* populates post-fix — that's why these fail cleanly pre-fix (assertion fires with an empty dict) and pass post-fix (mixin writes the marker).

- [ ] **Step 1: Write the failing test**

`.diagnostics/upstream-resume-tests/test_resume_integration.py`:
```python
"""Simulate a stop/resume cycle.

Each test asserts a post-fix invariant: `train_progress.last_action_epoch`
is populated by the mixin as it fires actions. Pre-fix, the mixin never
touches that attribute, so the dict stays empty and every assertion fails.
"""
import json

import pytest

from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.TimeUnit import TimeUnit


class _Trainer(TimedActionMixin):
    pass


def _simulate_epoch(trainer, tp, epoch_length, stop_at_step=None):
    """One epoch of the GenericTrainer inner loop: always runs `epoch_length`
    iterations, mirroring the real trainer where the dataloader restarts fresh
    each epoch regardless of resume position.
    """
    fires = []
    for _ in range(epoch_length):
        if trainer.repeating_action_needed("validate", 1, TimeUnit.EPOCH, tp):
            fires.append((tp.epoch, tp.global_step))
        tp.next_step(1)
        if stop_at_step is not None and tp.epoch_step == stop_at_step:
            return fires
    return fires


def test_mixin_records_last_action_epoch_on_fire():
    """When validate fires at epoch 0, last_action_epoch['validate'] must be 0.
    Pre-fix: mixin never writes to last_action_epoch -> dict stays {} -> FAIL.
    """
    trainer = _Trainer()
    tp = TrainProgress()
    tp.last_action_epoch = {}

    assert trainer.repeating_action_needed(
        "validate", 1, TimeUnit.EPOCH, tp
    ) is True
    assert tp.last_action_epoch == {"validate": 0}


def test_marker_survives_meta_roundtrip_and_prevents_double_fire(tmp_path):
    """End-to-end: fire at epoch 3 start, serialize tp mid-epoch-3, reload,
    and a fresh trainer instance must NOT re-fire epoch 3 on the resumed
    first batch. Pre-fix has no marker to prevent a second fire when the
    resumed batch happens to land on epoch_step == 0 (it won't, but more
    importantly: last_action_epoch remains empty, so this assertion fails).
    """
    trainer = _Trainer()
    tp = TrainProgress()
    tp.last_action_epoch = {}

    # Simulate full epochs 0..2, then start epoch 3 and stop mid-way.
    for epoch in range(3):
        _simulate_epoch(trainer, tp, epoch_length=10)
        tp.next_epoch()
    # Fire at epoch 3 start, then walk to step 5.
    _simulate_epoch(trainer, tp, epoch_length=10, stop_at_step=5)

    # Post-fix invariant: marker records latest fired epoch.
    assert tp.last_action_epoch.get("validate") == 3

    # Serialize/reload (as backup save+load does).
    blob = json.dumps({
        "train_progress": {
            "epoch": tp.epoch, "epoch_step": tp.epoch_step,
            "epoch_sample": tp.epoch_sample, "global_step": tp.global_step,
        },
        "last_action_epoch": tp.last_action_epoch,
    })
    loaded = json.loads(blob)
    resumed_tp = TrainProgress(
        epoch=loaded["train_progress"]["epoch"],
        epoch_step=loaded["train_progress"]["epoch_step"],
        epoch_sample=loaded["train_progress"]["epoch_sample"],
        global_step=loaded["train_progress"]["global_step"],
    )
    resumed_tp.last_action_epoch = dict(loaded["last_action_epoch"])

    # Fresh mixin instance — no in-memory carry-over.
    trainer2 = _Trainer()

    # Resume: do NOT re-fire epoch 3. Check at the resumed first batch.
    assert trainer2.repeating_action_needed(
        "validate", 1, TimeUnit.EPOCH, resumed_tp
    ) is False

    # Finish epoch 3 (5 remaining steps); then epoch 4 must fire exactly once.
    _simulate_epoch(trainer2, resumed_tp, epoch_length=10)
    resumed_tp.next_epoch()
    fires_epoch_4 = _simulate_epoch(trainer2, resumed_tp, epoch_length=10)

    assert len(fires_epoch_4) == 1
    assert resumed_tp.last_action_epoch["validate"] == 4
```

- [ ] **Step 2: Run tests and confirm failure**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/test_resume_integration.py -v
```

Expected pre-fix failures:
- `test_mixin_records_last_action_epoch_on_fire` → FAIL on `assert tp.last_action_epoch == {"validate": 0}` (pre-fix mixin doesn't write to the dict, so it stays `{}`).
- `test_marker_survives_meta_roundtrip_and_prevents_double_fire` → FAIL on the first marker assertion for the same reason.

If either test errors instead of failing on assertion (e.g., `AttributeError`), that's still acceptable as "initially failing" per the user's rule — the key property is that *no test edits* are needed to make them pass once the fix is in.

---

## Task 5: Implement — `TrainProgress.last_action_epoch`

**Files:**
- Modify: `.worktrees/upstream-investigate/modules/util/TrainProgress.py`

- [ ] **Step 1: Add the field**

Replace the class body:
```python
class TrainProgress:
    def __init__(
            self,
            epoch: int = 0,
            epoch_step: int = 0,
            epoch_sample: int = 0,
            global_step: int = 0,
            last_action_epoch: dict | None = None,
    ):
        self.epoch = epoch
        self.epoch_step = epoch_step
        self.epoch_sample = epoch_sample
        self.global_step = global_step
        self.last_action_epoch = last_action_epoch if last_action_epoch is not None else {}

    def next_step(self, batch_size: int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    def next_epoch(self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1

    def filename_string(self):
        return f"{self.global_step}-{self.epoch}-{self.epoch_step}"
```

- [ ] **Step 2: Quick sanity check that import still works**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/test_timed_action_mixin.py::test_fires_once_per_epoch_fresh_run -v
```

At this point the test still fails because the mixin isn't using the new field yet — that's expected. Proceed to Task 6.

---

## Task 6: Implement — `TimedActionMixin` uses `last_action_epoch`

**Files:**
- Modify: `.worktrees/upstream-investigate/modules/util/TimedActionMixin.py`

- [ ] **Step 1: Change the EPOCH + start_at_zero=True branch**

In the `case TimeUnit.EPOCH:` arm, replace lines 26-33 with:

```python
            case TimeUnit.EPOCH:
                if int(interval) == 0:
                    return False
                if start_at_zero:
                    last = train_progress.last_action_epoch.get(name, -1)
                    fire = train_progress.epoch > last \
                        and train_progress.epoch % int(interval) == 0
                    if fire:
                        train_progress.last_action_epoch[name] = train_progress.epoch
                    return fire
                else:
                    # should actually be the last step of each epoch, but we don't know how many steps an epoch has
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0 \
                        and train_progress.epoch > 0
```

**Rationale:** Only the `start_at_zero=True` branch is affected. That covers `validate` and `sample`. `backup` and `save` use `start_at_zero=False` and keep their existing semantics (no persistent marker needed; they fire at `epoch_step == 0` of each interval-matching epoch *after* epoch 0).

Note: the `start_at_zero=False` branch has a known-but-out-of-scope bug too — it also relies on `epoch_step == 0` and would skip a backup on mid-epoch resume. We intentionally leave it alone; backup/save skipping one cycle on resume is a much lower-cost outcome than a missed validation, and the user's reported pain is validation. Flag for a future PR.

- [ ] **Step 2: Run the mixin tests — all should pass now**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/test_timed_action_mixin.py -v
```

Expected: 6 passed. If any fail, read the assertion, fix the mixin logic — do NOT edit the tests.

---

## Task 7: Implement — Backup `meta.json` writes new fields

**Files:**
- Modify: `.worktrees/upstream-investigate/modules/modelSaver/mixin/InternalModelSaverMixin.py`

- [ ] **Step 1: Add the two new fields**

Replace the `meta.json` dump (lines 34-42) with:

```python
        # meta
        tensorboard_subdir = getattr(model, "tensorboard_subdir", None)
        with open(os.path.join(destination, "meta.json"), "w") as meta_file:
            json.dump({
                'train_progress': {
                    'epoch': model.train_progress.epoch,
                    'epoch_step': model.train_progress.epoch_step,
                    'epoch_sample': model.train_progress.epoch_sample,
                    'global_step': model.train_progress.global_step,
                },
                'last_action_epoch': dict(getattr(model.train_progress, 'last_action_epoch', {})),
                'tensorboard_subdir': tensorboard_subdir,
            }, meta_file)
```

Using `getattr` with a default makes the saver tolerant to a `BaseModel` that hasn't had `tensorboard_subdir` set yet (defensive for any unusual code path).

---

## Task 8: Implement — Loader reads new fields

**Files:**
- Modify: `.worktrees/upstream-investigate/modules/modelLoader/BaseModelLoader.py`

- [ ] **Step 1: Read and populate the new fields**

Replace `_load_internal_state` body (lines 18-42) with:

```python
    def _load_internal_state(
            self,
            model: BaseModel,
            base_model_name: str,
    ):
        with open(os.path.join(base_model_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )
            train_progress.last_action_epoch = dict(meta.get('last_action_epoch', {}))
            resumed_tb_subdir = meta.get('tensorboard_subdir', None)

        # optimizer
        with contextlib.suppress(FileNotFoundError):
            model.optimizer_state_dict = torch.load(os.path.join(base_model_name, "optimizer", "optimizer.pt"),
                                                    weights_only=True)

        # ema
        with contextlib.suppress(FileNotFoundError):
            model.ema_state_dict = torch.load(os.path.join(base_model_name, "ema", "ema.pt"), weights_only=True)

        # meta
        model.train_progress = train_progress
        model.resumed_tensorboard_subdir = resumed_tb_subdir
```

`meta.get(..., default)` makes the loader tolerant of legacy backups created before this change.

---

## Task 9: Implement — `BaseModel` carries TB-subdir attributes

**Files:**
- Modify: `.worktrees/upstream-investigate/modules/model/BaseModel.py`

- [ ] **Step 1: Add the class-level annotations and init the attributes**

In the class field declarations block (currently lines 66-78, immediately after `train_dtype: DataType`), add:

```python
    resumed_tensorboard_subdir: str | None
    tensorboard_subdir: str | None
```

In `__init__` (currently lines 80-95), append after `self.train_dtype = DataType.FLOAT_32`:

```python
        self.resumed_tensorboard_subdir = None
        self.tensorboard_subdir = None
```

**Rationale for two attributes:** `resumed_tensorboard_subdir` is what the loader set (input from previous session); `tensorboard_subdir` is what the trainer decided to USE for this session. The trainer sets `tensorboard_subdir = resumed_tensorboard_subdir or <new_timestamp_dir>` and from then on that is the value saved into every subsequent backup. Keeping them distinct avoids ambiguity about which direction the value is flowing.

---

## Task 10: Implement — `GenericTrainer` reuses TB subdir on resume

**Files:**
- Modify: `.worktrees/upstream-investigate/modules/trainer/GenericTrainer.py`

**Context:** Currently TB is created in `__init__` (line 73), BEFORE `start()` loads the backup. We move the creation to the end of `start()` so `model.resumed_tensorboard_subdir` is available.

- [ ] **Step 1: Remove TB creation from `__init__`**

Replace lines 70-75:

```python
        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()
```

with:

```python
        if multi.is_master():
            # TB writer creation is deferred to start() so a backup's tensorboard_subdir
            # can be reused and logging continues in the same TB run on resume.
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()
        self.tensorboard = None
```

- [ ] **Step 2: Create TB writer at the end of `start()`**

After the `validation_data_loader` block ends (around line 162), before the method closes, add:

```python
        if multi.is_master():
            self._init_tensorboard_writer()
```

- [ ] **Step 3: Add the `_init_tensorboard_writer` helper**

Insert a new method in `GenericTrainer` (good placement: right after `start()`, before `__save_config_to_workspace`):

```python
    def _init_tensorboard_writer(self):
        """Construct the SummaryWriter.

        On resume (model.resumed_tensorboard_subdir is set and the directory still
        exists under workspace_dir/tensorboard), reuse the existing subdir so the
        TB UI shows a single continuous run across stop/resume cycles. Use
        purge_step to trim any scalars written after the backup point by a
        previous session that crashed mid-log.
        """
        tensorboard_log_dir = os.path.join(self.config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)

        resumed = getattr(self.model, "resumed_tensorboard_subdir", None)
        reuse_path = None
        if resumed:
            candidate = os.path.join(tensorboard_log_dir, resumed)
            if os.path.isdir(candidate):
                reuse_path = candidate

        if reuse_path is not None:
            self.model.tensorboard_subdir = resumed
            self.tensorboard = SummaryWriter(
                reuse_path,
                purge_step=self.model.train_progress.global_step,
            )
        else:
            subdir = f"{self.config.save_filename_prefix}{get_string_timestamp()}"
            self.model.tensorboard_subdir = subdir
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, subdir))
```

- [ ] **Step 4: Run the full test suite**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/ -v
```

Expected: all tests PASS. Do NOT edit tests. If anything fails, investigate the implementation.

---

## Task 11: Manual sanity check in worktree

- [ ] **Step 1: Syntax-verify every modified Python file**

```
./venv/Scripts/python.exe -m py_compile \
  .worktrees/upstream-investigate/modules/util/TrainProgress.py \
  .worktrees/upstream-investigate/modules/util/TimedActionMixin.py \
  .worktrees/upstream-investigate/modules/modelSaver/mixin/InternalModelSaverMixin.py \
  .worktrees/upstream-investigate/modules/modelLoader/BaseModelLoader.py \
  .worktrees/upstream-investigate/modules/model/BaseModel.py \
  .worktrees/upstream-investigate/modules/trainer/GenericTrainer.py
```

Expected: no output (clean compile).

- [ ] **Step 2: Re-run the test suite from scratch to confirm no flakiness**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/ -v --tb=short
```

Expected: all green.

---

## Task 12: Code-review agent pass

- [ ] **Step 1: Dispatch `feature-dev:code-reviewer` on the worktree diff**

Scope the review to only the six modified upstream files. Ask specifically:
1. Does the EPOCH + `start_at_zero=True` change preserve behavior for fresh runs (no regressions for users who never resume)?
2. Is the legacy-backup (no `last_action_epoch`, no `tensorboard_subdir`) path correct? I.e., a user resuming from a pre-fix backup should not crash.
3. Is deferring `SummaryWriter` creation from `__init__` to end of `start()` safe given other code that may reference `self.tensorboard` early? Check `BaseTrainer._start_tensorboard` and the start() flow.
4. Is `purge_step` the right semantic choice here? Risk of unexpected scalar loss?
5. `multi.is_master()` usage: does the TB init run only on the master process, and do other processes have a reasonable `self.tensorboard = None`?

Address any findings inline in the worktree before moving on. Re-run tests after each change.

---

## Task 13: Run upstream pre-commit hooks against the modified files

**Files:** No file changes in this task — this is a gate check before commit.

**Context:** Upstream `Nerogar/OneTrainer` runs pre-commit via `pre-commit.ci` on every PR. Config lives at `.worktrees/upstream-investigate/.pre-commit-config.yaml`:
- `pre-commit/pre-commit-hooks v6.0.0`: merge-conflict, case-conflict, illegal-windows-names, destroyed-symlinks, byte-order-marker, mixed-line-ending, trailing-whitespace, end-of-file-fixer, executables-have-shebangs, check-yaml
- `astral-sh/ruff-pre-commit v0.15.7`: ruff linter with `--fix` on python/pyi/jupyter
- `ci: autofix_prs: false` — pre-commit.ci will not push autofix commits, so our PR must be clean *before* the hooks run on CI.

Ruff is already in the venv; `pre-commit` is not. Install it first.

- [ ] **Step 1: Install pre-commit into the venv**

```bash
./venv/Scripts/pip.exe install pre-commit
```

Per the user's standing memory ("Use venv/Scripts/pip.exe, not system pip"), always use the venv's pip.

Verify:
```bash
./venv/Scripts/python.exe -m pre_commit --version
```

- [ ] **Step 2: Run pre-commit against only the six modified files**

From the worktree:
```bash
cd .worktrees/upstream-investigate
../../venv/Scripts/python.exe -m pre_commit run \
  --config .pre-commit-config.yaml \
  --files \
    modules/util/TrainProgress.py \
    modules/util/TimedActionMixin.py \
    modules/modelSaver/mixin/InternalModelSaverMixin.py \
    modules/modelLoader/BaseModelLoader.py \
    modules/model/BaseModel.py \
    modules/trainer/GenericTrainer.py
```

Expected: all hooks pass (green) OR pass with modifications applied (trailing whitespace / EOF fixer / ruff --fix). If modifications are applied, the hook exits non-zero the first time — this is by design.

- [ ] **Step 3: If hooks modified files, re-run tests and pre-commit**

If any hook auto-fixed a file:

1. Re-run the local test suite to confirm the fix didn't break behavior:
   ```
   ../../venv/Scripts/pytest.exe ../../.diagnostics/upstream-resume-tests/ -v
   ```
   Expected: all green. If something broke, investigate (likely an unused import was stripped by ruff — restore the import if it's actually needed, or remove the dead code that required it).

2. Re-run pre-commit to confirm it now passes cleanly:
   ```
   ../../venv/Scripts/python.exe -m pre_commit run --config .pre-commit-config.yaml --files <same six files>
   ```
   Expected: all hooks report `Passed`. If anything still fails, read the hook's error output and fix manually.

- [ ] **Step 4: If ruff reports an error it cannot auto-fix, fix it manually**

Common cases:
- Unused `import` in a file we modified but didn't fully use → remove the import.
- Line length (E501) in a line we added → wrap it or split.
- Whitespace or style nit → fix in place.

Do NOT add `# noqa` comments — upstream's style is to fix, not suppress. After fixing, re-run Step 2.

- [ ] **Step 5: Confirm worktree is clean of noise**

```bash
cd .worktrees/upstream-investigate
git status
git diff --stat
```

Expected: only the six target files show in the diff. No stray test files, no `.diagnostics/` content (that directory is local to the outer fork and should not appear in the worktree's git status at all).

---

## Task 14: Commit the upstream changes to a fresh branch in the worktree

- [ ] **Step 1: Create a branch in the worktree rooted at upstream master**

From the worktree directory:
```bash
cd .worktrees/upstream-investigate
git checkout -b fix/tensorboard-resume-continuity
```

- [ ] **Step 2: Stage and commit ONLY the six modified files**

```bash
git add modules/util/TrainProgress.py \
        modules/util/TimedActionMixin.py \
        modules/modelSaver/mixin/InternalModelSaverMixin.py \
        modules/modelLoader/BaseModelLoader.py \
        modules/model/BaseModel.py \
        modules/trainer/GenericTrainer.py

git status
```

Verify the status shows exactly those six files staged and nothing else. If anything else appears (test files, diagnostics, etc.), it's a bug — do not commit.

- [ ] **Step 3: Commit with a clear message**

Pre-commit hooks re-run on `git commit`. They should all pass cleanly now because Task 13 already validated and applied any fixes.

```bash
git commit -m "fix: preserve TensorBoard run and epoch validation across backup resume

When resuming via continue_last_backup, continue writing to the same
TensorBoard log subdirectory instead of spawning a new timestamped run,
and ensure epoch-scheduled validation still fires for epochs that were
resumed mid-way.

- Persist tensorboard_subdir and last_action_epoch in the backup meta.json.
- On resume, reuse the existing TB subdir with purge_step=global_step so
  phantom scalars past the backup point are trimmed.
- Gate the EPOCH + start_at_zero=True branch of TimedActionMixin on
  last_action_epoch instead of epoch_step == 0 so mid-epoch resume no
  longer drops the scheduled validation for that epoch.
- Legacy backups without the new fields load unchanged (graceful defaults)."
```

If the commit fails because a pre-commit hook modified a file, re-run Task 13's Step 3 cycle, then re-stage and re-commit (create a NEW commit, do NOT --amend). Per the user's standing memory: no `Co-Authored-By` attribution.

---

## Task 15: Final verification before opening a PR

- [ ] **Step 1: Re-run the full local test suite against the committed code**

```
./venv/Scripts/pytest.exe .diagnostics/upstream-resume-tests/ -v
```

Expected: all green.

- [ ] **Step 2: Re-run pre-commit against the committed range one more time**

```bash
cd .worktrees/upstream-investigate
../../venv/Scripts/python.exe -m pre_commit run --config .pre-commit-config.yaml --from-ref HEAD~1 --to-ref HEAD
```

Expected: all hooks `Passed`. This is the same invocation `pre-commit.ci` will use on the PR.

- [ ] **Step 3: Inspect the diff one final time**

```bash
cd .worktrees/upstream-investigate
git log -1 --stat
git diff HEAD~1 HEAD
```

Verify: six files changed, no test files, no docs, no formatting-only noise. Line count roughly:
- `TrainProgress.py`: +3 lines
- `TimedActionMixin.py`: ~+5 net lines
- `InternalModelSaverMixin.py`: +3 lines
- `BaseModelLoader.py`: +3 lines
- `BaseModel.py`: +4 lines
- `GenericTrainer.py`: ~+25 lines (helper + moved creation)

Total: ~45 lines of additive change. That is the "minimal PR" target.

- [ ] **Step 4: Report done**

Branch `fix/tensorboard-resume-continuity` in the worktree is ready to push to a fork and open a PR against `Nerogar/OneTrainer:master`. **Do not push without explicit user confirmation** — the user is the one who chooses when the PR goes up.

---

## Summary

| Layer | Change |
|---|---|
| State schema | Add `last_action_epoch` to `TrainProgress`; add `resumed_tensorboard_subdir` + `tensorboard_subdir` to `BaseModel` |
| Backup meta | Two new keys: `last_action_epoch`, `tensorboard_subdir` |
| Trigger logic | EPOCH + `start_at_zero=True` branch now gates on persistent `last_action_epoch[name]`, not transient `epoch_step == 0` |
| TB writer lifecycle | Deferred from `__init__` → end of `start()`; reuses existing subdir on resume with `purge_step` |
| Legacy compat | `getattr`/`meta.get` defaults everywhere a new field is read so old backups load cleanly |
| PR surface | Six upstream files, ~45 lines of diff |
| Local-only | Four test files in `.diagnostics/upstream-resume-tests/` (pytest, CPU-only) + this plan |
