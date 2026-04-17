# Live Config Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every UI field change persists to disk so that after closing and reopening the app the state matches exactly what the user last set.

**Architecture:** Treat `training_presets/#.json` as the canonical live user state (matches legacy CTk behavior). The backend already reads `#.json` on startup and writes it on every `update_config`; we remove the frontend `autoLoadPreset` call that overrides the persisted state, move first-run Z-Image seeding into the backend, and add a synchronous flush-on-close path through Electron so in-flight debounced writes (config, concepts, samples) complete before the process exits.

**Tech Stack:** FastAPI (Python 3.12), React 19 + Zustand, Electron 40 IPC, Vitest, pytest.

---

## File Structure

**Backend changes** (one file, one responsibility: make `#.json` the canonical state + seed on first run):
- Modify `web/backend/services/config_service.py`

**Backend tests:**
- Create `web/backend/tests/test_config_service_persistence.py`

**Frontend renderer changes** (split by responsibility):
- Modify `web/gui/src/renderer/store/configStore.ts` — add `flushPendingChanges`, delete `autoLoadPreset`, remove `localStorage "onetrainer_last_preset"` writes
- Modify `web/gui/src/renderer/App.tsx` — remove `autoLoadPreset()` from init, register flush-on-close IPC listener

**Frontend tests:**
- Create `web/gui/src/renderer/store/__tests__/configStore.flush.test.ts`

**Electron shared / main / preload** (add a request-flush IPC one-way channel, main → renderer):
- Modify `web/gui/src/shared/ipc-channels.ts`
- Modify `web/gui/src/shared/electron-api.ts`
- Modify `web/gui/src/main/preload.ts`
- Modify `web/gui/src/main/index.ts` — `cleanupAndQuit` awaits renderer flush before backend shutdown

---

## Task 1: Backend first-run seeding

**Files:**
- Modify: `web/backend/services/config_service.py` (`_load_default_preset` and add helper)

**Context:** `_load_default_preset()` currently loads `#.json` if it exists, otherwise leaves `TrainConfig.default_values()`. We want: if `#.json` does not exist, pick the first built-in preset whose filename (stripped of leading `#` and extension) contains `z-image` or `z_image`; if none match, use the first built-in preset (`#*.json` excluding `#.json`); if none exist at all, keep raw defaults. Whichever branch runs, write the result to `#.json` atomically so subsequent startups take the fast path.

- [ ] **Step 1: Write the failing test**

Create `web/backend/tests/test_config_service_persistence.py`:

```python
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def temp_presets_dir(tmp_path, monkeypatch):
    """Redirect PRESETS_DIR to an isolated tmp directory and reset singleton."""
    presets_dir = tmp_path / "training_presets"
    presets_dir.mkdir()
    monkeypatch.setattr("web.backend.paths.PRESETS_DIR", str(presets_dir))
    monkeypatch.setattr(
        "web.backend.services.config_service.PRESETS_DIR", str(presets_dir)
    )
    monkeypatch.setattr(
        "web.backend.services.config_service._DEFAULT_PRESET_PATH",
        str(presets_dir / "#.json"),
    )
    # Reset singleton so __init__ reruns with patched paths.
    from web.backend.services.config_service import ConfigService
    ConfigService._instance = None
    yield presets_dir
    ConfigService._instance = None


def _write_preset(path: Path, model_type: str = "Z_IMAGE") -> None:
    from modules.util.config.TrainConfig import TrainConfig
    cfg = TrainConfig.default_values()
    cfg.model_type = type(cfg.model_type)[model_type]
    data = cfg.to_settings_dict(secrets=False)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def test_first_run_seeds_from_z_image_preset(temp_presets_dir):
    """When #.json is absent, ConfigService seeds from a z-image built-in preset and writes #.json."""
    _write_preset(temp_presets_dir / "#z-image LoRA 16GB.json", "Z_IMAGE")

    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()

    default_file = temp_presets_dir / "#.json"
    assert default_file.exists(), "#.json should be created on first run"

    saved = json.loads(default_file.read_text(encoding="utf-8"))
    assert saved["model_type"] == "Z_IMAGE"


def test_first_run_falls_back_to_any_builtin(temp_presets_dir):
    """When no z-image preset exists, fall back to first #*.json preset."""
    _write_preset(temp_presets_dir / "#flux LoRA.json", "FLUX_DEV_1")

    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()

    saved = json.loads((temp_presets_dir / "#.json").read_text(encoding="utf-8"))
    assert saved["model_type"] == "FLUX_DEV_1"


def test_existing_default_preset_is_loaded_unchanged(temp_presets_dir):
    """If #.json already exists, it is loaded and not overwritten by a Z-Image seed."""
    _write_preset(temp_presets_dir / "#.json", "STABLE_DIFFUSION_XL_10_BASE")
    _write_preset(temp_presets_dir / "#z-image LoRA 16GB.json", "Z_IMAGE")
    original_mtime = (temp_presets_dir / "#.json").stat().st_mtime_ns

    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()

    assert service.config.model_type.name == "STABLE_DIFFUSION_XL_10_BASE"
    assert (temp_presets_dir / "#.json").stat().st_mtime_ns == original_mtime


def test_no_presets_at_all_keeps_defaults(temp_presets_dir):
    """With no preset files, keep raw TrainConfig defaults and do not create #.json."""
    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()
    assert service.config is not None
    assert not (temp_presets_dir / "#.json").exists()


def test_update_config_writes_default_preset_atomically(temp_presets_dir):
    """Every update_config persists to #.json."""
    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()

    current = service.get_config_dict()
    current["learning_rate"] = 0.000123
    service.update_config(current)

    saved = json.loads((temp_presets_dir / "#.json").read_text(encoding="utf-8"))
    assert saved["learning_rate"] == pytest.approx(0.000123)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest web/backend/tests/test_config_service_persistence.py -v`
Expected: `test_first_run_seeds_from_z_image_preset` and `test_first_run_falls_back_to_any_builtin` FAIL (no seeding yet). Others pass.

- [ ] **Step 3: Implement seeding in `_load_default_preset`**

Replace the existing `_load_default_preset` method in `web/backend/services/config_service.py` with:

```python
    def _load_default_preset(self) -> None:
        if os.path.isfile(_DEFAULT_PRESET_PATH):
            try:
                with open(_DEFAULT_PRESET_PATH, "r", encoding="utf-8") as fh:
                    loaded_dict: dict = json.load(fh)
                loaded_dict["__version"] = self.config.config_version
                self.config.from_dict(loaded_dict)
                logger.info("Restored config from %s", _DEFAULT_PRESET_PATH)
                return
            except Exception:
                logger.warning(
                    "Failed to load default preset %s, attempting first-run seed",
                    _DEFAULT_PRESET_PATH,
                    exc_info=True,
                )

        seed_path = self._find_first_run_seed_preset()
        if seed_path is None:
            return

        try:
            with open(seed_path, "r", encoding="utf-8") as fh:
                loaded_dict = json.load(fh)
            loaded_dict["__version"] = self.config.config_version
            self.config.from_dict(loaded_dict)
            logger.info("First-run: seeded config from %s", seed_path)
            self._save_default_preset()
        except Exception:
            logger.warning("Failed to seed first-run config from %s", seed_path, exc_info=True)

    def _find_first_run_seed_preset(self) -> str | None:
        if not os.path.isdir(PRESETS_DIR):
            return None

        try:
            entries = sorted(os.listdir(PRESETS_DIR))
        except OSError:
            return None

        def is_builtin(name: str) -> bool:
            return name.startswith("#") and name.endswith(".json") and name != "#.json"

        builtins = [name for name in entries if is_builtin(name)]
        if not builtins:
            return None

        for name in builtins:
            lowered = name.lower()
            if "z-image" in lowered or "z_image" in lowered:
                return os.path.join(PRESETS_DIR, name)

        return os.path.join(PRESETS_DIR, builtins[0])
```

- [ ] **Step 4: Run test to verify all pass**

Run: `pytest web/backend/tests/test_config_service_persistence.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Run ruff**

Run: `ruff check --fix web/backend/services/config_service.py web/backend/tests/test_config_service_persistence.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add web/backend/services/config_service.py web/backend/tests/test_config_service_persistence.py
git commit -m "feat(web/backend): seed #.json on first run from z-image built-in preset"
```

---

## Task 2: Frontend `flushPendingChanges` action

**Files:**
- Modify: `web/gui/src/renderer/store/configStore.ts`
- Create: `web/gui/src/renderer/store/__tests__/configStore.flush.test.ts`

**Context:** Today `updateField` schedules three independent debounced timers (`_syncTimer`, `_conceptSaveTimer`, `_sampleSaveTimer`). On window close these may not have fired. `flushPendingChanges` cancels any pending timers and runs the pending writes synchronously, so a close ≤500ms after the last edit is safe.

- [ ] **Step 1: Write the failing test**

Create `web/gui/src/renderer/store/__tests__/configStore.flush.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { configApi } from "@/api/configApi";
import { useConfigStore } from "@/store/configStore";

vi.mock("@/api/configApi", () => ({
  configApi: {
    getConfig: vi.fn(),
    updateConfig: vi.fn(),
    saveConcepts: vi.fn(),
    saveSamples: vi.fn(),
    getConcepts: vi.fn(),
    getSamples: vi.fn(),
  },
}));

const minimalConfig = {
  learning_rate: 0.0001,
  concepts: [{ name: "foo" }],
  samples: [{ prompt: "bar" }],
} as unknown as Parameters<typeof useConfigStore.setState>[0];

describe("configStore.flushPendingChanges", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    (configApi.updateConfig as ReturnType<typeof vi.fn>).mockResolvedValue(minimalConfig);
    (configApi.saveConcepts as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (configApi.saveSamples as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    useConfigStore.setState({
      config: { ...minimalConfig } as never,
      isDirty: false,
      _syncTimer: null,
      _conceptSaveTimer: null,
      _sampleSaveTimer: null,
    } as never);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    useConfigStore.getState().destroy();
  });

  it("flushes pending config sync immediately when dirty", async () => {
    useConfigStore.getState().updateField("learning_rate", 0.0002);
    expect(useConfigStore.getState().isDirty).toBe(true);
    expect(configApi.updateConfig).not.toHaveBeenCalled();

    await useConfigStore.getState().flushPendingChanges();

    expect(configApi.updateConfig).toHaveBeenCalledTimes(1);
    expect(useConfigStore.getState().isDirty).toBe(false);
    expect(useConfigStore.getState()._syncTimer).toBeNull();
  });

  it("flushes pending concepts save immediately", async () => {
    useConfigStore.getState().updateField("concepts", [{ name: "baz" }]);
    expect(useConfigStore.getState()._conceptSaveTimer).not.toBeNull();

    await useConfigStore.getState().flushPendingChanges();

    expect(configApi.saveConcepts).toHaveBeenCalledTimes(1);
    expect(useConfigStore.getState()._conceptSaveTimer).toBeNull();
  });

  it("flushes pending samples save immediately", async () => {
    useConfigStore.getState().updateField("samples", [{ prompt: "new" }]);
    expect(useConfigStore.getState()._sampleSaveTimer).not.toBeNull();

    await useConfigStore.getState().flushPendingChanges();

    expect(configApi.saveSamples).toHaveBeenCalledTimes(1);
    expect(useConfigStore.getState()._sampleSaveTimer).toBeNull();
  });

  it("is a no-op when nothing is dirty", async () => {
    await useConfigStore.getState().flushPendingChanges();
    expect(configApi.updateConfig).not.toHaveBeenCalled();
    expect(configApi.saveConcepts).not.toHaveBeenCalled();
    expect(configApi.saveSamples).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Verify test fails**

Run: `cd web/gui && npm test -- configStore.flush`
Expected: FAIL — `flushPendingChanges` is not defined.

- [ ] **Step 3: Add `flushPendingChanges` to the store**

In `web/gui/src/renderer/store/configStore.ts`, add the action to the `ConfigState` interface and to the store implementation.

Update the interface (near the other action declarations around line 88–102):

```typescript
  flushPendingChanges: () => Promise<void>;
```

Add the implementation (place it adjacent to `syncToBackend`, before `loadConcepts`):

```typescript
    flushPendingChanges: async () => {
      const { _syncTimer, _conceptSaveTimer, _sampleSaveTimer } = get();

      const tasks: Promise<unknown>[] = [];

      if (_syncTimer !== null) {
        clearTimeout(_syncTimer);
        set((draft) => {
          draft._syncTimer = null;
        });
      }
      if (get().isDirty) {
        tasks.push(get().syncToBackend());
      }

      if (_conceptSaveTimer !== null) {
        clearTimeout(_conceptSaveTimer);
        set((draft) => {
          draft._conceptSaveTimer = null;
        });
        const concepts = get().config?.concepts;
        if (concepts != null) {
          tasks.push(
            configApi.saveConcepts(concepts).catch((err) => {
              console.error("[configStore] flush: failed to save concepts:", err);
            }),
          );
        }
      }

      if (_sampleSaveTimer !== null) {
        clearTimeout(_sampleSaveTimer);
        set((draft) => {
          draft._sampleSaveTimer = null;
        });
        const samples = get().config?.samples;
        if (samples != null) {
          tasks.push(
            configApi.saveSamples(samples).catch((err) => {
              console.error("[configStore] flush: failed to save samples:", err);
            }),
          );
        }
      }

      await Promise.all(tasks);
    },
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd web/gui && npm test -- configStore.flush`
Expected: all 4 tests PASS.

- [ ] **Step 6: Run eslint**

Run: `cd web/gui && npm run lint`
Expected: no errors on `configStore.ts` or the new test file.

- [ ] **Step 7: Commit**

```bash
git add web/gui/src/renderer/store/configStore.ts web/gui/src/renderer/store/__tests__/configStore.flush.test.ts
git commit -m "feat(web/gui): add flushPendingChanges to configStore"
```

---

## Task 3: Remove `autoLoadPreset` override from startup

**Files:**
- Modify: `web/gui/src/renderer/store/configStore.ts`
- Modify: `web/gui/src/renderer/App.tsx`

**Context:** `App.tsx` calls `autoLoadPreset()` after `loadConfig()`, which replaces the backend-persisted `#.json` state with a stored preset path from `localStorage`. Remove the call; delete the action and its `localStorage` reads/writes. First-run seeding is handled by the backend (Task 1).

- [ ] **Step 1: Remove the call from App.tsx init**

In `web/gui/src/renderer/App.tsx`, replace the init effect (around line 95–105):

```typescript
  useEffect(() => {
    if (!backendConnected) return;
    const init = async () => {
      await loadConfig();
      await fetchTrainingStatus();
    };
    init().catch((err) => {
      console.error("App initialization failed:", err);
    });
  }, [backendConnected, loadConfig, fetchTrainingStatus]);
```

Remove the now-unused import / hook read:

```typescript
// Delete this line:
const autoLoadPreset = useConfigStore((s) => s.autoLoadPreset);
```

- [ ] **Step 2: Delete `autoLoadPreset` action from the store**

In `web/gui/src/renderer/store/configStore.ts`:

- Remove `autoLoadPreset: () => Promise<void>;` from the `ConfigState` interface.
- Remove the entire `autoLoadPreset: async () => { ... }` block from the store implementation.

- [ ] **Step 3: Remove `localStorage "onetrainer_last_preset"` writes**

Inside `loadPreset`, delete the try/catch that writes to `localStorage`:

```typescript
// Delete:
try {
  localStorage.setItem("onetrainer_last_preset", presetPath);
} catch {
  /* ignore */
}
```

- [ ] **Step 4: Grep for stray references and clean up**

Run: `grep -rn "onetrainer_last_preset\|autoLoadPreset" web/gui/src/`
Expected: no matches.

If any remain (e.g., in a test or comment), remove them.

- [ ] **Step 5: Typecheck + lint**

Run: `cd web/gui && npm run typecheck && npm run lint`
Expected: pass.

- [ ] **Step 6: Run existing tests**

Run: `cd web/gui && npm test`
Expected: all tests pass (includes the new flush test from Task 2).

- [ ] **Step 7: Commit**

```bash
git add web/gui/src/renderer/App.tsx web/gui/src/renderer/store/configStore.ts
git commit -m "fix(web/gui): remove autoLoadPreset override so persisted config survives restart"
```

---

## Task 4: Add IPC channel for main→renderer flush request

**Files:**
- Modify: `web/gui/src/shared/ipc-channels.ts`
- Modify: `web/gui/src/shared/electron-api.ts`
- Modify: `web/gui/src/main/preload.ts`

**Context:** The flush must originate in the Electron main process during `cleanupAndQuit`. Main will send a request over an IPC channel; the renderer will subscribe and reply with an ack once flush completes. We use `ipcRenderer.on` in the preload to expose a subscribe function.

- [ ] **Step 1: Add channel constants**

In `web/gui/src/shared/ipc-channels.ts`:

```typescript
export const IPC_CHANNELS = {
  OPEN_FILE: "dialog:openFile",
  OPEN_DIRECTORY: "dialog:openDirectory",
  SAVE_FILE: "dialog:saveFile",
  GET_APP_PATH: "app:getPath",
  RESTART_BACKEND: "backend:restart",
  GET_PLATFORM_INFO: "app:getPlatformInfo",
  GET_BACKEND_PORT: "backend:getPort",
  OPEN_MASK_EDITOR: "tools:openMaskEditor",
  FLUSH_REQUEST: "lifecycle:flushRequest",
  FLUSH_COMPLETE: "lifecycle:flushComplete",
} as const;

export type IpcChannel = (typeof IPC_CHANNELS)[keyof typeof IPC_CHANNELS];
```

- [ ] **Step 2: Extend the renderer-facing API surface**

In `web/gui/src/shared/electron-api.ts`, add to the `ElectronAPI` interface:

```typescript
  onFlushRequest: (handler: (requestId: string) => Promise<void> | void) => () => void;
  signalFlushComplete: (requestId: string) => void;
```

- [ ] **Step 3: Implement in preload**

In `web/gui/src/main/preload.ts`, add to the object exposed on `window.electronAPI`:

```typescript
  onFlushRequest: (handler) => {
    const listener = (_event: Electron.IpcRendererEvent, requestId: string) => {
      void Promise.resolve(handler(requestId));
    };
    ipcRenderer.on(IPC_CHANNELS.FLUSH_REQUEST, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.FLUSH_REQUEST, listener);
    };
  },
  signalFlushComplete: (requestId) => {
    ipcRenderer.send(IPC_CHANNELS.FLUSH_COMPLETE, requestId);
  },
```

- [ ] **Step 4: Typecheck**

Run: `cd web/gui && npm run typecheck`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add web/gui/src/shared/ipc-channels.ts web/gui/src/shared/electron-api.ts web/gui/src/main/preload.ts
git commit -m "feat(web/gui): add main→renderer flush IPC channels"
```

---

## Task 5: Renderer listens for flush and calls `flushPendingChanges`

**Files:**
- Modify: `web/gui/src/renderer/App.tsx`

**Context:** On mount, subscribe to `onFlushRequest`. When main dispatches the request, call `flushPendingChanges`, then `signalFlushComplete(requestId)`. Unsubscribe on unmount. If the window is not in Electron (e.g., vite dev server in a plain browser), `window.electronAPI` is undefined — skip registration.

- [ ] **Step 1: Update App.tsx**

Add the effect near the other `useEffect`s, below the init effect:

```typescript
  const flushPendingChanges = useConfigStore((s) => s.flushPendingChanges);

  useEffect(() => {
    const api = (window as unknown as { electronAPI?: import("../shared/electron-api").ElectronAPI }).electronAPI;
    if (!api) return;

    const unsubscribe = api.onFlushRequest(async (requestId) => {
      try {
        await flushPendingChanges();
      } catch (err) {
        console.error("[App] flush failed:", err);
      } finally {
        api.signalFlushComplete(requestId);
      }
    });

    return unsubscribe;
  }, [flushPendingChanges]);
```

- [ ] **Step 2: Typecheck + lint**

Run: `cd web/gui && npm run typecheck && npm run lint`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add web/gui/src/renderer/App.tsx
git commit -m "feat(web/gui): renderer flushes pending config writes on lifecycle request"
```

---

## Task 6: Main process awaits renderer flush before window closes

**Files:**
- Modify: `web/gui/src/main/index.ts`

**Context:** The existing `cleanupAndQuit` fires on `window-all-closed`, by which point the window is already destroyed and we can no longer IPC the renderer. The correct hook is the window's own `close` event: intercept it with `event.preventDefault()`, send `FLUSH_REQUEST`, await the ack (capped at 3 s), then call `win.destroy()` so normal teardown proceeds.

- [ ] **Step 1: Import `randomUUID`**

Near the other imports at the top of `web/gui/src/main/index.ts`:

```typescript
import { randomUUID } from "node:crypto";
```

- [ ] **Step 2: Add a `flushWindow` helper above `createWindow`**

```typescript
function flushWindow(win: BrowserWindow, timeoutMs = 3000): Promise<void> {
  if (win.isDestroyed() || win.webContents.isDestroyed()) return Promise.resolve();

  return new Promise<void>((resolve) => {
    const requestId = randomUUID();

    const timer = setTimeout(() => {
      ipcMain.removeListener(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
      console.warn(`[Electron] Flush timed out after ${timeoutMs}ms for window ${win.id}`);
      resolve();
    }, timeoutMs);

    const onComplete = (_event: Electron.IpcMainEvent, id: string) => {
      if (id !== requestId) return;
      clearTimeout(timer);
      ipcMain.removeListener(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
      resolve();
    };

    ipcMain.on(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
    try {
      win.webContents.send(IPC_CHANNELS.FLUSH_REQUEST, requestId);
    } catch (err) {
      console.warn(`[Electron] Failed to send flush request: ${String(err)}`);
      clearTimeout(timer);
      ipcMain.removeListener(IPC_CHANNELS.FLUSH_COMPLETE, onComplete);
      resolve();
    }
  });
}
```

- [ ] **Step 3: Intercept `close` on the main window**

Inside `createWindow()` in `web/gui/src/main/index.ts`, immediately after `win.on("closed", ...)` (around line 352), add:

```typescript
    let didFlush = false;
    win.on("close", (event) => {
      if (didFlush) return;
      event.preventDefault();
      void (async () => {
        try {
          await flushWindow(win);
        } finally {
          didFlush = true;
          if (!win.isDestroyed()) win.destroy();
        }
      })();
    });
```

Only guard the main window; secondary windows (mask editor) don't need flushing since they don't own the config store.

- [ ] **Step 4: Typecheck**

Run: `cd web/gui && npm run typecheck`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add web/gui/src/main/index.ts
git commit -m "feat(web/gui): main window flushes renderer before closing"
```

---

## Task 7: Manual end-to-end verification

**Context:** Automated tests verify the units; this task verifies the whole loop in a running app. Do not mark the feature complete without this.

- [ ] **Step 1: Wipe existing default preset**

```bash
rm -f training_presets/#.json
```

- [ ] **Step 2: Start dev app**

In one terminal:
```bash
cd web/backend && uvicorn main:app --reload --port 8000
```

In another:
```bash
cd web/gui && npm run dev
```

Then launch Electron:
```bash
cd web/gui && npm run electron:dev
```

(If `electron:dev` is not the script name, use whatever the repo uses to launch Electron against the vite dev server.)

- [ ] **Step 3: Verify first-run seed**

Expected: UI loads with a z-image preset's values (model_type Z_IMAGE, matching hyperparameters).
Check disk: `training_presets/#.json` exists and contains Z_IMAGE model_type.

- [ ] **Step 4: Edit a value**

Change learning rate to `0.000987`. Wait 1 second.
Check disk: `cat training_presets/#.json | grep learning_rate` shows `0.000987`.

- [ ] **Step 5: Edit and immediately close**

Change learning rate to `0.000111`. Within <200 ms, close the Electron window (faster than the 500 ms debounce).

Check disk: `cat training_presets/#.json | grep learning_rate` shows `0.000111`. (This verifies the flush-on-close.)

- [ ] **Step 6: Restart the app**

Launch again. Expected: learning rate in UI is `0.000111`. Top bar shows no preset label (the blank `#.json` slot).

- [ ] **Step 7: Load a named preset, edit, restart**

Load `#flux LoRA` from the top-bar dropdown. Change learning rate to `0.000222`. Close. Relaunch.
Expected: learning rate is `0.000222`. Top bar shows no preset label. `#.json` on disk reflects the flux-based state.

- [ ] **Step 8: Commit notes if any tweaks were needed**

If steps revealed issues and you adjusted code, commit those fixes with messages like:
```bash
git commit -m "fix(web/gui): <specific issue>"
```

---

## Task 8: Final lint pass (repo rule)

Per `CLAUDE.md`: "ALWAYS RUN RUFF AND ESLINT ON STRICT MODE BEFORE FINALISING ANY TASK."

- [ ] **Step 1: Ruff**

Run: `ruff check --fix .`
Expected: no errors.

- [ ] **Step 2: ESLint**

Run: `cd web/gui && npm run lint`
Expected: no errors.

- [ ] **Step 3: Typecheck**

Run: `cd web/gui && npm run typecheck`
Expected: pass.

- [ ] **Step 4: Full test suites**

Run: `pytest web/backend/tests/` and `cd web/gui && npm test`
Expected: all pass.

- [ ] **Step 5: Commit any auto-fixes**

```bash
git add -u
git diff --cached --stat
git commit -m "chore: lint autofixes" || echo "nothing to commit"
```
