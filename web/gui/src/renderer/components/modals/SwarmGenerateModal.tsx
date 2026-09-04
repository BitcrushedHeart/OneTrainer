import { AlertCircle, CheckCircle2, Loader2, Plug, Plus, Sparkles, StopCircle, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { swarmApi, type SwarmLora, type SwarmStatus } from "@/api/swarmApi";
import { Button, DirPicker, FormEntry, SectionCard, Select, Toggle } from "@/components/shared";

import { ModalBase } from "./ModalBase";

export interface SwarmGenerateModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.swarmGenerate";
/** The off-policy builder's folder settings, reused as defaults. */
const OFFPOLICY_KEY = "ot.offpolicyDpo.folders";

interface Settings {
  baseUrl: string;
  token: string;
  model: string;
  real: string;
  generated: string;
  output: string;
  captions: number;
  imagesPerCaption: number;
  keywords: string;
  excludeTagStyle: boolean;
  minCaptionWords: number;
  steps: string;
  cfg: string;
  negativePrompt: string;
  loras: SwarmLora[];
  paramSource: "last" | "preset" | "manual";
  presetName: string;
  concurrency: number;
}

const DEFAULTS: Settings = {
  baseUrl: "http://localhost:7801",
  token: "",
  model: "",
  real: "",
  generated: "",
  output: "",
  captions: 10,
  imagesPerCaption: 1,
  keywords: "",
  excludeTagStyle: true,
  minCaptionWords: 0,
  steps: "",
  cfg: "",
  negativePrompt: "",
  loras: [],
  paramSource: "last",
  presetName: "",
  concurrency: 3,
};

function loadSettings(): Settings {
  const settings = { ...DEFAULTS };
  try {
    // Fall back to the folders the off-policy builder already knows about.
    const shared = window.localStorage.getItem(OFFPOLICY_KEY);
    if (shared) {
      const folders = JSON.parse(shared) as { generated?: string; real?: string; output?: string };
      if (folders.generated) settings.generated = folders.generated;
      if (folders.real) settings.real = folders.real;
      if (folders.output) settings.output = folders.output;
    }
    const own = window.localStorage.getItem(STORAGE_KEY);
    if (own) Object.assign(settings, JSON.parse(own) as Partial<Settings>);
  } catch {
    /* storage unavailable — defaults are fine */
  }
  return settings;
}

const PHASE_LABEL: Record<SwarmStatus["phase"], string> = {
  idle: "Idle",
  starting: "Starting…",
  indexing: "Reading real photos and captions…",
  scanning: "Reading existing generations…",
  planning: "Choosing captions…",
  generating: "Generating in SwarmUI…",
  done: "Finished",
  error: "Failed",
  cancelling: "Stopping…",
  cancelled: "Stopped",
};

export function SwarmGenerateModal({ open, onClose }: SwarmGenerateModalProps) {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [models, setModels] = useState<string[]>([]);
  const [connection, setConnection] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<SwarmStatus | null>(null);
  const [availableLoras, setAvailableLoras] = useState<string[]>([]);
  const [loaded, setLoaded] = useState<string[]>([]);
  const [loraToAdd, setLoraToAdd] = useState("");
  const [loraWeight, setLoraWeight] = useState(1);
  const [presets, setPresets] = useState<string[]>([]);
  const [lastParams, setLastParams] = useState<Record<string, unknown> | null>(null);
  const pollRef = useRef<number | null>(null);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((previous) => {
      const next = { ...previous, ...patch };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
  }, []);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      void swarmApi
        .status()
        .then((state) => {
          setStatus(state);
          if (!state.running) clearPoll();
        })
        .catch(() => {
          /* transient */
        });
    }, 700);
  }, [clearPoll]);

  useEffect(() => {
    if (!open) {
      clearPoll();
      return;
    }
    let cancelled = false;
    void swarmApi.status().then((state) => {
      if (cancelled) return;
      setStatus(state);
      if (state.running) startPolling();
    });
    return () => {
      cancelled = true;
    };
  }, [open, clearPoll, startPolling]);

  useEffect(() => () => clearPoll(), [clearPoll]);

  const check = useCallback(async () => {
    setChecking(true);
    setError(null);
    setConnection(null);
    try {
      const result = await swarmApi.check(settings.baseUrl, settings.token);
      if (!result.ok) {
        setError(result.error ?? "Could not reach SwarmUI");
        return;
      }
      setModels(result.models ?? []);
      setAvailableLoras(result.loras ?? []);
      setLoaded(result.loaded ?? []);
      setPresets(result.presets ?? []);
      setConnection(
        `SwarmUI ${result.version ?? "?"} as ${result.user_id ?? "?"} — ${result.models?.length ?? 0} models, ${
          result.loras?.length ?? 0
        } LoRAs`,
      );
      // Only pre-fill a model when the user is choosing settings by hand;
      // otherwise the settings source supplies it.
      if (!settings.model && settings.paramSource === "manual") {
        const preferred = result.loaded?.[0] ?? result.models?.[0];
        if (preferred) update({ model: preferred });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  }, [settings.baseUrl, settings.token, settings.model, settings.paramSource, update]);

  const run = useCallback(
    async (dryRun: boolean) => {
      setBusy(true);
      setError(null);
      try {
        const result = await swarmApi.generate({
          real_folder: settings.real,
          generated_folder: settings.generated,
          output_dir: settings.output,
          model: settings.model,
          base_url: settings.baseUrl,
          token: settings.token,
          captions: Number(settings.captions) || 1,
          images_per_caption: Number(settings.imagesPerCaption) || 1,
          keywords: settings.keywords,
          exclude_tag_style: settings.excludeTagStyle,
          min_caption_words: Number(settings.minCaptionWords) || 0,
          steps: settings.steps.trim() === "" ? null : Number(settings.steps),
          cfg_scale: settings.cfg.trim() === "" ? null : Number(settings.cfg),
          negative_prompt: settings.negativePrompt,
          loras: settings.loras,
          param_source: settings.paramSource,
          preset_name: settings.presetName,
          concurrency: Number(settings.concurrency) || 1,
          dry_run: dryRun,
        });
        if (!result.ok) {
          setError(result.error ?? "Could not start");
          return;
        }
        startPolling();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [settings, startPolling],
  );

  const cancel = useCallback(async () => {
    try {
      await swarmApi.cancel();
    } catch {
      /* best effort */
    }
  }, []);

  const refreshLastParams = useCallback(async () => {
    if (!settings.generated) return;
    try {
      const result = await swarmApi.lastParams(settings.generated);
      setLastParams(result.ok ? (result.params ?? {}) : null);
    } catch {
      setLastParams(null);
    }
  }, [settings.generated]);

  const addLora = useCallback(() => {
    const name = loraToAdd.trim();
    if (!name || settings.loras.some((entry) => entry.name === name)) return;
    update({ loras: [...settings.loras, { name, weight: Number(loraWeight) || 1 }] });
  }, [loraToAdd, loraWeight, settings.loras, update]);

  const removeLora = useCallback(
    (name: string) => update({ loras: settings.loras.filter((entry) => entry.name !== name) }),
    [settings.loras, update],
  );

  // Refresh the review list whenever the modal opens and whenever a run ends.
  useEffect(() => {
    if (open) void refreshLastParams();
  }, [open, refreshLastParams]);

  const running = status?.running ?? false;
  const foldersReady = Boolean(settings.real && settings.generated && settings.output);

  return (
    <ModalBase open={open} onClose={onClose} title="Generate Off-policy Negatives (SwarmUI)" size="xl">
      <div className="flex flex-col gap-4">
        <div className="text-sm text-[var(--color-on-surface-secondary)]">
          Picks captions from your real photos that have no usable generation yet and generates them in SwarmUI at each
          photo&apos;s <span className="font-semibold">exact trainer bucket size</span> — so the pair needs no cropping
          at all. Images land in SwarmUI&apos;s own output folder, which is the folder the Off-policy DPO Builder scans,
          so they are ready to review there as pairs on its next run.
        </div>

        <SectionCard title="SwarmUI">
          <div className="grid grid-cols-4 gap-3 items-end">
            <FormEntry label="Server URL" value={settings.baseUrl} onChange={(v) => update({ baseUrl: String(v) })} />
            <FormEntry
              label="Auth token (optional)"
              value={settings.token}
              onChange={(v) => update({ token: String(v) })}
              tooltip="Only needed if your SwarmUI instance requires a swarm_token cookie."
            />
            <div className="col-span-2 flex items-end gap-2">
              <Button variant="secondary" onClick={() => void check()} disabled={checking}>
                {checking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plug className="w-4 h-4" />}
                Check connection
              </Button>
              {connection && (
                <span className="flex items-center gap-1 text-xs text-[var(--color-success-500)]">
                  <CheckCircle2 className="w-4 h-4" />
                  {connection}
                </span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-4 gap-3 items-end pt-3">
            <div className="col-span-2">
              <Select
                label="Generation settings"
                value={settings.paramSource}
                onChange={(value) => update({ paramSource: value as Settings["paramSource"] })}
                options={[
                  { label: "Reuse my last generation", value: "last" },
                  { label: "Use a SwarmUI preset", value: "preset" },
                  { label: "Only the fields below", value: "manual" },
                ]}
              />
            </div>
            {settings.paramSource === "preset" && (
              <div className="col-span-2">
                <Select
                  label="Preset"
                  value={settings.presetName}
                  onChange={(value) => update({ presetName: value })}
                  options={presets.length ? ["", ...presets] : [""]}
                  formatLabel={(value) => value || (presets.length ? "Pick a preset…" : "Check the connection first")}
                />
              </div>
            )}
          </div>
          {settings.paramSource === "last" && (
            <div className="text-xs text-[var(--color-on-surface-secondary)] pt-2">
              {lastParams && Object.keys(lastParams).length > 0 ? (
                <>
                  Reusing your last generation:{" "}
                  <span className="text-[var(--color-on-surface)]">
                    {Object.entries(lastParams)
                      .filter(([key]) => key !== "negativeprompt")
                      .map(([key, value]) => `${key} ${Array.isArray(value) ? value.join("+") : String(value)}`)
                      .join(" · ")}
                  </span>
                  . The prompt, size and seed are set per photo; anything you fill in below overrides the rest.
                </>
              ) : (
                <>
                  No previous generation found in that folder yet — generate once in SwarmUI, or pick a preset / fill in
                  the fields below.
                </>
              )}
            </div>
          )}
          <div className="grid grid-cols-4 gap-3 items-end pt-3">
            <div className="col-span-2">
              <Select
                label={settings.paramSource === "manual" ? "Model" : "Model (override)"}
                value={settings.model}
                onChange={(value) => update({ model: value })}
                options={models.length ? models : settings.model ? [settings.model] : [""]}
                formatLabel={(value) => value || "Check the connection to list models"}
              />
            </div>
            <FormEntry
              label="Steps (blank = from settings)"
              value={settings.steps}
              onChange={(v) => update({ steps: String(v) })}
            />
            <FormEntry
              label="CFG (blank = from settings)"
              value={settings.cfg}
              onChange={(v) => update({ cfg: String(v) })}
            />
          </div>
          <div className="pt-3">
            <FormEntry
              label="Negative prompt (blank = from settings)"
              value={settings.negativePrompt}
              onChange={(v) => update({ negativePrompt: String(v) })}
            />
          </div>
          {loaded.length > 0 && loaded[0] !== settings.model && (
            <div className="pt-2 text-xs text-[var(--color-on-surface-secondary)]">
              SwarmUI currently has <span className="font-semibold">{loaded[0]}</span> loaded.{" "}
              <button type="button" className="underline" onClick={() => update({ model: loaded[0] })}>
                Use it
              </button>
            </div>
          )}
        </SectionCard>

        <SectionCard title="LoRAs">
          <div className="grid grid-cols-4 gap-3 items-end">
            <div className="col-span-2">
              <Select
                label="Add a LoRA"
                value={loraToAdd}
                onChange={setLoraToAdd}
                options={availableLoras.length ? ["", ...availableLoras] : [""]}
                formatLabel={(value) =>
                  value || (availableLoras.length ? "Pick a LoRA…" : "Check the connection first")
                }
              />
            </div>
            <FormEntry label="Weight" type="number" value={loraWeight} onChange={(v) => setLoraWeight(Number(v))} />
            <Button variant="secondary" onClick={addLora} disabled={!loraToAdd.trim()}>
              <Plus className="w-4 h-4" /> Add
            </Button>
          </div>
          {settings.loras.length > 0 && (
            <div className="flex flex-col gap-1 pt-3">
              {settings.loras.map((entry) => (
                <div
                  key={entry.name}
                  className="flex items-center gap-2 text-sm bg-[var(--color-surface-container)] rounded px-3 py-1"
                >
                  <span className="flex-1 truncate" title={entry.name}>
                    {entry.name}
                  </span>
                  <span className="tabular-nums text-[var(--color-on-surface-secondary)]">×{entry.weight}</span>
                  <Button variant="ghost" size="sm" onClick={() => removeLora(entry.name)} title="Remove">
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
          {settings.loras.length === 0 && (
            <div className="pt-2 text-xs text-[var(--color-on-surface-secondary)]">
              {settings.paramSource === "manual"
                ? "No LoRAs — generations use the base model alone."
                : "None added, so the LoRAs from your chosen settings are used as-is. Adding any replaces them."}
            </div>
          )}
        </SectionCard>

        <SectionCard title="Folders">
          <div className="space-y-3">
            <DirPicker
              label="Real photos (with .txt captions)"
              value={settings.real}
              onChange={(v) => update({ real: v })}
            />
            <DirPicker
              label="Generated images (SwarmUI output)"
              value={settings.generated}
              onChange={(v) => update({ generated: v })}
            />
            <DirPicker
              label="Off-policy DPO output folder (for its caches and pairs)"
              value={settings.output}
              onChange={(v) => update({ output: v })}
            />
          </div>
        </SectionCard>

        <SectionCard title="What to generate">
          <div className="grid grid-cols-4 gap-3 items-end">
            <FormEntry
              label="Captions"
              type="number"
              value={settings.captions}
              onChange={(v) => update({ captions: Number(v) })}
              tooltip="How many different captions to generate for this run."
            />
            <FormEntry
              label="Images per caption"
              type="number"
              value={settings.imagesPerCaption}
              onChange={(v) => update({ imagesPerCaption: Number(v) })}
              tooltip="More images per caption gives the builder a harder negative to choose from."
            />
            <FormEntry
              label="Caption keywords"
              value={settings.keywords}
              onChange={(v) => update({ keywords: String(v) })}
              placeholder="comma-separated, all required"
            />
            <FormEntry
              label="Min caption words (0 = off)"
              type="number"
              value={settings.minCaptionWords}
              onChange={(v) => update({ minCaptionWords: Number(v) })}
              tooltip="Very short captions under-specify the scene, so the generation shares little with the photo."
            />
            <FormEntry
              label="Requests in flight"
              type="number"
              value={settings.concurrency}
              onChange={(v) => update({ concurrency: Number(v) })}
              tooltip="How many generation requests to keep queued at SwarmUI at once. Above 1 the next request is already waiting when the current one finishes, so the GPU never sits idle between batches."
            />
          </div>
          <div className="pt-3">
            <Toggle
              label="Skip tag-style captions"
              value={settings.excludeTagStyle}
              onChange={(value) => update({ excludeTagStyle: value })}
              tooltip="Booru-style tag lists ('blonde hair, blue eyes, outdoors') make poor realism prompts: the generation shares only keywords with the photo, so the pair teaches composition rather than realism."
            />
          </div>
        </SectionCard>

        <div className="flex flex-wrap items-center gap-2">
          <Button variant="secondary" onClick={() => void run(true)} disabled={busy || !foldersReady}>
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Preview plan
          </Button>
          <Button
            variant="primary"
            onClick={() => void run(false)}
            disabled={busy || !foldersReady || (settings.paramSource === "manual" && !settings.model)}
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {running ? "Queue another run" : "Generate"}
          </Button>
          {running && (
            <Button variant="danger" onClick={() => void cancel()}>
              <StopCircle className="w-4 h-4" /> Stop all
            </Button>
          )}
          {(status?.queued?.length ?? 0) > 0 && (
            <Button variant="ghost" onClick={() => void swarmApi.clearQueue()}>
              Clear queue ({status?.queued.length})
            </Button>
          )}
          {!foldersReady && (
            <span className="text-xs text-[var(--color-on-surface-secondary)]">Pick all three folders first.</span>
          )}
        </div>

        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)] bg-[var(--color-surface-container)] rounded p-3">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        {status && status.phase !== "idle" && (
          <div className="flex flex-col gap-2 bg-[var(--color-surface-container)] rounded p-3 text-sm">
            <div className="flex items-center gap-2">
              {running && <Loader2 className="w-4 h-4 animate-spin" />}
              <span>{PHASE_LABEL[status.phase]}</span>
              {status.dry_run && <span className="text-xs text-[var(--color-on-surface-secondary)]">(plan only)</span>}
            </div>
            {status.job_total > 1 && (
              <div className="text-xs text-[var(--color-on-surface-secondary)]">
                run {status.job_index} of {status.job_total}
                {status.current_job?.keywords ? ` — "${status.current_job.keywords}"` : ""}
                {status.queued.length > 0 && (
                  <> · next: {status.queued.map((j) => j.keywords || "(no keywords)").join(", ")}</>
                )}
              </div>
            )}
            <div className="flex flex-wrap gap-x-4 text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
              <span>{status.pool} captions need generating</span>
              <span>{status.planned} planned</span>
              <span>
                {status.done} done · {status.images} images
              </span>
              {status.failed > 0 && <span className="text-[var(--color-error-500)]">{status.failed} failed</span>}
              {status.skipped_tag_style > 0 && <span>{status.skipped_tag_style} tag-style skipped</span>}
              {status.skipped_short > 0 && <span>{status.skipped_short} too short</span>}
            </div>
            {status.param_source && (
              <div className="text-xs text-[var(--color-on-surface-secondary)]">
                Settings from <span className="text-[var(--color-on-surface)]">{status.param_source}</span>
                {Object.keys(status.params ?? {}).length > 0 && (
                  <>
                    {" — "}
                    {Object.entries(status.params)
                      .filter(([key]) => key !== "negativeprompt")
                      .map(([key, value]) => `${key} ${Array.isArray(value) ? value.join("+") : String(value)}`)
                      .join(" · ")}
                  </>
                )}
              </div>
            )}
            {status.error && <div className="text-xs text-[var(--color-error-500)]">{status.error}</div>}
          </div>
        )}

        {status && status.results.length > 0 && (
          <div className="flex flex-col gap-2 max-h-[40vh] overflow-y-auto">
            {status.results.map((result, index) => (
              <div key={`${result.real_path}-${index}`} className="bg-[var(--color-surface-container)] rounded p-3">
                <div className="text-sm text-[var(--color-on-surface)] whitespace-pre-wrap">{result.caption}</div>
                <div className="flex flex-wrap gap-x-4 pt-1 text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
                  <span>
                    {result.size[0]}×{result.size[1]} · {result.images} image{result.images === 1 ? "" : "s"}
                  </span>
                  <span>{result.caption_type}</span>
                  {result.planned_only && <span>planned</span>}
                  {result.ok && !result.planned_only && (
                    <span className="text-[var(--color-success-500)]">generated</span>
                  )}
                  {result.ok === false && <span className="text-[var(--color-error-500)]">{result.error}</span>}
                </div>
                {result.real_path && (
                  <div
                    className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                    title={result.real_path}
                  >
                    {result.real_path}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </ModalBase>
  );
}
