import {
  AlertCircle,
  CheckCircle2,
  ImageOff,
  Loader2,
  Plug,
  Plus,
  RotateCcw,
  Sparkles,
  StopCircle,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { swarmApi, type SwarmBlackImage, type SwarmLora, type SwarmStatus } from "@/api/swarmApi";
import { Button, DirPicker, FormEntry, SectionCard, Select, Toggle } from "@/components/shared";

import { ModalBase } from "./ModalBase";

/**
 * Fields another tool can pre-fill when it opens the modal (the Character
 * DPO flow hands over its folders and LoRA). Applied, and persisted like a
 * manual edit, each time `open` turns true; fields left undefined keep the
 * stored values. `loras` entries are SwarmUI model-list names.
 */
export interface SwarmGenerateInitial {
  realFolder?: string;
  generatedFolder?: string;
  outputDir?: string;
  loras?: Array<{ name: string; weight: number }>;
  imagesPerCaption?: number;
  captions?: number;
}

export interface SwarmGenerateModalProps {
  open: boolean;
  onClose: () => void;
  initial?: SwarmGenerateInitial;
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
  /** One run per comma-separated keyword instead of one run requiring them all. */
  bulkKeywords: boolean;
  excludeTagStyle: boolean;
  minCaptionWords: number;
  steps: string;
  cfg: string;
  negativePrompt: string;
  loras: SwarmLora[];
  paramSource: "last" | "preset" | "manual";
  presetName: string;
  concurrency: number;
  autoRequeueBlack: boolean;
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
  bulkKeywords: false,
  excludeTagStyle: true,
  minCaptionWords: 0,
  steps: "",
  cfg: "",
  negativePrompt: "",
  loras: [],
  paramSource: "last",
  presetName: "",
  concurrency: 3,
  autoRequeueBlack: true,
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

export function SwarmGenerateModal({ open, onClose, initial }: SwarmGenerateModalProps) {
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
  // Black (failed) generations found by the last scan; null = not scanned yet.
  const [blackImages, setBlackImages] = useState<SwarmBlackImage[] | null>(null);
  const [blackScanning, setBlackScanning] = useState(false);
  const [blackBusy, setBlackBusy] = useState(false);
  const [blackNotice, setBlackNotice] = useState<string | null>(null);
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

  // Seed from `initial` on each open, through `update` so the values persist
  // exactly as if typed. Read via a ref so a caller rebuilding the object
  // literal every render does not re-apply it while the modal is open.
  const initialRef = useRef(initial);
  initialRef.current = initial;
  useEffect(() => {
    if (!open) return;
    const seed = initialRef.current;
    if (!seed) return;
    const patch: Partial<Settings> = {};
    if (seed.realFolder !== undefined) patch.real = seed.realFolder;
    if (seed.generatedFolder !== undefined) patch.generated = seed.generatedFolder;
    if (seed.outputDir !== undefined) patch.output = seed.outputDir;
    if (seed.loras !== undefined) {
      patch.loras = seed.loras.map((entry) => ({ name: entry.name, weight: Number(entry.weight) || 1 }));
    }
    if (seed.imagesPerCaption !== undefined) patch.imagesPerCaption = Math.max(1, Number(seed.imagesPerCaption) || 1);
    if (seed.captions !== undefined) patch.captions = Math.max(1, Number(seed.captions) || 1);
    if (Object.keys(patch).length > 0) update(patch);
  }, [open, update]);

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

  /** Bulk mode: the keyword field split into one entry per run. */
  const bulkJobs = useMemo(() => {
    const seen = new Set<string>();
    return settings.keywords
      .split(",")
      .map((entry) => entry.trim())
      .filter((entry) => {
        if (!entry || seen.has(entry.toLowerCase())) return false;
        seen.add(entry.toLowerCase());
        return true;
      });
  }, [settings.keywords]);
  const bulkCaptions = Math.max(1, Number(settings.captions) || 1);
  const bulkPerCaption = Math.max(1, Number(settings.imagesPerCaption) || 1);

  const run = useCallback(
    async (dryRun: boolean) => {
      if (settings.bulkKeywords && bulkJobs.length === 0) {
        setError("Enter at least one keyword, or turn off One run per keyword");
        return;
      }
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
          keywords: settings.bulkKeywords ? "" : settings.keywords,
          keyword_jobs: settings.bulkKeywords ? bulkJobs : [],
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
          auto_requeue_black: settings.autoRequeueBlack,
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
    [settings, startPolling, bulkJobs],
  );

  const cancel = useCallback(async () => {
    try {
      await swarmApi.cancel();
    } catch {
      /* best effort */
    }
  }, []);

  const scanBlack = useCallback(async () => {
    if (!settings.generated) return;
    setBlackScanning(true);
    setBlackNotice(null);
    setError(null);
    try {
      const result = await swarmApi.scanBlack(settings.generated);
      if (!result.ok) {
        setError(result.error ?? "Scan failed");
        return;
      }
      setBlackImages(result.images ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBlackScanning(false);
    }
  }, [settings.generated]);

  const requeueBlack = useCallback(async () => {
    if (!settings.generated) return;
    setBlackBusy(true);
    setBlackNotice(null);
    setError(null);
    try {
      const result = await swarmApi.requeueBlack({
        generated_folder: settings.generated,
        base_url: settings.baseUrl,
        token: settings.token,
        model: settings.model,
        concurrency: Number(settings.concurrency) || 1,
        // Everything the scan found; the server re-checks each is still black.
        paths: (blackImages ?? []).map((image) => image.path),
      });
      if (!result.ok) {
        setError(result.error ?? "Could not requeue");
        return;
      }
      const unresendable = result.unresendable ?? [];
      const parts = [`${result.moved ?? 0} moved to .black`, `${result.requeued ?? 0} requeued`];
      if (unresendable.length > 0) parts.push(`${unresendable.length} without parameters (retired only)`);
      if (result.queued) parts.push("waiting behind the current run");
      setBlackNotice(parts.join(" · "));
      setBlackImages([]);
      if ((result.requeued ?? 0) > 0) startPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBlackBusy(false);
    }
  }, [settings, blackImages, startPolling]);

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
              label={settings.bulkKeywords ? "Captions per keyword" : "Captions"}
              type="number"
              value={settings.captions}
              onChange={(v) => update({ captions: Number(v) })}
              tooltip={
                settings.bulkKeywords
                  ? "How many different captions to generate for each keyword's run."
                  : "How many different captions to generate for this run."
              }
            />
            <FormEntry
              label="Images per caption"
              type="number"
              value={settings.imagesPerCaption}
              onChange={(v) => update({ imagesPerCaption: Number(v) })}
              tooltip="More images per caption gives the builder a harder negative to choose from."
            />
            <FormEntry
              label={settings.bulkKeywords ? "Bulk keywords" : "Caption keywords"}
              value={settings.keywords}
              onChange={(v) => update({ keywords: String(v) })}
              placeholder={
                settings.bulkKeywords ? "comma-separated, one run each (a+b = both)" : "comma-separated, all required"
              }
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
          {settings.bulkKeywords && (
            <div className="text-xs text-[var(--color-on-surface-secondary)] pt-2 tabular-nums">
              {bulkJobs.length === 0
                ? "Enter at least one keyword: each becomes its own queued run."
                : `${bulkJobs.length} run${bulkJobs.length === 1 ? "" : "s"} × ${bulkCaptions} caption${
                    bulkCaptions === 1 ? "" : "s"
                  } × ${bulkPerCaption} image${bulkPerCaption === 1 ? "" : "s"} = up to ${
                    bulkJobs.length * bulkCaptions * bulkPerCaption
                  } images. A keyword with fewer un-generated captions than that gives fewer.`}
            </div>
          )}
          <div className="flex flex-wrap gap-6 pt-3">
            <Toggle
              label="One run per keyword"
              value={settings.bulkKeywords}
              onChange={(value) => update({ bulkKeywords: value })}
              tooltip="Bulk add: every comma-separated keyword becomes its own queued run with the full captions count, e.g. 3 keywords at 30 captions = 90 images. Join words with + when one run needs them all (beach+sunset). Runs go in order and a caption two keywords share is generated only once."
            />
            <Toggle
              label="Skip tag-style captions"
              value={settings.excludeTagStyle}
              onChange={(value) => update({ excludeTagStyle: value })}
              tooltip="Booru-style tag lists ('blonde hair, blue eyes, outdoors') make poor realism prompts: the generation shares only keywords with the photo, so the pair teaches composition rather than realism."
            />
            <Toggle
              label="Auto-regenerate black images"
              value={settings.autoRequeueBlack}
              onChange={(value) => update({ autoRequeueBlack: value })}
              tooltip="At the start of each run, black images already in the generated folder (failed generations) are moved to .black and regenerated from their own settings ahead of the new captions. During the run, an output that comes back black is retried with a fresh seed up to two more times before it counts as a failure."
            />
          </div>
        </SectionCard>

        <SectionCard title="Black (failed) generations">
          <div className="text-xs text-[var(--color-on-surface-secondary)]">
            A generation that fails inside SwarmUI&apos;s backend (a dtype error mid-run, for instance) still lands in
            the output folder as a completely black image. It carries the prompt, so the builder counts its caption as
            generated and can never pair the photo properly. Scanning finds them; requeuing moves them into a{" "}
            <span className="font-mono">.black</span> folder beside themselves (which every scan skips) and generates
            each one again from its own recorded settings with a fresh seed. With &quot;Auto-regenerate black
            images&quot; on, every run does this by itself first; use this section to do it without generating anything
            new.
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-3">
            <Button
              variant="secondary"
              onClick={() => void scanBlack()}
              disabled={blackScanning || !settings.generated}
            >
              {blackScanning ? <Loader2 className="w-4 h-4 animate-spin" /> : <ImageOff className="w-4 h-4" />}
              Scan for black images
            </Button>
            <Button
              variant="primary"
              onClick={() => void requeueBlack()}
              disabled={blackBusy || !blackImages || blackImages.length === 0}
            >
              {blackBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}
              Retire and requeue{blackImages && blackImages.length > 0 ? ` (${blackImages.length})` : ""}
            </Button>
            {blackImages !== null && !blackNotice && (
              <span className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
                {blackImages.length === 0
                  ? "No black images found."
                  : `${blackImages.length} black image${blackImages.length === 1 ? "" : "s"} found${
                      blackImages.some((image) => !image.has_params)
                        ? ` · ${blackImages.filter((image) => !image.has_params).length} without parameters`
                        : ""
                    }`}
              </span>
            )}
            {blackNotice && <span className="text-xs text-[var(--color-success-500)]">{blackNotice}</span>}
          </div>
          {blackImages && blackImages.length > 0 && (
            <div className="flex flex-col gap-1 pt-3 max-h-[24vh] overflow-y-auto">
              {blackImages.slice(0, 200).map((image) => (
                <div
                  key={image.path}
                  className="flex items-center gap-2 text-xs bg-[var(--color-surface-container)] rounded px-3 py-1"
                >
                  <span className="flex-1 truncate" title={image.prompt || image.path}>
                    {image.prompt || <span className="italic">no prompt recorded</span>}
                  </span>
                  <span className="tabular-nums text-[var(--color-on-surface-secondary)]">
                    {image.size ? `${image.size[0]}×${image.size[1]}` : "?"}
                  </span>
                  {!image.has_params && <span className="text-[var(--color-error-500)]">no parameters</span>}
                  <span
                    className="text-[10px] text-[var(--color-on-surface-secondary)] truncate max-w-[30%]"
                    title={image.path}
                  >
                    {image.path.split(/[\\/]/).pop()}
                  </span>
                </div>
              ))}
              {blackImages.length > 200 && (
                <div className="text-[10px] text-[var(--color-on-surface-secondary)]">
                  … and {blackImages.length - 200} more
                </div>
              )}
            </div>
          )}
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
                {status.current_job?.mode === "requeue"
                  ? ` — requeue of ${status.current_job.images} black image${status.current_job.images === 1 ? "" : "s"}`
                  : status.current_job?.keywords
                    ? ` — "${status.current_job.keywords}"`
                    : ""}
                {status.queued.length > 0 && (
                  <>
                    {" "}
                    · next:{" "}
                    {status.queued
                      .map((j) => (j.mode === "requeue" ? `requeue ${j.images} black` : j.keywords || "(no keywords)"))
                      .join(", ")}
                  </>
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
              {status.black_retired > 0 && <span>{status.black_retired} black retired and requeued</span>}
              {status.black_retried > 0 && <span>{status.black_retried} black outputs retried</span>}
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
                  {result.requeued_from && <span>requeued</span>}
                  {result.ok && !result.planned_only && (
                    <span className="text-[var(--color-success-500)]">generated</span>
                  )}
                  {(result.black ?? 0) > 0 && (
                    <span className="text-[var(--color-error-500)]">
                      {result.black} black output{result.black === 1 ? "" : "s"} retired
                    </span>
                  )}
                  {result.ok === false && <span className="text-[var(--color-error-500)]">{result.error}</span>}
                </div>
                {result.requeued_from && (
                  <div
                    className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                    title={result.requeued_from}
                  >
                    was {result.requeued_from}
                  </div>
                )}
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
