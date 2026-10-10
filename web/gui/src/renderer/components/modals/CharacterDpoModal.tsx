import { Check, Eye, FolderInput, Heart, Loader2, PackageCheck, RefreshCw, Upload, Wand2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { type Character, characterDpoApi, type FinalizeStatus, type ImportCandidate } from "@/api/characterDpoApi";
import { Button, DirPicker, FormEntry, Select } from "@/components/shared";
import { TEXTAREA_FULL } from "@/utils/inputStyles";

import { GenerateProgress } from "./characterDpo/GenerateProgress";
import { QueueStep } from "./characterDpo/QueueStep";
import { RejectedPicker } from "./characterDpo/RejectedPicker";
import {
  Bar,
  Chip,
  errorText,
  HINT,
  joinPath,
  loadOffPolicyDefaults,
  loraStem,
  Notice,
  SECTION,
  SECTION_TITLE,
  splitLabel,
} from "./characterDpo/ui";
import { type DPOToolInitial, DPOToolModal } from "./DPOToolModal";
import { ModalBase } from "./ModalBase";
import { type OffPolicyDpoInitial, OffPolicyDpoModal } from "./OffPolicyDpoModal";
import { type SwarmGenerateInitial, SwarmGenerateModal } from "./SwarmGenerateModal";

export interface CharacterDpoModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.characterDpo";
/** Checked character names, kept across close/reopen. */
const SELECTED_KEY = "ot.characterDpo.selected";
/** The off-policy generator's connection settings, shown read-only here. */
const SWARM_KEY = "ot.swarmGenerate";

const MASK_MODELS = ["Rembg-Human", "Rembg", "ClipSeg"];

type Step = 1 | 2 | 3 | 4;
const STEPS: Array<{ id: Step; label: string }> = [
  { id: 1, label: "Characters" },
  { id: 2, label: "Generate" },
  { id: 3, label: "Pick rejecteds" },
  { id: 4, label: "Queue" },
];

interface Settings {
  root: string;
  /** "" = backend default. */
  modelsDir: string;
  swarmLoraDir: string;
  /** Folder whose newest generation supplies the Swarm parameters (default: the Swarm output folder). */
  paramFolder: string;
  gensPerCaption: number;
  /** Photos per character to generate negatives for in one run (0 = every caption). */
  photosPerCharacter: number;
  valPct: number;
  addOnWeight: number;
  /** Starting DPO beta for the entries (adaptive beta is switched on). */
  dpoBeta: number;
  maskModel: string;
  /** Resolution cap for the DPO concepts; adaptive resolution snaps each pair to the ladder below it. */
  dpoResolution: string;
  /** Concept files to scan for pairs that belong to a character (one path per line in the UI). */
  conceptFiles: string[];
}

interface SwarmConnection {
  baseUrl: string;
  token: string;
  model: string;
  /** SwarmUI's output folder as the generator knows it. */
  generated: string;
}

function loadSwarmConnection(): SwarmConnection {
  const fallback: SwarmConnection = { baseUrl: "http://localhost:7801", token: "", model: "", generated: "" };
  try {
    const raw = window.localStorage.getItem(SWARM_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<SwarmConnection>;
    return {
      baseUrl: parsed.baseUrl || fallback.baseUrl,
      token: parsed.token ?? "",
      model: parsed.model ?? "",
      generated: parsed.generated ?? "",
    };
  } catch {
    return fallback;
  }
}

const DEFAULTS: Settings = {
  root: "F:\\Datasets\\People",
  modelsDir: "",
  swarmLoraDir: "E:\\AI\\Data\\Models\\Lora\\SoRealPeople!",
  paramFolder: "",
  gensPerCaption: 2,
  photosPerCharacter: 10,
  valPct: 10,
  addOnWeight: 0.7,
  dpoBeta: 200,
  maskModel: "Rembg-Human",
  dpoResolution: "1024",
  conceptFiles: [],
};

function loadSettings(): Settings {
  const merged = { ...DEFAULTS };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) Object.assign(merged, JSON.parse(raw) as Partial<Settings>);
  } catch {
    /* storage unavailable */
  }
  if (!Array.isArray(merged.conceptFiles)) merged.conceptFiles = [];
  if (!merged.paramFolder) merged.paramFolder = loadSwarmConnection().generated;
  return merged;
}

function loadSelected(): Set<string> {
  try {
    const raw = window.localStorage.getItem(SELECTED_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    return new Set(Array.isArray(parsed) ? parsed.filter((n): n is string => typeof n === "string") : []);
  } catch {
    return new Set();
  }
}

function saveSelected(names: Set<string>) {
  try {
    window.localStorage.setItem(SELECTED_KEY, JSON.stringify([...names]));
  } catch {
    /* storage unavailable */
  }
}

/** Chip tooltip: the LoRA file and, when known, which of the LoRA folders it came from. */
function loraTitle(c: Character): string | undefined {
  if (!c.lora.path) return undefined;
  return c.lora.dir ? `${c.lora.path}\nfound in ${c.lora.dir}` : c.lora.path;
}

interface ParamSummary {
  ok: boolean;
  file?: string;
  model?: string;
  steps?: string;
  size?: string;
}

function summariseParams(params: Record<string, unknown>): ParamSummary {
  const str = (k: string) => (params[k] == null || params[k] === "" ? undefined : String(params[k]));
  const w = str("width");
  const h = str("height");
  return {
    ok: true,
    file: ["source", "source_image", "newest", "file", "path", "image"].map(str).find(Boolean),
    model: str("model"),
    steps: str("steps"),
    size: w && h ? `${w}×${h}` : undefined,
  };
}

export function CharacterDpoModal({ open, onClose }: CharacterDpoModalProps) {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [conceptFilesText, setConceptFilesText] = useState(() => loadSettings().conceptFiles.join("\n"));
  const [swarm, setSwarm] = useState<SwarmConnection>(loadSwarmConnection);
  const [paramSummary, setParamSummary] = useState<ParamSummary | null>(null);

  const [step, setStep] = useState<Step>(1);
  const [stepHint, setStepHint] = useState<string | null>(null);

  const [characters, setCharacters] = useState<Character[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [checked, setChecked] = useState<Set<string>>(loadSelected);

  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Manual fallback tools, each with the folders snapshotted when it was launched.
  const [swarmInitial, setSwarmInitial] = useState<SwarmGenerateInitial | null>(null);
  const [offPolicyInitial, setOffPolicyInitial] = useState<OffPolicyDpoInitial | null>(null);
  const [curateInitial, setCurateInitial] = useState<DPOToolInitial | null>(null);

  // Advanced: publish / import / finalize-alone for the active character
  const [publishBusy, setPublishBusy] = useState(false);
  const [publishResult, setPublishResult] = useState<{ ok: boolean; text: string; needsForce: boolean } | null>(null);
  const [importScanning, setImportScanning] = useState(false);
  const [importCandidates, setImportCandidates] = useState<ImportCandidate[] | null>(null);
  const [importPicks, setImportPicks] = useState<Set<string>>(() => new Set());
  const [importApplying, setImportApplying] = useState(false);
  const [finalizeStatus, setFinalizeStatus] = useState<FinalizeStatus | null>(null);
  const [finalizeBusy, setFinalizeBusy] = useState(false);
  const finalizePollRef = useRef<number | null>(null);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
  }, []);

  const active = useMemo(() => characters.find((c) => c.name === activeName) ?? null, [characters, activeName]);
  const childOpen = swarmInitial !== null || offPolicyInitial !== null || curateInitial !== null;
  const checkedChars = useMemo(() => characters.filter((c) => checked.has(c.name)), [characters, checked]);
  const pickable = useMemo(() => checkedChars.filter((c) => c.generated_count > 0), [checkedChars]);
  const pickableKey = pickable.map((c) => c.name).join("\u0000");

  const refresh = useCallback(async () => {
    if (!settings.root.trim()) {
      setListError("Set the characters root folder first.");
      return;
    }
    setListLoading(true);
    setListError(null);
    try {
      const res = await characterDpoApi.list({
        root: settings.root,
        modelsDir: settings.modelsDir || undefined,
        swarmUrl: swarm.baseUrl || undefined,
        swarmToken: swarm.token || undefined,
      });
      const list = res.characters ?? [];
      setCharacters(list);
      const names = new Set(list.map((c) => c.name));
      setChecked((prev) => {
        const next = new Set([...prev].filter((n) => names.has(n)));
        if (next.size !== prev.size) saveSelected(next);
        return next;
      });
      setActiveName((prev) => (prev && names.has(prev) ? prev : null));
    } catch (e) {
      setListError(errorText(e));
    } finally {
      setListLoading(false);
    }
  }, [settings.root, settings.modelsDir, swarm.baseUrl, swarm.token]);

  const refreshParams = useCallback(async () => {
    if (!settings.paramFolder.trim()) {
      setParamSummary({ ok: false });
      return;
    }
    try {
      const res = await characterDpoApi.swarmLastParams(settings.paramFolder);
      setParamSummary(
        res.ok && res.params && Object.keys(res.params).length > 0
          ? { ...summariseParams(res.params), ...(res.source_file ? { file: res.source_file } : {}) }
          : { ok: false },
      );
    } catch {
      setParamSummary({ ok: false });
    }
  }, [settings.paramFolder]);

  const clearFinalizePoll = useCallback(() => {
    if (finalizePollRef.current !== null) {
      window.clearInterval(finalizePollRef.current);
      finalizePollRef.current = null;
    }
  }, []);

  const pollFinalize = useCallback(
    (name: string, root: string) => {
      clearFinalizePoll();
      finalizePollRef.current = window.setInterval(() => {
        void characterDpoApi
          .finalizeStatus(name, root)
          .then((s) => {
            setFinalizeStatus(s);
            if (!s.running) {
              clearFinalizePoll();
              void refresh();
            }
          })
          .catch(() => undefined);
      }, 1000);
    },
    [clearFinalizePoll, refresh],
  );

  // Load on open; re-read the Swarm connection each time in case it changed.
  useEffect(() => {
    if (!open) {
      clearFinalizePoll();
      return;
    }
    setSwarm(loadSwarmConnection());
    setError(null);
    setNotice(null);
    setStepHint(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (open) void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, swarm.baseUrl, swarm.token]);

  useEffect(() => {
    if (open) void refreshParams();
  }, [open, refreshParams]);

  useEffect(() => () => clearFinalizePoll(), [clearFinalizePoll]);

  // Per-character views reset when the active row changes; resume a running finalize.
  useEffect(() => {
    setPublishResult(null);
    setImportCandidates(null);
    setImportPicks(new Set());
    setFinalizeStatus(null);
    clearFinalizePoll();
    if (!active) return;
    let cancelled = false;
    void characterDpoApi
      .finalizeStatus(active.name, settings.root)
      .then((s) => {
        if (cancelled || !s.running) return;
        setFinalizeStatus(s);
        pollFinalize(active.name, settings.root);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeName]);

  const setCheckedPersist = useCallback((next: Set<string>) => {
    saveSelected(next);
    setChecked(next);
  }, []);

  const toggleChecked = useCallback(
    (name: string) => {
      const next = new Set(checked);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      setCheckedPersist(next);
    },
    [checked, setCheckedPersist],
  );

  const allChecked = characters.length > 0 && characters.every((c) => checked.has(c.name));
  const toggleAll = useCallback(() => {
    setCheckedPersist(allChecked ? new Set() : new Set(characters.map((c) => c.name)));
  }, [allChecked, characters, setCheckedPersist]);

  const goStep = useCallback(
    (s: Step) => {
      if (s > 1 && checked.size === 0) {
        setStepHint("Check at least one character in step 1 first.");
        setStep(1);
        return;
      }
      setStepHint(null);
      setStep(s);
    },
    [checked.size],
  );

  // ---- Advanced actions for the active character ----------------------

  const publish = useCallback(
    async (force = false) => {
      if (!active || !active.lora.path) return;
      setPublishBusy(true);
      setPublishResult(null);
      try {
        const res = await characterDpoApi.publishLora({
          name: active.name,
          root: settings.root,
          models_dir: settings.modelsDir || undefined,
          swarm_lora_dir: settings.swarmLoraDir,
          force,
        });
        if (res.ok) {
          setPublishResult({
            ok: true,
            text: res.unchanged
              ? `Already in SwarmUI (${res.path ?? settings.swarmLoraDir})`
              : `Published to ${res.path ?? settings.swarmLoraDir}`,
            needsForce: false,
          });
          void refresh();
        } else {
          const msg = res.error ?? "Publish failed";
          setPublishResult({ ok: false, text: msg, needsForce: !force && /differ|exist|already|overwrite/i.test(msg) });
        }
      } catch (e) {
        setPublishResult({ ok: false, text: errorText(e), needsForce: false });
      } finally {
        setPublishBusy(false);
      }
    },
    [active, settings.root, settings.modelsDir, settings.swarmLoraDir, refresh],
  );

  const openGenerate = useCallback(() => {
    if (!active) return;
    setSwarmInitial({
      realFolder: joinPath(active.path, "train"),
      generatedFolder: joinPath(active.path, "dpo", "generated"),
      outputDir: joinPath(active.path, "dpo", "offpolicy"),
      loras: active.lora.swarm_name ? [{ name: active.lora.swarm_name, weight: 1 }] : [],
      imagesPerCaption: settings.gensPerCaption,
      captions: active.train_count,
    });
  }, [active, settings.gensPerCaption]);

  const openPair = useCallback(() => {
    if (!active) return;
    setOffPolicyInitial({
      generated: joinPath(active.path, "dpo", "generated"),
      real: joinPath(active.path, "train"),
      output: joinPath(active.path, "dpo", "offpolicy"),
    });
  }, [active]);

  const openCurate = useCallback(() => {
    if (!active) return;
    setCurateInitial({
      sourceFolder: joinPath(active.path, "dpo", "generated"),
      outputDir: joinPath(active.path, "dpo", "aivai"),
      captionFilter: active.name,
      loraFilter: loraStem(active),
    });
  }, [active]);

  const closeChild = useCallback(() => {
    setSwarmInitial(null);
    setOffPolicyInitial(null);
    setCurateInitial(null);
    setSwarm(loadSwarmConnection());
    void refresh();
  }, [refresh]);

  const importScan = useCallback(async () => {
    if (!active) return;
    setImportScanning(true);
    setImportCandidates(null);
    setImportPicks(new Set());
    setError(null);
    try {
      const res = await characterDpoApi.importScan({
        name: active.name,
        root: settings.root,
        concept_files: settings.conceptFiles,
        models_dir: settings.modelsDir || undefined,
      });
      const cands = res.candidates ?? [];
      setImportCandidates(cands);
      setImportPicks(new Set(cands.filter((c) => !c.already_imported).map((c) => c.key)));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setImportScanning(false);
    }
  }, [active, settings.root, settings.conceptFiles, settings.modelsDir]);

  const importApply = useCallback(async () => {
    if (!active || !importCandidates) return;
    const picks = importCandidates
      .filter((c) => importPicks.has(c.key))
      .map((c) => ({ concept_dir: c.concept_dir, pair_id: c.pair_id }));
    if (picks.length === 0) return;
    setImportApplying(true);
    setError(null);
    try {
      const res = await characterDpoApi.importApply({ name: active.name, root: settings.root, picks });
      const parts = [`Imported ${res.imported}`, `skipped ${res.skipped}`];
      if (res.errors?.length) parts.push(`${res.errors.length} error(s)`);
      setNotice(`${parts.join(" · ")} for ${active.name}.`);
      if (res.errors?.length) setError(res.errors.join("\n"));
      setImportCandidates(null);
      setImportPicks(new Set());
      await refresh();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setImportApplying(false);
    }
  }, [active, importCandidates, importPicks, settings.root, refresh]);

  const importSelectable = useMemo(
    () => (importCandidates ?? []).filter((c) => !c.already_imported),
    [importCandidates],
  );
  const importAllPicked = importSelectable.length > 0 && importSelectable.every((c) => importPicks.has(c.key));

  const finalize = useCallback(async () => {
    if (!active) return;
    setFinalizeBusy(true);
    setError(null);
    try {
      const res = await characterDpoApi.finalize({
        name: active.name,
        root: settings.root,
        val_pct: settings.valPct,
        mask_model: settings.maskModel,
        dpo_resolution: settings.dpoResolution,
      });
      if (!res.ok) {
        setError(res.error ?? "Could not start finalize");
        return;
      }
      setFinalizeStatus({
        running: true,
        stage: "starting",
        done: 0,
        total: 0,
        message: "",
        error: null,
        result: null,
      });
      pollFinalize(active.name, settings.root);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setFinalizeBusy(false);
    }
  }, [active, settings.root, settings.valPct, settings.maskModel, settings.dpoResolution, pollFinalize]);

  const paramLine = !paramSummary
    ? "reading…"
    : !paramSummary.ok
      ? "no generation found in the params folder"
      : `Params from: ${paramSummary.file ?? "newest image"} (model ${paramSummary.model ?? "?"}${
          paramSummary.steps ? `, ${paramSummary.steps} steps` : ""
        }${paramSummary.size ? `, ${paramSummary.size}` : ""}) — adapters stripped`;

  return (
    <ModalBase
      open={open}
      onClose={onClose}
      title="Character DPO"
      size="full"
      closeOnEscape={!childOpen}
      closeOnBackdrop={!childOpen}
    >
      <div className="flex flex-col gap-3 h-full min-h-0">
        {/* ---------------- Stepper ---------------- */}
        <div className="flex items-center gap-1 flex-wrap">
          {STEPS.map((s, i) => {
            const enabled = s.id === 1 || checked.size > 0;
            const current = s.id === step;
            return (
              <div key={s.id} className="flex items-center gap-1">
                {i > 0 && <span className="text-[var(--color-on-surface-secondary)] px-1">›</span>}
                <button
                  type="button"
                  onClick={() => goStep(s.id)}
                  disabled={!enabled}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors ${
                    current
                      ? "bg-[var(--color-cobalt-600)] text-white font-semibold"
                      : enabled
                        ? "text-[var(--color-on-surface)] hover:bg-[var(--color-surface-container)]"
                        : "text-[var(--color-on-surface-secondary)] opacity-60 cursor-not-allowed"
                  }`}
                  title={enabled ? undefined : "Check at least one character first"}
                >
                  <span
                    className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[11px] font-bold tabular-nums ${
                      current ? "bg-white/20" : "bg-[var(--color-surface-container)]"
                    }`}
                  >
                    {s.id}
                  </span>
                  {s.label}
                  {s.id === 1 && checked.size > 0 && <span className="text-xs opacity-80">({checked.size})</span>}
                  {s.id === 3 && checked.size > 0 && (
                    <span className="text-xs opacity-80">
                      ({pickable.length}/{checkedChars.length} with gens)
                    </span>
                  )}
                </button>
              </div>
            );
          })}
          <span className={`ml-auto ${HINT}`}>
            {settings.root} · SwarmUI {swarm.baseUrl}
            {swarm.model ? ` · ${swarm.model}` : ""}
          </span>
        </div>
        {stepHint && <div className="text-xs text-[var(--color-warning-500,#eab308)]">{stepHint}</div>}

        {/* ---------------- Settings drawer ---------------- */}
        <details className={SECTION}>
          <summary className="cursor-pointer select-none text-sm font-semibold text-[var(--color-on-surface)]">
            Settings
            <span className={`ml-2 font-normal ${HINT}`}>{paramLine}</span>
          </summary>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 pt-2">
            <DirPicker
              label="Characters root"
              value={settings.root}
              onChange={(v) => update({ root: v })}
              tooltip="One sub-folder per character: <name>/train, <name>/val, <name>/dpo/…"
            />
            <FormEntry
              label="LoRA folders"
              value={settings.modelsDir}
              onChange={(v) => update({ modelsDir: String(v) })}
              placeholder="models;E:\AI\Data\Models\Lora\SoRealPeople!"
              tooltip="Semicolon-separated, searched in order; leave empty for models/ + SoRealPeople!"
            />
            <DirPicker
              label="SwarmUI LoRA folder"
              value={settings.swarmLoraDir}
              onChange={(v) => update({ swarmLoraDir: v })}
              tooltip="Publish copies the character's best LoRA here so SwarmUI can load it."
            />
            <DirPicker
              label="Generation params folder"
              value={settings.paramFolder}
              onChange={(v) => update({ paramFolder: v })}
              tooltip="The newest image here supplies model, steps, CFG, sampler… for every character's run (its LoRAs are dropped; each character brings its own). Defaults to your SwarmUI output folder."
            />
            <FormEntry
              label="Photos per character"
              type="number"
              value={settings.photosPerCharacter}
              onChange={(v) => update({ photosPerCharacter: Math.max(0, Math.round(Number(v)) || 0) })}
              tooltip="How many train photos (captions) get negatives per Generate run, picked at random among those without a pair yet. 0 = all of them. Run Generate again later to cover more."
            />
            <FormEntry
              label="Generations per caption"
              type="number"
              value={settings.gensPerCaption}
              onChange={(v) => update({ gensPerCaption: Math.max(1, Math.round(Number(v)) || 1) })}
              tooltip="Renders per photo; the hardest (most photo-like) one is pre-selected as the rejected image, the others stay available to pick."
            />
            <FormEntry
              label="Validation split (%)"
              type="number"
              value={settings.valPct}
              onChange={(v) => update({ valPct: Number(v) })}
              tooltip="Share of unsplit pairs moved to val when finalizing."
            />
            <FormEntry
              label="DPO resolution cap"
              value={settings.dpoResolution}
              onChange={(v) => update({ dpoResolution: String(v).trim() || "1024" })}
              tooltip="Resolution override written on the four DPO concepts. With adaptive resolution each pair trains at the largest ladder tier at or below its own size, capped here — so 1024 renders stay 1024 while the SFT concepts keep the global 512/768."
            />
            <FormEntry
              label="DPO add-on weight (training)"
              type="number"
              value={settings.addOnWeight}
              onChange={(v) => update({ addOnWeight: Number(v) })}
              tooltip="rlhf_dpo_addon_weight on the queue entries: multiplier on the DPO loss when a pair is interleaved into the SFT run. Nothing to do with SwarmUI — the LoRA is always applied at strength 1.0 when generating negatives."
            />
            <FormEntry
              label="DPO beta (start)"
              type="number"
              value={settings.dpoBeta}
              onChange={(v) => update({ dpoBeta: Math.max(1, Number(v) || 200) })}
              tooltip="rlhf_dpo_beta on the entries, with adaptive beta on: starts here and is scaled per step from the smoothed reward margin within [beta/4, beta*4]. A warm start from the character's own LoRA begins at zero margin, so keep this well below a cold-start beta."
            />
            <Select
              label="Mask model"
              value={settings.maskModel}
              onChange={(v) => update({ maskModel: v })}
              options={MASK_MODELS}
              formatLabel={(v) => v}
            />
            <div className="flex flex-col gap-1 lg:col-span-1">
              <span className="text-sm font-medium text-[var(--color-on-surface)]">
                Concept files to scan for Import (one path per line)
              </span>
              <textarea
                value={conceptFilesText}
                onChange={(e) => {
                  setConceptFilesText(e.target.value);
                  update({
                    conceptFiles: e.target.value
                      .split(/\r?\n/)
                      .map((l) => l.trim())
                      .filter(Boolean),
                  });
                }}
                rows={2}
                className={`${TEXTAREA_FULL} resize-y font-mono text-xs`}
                placeholder="E:\path\to\concepts.json"
              />
            </div>
            <div className={`lg:col-span-3 ${HINT}`}>
              SwarmUI: <span className="text-[var(--color-on-surface)]">{swarm.baseUrl}</span>
              {" · "}
              {swarm.token ? "token set" : "no token"} — read from Generate Off-policy Negatives (change it there).
              {" · "}
              <button type="button" className="underline" onClick={() => void refreshParams()}>
                re-read params
              </button>
            </div>
          </div>
        </details>

        {error && <Notice kind="error" text={error} onClose={() => setError(null)} />}
        {notice && <Notice kind="info" text={notice} onClose={() => setNotice(null)} />}

        {/* ---------------- Step 1: Characters ---------------- */}
        {step === 1 && (
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(360px,2fr)_3fr] gap-3 flex-1 min-h-0">
            <div className={`${SECTION} min-h-0`}>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={allChecked}
                  onChange={toggleAll}
                  disabled={characters.length === 0}
                  title="Select all"
                  className="cursor-pointer"
                />
                <span className={SECTION_TITLE}>Characters{characters.length > 0 && ` (${characters.length})`}</span>
                <span className={HINT}>{checked.size > 0 && `${checked.size} checked`}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="ml-auto"
                  onClick={() => void refresh()}
                  disabled={listLoading}
                  title="Re-read the characters root"
                >
                  {listLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                  Refresh
                </Button>
                <Button variant="primary" size="sm" onClick={() => goStep(2)} disabled={checked.size === 0}>
                  Next: Generate ›
                </Button>
              </div>
              {listError && <div className="text-xs text-[var(--color-error-500)]">{listError}</div>}
              {!listError && !listLoading && characters.length === 0 && (
                <div className={HINT}>No characters found under {settings.root || "(no root)"}.</div>
              )}
              <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-1">
                {characters.map((c) => {
                  const isActive = c.name === activeName;
                  const missingMasks = c.offpolicy.missing_masks + c.aivai.missing_masks;
                  return (
                    <div
                      key={c.name}
                      role="button"
                      tabIndex={0}
                      onClick={() => setActiveName(c.name)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setActiveName(c.name);
                        }
                      }}
                      className={`flex items-start gap-2 rounded px-2 py-1.5 cursor-pointer border-l-2 transition-colors ${
                        isActive
                          ? "bg-[var(--color-surface-container)] border-[var(--color-cobalt-600)]"
                          : "border-transparent hover:bg-[var(--color-surface-container)]"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked.has(c.name)}
                        onChange={() => toggleChecked(c.name)}
                        onClick={(e) => e.stopPropagation()}
                        className="mt-1 cursor-pointer"
                        title="Include in the run"
                      />
                      <div className="flex-1 min-w-0 flex flex-col gap-1">
                        <div className="flex items-center gap-2 min-w-0">
                          <span
                            className="text-sm font-semibold text-[var(--color-on-surface)] truncate"
                            title={c.path}
                          >
                            {c.name}
                          </span>
                          <span className={`${HINT} tabular-nums`}>
                            {c.train_count} train · {c.val_count} val
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {c.lora.kind === "best" && (
                            <Chip tone="ok" title={loraTitle(c)}>
                              LoRA ✓ best
                            </Chip>
                          )}
                          {c.lora.kind === "second" && (
                            <Chip tone="warn" title={loraTitle(c)}>
                              LoRA ✓ second
                            </Chip>
                          )}
                          {!c.lora.kind && <Chip tone="bad">LoRA ✗</Chip>}
                          <Chip tone={c.generated_count > 0 ? "neutral" : "warn"}>generated {c.generated_count}</Chip>
                          <Chip
                            tone={c.offpolicy.train + c.offpolicy.val + c.offpolicy.unsplit > 0 ? "neutral" : "bad"}
                          >
                            off-policy {splitLabel(c.offpolicy)}
                          </Chip>
                          <Chip tone={c.aivai.train + c.aivai.val + c.aivai.unsplit > 0 ? "neutral" : "bad"}>
                            AI-vs-AI {splitLabel(c.aivai)}
                          </Chip>
                          {missingMasks === 0 ? (
                            <Chip tone="ok">masks ✓</Chip>
                          ) : (
                            <Chip tone="bad" title="Pairs without a mask">
                              masks ✗ {missingMasks}
                            </Chip>
                          )}
                          {c.finalized && <Chip tone="ok">finalized</Chip>}
                          {c.queue_entry && (
                            <Chip tone="info" title={c.queue_entry}>
                              in queue
                            </Chip>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Advanced panel for the active character */}
            <details className={`${SECTION} min-h-0 overflow-y-auto`} open={active !== null}>
              <summary className="cursor-pointer select-none text-sm font-semibold text-[var(--color-on-surface)]">
                Advanced{active ? ` — ${active.name}` : ""}
                <span className={`ml-2 font-normal ${HINT}`}>
                  {active ? active.path : "click a character row to work on it alone"}
                </span>
              </summary>
              {active && (
                <div className="flex flex-col gap-2 pt-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => void publish(false)}
                      disabled={publishBusy || !active.lora.path}
                      title={active.lora.path ?? "No trained LoRA found for this character"}
                    >
                      {publishBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                      Publish LoRA
                    </Button>
                    <span className={`${HINT} truncate`}>
                      {active.lora.path
                        ? `${active.lora.kind ?? ""} → ${active.lora.swarm_name ?? loraStem(active)}`
                        : "no LoRA"}
                    </span>
                    <Button variant="secondary" size="sm" onClick={openCurate} title="Opens the DPO Pair Tool">
                      <Eye className="w-4 h-4" /> Curate AI-vs-AI
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={openGenerate}
                      title="Manual fallback: opens Generate Off-policy Negatives"
                    >
                      <Wand2 className="w-4 h-4" /> Generate (manual)
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={openPair}
                      title="Manual fallback: opens the Off-policy DPO Builder"
                    >
                      <Heart className="w-4 h-4" /> Pair (manual)
                    </Button>
                  </div>
                  {publishResult && (
                    <div
                      className={`flex flex-wrap items-center gap-2 text-xs ${
                        publishResult.ok ? "text-[var(--color-success-500)]" : "text-[var(--color-error-500)]"
                      }`}
                    >
                      <span className="break-all">{publishResult.text}</span>
                      {publishResult.needsForce && (
                        <Button variant="danger" size="sm" onClick={() => void publish(true)} disabled={publishBusy}>
                          Overwrite the existing file
                        </Button>
                      )}
                    </div>
                  )}

                  {/* Import */}
                  <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-[var(--color-border-subtle)]">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => void importScan()}
                      disabled={importScanning || importApplying}
                      title={
                        settings.conceptFiles.length
                          ? `Scan ${settings.conceptFiles.length} concept file(s) for this character's pairs`
                          : "Add concept files in Settings first"
                      }
                    >
                      {importScanning ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <FolderInput className="w-4 h-4" />
                      )}
                      Import existing pairs
                    </Button>
                    {importCandidates && (
                      <>
                        <span className={`${HINT} tabular-nums`}>
                          {importCandidates.length} candidate(s) · {importPicks.size} picked ·{" "}
                          {importCandidates.length - importSelectable.length} already imported
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() =>
                            setImportPicks(importAllPicked ? new Set() : new Set(importSelectable.map((c) => c.key)))
                          }
                          disabled={importSelectable.length === 0}
                        >
                          {importAllPicked ? "Select none" : "Select all"}
                        </Button>
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => void importApply()}
                          disabled={importApplying || importPicks.size === 0}
                        >
                          {importApplying ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Check className="w-4 h-4" />
                          )}
                          Apply ({importPicks.size})
                        </Button>
                      </>
                    )}
                  </div>
                  {importCandidates && importCandidates.length === 0 && (
                    <div className={HINT}>No pairs matching {active.name} in the configured concept files.</div>
                  )}
                  {importCandidates && importCandidates.length > 0 && (
                    <div className="grid grid-cols-1 2xl:grid-cols-2 gap-2 max-h-[40vh] overflow-y-auto pr-1">
                      {importCandidates.map((c) => (
                        <label
                          key={c.key}
                          className={`flex items-start gap-2 rounded p-2 bg-[var(--color-surface-container)] ${
                            c.already_imported ? "opacity-50" : "cursor-pointer"
                          }`}
                          title={`${c.concept_name} · ${c.pair_id}`}
                        >
                          <input
                            type="checkbox"
                            checked={importPicks.has(c.key)}
                            disabled={c.already_imported}
                            onChange={() =>
                              setImportPicks((prev) => {
                                const next = new Set(prev);
                                if (next.has(c.key)) next.delete(c.key);
                                else next.add(c.key);
                                return next;
                              })
                            }
                            className="mt-1"
                          />
                          <div className="flex gap-1 shrink-0">
                            <img
                              src={characterDpoApi.imageUrl(c.chosen_path)}
                              alt="chosen"
                              loading="lazy"
                              className="w-20 h-20 object-cover rounded border border-[var(--color-success-500)]"
                            />
                            <img
                              src={characterDpoApi.imageUrl(c.rejected_path)}
                              alt="rejected"
                              loading="lazy"
                              className="w-20 h-20 object-cover rounded border border-[var(--color-error-500)]"
                            />
                          </div>
                          <div className="flex-1 min-w-0 flex flex-col gap-1">
                            <div className="flex flex-wrap gap-1">
                              <Chip tone={c.kind === "offpolicy" ? "info" : "neutral"}>
                                {c.kind === "offpolicy" ? "off-policy" : "AI-vs-AI"}
                              </Chip>
                              <Chip tone={c.match === "both" ? "ok" : "neutral"}>match: {c.match}</Chip>
                              <Chip title={c.concept_dir}>{c.concept_name}</Chip>
                              {c.already_imported && <Chip tone="warn">already imported</Chip>}
                            </div>
                            <div
                              className="text-xs text-[var(--color-on-surface)] truncate"
                              title={c.prompt || c.caption}
                            >
                              {c.prompt || c.caption}
                            </div>
                          </div>
                        </label>
                      ))}
                    </div>
                  )}

                  {/* Finalize alone */}
                  <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-[var(--color-border-subtle)]">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => void finalize()}
                      disabled={finalizeBusy || finalizeStatus?.running === true}
                      title={`Split unsplit pairs ${100 - settings.valPct}/${settings.valPct}, generate missing masks with ${settings.maskModel}, write the concepts file`}
                    >
                      {finalizeBusy || finalizeStatus?.running ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <PackageCheck className="w-4 h-4" />
                      )}
                      Finalize alone
                    </Button>
                    <span className={HINT}>
                      val {settings.valPct}% · masks {settings.maskModel}
                      {active.concepts_file && (
                        <>
                          {" · "}
                          <span title={active.concepts_file}>concepts file present</span>
                        </>
                      )}
                    </span>
                  </div>
                  {finalizeStatus && (
                    <div className="flex flex-col gap-1 text-xs">
                      {finalizeStatus.running && (
                        <>
                          <div className="flex items-center justify-between text-[var(--color-on-surface-secondary)]">
                            <span>
                              {finalizeStatus.stage}
                              {finalizeStatus.message ? ` — ${finalizeStatus.message}` : ""}
                            </span>
                            <span className="tabular-nums">
                              {finalizeStatus.total > 0 ? `${finalizeStatus.done} / ${finalizeStatus.total}` : "…"}
                            </span>
                          </div>
                          <Bar done={finalizeStatus.done} total={finalizeStatus.total} />
                        </>
                      )}
                      {!finalizeStatus.running && finalizeStatus.error && (
                        <div className="text-[var(--color-error-500)] whitespace-pre-wrap">{finalizeStatus.error}</div>
                      )}
                      {!finalizeStatus.running && !finalizeStatus.error && (
                        <div className="text-[var(--color-success-500)]">
                          Finalized{finalizeStatus.message ? ` — ${finalizeStatus.message}` : "."}
                        </div>
                      )}
                      {!finalizeStatus.running && finalizeStatus.result && (
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(finalizeStatus.result).map(([k, v]) => (
                            <Chip key={k}>
                              {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
                            </Chip>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </details>
          </div>
        )}

        {/* ---------------- Step 2: Generate ---------------- */}
        {step === 2 && (
          <>
            <GenerateProgress
              characters={checkedChars}
              root={settings.root}
              modelsDir={settings.modelsDir}
              swarmUrl={swarm.baseUrl}
              swarmToken={swarm.token}
              swarmLoraDir={settings.swarmLoraDir}
              paramFolder={settings.paramFolder}
              gensPerCaption={settings.gensPerCaption}
              photosPerCharacter={settings.photosPerCharacter}
              onFinished={() => void refresh()}
            />
            <div className="flex justify-end">
              <Button variant="secondary" size="sm" onClick={() => goStep(3)}>
                Next: Pick rejecteds ›
              </Button>
            </div>
          </>
        )}

        {/* ---------------- Step 3: Pick rejecteds ---------------- */}
        {step === 3 && (
          <>
            <RejectedPicker
              key={pickableKey}
              characters={pickable}
              defaults={loadOffPolicyDefaults()}
              onCharacterDone={() => void refresh()}
            />
            <div className="flex justify-end">
              <Button variant="secondary" size="sm" onClick={() => goStep(4)}>
                Next: Queue ›
              </Button>
            </div>
          </>
        )}

        {/* ---------------- Step 4: Queue ---------------- */}
        {step === 4 && (
          <QueueStep
            characters={checkedChars}
            root={settings.root}
            modelsDir={settings.modelsDir}
            valPct={settings.valPct}
            maskModel={settings.maskModel}
            dpoResolution={settings.dpoResolution}
            addOnWeight={settings.addOnWeight}
            dpoBeta={settings.dpoBeta}
            onNotice={setNotice}
            onError={setError}
            onChanged={() => void refresh()}
          />
        )}
      </div>

      {swarmInitial && <SwarmGenerateModal open onClose={closeChild} initial={swarmInitial} />}
      {offPolicyInitial && <OffPolicyDpoModal open onClose={closeChild} initial={offPolicyInitial} />}
      {curateInitial && <DPOToolModal open onClose={closeChild} initial={curateInitial} />}
    </ModalBase>
  );
}
