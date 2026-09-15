import { AlertCircle, Archive, Loader2, Play, RotateCcw, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  dpoSaturationApi,
  type RetiredPair,
  type SaturationConcept,
  type SaturationConceptsResponse,
  type SaturationRow,
  type SaturationRule,
  type SaturationStatus,
  type SaturationSummary,
} from "@/api/dpoSaturationApi";
import { PreviewOverlay } from "@/components/modals/dpo/PreviewOverlay";
import { ModalBase } from "@/components/modals/ModalBase";
import {
  Button,
  DirPicker,
  FilePicker,
  FormEntry,
  SectionCard,
  Select,
  SliderEntry,
  Toggle,
} from "@/components/shared";
import { ScalarChart } from "@/components/shared/ScalarChart";

export interface DPOSaturationModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.dpoSaturation";
const POLL_MS = 700;
const SUMMARY_DEBOUNCE_MS = 250;

type ScoreMode = "checkpoints" | "adapter";

interface Settings {
  mode: ScoreMode;
  current: string;
  origin: string;
  adapter: string;
  draws: number;
  timesteps: string;
  seeds: number;
  cacheDir: string;
  forceOrigin: boolean;
  concept: string;
  /** Per-concept beta the user dialled in; absent = the concept's own beta. */
  betaByConcept: Record<string, number>;
  deadWeight: number;
  requireAllBuckets: boolean;
  minDraws: number;
}

const DEFAULTS: Settings = {
  mode: "checkpoints",
  current: "",
  origin: "",
  adapter: "",
  draws: 8,
  timesteps: "",
  seeds: 2,
  cacheDir: "",
  forceOrigin: false,
  concept: "",
  betaByConcept: {},
  deadWeight: 0.05,
  requireAllBuckets: false,
  minDraws: 1,
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULTS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    // ignore a corrupt entry
  }
  return { ...DEFAULTS };
}

const fmt = (x: number | null | undefined, digits = 2) => (x == null || Number.isNaN(x) ? "-" : x.toFixed(digits));
const pct = (done: number, total: number) => (total > 0 ? Math.min(100, (100 * done) / total) : 0);
const fileName = (path: string | null | undefined) => (path ? (path.split(/[/\\]/).pop() ?? path) : "-");
const minutes = (s: number) =>
  `${Math.floor(s / 60)}m ${Math.round(s % 60)
    .toString()
    .padStart(2, "0")}s`;

/** Logit at which the gradient weight drops to `deadWeight`: sigmoid(-x) = w -> x = ln((1-w)/w). */
const deadLogit = (deadWeight: number) => Math.log((1 - deadWeight) / Math.max(1e-9, deadWeight));

function conceptLabel(c: SaturationConcept): string {
  const base =
    c.path
      .replace(/[/\\]+$/, "")
      .split(/[/\\]/)
      .pop() ?? c.path;
  return c.name ? `${c.name} (${base})` : base;
}

function defaultBeta(c: SaturationConcept | undefined, globalBeta: number): number {
  if (!c) return globalBeta;
  return c.beta_override > 0 ? c.beta_override : globalBeta;
}

// --------------------------------------------------------------------- //
// Histogram of beta * margin with the dead-weight threshold marked
// --------------------------------------------------------------------- //

function LogitHistogram({ edges, counts, threshold }: { edges: number[]; counts: number[]; threshold: number }) {
  const width = 560;
  const height = 180;
  const pad = { top: 12, right: 12, bottom: 28, left: 40 };
  if (edges.length < 2 || counts.length === 0) {
    return <div className="text-sm text-[var(--color-on-surface-secondary)]">No scored pairs yet.</div>;
  }
  const lo = Math.min(edges[0], 0, threshold);
  const hi = Math.max(edges[edges.length - 1], threshold);
  const maxCount = Math.max(1, ...counts);
  const x = (v: number) => pad.left + ((v - lo) / (hi - lo || 1)) * (width - pad.left - pad.right);
  const y = (c: number) => pad.top + (1 - c / maxCount) * (height - pad.top - pad.bottom);
  const baseline = height - pad.bottom;
  const ticks = 5;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-[640px]" role="img">
      <title>Distribution of absorbed preference (beta times margin)</title>
      {counts.map((c, i) => {
        const x0 = x(edges[i]);
        const x1 = x(edges[i + 1]);
        const dead = edges[i] >= threshold;
        return (
          <rect
            key={i}
            x={x0}
            y={y(c)}
            width={Math.max(1, x1 - x0 - 1)}
            height={baseline - y(c)}
            fill={dead ? "var(--color-warning-500)" : "var(--color-cobalt-600)"}
            opacity={0.85}
            rx={1}
          />
        );
      })}
      <line x1={pad.left} x2={width - pad.right} y1={baseline} y2={baseline} stroke="var(--color-border-subtle)" />
      <line
        x1={x(0)}
        x2={x(0)}
        y1={pad.top}
        y2={baseline}
        stroke="var(--color-on-surface-secondary)"
        strokeDasharray="3 3"
      />
      <line x1={x(threshold)} x2={x(threshold)} y1={pad.top} y2={baseline} stroke="var(--color-error-500)" />
      <text x={x(threshold) + 4} y={pad.top + 10} fontSize={10} fill="var(--color-error-500)">
        dead ≥ {threshold.toFixed(1)}
      </text>
      {Array.from({ length: ticks + 1 }, (_, i) => lo + ((hi - lo) * i) / ticks).map((v) => (
        <text
          key={v}
          x={x(v)}
          y={height - 8}
          fontSize={10}
          textAnchor="middle"
          fill="var(--color-on-surface-secondary)"
        >
          {v.toFixed(1)}
        </text>
      ))}
      <text x={pad.left - 6} y={pad.top + 8} fontSize={10} textAnchor="end" fill="var(--color-on-surface-secondary)">
        {maxCount}
      </text>
      <text x={width / 2} y={height - 18} fontSize={10} textAnchor="middle" fill="var(--color-on-surface-secondary)">
        beta × margin (nats absorbed; left of 0 = current prefers the rejected image)
      </text>
    </svg>
  );
}

// --------------------------------------------------------------------- //
// Pair tile: chosen and rejected side by side with the pair's numbers
// --------------------------------------------------------------------- //

function PairTile({
  row,
  onPreview,
  action,
}: {
  row: SaturationRow;
  onPreview: (path: string) => void;
  action?: { label: string; onClick: () => void; disabled?: boolean; danger?: boolean };
}) {
  const buckets = row.bucket_weight.map((w) => (w == null ? "·" : fmt(w, 2))).join(" ");
  return (
    <figure className="flex flex-col gap-1 rounded bg-[var(--color-surface-container)] p-2">
      <div className="grid grid-cols-2 gap-1">
        {[
          { path: row.chosen, tag: "chosen", ring: "ring-[var(--color-success-500)]" },
          { path: row.rejected, tag: "rejected", ring: "ring-[var(--color-error-500)]" },
        ].map(({ path, tag, ring }) => (
          <button
            key={tag}
            type="button"
            className={`relative aspect-square overflow-hidden rounded ring-2 ${ring} bg-black/20`}
            onClick={() => path && onPreview(path)}
            title={path ?? ""}
          >
            {path ? (
              <img
                src={dpoSaturationApi.imageUrl(path)}
                alt={tag}
                className="h-full w-full object-cover"
                loading="lazy"
              />
            ) : (
              <span className="text-xs">missing</span>
            )}
            <span className="absolute left-1 top-1 rounded bg-black/60 px-1 text-[10px] text-white">{tag}</span>
          </button>
        ))}
      </div>
      <figcaption className="flex flex-col gap-0.5 text-xs tabular-nums">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate" title={row.key}>
            {row.key}
          </span>
          <span
            className={row.saturated ? "text-[var(--color-warning-500)]" : "text-[var(--color-on-surface-secondary)]"}
          >
            {row.saturated ? "dead" : "live"}
          </span>
        </div>
        <div className="flex justify-between text-[var(--color-on-surface-secondary)]">
          <span>β·m {fmt(row.mean_logit, 2)}</span>
          <span>w {fmt(row.mean_weight, 3)}</span>
          <span>wins {Math.round(100 * row.wins)}%</span>
        </div>
        <div
          className="text-[var(--color-on-surface-secondary)]"
          title="mean weight per timestep quartile, low noise first"
        >
          q: {buckets}
        </div>
        {action && (
          <Button
            variant={action.danger ? "danger" : "secondary"}
            size="sm"
            onClick={action.onClick}
            disabled={action.disabled}
          >
            {action.label}
          </Button>
        )}
      </figcaption>
    </figure>
  );
}

// --------------------------------------------------------------------- //
// Modal
// --------------------------------------------------------------------- //

export function DPOSaturationModal({ open, onClose }: DPOSaturationModalProps) {
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [conceptsInfo, setConceptsInfo] = useState<SaturationConceptsResponse | null>(null);
  const [status, setStatus] = useState<SaturationStatus | null>(null);
  const [summary, setSummary] = useState<SaturationSummary | null>(null);
  const [retired, setRetired] = useState<RetiredPair[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmRetire, setConfirmRetire] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const pollRef = useRef<number | null>(null);
  const summaryTimer = useRef<number | null>(null);
  const summarySeq = useRef(0);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // storage full or unavailable
      }
      return next;
    });
  }, []);

  const concepts = useMemo(() => conceptsInfo?.concepts ?? [], [conceptsInfo]);
  const globalBeta = conceptsInfo?.global_beta ?? 0;
  const concept = useMemo(
    () => concepts.find((c) => c.path === settings.concept) ?? concepts.find((c) => c.table) ?? concepts[0],
    [concepts, settings.concept],
  );
  const conceptBeta = defaultBeta(concept, globalBeta);
  const beta = concept ? (settings.betaByConcept[concept.path] ?? conceptBeta) : conceptBeta;
  const rule: SaturationRule = useMemo(
    () => ({
      beta,
      deadWeight: settings.deadWeight,
      requireAllBuckets: settings.requireAllBuckets,
      minDraws: settings.minDraws,
    }),
    [beta, settings.deadWeight, settings.requireAllBuckets, settings.minDraws],
  );
  const running = status?.running ?? false;

  const clearPoll = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const loadConcepts = useCallback(async () => {
    try {
      const res = await dpoSaturationApi.concepts();
      setConceptsInfo(res);
      if (res.warning) setNotice(res.warning);
      return res;
    } catch (ex) {
      setError(String(ex));
      return null;
    }
  }, []);

  const loadSummary = useCallback(async (conceptPath: string, r: SaturationRule) => {
    const seq = ++summarySeq.current;
    try {
      const res = await dpoSaturationApi.summary(conceptPath, r);
      if (seq !== summarySeq.current) return;
      if (!res.ok) {
        setSummary(null);
        setNotice(res.error ?? "No summary.");
        return;
      }
      setSummary(res);
      setNotice(null);
    } catch (ex) {
      if (seq === summarySeq.current) setError(String(ex));
    }
  }, []);

  const loadRetired = useCallback(async (conceptPath: string) => {
    try {
      const res = await dpoSaturationApi.retired(conceptPath);
      setRetired(res.ok ? res.pairs : []);
    } catch {
      setRetired([]);
    }
  }, []);

  const refreshConcept = useCallback(() => {
    if (!concept?.table) {
      setSummary(null);
      setRetired([]);
      if (concept) void loadRetired(concept.path);
      return;
    }
    void loadSummary(concept.path, rule);
    void loadRetired(concept.path);
  }, [concept, rule, loadSummary, loadRetired]);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(async () => {
      try {
        const s = await dpoSaturationApi.status();
        setStatus(s);
        if (!s.running) {
          clearPoll();
          if (s.phase === "error" && s.error) setError(s.error);
          await loadConcepts();
        }
      } catch (ex) {
        setError(String(ex));
        clearPoll();
      }
    }, POLL_MS);
  }, [clearPoll, loadConcepts]);

  // Open: concepts, current job state, and resume polling if a run is going.
  useEffect(() => {
    if (!open) return;
    setError(null);
    void (async () => {
      await loadConcepts();
      try {
        const s = await dpoSaturationApi.status();
        setStatus(s);
        if (s.running) startPolling();
      } catch (ex) {
        setError(String(ex));
      }
    })();
  }, [open, loadConcepts, startPolling]);

  useEffect(() => () => clearPoll(), [clearPoll]);

  // Debounced summary reload whenever the concept or the rule changes.
  useEffect(() => {
    if (!open) return;
    if (summaryTimer.current != null) window.clearTimeout(summaryTimer.current);
    summaryTimer.current = window.setTimeout(refreshConcept, SUMMARY_DEBOUNCE_MS);
    return () => {
      if (summaryTimer.current != null) window.clearTimeout(summaryTimer.current);
    };
  }, [open, refreshConcept]);

  useEffect(() => setConfirmRetire(false), [concept?.path, rule]);

  const start = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await dpoSaturationApi.start({
        current: settings.mode === "checkpoints" ? settings.current : "",
        origin: settings.mode === "checkpoints" ? settings.origin : "",
        adapter: settings.mode === "adapter" ? settings.adapter : "",
        draws: settings.draws,
        timesteps: settings.timesteps.trim(),
        seeds: settings.seeds,
        cacheDir: settings.cacheDir,
        forceOrigin: settings.forceOrigin,
        concepts: [],
      });
      if (!res.ok) {
        setError(res.error ?? "Could not start scoring.");
        return;
      }
      setStatus(await dpoSaturationApi.status());
      startPolling();
    } catch (ex) {
      setError(String(ex));
    } finally {
      setBusy(false);
    }
  }, [settings, startPolling]);

  const cancel = useCallback(async () => {
    setBusy(true);
    try {
      await dpoSaturationApi.cancel();
      setStatus(await dpoSaturationApi.status());
    } catch (ex) {
      setError(String(ex));
    } finally {
      setBusy(false);
    }
  }, []);

  const retireKeys = useCallback(
    async (keys: string[], reason: string) => {
      if (!concept) return;
      setBusy(true);
      setError(null);
      try {
        const res = await dpoSaturationApi.retireKeys(concept.path, keys, reason);
        if (!res.ok) {
          setError(res.error ?? "Retire failed.");
          return;
        }
        setNotice(`Retired ${res.moved.length} pair(s).`);
        await loadConcepts();
        refreshConcept();
      } catch (ex) {
        setError(String(ex));
      } finally {
        setBusy(false);
      }
    },
    [concept, loadConcepts, refreshConcept],
  );

  const retireSaturated = useCallback(async () => {
    if (!concept) return;
    if (!confirmRetire) {
      setConfirmRetire(true);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await dpoSaturationApi.retireSaturated(concept.path, rule, false);
      if (!res.ok) {
        setError(res.error ?? "Retire failed.");
        return;
      }
      setNotice(`Retired ${res.moved.length} saturated pair(s); ${res.retired} retired in total.`);
      setConfirmRetire(false);
      await loadConcepts();
      refreshConcept();
    } catch (ex) {
      setError(String(ex));
    } finally {
      setBusy(false);
    }
  }, [concept, confirmRetire, rule, loadConcepts, refreshConcept]);

  const restore = useCallback(
    async (keys: string[] | null) => {
      if (!concept) return;
      setBusy(true);
      setError(null);
      try {
        const res = await dpoSaturationApi.restore(concept.path, keys);
        if (!res.ok) {
          setError(res.error ?? "Restore failed.");
          return;
        }
        setNotice(`Restored ${res.restored.length} pair(s).`);
        await loadConcepts();
        refreshConcept();
      } catch (ex) {
        setError(String(ex));
      } finally {
        setBusy(false);
      }
    },
    [concept, loadConcepts, refreshConcept],
  );

  const sweepText = useMemo(
    () =>
      summary ? summary.beta_sweep.map((s) => `${s.multiplier}× (${Math.round(s.beta)}): ${s.dead}`).join("   ") : "",
    [summary],
  );

  const utilityPoints = useMemo(
    () => (summary ? summary.utility_curve.map((p) => ({ step: p.pairs, value: 100 * p.share })) : []),
    [summary],
  );

  const betaMax = Math.max(100, Math.round(conceptBeta * 4));

  return (
    <ModalBase open={open} onClose={onClose} title="DPO Saturation" size="full" closeOnBackdrop={!running}>
      <div className="flex flex-col gap-4">
        <div className="text-sm text-[var(--color-on-surface-secondary)]">
          Scores every DPO pair against the pre-DPO <em>origin</em> base of the lineage: how much the current checkpoint
          already prefers the chosen image over the rejected one, in the units the DPO loss sees (β × margin, nats). A
          pair whose expected gradient weight <code>sigmoid(−β·margin)</code> is below the budget is dead weight for the
          next run. Scoring reads the training cache and runs the trainer&apos;s own forward pass in a separate process;
          the origin pass is cached for good, a rescore only runs the current pass. Retired pairs move into{" "}
          <code>.retired/</code> under their side folder and can be restored.
        </div>

        {notice && <div className="text-sm text-[var(--color-on-surface-secondary)]">{notice}</div>}
        {error && (
          <div className="flex items-center gap-2 rounded bg-[var(--color-surface-container)] p-3 text-sm text-[var(--color-error-500)]">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        <SectionCard title="Score">
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <Select
                label="Mode"
                value={settings.mode}
                onChange={(v) => update({ mode: v as ScoreMode })}
                options={[
                  { label: "Two checkpoints (origin → current)", value: "checkpoints" },
                  { label: "Adapter on the config's base", value: "adapter" },
                ]}
                formatLabel={(v) => v}
                disabled={running}
              />
              <FormEntry
                label="Draws per pair (training distribution)"
                type="number"
                value={settings.draws}
                onChange={(v) => update({ draws: Math.max(1, Number(v) || 1) })}
                disabled={running || settings.timesteps.trim() !== ""}
                tooltip="Each draw is a real training step (seed = global step): noise and timestep from the run's own distribution."
              />
              <FormEntry
                label="Fixed noise levels (optional, e.g. 0.1,0.5,0.9)"
                value={settings.timesteps}
                onChange={(v) => update({ timesteps: String(v) })}
                disabled={running}
                tooltip="Overrides draws: every listed level gets the number of noise seeds below, for controlled per-bucket coverage."
              />
            </div>
            {settings.mode === "checkpoints" ? (
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <FilePicker
                  label="Current checkpoint (transformer safetensors)"
                  value={settings.current}
                  onChange={(v) => update({ current: v })}
                  disabled={running}
                />
                <FilePicker
                  label={`Origin checkpoint (empty = ${fileName(conceptsInfo?.default_origin) || "config transformer"})`}
                  value={settings.origin}
                  onChange={(v) => update({ origin: v })}
                  disabled={running}
                  tooltip="The pre-DPO base of the lineage. Keep it fixed across generations so scores stay comparable."
                />
              </div>
            ) : (
              <FilePicker
                label={`Adapter (empty = ${fileName(conceptsInfo?.adapter) || "none configured"})`}
                value={settings.adapter}
                onChange={(v) => update({ adapter: v })}
                disabled={running}
                tooltip="Origin = the config's base without the adapter, current = with it. One model load."
              />
            )}
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <FormEntry
                label="Noise seeds per fixed level"
                type="number"
                value={settings.seeds}
                onChange={(v) => update({ seeds: Math.max(1, Number(v) || 1) })}
                disabled={running || settings.timesteps.trim() === ""}
              />
              <DirPicker
                label={`Latent cache (empty = ${conceptsInfo?.cache_dir || "config cache"})`}
                value={settings.cacheDir}
                onChange={(v) => update({ cacheDir: v })}
                disabled={running}
                tooltip="Point at a separate folder while a training run is using the config's cache."
              />
              <div className="flex items-end gap-3">
                <Toggle
                  label="Rescore origin"
                  value={settings.forceOrigin}
                  onChange={(v) => update({ forceOrigin: v })}
                  disabled={running}
                  tooltip="Ignore cached origin scalars and run the origin pass again."
                />
                {running ? (
                  <Button variant="danger" onClick={() => void cancel()} disabled={busy}>
                    <Square className="h-4 w-4" /> Cancel
                  </Button>
                ) : (
                  <Button
                    variant="primary"
                    onClick={() => void start()}
                    disabled={
                      busy ||
                      (settings.mode === "checkpoints"
                        ? !settings.current
                        : !(settings.adapter || conceptsInfo?.adapter))
                    }
                  >
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                    Score all DPO concepts
                  </Button>
                )}
              </div>
            </div>
            {status && status.phase !== "idle" && (
              <div className="flex flex-col gap-2 rounded bg-[var(--color-surface-container)] p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    {running && <Loader2 className="h-4 w-4 animate-spin" />}
                    {status.phase === "running" ? status.progress_phase || "starting" : status.phase}
                  </span>
                  <span className="tabular-nums text-[var(--color-on-surface-secondary)]">
                    {status.total > 0
                      ? `${status.done.toLocaleString()} / ${status.total.toLocaleString()} · ${pct(status.done, status.total).toFixed(0)}%`
                      : ""}{" "}
                    · {minutes(status.elapsed)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded bg-[var(--color-border-subtle)]">
                  <div
                    className={`h-full transition-all ${status.phase === "error" ? "bg-[var(--color-error-500)]" : "bg-[var(--color-cobalt-600)]"}`}
                    style={{ width: `${status.total > 0 ? pct(status.done, status.total) : running ? 100 : 0}%` }}
                  />
                </div>
                <button
                  type="button"
                  className="self-start text-xs text-[var(--color-on-surface-secondary)] underline"
                  onClick={() => setShowLog((v) => !v)}
                >
                  {showLog ? "hide log" : "show log"}
                </button>
                {showLog && (
                  <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs">{status.log.join("\n")}</pre>
                )}
              </div>
            )}
          </div>
        </SectionCard>

        <SectionCard title="Analyse">
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
              <Select
                label="Concept"
                value={concept?.path ?? ""}
                onChange={(v) => update({ concept: v })}
                options={concepts.map((c) => ({
                  label: `${conceptLabel(c)}${c.table ? ` · ${c.table.scored} scored` : " · not scored"}${c.retired ? ` · ${c.retired} retired` : ""}`,
                  value: c.path,
                }))}
                formatLabel={(v) => v}
              />
              <SliderEntry
                label={`Beta (concept default ${Math.round(conceptBeta)})`}
                min={1}
                max={betaMax}
                step={1}
                value={Math.round(beta)}
                onChange={(v) => concept && update({ betaByConcept: { ...settings.betaByConcept, [concept.path]: v } })}
                tooltip="The beta the next run will use for this concept. Only the reading changes: the table is beta-independent."
              />
              <SliderEntry
                label="Dead below gradient weight"
                min={0.005}
                max={0.5}
                step={0.005}
                value={settings.deadWeight}
                onChange={(v) => update({ deadWeight: v })}
                tooltip="sigmoid(-beta*margin) under this = the pair contributes almost nothing. 0.5 is an untouched pair."
              />
              <div className="flex flex-col gap-2">
                <Toggle
                  label="Dead in every timestep bucket"
                  value={settings.requireAllBuckets}
                  onChange={(v) => update({ requireAllBuckets: v })}
                  tooltip="Strict: a pair still alive at any noise level (low noise = detail work) is kept."
                />
                <FormEntry
                  label="Min draws"
                  type="number"
                  value={settings.minDraws}
                  onChange={(v) => update({ minDraws: Math.max(1, Number(v) || 1) })}
                  width="w-24"
                />
              </div>
            </div>

            {!concept?.table && (
              <div className="text-sm text-[var(--color-on-surface-secondary)]">
                This concept has no saturation table yet. Score it above.
              </div>
            )}

            {summary && (
              <>
                <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs tabular-nums">
                  <span>
                    scored <b>{summary.pairs}</b> / {summary.pairs_in_table} pairs · {summary.draws_per_pair} draws
                  </span>
                  <span>
                    saturated{" "}
                    <b className="text-[var(--color-warning-500)]">
                      {summary.saturated} (
                      {summary.pairs ? ((100 * summary.saturated) / summary.pairs).toFixed(1) : "0"}%)
                    </b>
                  </span>
                  <span>
                    chosen preferred on {summary.wins_mean == null ? "-" : `${(100 * summary.wins_mean).toFixed(1)}%`}{" "}
                    of draws
                  </span>
                  <span>
                    β·m median {fmt(summary.logit_percentiles.p50, 2)} · p95 {fmt(summary.logit_percentiles.p95, 2)}
                  </span>
                  <span>
                    buckets dead/scored:{" "}
                    {summary.bucket_dead.map((d, i) => `q${i + 1} ${d}/${summary.bucket_counts[i]}`).join("  ")}
                  </span>
                  <span>retired {summary.retired}</span>
                  <span className="text-[var(--color-on-surface-secondary)]">
                    origin {fileName(summary.origin)} → current {fileName(summary.current)}
                  </span>
                </div>
                <div className="text-xs tabular-nums text-[var(--color-on-surface-secondary)]">
                  saturated if the run used: {sweepText}
                </div>

                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div>
                    <div className="mb-1 text-xs uppercase text-[var(--color-on-surface-secondary)]">
                      Absorbed preference per pair
                    </div>
                    <LogitHistogram
                      edges={summary.logit_histogram.edges}
                      counts={summary.logit_histogram.counts}
                      threshold={deadLogit(summary.dead_weight)}
                    />
                  </div>
                  <div>
                    <div className="mb-1 text-xs uppercase text-[var(--color-on-surface-secondary)]">
                      Utility curve: pairs kept (best first) → % of total gradient weight
                    </div>
                    <ScalarChart tag="dpo-saturation-utility" points={utilityPoints} width={520} height={200} />
                    <div className="text-xs tabular-nums text-[var(--color-on-surface-secondary)]">
                      {Object.entries(summary.utility.share)
                        .map(([frac, n]) => `${Math.round(100 * Number(frac))}% of weight in ${n} pairs`)
                        .join(" · ")}
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    variant={confirmRetire ? "danger" : "secondary"}
                    onClick={() => void retireSaturated()}
                    disabled={busy || running || summary.saturated === 0}
                  >
                    <Archive className="h-4 w-4" />
                    {confirmRetire
                      ? `Confirm: retire ${summary.saturated} saturated pairs`
                      : `Retire all ${summary.saturated} saturated pairs`}
                  </Button>
                  {confirmRetire && (
                    <Button variant="ghost" onClick={() => setConfirmRetire(false)}>
                      Keep them
                    </Button>
                  )}
                  {running && (
                    <span className="text-xs text-[var(--color-on-surface-secondary)]">
                      Moves are locked while scoring runs.
                    </span>
                  )}
                </div>

                <details>
                  <summary className="cursor-pointer text-sm">
                    Most saturated ({summary.most_saturated.length} of{" "}
                    {summary.saturated_rows.length + summary.saturated_truncated} dead shown first)
                  </summary>
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
                    {summary.most_saturated.map((row) => (
                      <PairTile
                        key={row.key}
                        row={row}
                        onPreview={setPreview}
                        action={{
                          label: "Retire",
                          onClick: () => void retireKeys([row.key], `manual, beta*margin ${row.mean_logit.toFixed(2)}`),
                          disabled: busy || running,
                        }}
                      />
                    ))}
                  </div>
                </details>

                <details>
                  <summary className="cursor-pointer text-sm">
                    Most negative: the current model prefers the rejected image ({summary.most_negative.length}) — check
                    the label
                  </summary>
                  <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
                    {summary.most_negative.length === 0 && (
                      <div className="text-sm text-[var(--color-on-surface-secondary)]">None.</div>
                    )}
                    {summary.most_negative.map((row) => (
                      <PairTile
                        key={row.key}
                        row={row}
                        onPreview={setPreview}
                        action={{
                          label: "Retire",
                          onClick: () =>
                            void retireKeys([row.key], `manual, negative margin ${row.mean_logit.toFixed(2)}`),
                          disabled: busy || running,
                          danger: true,
                        }}
                      />
                    ))}
                  </div>
                </details>
              </>
            )}

            {concept && retired.length > 0 && (
              <details>
                <summary className="cursor-pointer text-sm">Retired pairs ({retired.length})</summary>
                <div className="mt-2 flex flex-col gap-2">
                  <Button variant="secondary" size="sm" onClick={() => void restore(null)} disabled={busy || running}>
                    <RotateCcw className="h-4 w-4" /> Restore all
                  </Button>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
                    {retired.map((r) => (
                      <figure
                        key={r.key}
                        className="flex flex-col gap-1 rounded bg-[var(--color-surface-container)] p-2"
                      >
                        <div className="grid grid-cols-2 gap-1">
                          {[r.chosen, r.rejected].map((path, i) => (
                            <button
                              key={i}
                              type="button"
                              className="aspect-square overflow-hidden rounded bg-black/20"
                              onClick={() => path && setPreview(path)}
                            >
                              {path && (
                                <img
                                  src={dpoSaturationApi.imageUrl(path)}
                                  alt={i === 0 ? "chosen" : "rejected"}
                                  className="h-full w-full object-cover"
                                  loading="lazy"
                                />
                              )}
                            </button>
                          ))}
                        </div>
                        <figcaption className="text-xs">
                          <div className="truncate" title={r.key}>
                            {r.key}
                          </div>
                          <div className="truncate text-[var(--color-on-surface-secondary)]" title={r.reason}>
                            {r.reason}
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => void restore([r.key])}
                            disabled={busy || running}
                          >
                            Restore
                          </Button>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              </details>
            )}
          </div>
        </SectionCard>
      </div>
      {preview && <PreviewOverlay path={preview} onClose={() => setPreview(null)} />}
    </ModalBase>
  );
}
