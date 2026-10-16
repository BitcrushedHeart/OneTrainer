import { AlertCircle, Check, ChevronLeft, ChevronRight, Loader2, RotateCcw, SkipForward, Undo2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  dpoNightDarkenApi,
  NEUTRAL_ADJUST,
  type NightDarkenAdjust,
  type NightDarkenCandidate,
  type NightDarkenJob,
  type NightDarkenState,
} from "@/api/dpoNightDarkenApi";
import { ModalBase } from "@/components/modals/ModalBase";
import { Button, DirPicker, FormEntry, ProgressBar, SectionCard, SliderEntry, Toggle } from "@/components/shared";

export interface NightDarkenModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.nightDarken";
const POLL_MS = 500;
const PREVIEW_DEBOUNCE_MS = 120;
const HUE_RANGE = 180;
const SATURATION_MAX = 2;

interface Settings {
  conceptDir: string;
  nightEditsDir: string;
  fraction: number;
  darkMedian: number;
  darkMean: number;
  /** The last accepted settings; the starting point for the next candidate. */
  adjust: NightDarkenAdjust;
  showCaption: boolean;
  onlyUndecided: boolean;
}

const DEFAULTS: Settings = {
  conceptDir: "",
  nightEditsDir: "",
  fraction: 0.33,
  darkMedian: 60,
  darkMean: 80,
  adjust: { ...NEUTRAL_ADJUST, strength: 0.5 },
  showCaption: true,
  onlyUndecided: true,
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Settings>;
      return { ...DEFAULTS, ...parsed, adjust: { ...DEFAULTS.adjust, ...(parsed.adjust ?? {}) } };
    }
  } catch {
    // ignore a corrupt entry
  }
  return { ...DEFAULTS };
}

const fmt = (x: number | null | undefined, digits = 1) => (x == null || Number.isNaN(x) ? "-" : x.toFixed(digits));
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const clamp01 = (x: number) => clamp(x, 0, 1);
const when = (t: number | null | undefined) => (t ? new Date(t * 1000).toLocaleString() : "-");
const round2 = (x: number) => Math.round(x * 100) / 100;

function normalizeAdjust(a: NightDarkenAdjust): NightDarkenAdjust {
  return {
    strength: clamp01(a.strength),
    noise: clamp01(a.noise),
    hue: clamp(a.hue, -HUE_RANGE, HUE_RANGE),
    saturation: clamp(a.saturation, 0, SATURATION_MAX),
    preset: a.preset ?? "",
  };
}

function adjustLabel(a: NightDarkenAdjust | undefined, strength?: number): string {
  if (!a) return strength != null ? `strength ${fmt(strength, 2)}` : "";
  const parts = [`strength ${fmt(a.strength, 2)}`];
  if (a.preset) parts.unshift(a.preset);
  if (a.noise > 0) parts.push(`noise ${fmt(a.noise, 2)}`);
  if (a.hue !== 0) parts.push(`hue ${a.hue > 0 ? "+" : ""}${fmt(a.hue, 0)}°`);
  if (a.saturation !== 1) parts.push(`sat ×${fmt(a.saturation, 2)}`);
  return parts.join(", ");
}

/** The previous image stays up until the next one has loaded, so dragging a slider never flashes white. */
function useLoadedSrc(src: string | null): string | null {
  const [loaded, setLoaded] = useState<string | null>(null);
  useEffect(() => {
    if (!src) {
      setLoaded(null);
      return;
    }
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (live) setLoaded(src);
    };
    img.src = src;
    return () => {
      live = false;
    };
  }, [src]);
  return loaded;
}

// --------------------------------------------------------------------- //
// Before/after wipe: original on the left of the divider, adjusted right
// --------------------------------------------------------------------- //

function WipeCompare({
  beforeSrc,
  afterSrc,
  position,
  onPosition,
}: {
  beforeSrc: string | null;
  afterSrc: string | null;
  position: number;
  onPosition: (p: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const update = useCallback(
    (clientX: number) => {
      const box = ref.current?.getBoundingClientRect();
      if (!box || box.width === 0) return;
      onPosition(clamp01((clientX - box.left) / box.width));
    },
    [onPosition],
  );

  return (
    <div
      ref={ref}
      className="relative w-full h-[58vh] min-h-[300px] select-none overflow-hidden rounded bg-black/70 cursor-col-resize touch-none"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        dragging.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        update(e.clientX);
      }}
      onPointerMove={(e) => {
        if (dragging.current) update(e.clientX);
      }}
      onPointerUp={() => {
        dragging.current = false;
      }}
      onPointerCancel={() => {
        dragging.current = false;
      }}
    >
      {beforeSrc && (
        <img
          src={beforeSrc}
          alt="original"
          className="absolute inset-0 h-full w-full object-contain"
          draggable={false}
        />
      )}
      {afterSrc && (
        <img
          src={afterSrc}
          alt="adjusted"
          className="absolute inset-0 h-full w-full object-contain"
          style={{ clipPath: `inset(0 0 0 ${(position * 100).toFixed(2)}%)` }}
          draggable={false}
        />
      )}
      <div
        className="absolute top-0 bottom-0 w-px bg-white/90 shadow-[0_0_4px_rgba(0,0,0,0.9)] pointer-events-none"
        style={{ left: `${(position * 100).toFixed(2)}%` }}
      >
        <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 h-8 w-8 rounded-full border border-white/80 bg-black/50 flex items-center justify-center text-[10px] text-white">
          ⇔
        </div>
      </div>
      <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white pointer-events-none">
        original
      </span>
      <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white pointer-events-none">
        adjusted
      </span>
    </div>
  );
}

// --------------------------------------------------------------------- //
// Modal
// --------------------------------------------------------------------- //

export function NightDarkenModal({ open, onClose }: NightDarkenModalProps) {
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [state, setState] = useState<NightDarkenState | null>(null);
  const [candidates, setCandidates] = useState<NightDarkenCandidate[]>([]);
  const [index, setIndex] = useState(0);
  const [adjust, setAdjust] = useState<NightDarkenAdjust>(settings.adjust);
  const [previewAdjust, setPreviewAdjust] = useState<NightDarkenAdjust>(settings.adjust);
  const [wipe, setWipe] = useState(0.5);
  const [caption, setCaption] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmCommit, setConfirmCommit] = useState(false);
  const pollRef = useRef<number | null>(null);
  const debounceRef = useRef<number | null>(null);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // storage unavailable
      }
      return next;
    });
  }, []);

  // A hand edit of any slider drops the preset tag: the values are no longer the preset's.
  const patchAdjust = useCallback((patch: Partial<NightDarkenAdjust>) => {
    setAdjust((prev) => normalizeAdjust({ ...prev, ...patch, preset: "" }));
  }, []);

  const job: NightDarkenJob | null = state?.job ?? null;
  const running = job?.running ?? false;
  const opened = Boolean(state?.concept_dir);
  const current: NightDarkenCandidate | undefined = candidates[index];
  const total = candidates.length;
  const nightEditsDir = settings.nightEditsDir || state?.default_night_edits_dir || "";

  // ---- data loading ----------------------------------------------------

  const refresh = useCallback(async () => {
    try {
      const next = await dpoNightDarkenApi.state();
      setState(next);
      if (next.concept_dir) {
        const list = await dpoNightDarkenApi.candidates();
        setCandidates(list.candidates ?? []);
      } else {
        setCandidates([]);
      }
      return next;
    } catch (ex) {
      setError(String(ex));
      return null;
    }
  }, []);

  const clearPoll = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPoll = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(async () => {
      try {
        const res = await dpoNightDarkenApi.status();
        setState((prev) => (prev ? { ...prev, job: res.job } : prev));
        if (res.job && !res.job.running) {
          clearPoll();
          await refresh();
          if (res.job.error) setError(res.job.error);
        }
      } catch (ex) {
        clearPoll();
        setError(String(ex));
      }
    }, POLL_MS);
  }, [clearPoll, refresh]);

  useEffect(() => {
    if (!open) {
      clearPoll();
      return;
    }
    void refresh().then((next) => {
      if (next?.job?.running) startPoll();
    });
    return clearPoll;
  }, [open, refresh, startPoll, clearPoll]);

  // First undecided candidate when the list changes.
  useEffect(() => {
    if (candidates.length === 0) {
      setIndex(0);
      return;
    }
    setIndex((i) => {
      if (i < candidates.length && (!settings.onlyUndecided || !candidates[i].decision)) return i;
      const first = settings.onlyUndecided ? candidates.findIndex((c) => !c.decision) : 0;
      return first >= 0 ? first : Math.min(i, candidates.length - 1);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates]);

  // Caption + the accepted settings for the current candidate.
  useEffect(() => {
    if (!current) {
      setCaption("");
      return;
    }
    if (current.decision?.state === "accepted") {
      const accepted = current.decision.adjust ?? {
        ...NEUTRAL_ADJUST,
        strength: current.decision.strength ?? 0,
      };
      setAdjust(normalizeAdjust(accepted));
      setPreviewAdjust(normalizeAdjust(accepted));
    }
    let live = true;
    if (settings.showCaption) {
      void dpoNightDarkenApi.caption(current.rel).then((res) => {
        if (live) setCaption(res.caption ?? "");
      });
    }
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.rel, settings.showCaption]);

  // Debounced preview re-render while a slider moves.
  useEffect(() => {
    if (debounceRef.current != null) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => setPreviewAdjust(adjust), PREVIEW_DEBOUNCE_MS);
    return () => {
      if (debounceRef.current != null) window.clearTimeout(debounceRef.current);
    };
  }, [adjust]);

  const beforeSrc = useLoadedSrc(current ? dpoNightDarkenApi.previewUrl(current.rel, NEUTRAL_ADJUST) : null);
  const afterSrc = useLoadedSrc(current ? dpoNightDarkenApi.previewUrl(current.rel, previewAdjust) : null);

  // ---- actions -----------------------------------------------------------

  const guard = useCallback(
    async <T extends { ok: boolean; error?: string }>(fn: () => Promise<T>): Promise<T | null> => {
      setBusy(true);
      setError(null);
      try {
        const res = await fn();
        if (!res.ok) setError(res.error ?? "Request failed");
        return res;
      } catch (ex) {
        setError(String(ex));
        return null;
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const openConcept = useCallback(async () => {
    if (!settings.conceptDir) {
      setError("Choose the source concept folder first");
      return;
    }
    const res = await guard(() => dpoNightDarkenApi.open(settings.conceptDir));
    if (res?.ok) {
      setNotice(null);
      await refresh();
    }
  }, [settings.conceptDir, guard, refresh]);

  const scan = useCallback(async () => {
    if (!opened) {
      await openConcept();
    }
    const res = await guard(() => dpoNightDarkenApi.scan({ median: settings.darkMedian, mean: settings.darkMean }, []));
    if (res?.ok) {
      await refresh();
      startPoll();
    }
  }, [opened, openConcept, guard, settings.darkMedian, settings.darkMean, refresh, startPoll]);

  const step = useCallback(
    (delta: number) => {
      setIndex((i) => Math.min(Math.max(0, i + delta), Math.max(0, total - 1)));
    },
    [total],
  );

  const nextAfterDecision = useCallback(
    (from: number, list: NightDarkenCandidate[]) => {
      if (settings.onlyUndecided) {
        for (let i = from + 1; i < list.length; i++) if (!list[i].decision) return i;
        for (let i = 0; i < from; i++) if (!list[i].decision) return i;
        return Math.min(from + 1, list.length - 1);
      }
      return Math.min(from + 1, list.length - 1);
    },
    [settings.onlyUndecided],
  );

  const decide = useCallback(
    async (kind: "accepted" | "skipped") => {
      if (!current || busy) return;
      const settingsNow = kind === "accepted" ? normalizeAdjust(adjust) : null;
      const res = await guard(() => dpoNightDarkenApi.decide(current.rel, kind, settingsNow));
      if (!res?.ok || !res.decision) return;
      if (settingsNow) update({ adjust: settingsNow });
      const decision = res.decision;
      const list = candidates.map((c) => (c.rel === current.rel ? { ...c, decision } : c));
      setCandidates(list);
      setIndex(nextAfterDecision(index, list));
      void dpoNightDarkenApi.state().then(setState);
    },
    [current, candidates, busy, adjust, guard, update, nextAfterDecision, index],
  );

  const undo = useCallback(async () => {
    if (!current || busy) return;
    const res = await guard(() => dpoNightDarkenApi.undo(current.rel));
    if (!res?.ok) return;
    setCandidates((prev) => prev.map((c) => (c.rel === current.rel ? { ...c, decision: null } : c)));
    void dpoNightDarkenApi.state().then(setState);
  }, [current, busy, guard]);

  const back = useCallback(() => {
    // Back = the previous candidate, whatever its state, so a mistaken
    // decision can be undone there (U) or re-decided.
    setIndex((i) => Math.max(0, i - 1));
  }, []);

  /** Apply a night preset ("more" = the next stronger one) with the image's own jitter. */
  const applyPreset = useCallback(
    async (presetId: string) => {
      if (!current || busy) return;
      const res = await guard(() => dpoNightDarkenApi.preset(current.rel, presetId, normalizeAdjust(adjust)));
      if (res?.ok && res.adjust) {
        const next = normalizeAdjust(res.adjust);
        setAdjust(next);
        setPreviewAdjust(next);
      }
    },
    [current, busy, guard, adjust],
  );

  const commit = useCallback(async () => {
    setConfirmCommit(false);
    const res = await guard(() => dpoNightDarkenApi.commit(nightEditsDir, settings.fraction));
    if (res?.ok) {
      setNotice(`Committing ${res.pending ?? 0} images, ${res.pairs_planned ?? 0} Night Edits pairs…`);
      await refresh();
      startPoll();
    }
  }, [guard, nightEditsDir, settings.fraction, refresh, startPoll]);

  const restore = useCallback(
    async (commitId: string) => {
      const res = await guard(() => dpoNightDarkenApi.restore(commitId));
      if (res?.ok) {
        const missing = res.missing && res.missing.length ? `; ${res.missing.length} backups missing` : "";
        setNotice(`Restored ${res.restored ?? 0} originals, removed ${res.pairs_removed ?? 0} pairs${missing}`);
        await refresh();
      }
    },
    [guard, refresh],
  );

  // ---- keyboard ----------------------------------------------------------

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable) return;
      if (confirmCommit) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const key = e.key;
      if (key === "a" || key === "A" || key === "Enter") void decide("accepted");
      else if (key === "s" || key === "S" || key === " ") void decide("skipped");
      else if (key === "u" || key === "U") void undo();
      else if (key === "Backspace" || key === "ArrowLeft") back();
      else if (key === "ArrowRight") step(1);
      else if (key === "[") patchAdjust({ strength: round2(adjust.strength - 0.05) });
      else if (key === "]") patchAdjust({ strength: round2(adjust.strength + 0.05) });
      else if (key === "r" || key === "R") setAdjust({ ...NEUTRAL_ADJUST, strength: adjust.strength });
      else if (key === "d" || key === "D") void applyPreset("more");
      else if (/^[0-9]$/.test(key)) patchAdjust({ strength: key === "0" ? 1 : Number(key) / 10 });
      else return;
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, decide, undo, back, step, confirmCommit, patchAdjust, applyPreset, adjust.strength]);

  // ---- render --------------------------------------------------------------

  const counts = state?.counts;
  const scanInfo = state?.scan;
  const defaults = state?.defaults;
  const pairsPlanned = useMemo(() => {
    const n = counts?.accepted_uncommitted ?? 0;
    if (n === 0 || settings.fraction <= 0) return 0;
    return Math.max(1, Math.round(n * Math.min(1, settings.fraction)));
  }, [counts?.accepted_uncommitted, settings.fraction]);

  const decisionLabel = current?.decision
    ? current.decision.state === "accepted"
      ? `accepted (${adjustLabel(current.decision.adjust, current.decision.strength)}, ΔE ${fmt(current.decision.delta_e, 1)})${current.decision.committed ? " committed" : ""}`
      : "skipped"
    : "undecided";

  return (
    <ModalBase open={open} onClose={onClose} title="Night Darken" size="2xl" closeOnEscape={!confirmCommit}>
      <div className="flex flex-col gap-4">
        {error && (
          <div className="flex items-start gap-2 rounded bg-[var(--color-error-500)]/10 p-2 text-sm text-[var(--color-error-500)]">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="break-all">{error}</span>
          </div>
        )}
        {notice && !error && <div className="text-sm text-[var(--color-on-surface-secondary)]">{notice}</div>}

        <SectionCard title="Source">
          <div className="flex flex-col gap-2">
            <DirPicker
              label="DPO concept (chosen/ + rejected/)"
              value={settings.conceptDir}
              onChange={(v) => update({ conceptDir: v })}
              tooltip="The RLHF concept whose chosen images are reviewed. Minor Edits and Off-policy datasets are refused."
              disabled={running}
            />
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <FormEntry
                label="Already dark: median luma ≤"
                type="number"
                value={settings.darkMedian}
                onChange={(v) => update({ darkMedian: Number(v) })}
                tooltip={`An image whose median AND mean luma (0-255) are at or below these is left alone. Defaults ${defaults?.thresholds.median ?? 60} / ${defaults?.thresholds.mean ?? 80}.`}
                disabled={running}
              />
              <FormEntry
                label="Already dark: mean luma ≤"
                type="number"
                value={settings.darkMean}
                onChange={(v) => update({ darkMean: Number(v) })}
                disabled={running}
              />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="secondary" size="sm" onClick={openConcept} disabled={busy || running}>
                Open
              </Button>
              <Button variant="primary" size="sm" onClick={scan} disabled={busy || running || !settings.conceptDir}>
                {running && job?.kind === "scan" ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Scan captions + brightness
              </Button>
              {running && (
                <Button variant="ghost" size="sm" onClick={() => void dpoNightDarkenApi.cancel()}>
                  Cancel
                </Button>
              )}
              {defaults && (
                <span
                  className="text-xs text-[var(--color-on-surface-secondary)]"
                  title={`Phrases: ${defaults.phrases.join(", ")}`}
                >
                  {defaults.phrases.length} night/dark phrases (whole word, case-insensitive)
                </span>
              )}
            </div>
            {running && job && (
              <ProgressBar
                value={job.total > 0 ? (100 * job.done) / job.total : 0}
                indeterminate={job.total === 0}
                label={`${job.kind}: ${job.phase} ${job.done}/${job.total}`}
              />
            )}
            {scanInfo && (
              <div className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
                Scanned {when(scanInfo.at)}: {scanInfo.counts.images} chosen images, {scanInfo.counts.caption_matches}{" "}
                caption matches, {scanInfo.counts.already_dark} already dark (skipped), {scanInfo.counts.unreadable}{" "}
                unreadable, <b>{scanInfo.counts.candidates} candidates</b>.
              </div>
            )}
          </div>
        </SectionCard>

        {opened && total > 0 && current && (
          <SectionCard title={`Review ${index + 1} / ${total}`}>
            <div className="flex flex-col gap-2">
              <WipeCompare beforeSrc={beforeSrc} afterSrc={afterSrc} position={wipe} onPosition={setWipe} />
              <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs tabular-nums text-[var(--color-on-surface-secondary)]">
                <span className="truncate" title={current.rel}>
                  {current.rel}
                </span>
                <span>
                  {current.size[0]}×{current.size[1]} · median {fmt(current.stats.median, 0)} · mean{" "}
                  {fmt(current.stats.mean, 0)} · p90 {fmt(current.stats.p90, 0)} · bright{" "}
                  {fmt(100 * current.stats.bright_fraction, 0)}%
                </span>
                <span>matched: {current.phrases.join(", ")}</span>
                <span
                  className={
                    current.decision?.state === "accepted"
                      ? "text-[var(--color-success-500)]"
                      : current.decision?.state === "skipped"
                        ? "text-[var(--color-warning-500)]"
                        : ""
                  }
                >
                  {decisionLabel}
                </span>
              </div>
              {settings.showCaption && caption && (
                <div className="max-h-16 overflow-auto rounded bg-[var(--color-surface-container)] p-2 text-xs text-[var(--color-on-surface)]">
                  {caption}
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-[var(--color-on-surface-secondary)]">Night presets:</span>
                {(defaults?.presets ?? []).map((p) => (
                  <Button
                    key={p.id}
                    variant={adjust.preset === p.id ? "primary" : "secondary"}
                    size="sm"
                    onClick={() => void applyPreset(p.id)}
                    disabled={busy || running}
                    title={`brightness ×${p.brightness}, saturation ×${p.saturation}, hue ${p.hue}°, noise σ ${p.noise_sigma} → darken ${fmt(p.adjust.strength, 2)}, noise ${fmt(p.adjust.noise, 2)} (±${Math.round(100 * (defaults?.preset_jitter ?? 0.1))}% per image)`}
                  >
                    {p.label}
                  </Button>
                ))}
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => void applyPreset("more")}
                  disabled={busy || running}
                  title="Next stronger preset: none → Dusk → Urban → Moonlit"
                >
                  Darken more (D)
                </Button>
                {adjust.preset && (
                  <span className="text-[var(--color-on-surface-secondary)]">
                    preset {adjust.preset} (jittered for this image)
                  </span>
                )}
              </div>
              <div className="grid grid-cols-1 gap-x-6 gap-y-1 md:grid-cols-2">
                <SliderEntry
                  label={`Darken ${adjust.strength.toFixed(2)}`}
                  value={adjust.strength}
                  onChange={(v) => patchAdjust({ strength: v })}
                  min={0}
                  max={1}
                  step={0.01}
                  tooltip="0 = unchanged. Exposure drop and gamma in linear light; clipped highlights keep part of their brightness."
                />
                <SliderEntry
                  label={`Noise ${adjust.noise.toFixed(2)}`}
                  value={adjust.noise}
                  onChange={(v) => patchAdjust({ noise: v })}
                  min={0}
                  max={1}
                  step={0.01}
                  tooltip="Sensor-style noise, strongest in the shadows, with mild chroma noise. Seeded from the file name, so the preview and the committed image get the same field. Applied last."
                />
                <SliderEntry
                  label={`Hue ${adjust.hue > 0 ? "+" : ""}${adjust.hue.toFixed(0)}°`}
                  value={adjust.hue}
                  onChange={(v) => patchAdjust({ hue: Math.round(v) })}
                  min={-HUE_RANGE}
                  max={HUE_RANGE}
                  step={1}
                  tooltip="Chroma rotation in Y'CbCr; luma untouched. 0 = unchanged."
                />
                <SliderEntry
                  label={`Saturation ×${adjust.saturation.toFixed(2)}`}
                  value={adjust.saturation}
                  onChange={(v) => patchAdjust({ saturation: v })}
                  min={0}
                  max={SATURATION_MAX}
                  step={0.01}
                  tooltip="Chroma multiplier. 1 = unchanged, 0 = greyscale."
                />
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="secondary" size="sm" onClick={back} disabled={busy || index === 0}>
                  <ChevronLeft className="h-4 w-4" /> Back
                </Button>
                <Button variant="primary" size="sm" onClick={() => void decide("accepted")} disabled={busy || running}>
                  <Check className="h-4 w-4" /> Accept (A / Enter)
                </Button>
                <Button variant="secondary" size="sm" onClick={() => void decide("skipped")} disabled={busy || running}>
                  <SkipForward className="h-4 w-4" /> Skip (S / Space)
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={undo}
                  disabled={busy || running || !current.decision || Boolean(current.decision.committed)}
                >
                  <Undo2 className="h-4 w-4" /> Undo decision (U)
                </Button>
                <Button variant="ghost" size="sm" onClick={() => step(1)} disabled={busy || index >= total - 1}>
                  Next <ChevronRight className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setAdjust({ ...NEUTRAL_ADJUST, strength: adjust.strength })}
                  disabled={busy}
                  title="Noise 0, hue 0, saturation 1; darken kept"
                >
                  Reset colour/noise (R)
                </Button>
                <div className="ml-auto flex items-center gap-3">
                  <Toggle
                    label="Caption"
                    value={settings.showCaption}
                    onChange={(v) => update({ showCaption: v })}
                    labelPosition="left"
                  />
                  <Toggle
                    label="Skip decided"
                    value={settings.onlyUndecided}
                    onChange={(v) => update({ onlyUndecided: v })}
                    labelPosition="left"
                    tooltip="After a decision, jump to the next undecided candidate."
                  />
                </div>
              </div>
              <div className="text-[11px] text-[var(--color-on-surface-secondary)]">
                Keys: A/Enter accept · S/Space skip · U undo · ←/Backspace back · → next · D darken more (next preset) ·
                1-9 darken 0.1-0.9, 0 = 1.0 · [ ] ±0.05 · R reset colour/noise · drag the divider to wipe. Order:
                darken, then hue/saturation, then noise.
              </div>
            </div>
          </SectionCard>
        )}

        {opened && total === 0 && scanInfo && (
          <div className="text-sm text-[var(--color-on-surface-secondary)]">No candidates in the last scan.</div>
        )}

        {opened && (
          <SectionCard title="Commit">
            <div className="flex flex-col gap-2">
              <div className="text-xs tabular-nums text-[var(--color-on-surface-secondary)]">
                {counts ? (
                  <>
                    {counts.accepted_uncommitted} accepted and waiting · {counts.skipped} skipped · {counts.undecided}{" "}
                    undecided · {counts.accepted - counts.accepted_uncommitted} already committed
                  </>
                ) : null}
              </div>
              <DirPicker
                label="Night Edits destination (adjusted = chosen, original = rejected)"
                value={settings.nightEditsDir}
                onChange={(v) => update({ nightEditsDir: v })}
                tooltip={`A separate DPO dataset with its own manifest; created on the first commit. Empty = ${state?.default_night_edits_dir ?? "<workspace>/DPO Night Edits"}. Folders the Optimise workspace already registers (Minor Edits, Off-policy, RLHF_8) are refused.`}
                disabled={running}
              />
              {!settings.nightEditsDir && state?.default_night_edits_dir && (
                <div className="text-[11px] text-[var(--color-on-surface-secondary)] break-all">
                  Default: {state.default_night_edits_dir}
                </div>
              )}
              <SliderEntry
                label={`Share of accepted images that also become Night Edits pairs (${pairsPlanned} now)`}
                value={settings.fraction}
                onChange={(v) => update({ fraction: clamp01(v) })}
                min={0}
                max={1}
                step={0.01}
                tooltip="The most changed images first, by mean CIELAB ΔE between original and result (darkening, colour and noise all count). Default 0.33."
                disabled={running}
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => setConfirmCommit(true)}
                  disabled={busy || running || !counts || counts.accepted_uncommitted === 0}
                >
                  Commit {counts?.accepted_uncommitted ?? 0} accepted…
                </Button>
                {state?.backup_dir && (
                  <span className="text-[11px] text-[var(--color-on-surface-secondary)] break-all">
                    Backups + manifest.csv: {state.backup_dir}
                  </span>
                )}
              </div>
              {confirmCommit && (
                <div className="flex flex-col gap-2 rounded border border-[var(--color-warning-500)] p-2 text-sm">
                  <div>
                    Replace {counts?.accepted_uncommitted ?? 0} chosen images in place with their adjusted versions
                    (originals backed up) and add {pairsPlanned} pairs to {nightEditsDir || "(no Night Edits folder)"}?
                  </div>
                  <div className="flex gap-2">
                    <Button variant="ghost" size="sm" onClick={() => setConfirmCommit(false)}>
                      Cancel
                    </Button>
                    <Button variant="danger" size="sm" onClick={commit}>
                      Commit
                    </Button>
                  </div>
                </div>
              )}
              {state?.commits && state.commits.length > 0 && (
                <div className="flex flex-col gap-1">
                  {state.commits
                    .slice()
                    .reverse()
                    .map((c) => (
                      <div
                        key={c.id}
                        className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded bg-[var(--color-surface-container)] px-2 py-1 text-xs tabular-nums"
                      >
                        <span className="font-mono">{c.id}</span>
                        <span>{when(c.at)}</span>
                        <span>
                          {c.items} adjusted · {c.pairs} pairs · mean ΔE {fmt(c.mean_delta_e, 1)}
                        </span>
                        {c.night_edits_dir && c.pairs > 0 && (
                          <span className="break-all text-[var(--color-on-surface-secondary)]">
                            {c.night_edits_dir}
                          </span>
                        )}
                        {c.error && <span className="text-[var(--color-error-500)]">error: {c.error}</span>}
                        {c.restored ? (
                          <span className="text-[var(--color-on-surface-secondary)]">
                            restored {when(c.restored_at)}
                          </span>
                        ) : (
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => void restore(c.id)}
                            disabled={busy || running}
                          >
                            <RotateCcw className="h-3 w-3" /> Restore
                          </Button>
                        )}
                        {c.restore_missing.length > 0 && (
                          <span className="text-[var(--color-warning-500)]">
                            {c.restore_missing.length} backups missing
                          </span>
                        )}
                      </div>
                    ))}
                </div>
              )}
            </div>
          </SectionCard>
        )}
      </div>
    </ModalBase>
  );
}
