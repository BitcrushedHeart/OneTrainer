import {
  AlertCircle,
  Ban,
  Check,
  Copy,
  Dices,
  Loader2,
  RefreshCw,
  Search,
  SkipForward,
  Trash2,
  Undo2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  dpoApi,
  dpoOffPolicyApi,
  type OffPolicyBuckets,
  type OffPolicyCandidate,
  type OffPolicyPhotoBrief,
  type OffPolicyStatus,
  type OffPolicySuggestion,
} from "@/api/dpoApi";
import { Button, DirPicker, FormEntry, SectionCard, Select } from "@/components/shared";

import { PreviewOverlay } from "./dpo/PreviewOverlay";
import { ModalBase } from "./ModalBase";

export interface OffPolicyDpoModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.offpolicyDpo.folders";

interface Folders {
  generated: string;
  real: string;
  output: string;
  tolerance: number;
  baseResolution: number;
  // Allowed photo upscale relative to its generation, in percent.
  upscalePct: number;
}

function loadFolders(): Folders {
  const fallback: Folders = {
    generated: "",
    real: "",
    output: "",
    tolerance: 0.15,
    baseResolution: 1024,
    upscalePct: 10,
  };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    return { ...fallback, ...(JSON.parse(raw) as Partial<Folders>) };
  } catch {
    return fallback;
  }
}

function saveFolders(f: Folders) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(f));
  } catch {
    /* storage unavailable — fine */
  }
}

const PHASE_LABEL: Record<OffPolicyStatus["phase"], string> = {
  idle: "Idle",
  indexing: "Reading real photos and captions…",
  scanning: "Reading generation metadata…",
  matching: "Matching captions…",
  scoring: "Picking negatives from cached realism scores…",
  ready: "All candidates scored",
  error: "Scan failed",
};

function pct(done: number, total: number) {
  return total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
}

export function OffPolicyDpoModal({ open, onClose }: OffPolicyDpoModalProps) {
  const [folders, setFolders] = useState<Folders>(loadFolders);
  const [active, setActive] = useState(false);
  const [status, setStatus] = useState<OffPolicyStatus | null>(null);
  const [candidate, setCandidate] = useState<OffPolicyCandidate | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [armDelete, setArmDelete] = useState(false);
  // Full-size preview of one side of the pair (click an image).
  const [preview, setPreview] = useState<string | null>(null);
  // undefined = nothing asked yet; [] = nothing left for that selection.
  const [suggestions, setSuggestions] = useState<OffPolicySuggestion[] | undefined>(undefined);
  const [remaining, setRemaining] = useState<number | null>(null);
  const [buckets, setBuckets] = useState<OffPolicyBuckets | null>(null);
  const [bucket, setBucket] = useState<string>("");
  const [suggestCount, setSuggestCount] = useState<number>(10);
  const [keywords, setKeywords] = useState<string>("");
  const bucketsLoadedRef = useRef(false);
  const [suggesting, setSuggesting] = useState(false);
  const [lookupName, setLookupName] = useState("");
  const [lookupResults, setLookupResults] = useState<OffPolicyPhotoBrief[] | null>(null);
  const [lookupNote, setLookupNote] = useState<string | null>(null);
  const [lookingUp, setLookingUp] = useState(false);
  const pollRef = useRef<number | null>(null);
  const fetchingRef = useRef(false);
  // Mirror `done` / `candidate` for the poll interval, whose closure is created once.
  const doneRef = useRef(false);
  const candidateRef = useRef<OffPolicyCandidate | null>(null);
  candidateRef.current = candidate;

  const updateFolders = useCallback((patch: Partial<Folders>) => {
    setFolders((prev) => {
      const next = { ...prev, ...patch };
      saveFolders(next);
      return next;
    });
  }, []);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchNext = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    try {
      const res = await dpoOffPolicyApi.next();
      if (!res.ok) {
        setError(res.error ?? "Failed to fetch next candidate");
        return;
      }
      if (res.candidate) {
        setCandidate(res.candidate);
        setArmDelete(false);
        setPreview(null);
      } else if (res.done) {
        setCandidate(null);
        setDone(true);
        doneRef.current = true;
      }
      // waiting: the poll will call again once something is ready
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      fetchingRef.current = false;
    }
  }, []);

  const loadBuckets = useCallback(async (filter = "") => {
    try {
      const res = await dpoOffPolicyApi.buckets(filter);
      if (res.ok && !res.waiting && res.buckets) {
        setBuckets(res.buckets);
        setRemaining(res.remaining ?? null);
      }
    } catch {
      /* transient */
    }
  }, []);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      void dpoOffPolicyApi
        .status()
        .then((s) => {
          setStatus(s);
          if (!s.active) {
            clearPoll();
            setActive(false);
            return;
          }
          if (s.error) setError(s.error);
          if (s.phase !== "indexing" && s.phase !== "scanning" && !bucketsLoadedRef.current) {
            bucketsLoadedRef.current = true;
            void loadBuckets();
          }
          if (s.has_candidate && !candidateRef.current) {
            // The backend is holding a candidate this UI never saw (reload,
            // reopen after a hot swap): show it rather than waiting forever.
            void dpoOffPolicyApi
              .current()
              .then((c) => {
                if (c.ok && c.candidate) {
                  setCandidate(c.candidate);
                  setArmDelete(false);
                }
              })
              .catch(() => {
                /* transient */
              });
            return;
          }
          // Keep fetching while there is no candidate on screen.
          if (!s.has_candidate && !doneRef.current && (s.ready > 0 || s.worker_finished)) {
            void fetchNext();
          }
        })
        .catch(() => {
          /* transient */
        });
    }, 700);
  }, [clearPoll, fetchNext, loadBuckets]);

  // Reflect a session that is already running when the modal is reopened.
  useEffect(() => {
    if (!open) {
      clearPoll();
      return;
    }
    let cancelled = false;
    void dpoOffPolicyApi.status().then((s) => {
      if (cancelled) return;
      setStatus(s);
      if (s.active) {
        setActive(true);
        setDone(false);
        doneRef.current = false;
        startPolling();
      }
    });
    return () => {
      cancelled = true;
    };
  }, [open, clearPoll, startPolling]);

  useEffect(() => () => clearPoll(), [clearPoll]);

  const start = useCallback(async () => {
    setError(null);
    setNotice(null);
    setDone(false);
    doneRef.current = false;
    setCandidate(null);
    setSuggestions(undefined);
    setBuckets(null);
    bucketsLoadedRef.current = false;
    setBusy(true);
    try {
      const res = await dpoOffPolicyApi.start(
        folders.generated,
        folders.real,
        folders.output,
        folders.tolerance,
        folders.baseResolution,
        1 + Math.max(0, Number(folders.upscalePct) || 0) / 100,
      );
      if (!res.ok) {
        setError(res.error ?? "Could not start");
        return;
      }
      setActive(true);
      const notes: string[] = [];
      if ((res.existing_pairs ?? 0) > 0)
        notes.push(`Resuming — ${res.existing_pairs} pair(s) already in the output folder.`);
      if ((res.cache_imported ?? 0) > 0) {
        notes.push(`Reusing ${res.cache_imported} scan-cache entries from your other DPO concept(s).`);
      }
      if ((res.permanently_skipped ?? 0) > 0) {
        notes.push(`${res.permanently_skipped} photo(s) on the "never again" list are excluded.`);
      }
      if (notes.length) setNotice(notes.join(" "));
      startPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [folders, startPolling]);

  const finish = useCallback(async () => {
    clearPoll();
    try {
      await dpoOffPolicyApi.cancel();
    } catch {
      /* best effort */
    }
    setActive(false);
    setCandidate(null);
    setDone(false);
    setSuggestions(undefined);
    setBuckets(null);
    bucketsLoadedRef.current = false;
  }, [clearPoll]);

  const act = useCallback(
    async (action: () => Promise<{ ok: boolean; error?: string }>) => {
      if (busy || !candidate) return;
      setBusy(true);
      setError(null);
      try {
        const res = await action();
        if (!res.ok) {
          setError(res.error ?? "Action failed");
          return;
        }
        setCandidate(null);
        setArmDelete(false);
        await fetchNext();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, candidate, fetchNext],
  );

  const confirm = useCallback(() => act(() => dpoOffPolicyApi.confirm()), [act]);
  const skip = useCallback(() => act(() => dpoOffPolicyApi.skip()), [act]);
  const skipForever = useCallback(() => act(() => dpoOffPolicyApi.skipForever()), [act]);
  const reroll = useCallback(async () => {
    if (busy || !candidate || candidate.generated_count <= 1) return;
    setBusy(true);
    setError(null);
    try {
      const res = await dpoOffPolicyApi.reroll();
      if (!res.ok) {
        setNotice(res.error ?? "Could not re-roll");
        return;
      }
      if (res.candidate) {
        setCandidate(res.candidate);
        setPreview(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [busy, candidate]);

  const removeCaption = useCallback(() => {
    if (!armDelete) {
      setArmDelete(true);
      return;
    }
    void act(() => dpoOffPolicyApi.removeCaption());
  }, [act, armDelete]);

  const undo = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await dpoOffPolicyApi.undo();
      if (!res.ok) {
        setNotice(res.error ?? "Nothing to undo");
        return;
      }
      if (res.candidate) {
        setCandidate(res.candidate);
        setDone(false);
        doneRef.current = false;
        setArmDelete(false);
        setNotice(`Undid ${res.undone ?? "last action"}.`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [busy]);

  const suggest = useCallback(async () => {
    setSuggesting(true);
    try {
      const n = Math.max(1, Math.min(100, Math.round(Number(suggestCount)) || 1));
      const res = await dpoOffPolicyApi.suggest(bucket || null, n, keywords);
      if (!res.ok) {
        setError(res.error ?? "Could not pick captions");
        return;
      }
      if (res.waiting) {
        setNotice("Still reading the folders — try again in a moment.");
        return;
      }
      setSuggestions(res.suggestions ?? []);
      setRemaining(res.remaining ?? null);
      if (res.buckets) setBuckets(res.buckets);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSuggesting(false);
    }
  }, [bucket, suggestCount, keywords]);

  const lookup = useCallback(async () => {
    const name = lookupName.trim();
    if (!name) return;
    setLookingUp(true);
    setLookupNote(null);
    try {
      const res = await dpoOffPolicyApi.lookup(name);
      if (!res.ok) {
        setLookupNote(res.error ?? "Lookup failed");
        return;
      }
      if (res.waiting) {
        setLookupNote("Still reading the real photo folder — try again in a moment.");
        return;
      }
      setLookupResults(res.matches ?? []);
      if (res.generations_scanned === false) {
        setLookupNote("Generations are still being scanned; generation status may be incomplete.");
      }
    } catch (e) {
      setLookupNote(e instanceof Error ? e.message : String(e));
    } finally {
      setLookingUp(false);
    }
  }, [lookupName]);

  const copy = useCallback((text: string) => {
    void navigator.clipboard?.writeText(text).catch(() => {
      /* clipboard blocked */
    });
  }, []);

  // Keyboard: Enter/Y confirm, S skip, D delete (twice), N other generation, R re-roll suggestion, Ctrl/Cmd+Z undo.
  useEffect(() => {
    if (!open || !active) return;
    const handler = (e: KeyboardEvent) => {
      if (preview) return;
      const target = e.target as HTMLElement | null;
      // Typing must never trigger a shortcut.
      if (target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable)) return;
      // A focused button keeps Enter and Space for its own activation (Enter
      // on a focused Skip must skip, not also confirm), but the letter keys
      // are ours: clicking any action button leaves focus on it, and without
      // this the next X / S / D press would silently do nothing.
      if (target?.tagName === "BUTTON" && (e.key === "Enter" || e.key === " " || e.key === "Spacebar")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        void undo();
        return;
      }
      // Other modifier chords (Ctrl+C, Alt+Tab, ...) are not ours.
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "Enter" || e.key.toLowerCase() === "y") {
        e.preventDefault();
        void confirm();
      } else if (e.key.toLowerCase() === "s") {
        e.preventDefault();
        void skip();
      } else if (e.key.toLowerCase() === "x") {
        e.preventDefault();
        void skipForever();
      } else if (e.key.toLowerCase() === "d") {
        e.preventDefault();
        removeCaption();
      } else if (e.key.toLowerCase() === "r") {
        e.preventDefault();
        void suggest();
      } else if (e.key.toLowerCase() === "n") {
        e.preventDefault();
        void reroll();
      } else if (e.key === "Escape") {
        setArmDelete(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, active, confirm, skip, skipForever, removeCaption, suggest, undo, reroll, preview]);

  const scanning = status ? !status.worker_finished : false;

  return (
    <ModalBase open={open} onClose={onClose} title="Off-policy DPO Builder" size="full" closeOnEscape={!armDelete}>
      <div className="flex flex-col gap-4">
        {!active && (
          <>
            <div className="text-sm text-[var(--color-on-surface-secondary)]">
              Finds real photos whose caption (any line of the sidecar, matched in full) has a generated image in the
              generated folder, picks the most photorealistic generation as the hardest negative, and asks you to
              confirm the caption fits the photo. Confirmed pairs export photo-as-chosen / generation-as-rejected into
              the output folder. Captions that already have a pair in any configured DPO concept, or in the output
              folder, are excluded, so the tool can be re-run as the generated folder grows.
            </div>
            <SectionCard title="Folders">
              <div className="space-y-3">
                <DirPicker
                  label="Generated images"
                  value={folders.generated}
                  onChange={(v) => updateFolders({ generated: v })}
                />
                <DirPicker
                  label="Real photos (with .txt captions)"
                  value={folders.real}
                  onChange={(v) => updateFolders({ real: v })}
                />
                <DirPicker
                  label="Output folder (off-policy DPO concept)"
                  value={folders.output}
                  onChange={(v) => updateFolders({ output: v })}
                />
              </div>
            </SectionCard>
            <div className="grid grid-cols-5 gap-3 items-end">
              <FormEntry
                label="Aspect tolerance (0–1)"
                type="number"
                value={folders.tolerance}
                onChange={(v) => updateFolders({ tolerance: Number(v) })}
                tooltip="How far the generation's aspect may differ from the photo's, as a fraction. The trainer cover-scales the generation (aspect kept) and crops it to the photo's crop box, so this bounds how much of its long axis is trimmed — generate at the suggested size to keep it at 0."
              />
              <FormEntry
                label="Allowed photo upscale (%)"
                type="number"
                value={folders.upscalePct}
                onChange={(v) => updateFolders({ upscalePct: Number(v) })}
                tooltip="How much smaller than its generation a photo may be, per axis, and still be paired: 10 means the photo may be up to 10% smaller (it gets upscaled that much at train time). Generators emit fixed sizes that equal the training crop, and many photos sit a few percent under that."
              />
              <FormEntry
                label="Generation base resolution"
                type="number"
                value={folders.baseResolution}
                onChange={(v) => updateFolders({ baseResolution: Number(v) })}
                tooltip="Pixel budget (side of a square) used for the suggested generation size."
              />
              <div className="col-span-2 flex items-end">
                <Button
                  variant="primary"
                  onClick={() => void start()}
                  disabled={busy || !folders.generated || !folders.real || !folders.output}
                >
                  {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                  Start
                </Button>
              </div>
            </div>
          </>
        )}

        {active && status && !scanning && (
          <details className="bg-[var(--color-surface-container)] rounded px-3 py-2 text-xs tabular-nums">
            <summary className="cursor-pointer select-none flex flex-wrap items-center gap-x-4 text-[var(--color-on-surface-secondary)]">
              <span className="text-[var(--color-on-surface)] font-semibold">
                confirmed {status.confirmed} · skipped {status.skipped} · never again {status.skipped_forever} ·
                captions fixed {status.captions_removed}
              </span>
              <span>{status.candidates} candidates</span>
              <span>{status.total_pairs} pairs in output</span>
              {status.unscored > 0 && <span>{status.unscored} unscored</span>}
              <span className="ml-auto">scan details…</span>
            </summary>
            <div className="pt-2 flex flex-wrap gap-x-4 text-[var(--color-on-surface-secondary)]">
              <span>{status.photos.toLocaleString()} photos</span>
              <span>{status.generated.toLocaleString()} generations</span>
              <span>{status.excluded_captions} captions already paired</span>
              {status.unscored > 0 && <span>{status.unscored} unscored (random negative)</span>}
              {status.too_small > 0 && <span>{status.too_small} photos smaller than their generation (excluded)</span>}
              {status.permanently_skipped > 0 && <span>{status.permanently_skipped} permanently skipped photos</span>}
              {status.cache_imported > 0 && <span>{status.cache_imported} scan-cache entries reused</span>}
            </div>
            {notice && <div className="pt-1 text-[var(--color-on-surface-secondary)]">{notice}</div>}
            {status.warning && <div className="pt-1 text-[var(--color-warning-500,#eab308)]">{status.warning}</div>}
          </details>
        )}

        {active && status && scanning && (
          <div className="flex flex-col gap-2 bg-[var(--color-surface-container)] rounded p-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                {scanning && <Loader2 className="w-4 h-4 animate-spin" />}
                {PHASE_LABEL[status.phase]}
              </span>
              {scanning && (
                <span className="tabular-nums text-[var(--color-on-surface-secondary)]">
                  {status.phase === "indexing" && status.progress_total === 0
                    ? `listing folders… ${status.listed.toLocaleString()} files`
                    : `${status.progress_done.toLocaleString()} / ${status.progress_total.toLocaleString()} · ${pct(status.progress_done, status.progress_total)}%`}
                </span>
              )}
            </div>
            {scanning && (
              <div className="h-2 rounded bg-[var(--color-border-subtle)] overflow-hidden">
                <div
                  className="h-full bg-[var(--color-cobalt-600)] transition-all"
                  style={{ width: `${pct(status.progress_done, status.progress_total)}%` }}
                />
              </div>
            )}
            <div className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums flex flex-wrap gap-x-4">
              <span>{status.photos} photos</span>
              <span>{status.generated} generations</span>
              <span>{status.excluded_captions} captions already paired</span>
              {status.unscored > 0 && <span>{status.unscored} unscored (random negative)</span>}
              {status.too_small > 0 && <span>{status.too_small} photos smaller than their generation (excluded)</span>}
              <span>{status.candidates} candidates</span>
              <span className="text-[var(--color-on-surface)]">
                confirmed {status.confirmed} · skipped {status.skipped} · never again {status.skipped_forever} ·
                captions fixed {status.captions_removed}
              </span>
              <span>{status.total_pairs} pairs in output</span>
            </div>
            {status.warning && <div className="text-xs text-[var(--color-warning-500,#eab308)]">{status.warning}</div>}
          </div>
        )}

        {notice && (!active || scanning) && (
          <div className="text-sm text-[var(--color-on-surface-secondary)]">{notice}</div>
        )}
        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)] bg-[var(--color-surface-container)] rounded p-3">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        {active && candidate && (
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3 h-[55vh] min-h-[320px]">
              <figure className="flex flex-col min-h-0 bg-[var(--color-surface-container)] rounded p-2">
                <figcaption className="text-xs font-semibold mb-1 text-[var(--color-success-500)]">
                  CHOSEN — real photo · {candidate.real_size[0]}×{candidate.real_size[1]} · bucket{" "}
                  {candidate.aspect_bucket}
                  {candidate.real_score != null && ` · P(AI) ${candidate.real_score.toFixed(2)}`}
                </figcaption>
                <img
                  src={dpoApi.imageUrl(candidate.real_path)}
                  alt="real photo"
                  title="Click to expand"
                  className="flex-1 min-h-0 object-contain w-full cursor-zoom-in"
                  onClick={() => setPreview(candidate.real_path)}
                />
                <div
                  className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                  title={candidate.real_path}
                >
                  {candidate.real_path}
                </div>
              </figure>
              <figure className="flex flex-col min-h-0 bg-[var(--color-surface-container)] rounded p-2">
                <figcaption className="text-xs font-semibold mb-1 text-[var(--color-error-500)]">
                  REJECTED — generation
                  {candidate.rejected_size && ` · ${candidate.rejected_size[0]}×${candidate.rejected_size[1]}`}
                  {candidate.rejected_score != null && ` · P(AI) ${candidate.rejected_score.toFixed(2)}`}
                  {candidate.aspect_mismatch != null &&
                    ` · aspect Δ ${(candidate.aspect_mismatch * 100).toFixed(0)}% (cropped)`}
                  {candidate.generated_count > 1 &&
                    ` · ${candidate.scored ? "hardest" : "random"} of ${candidate.generated_count} (${candidate.shown_count} seen)`}
                </figcaption>
                {!candidate.scored && (
                  <div className="mb-1 rounded px-2 py-1 text-xs font-bold tracking-wide text-center bg-[var(--color-warning-500,#eab308)] text-black">
                    UNSCORED — no cached realism score; this generation was picked at random
                  </div>
                )}
                {candidate.rejected_path && (
                  <img
                    src={dpoApi.imageUrl(candidate.rejected_path)}
                    alt="generated image"
                    title="Click to expand"
                    className="flex-1 min-h-0 object-contain w-full cursor-zoom-in"
                    onClick={() => candidate.rejected_path && setPreview(candidate.rejected_path)}
                  />
                )}
                <div
                  className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                  title={candidate.rejected_path ?? ""}
                >
                  {candidate.rejected_path}
                </div>
              </figure>
            </div>

            {candidate.real_score != null &&
              candidate.rejected_score != null &&
              candidate.real_score > candidate.rejected_score && (
                <div className="flex items-center gap-2 text-xs text-[var(--color-warning-500,#eab308)]">
                  <AlertCircle className="w-4 h-4" />
                  The realism head rates the photo as more AI-looking than the generation. Check it really is a photo
                  before exporting it as chosen.
                </div>
              )}
            <div className="bg-[var(--color-surface-container)] rounded p-3">
              <div className="text-xs text-[var(--color-on-surface-secondary)] mb-1">
                Does this caption describe the photo? (candidate {candidate.index} of {candidate.total})
              </div>
              <div className="text-sm text-[var(--color-on-surface)] whitespace-pre-wrap">{candidate.caption}</div>
              {candidate.caption_lines.length > 1 && (
                <details className="mt-1 text-xs text-[var(--color-on-surface-secondary)]">
                  <summary>{candidate.caption_lines.length - 1} other caption line(s) on this photo</summary>
                  <ul className="list-disc ml-4 mt-1">
                    {candidate.caption_lines
                      .filter((l) => l !== candidate.caption)
                      .map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                  </ul>
                </details>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button variant="primary" onClick={() => void confirm()} disabled={busy}>
                <Check className="w-4 h-4" /> Caption fits — export pair (Enter)
              </Button>
              <Button variant="secondary" onClick={() => void skip()} disabled={busy}>
                <SkipForward className="w-4 h-4" /> Skip this session (S)
              </Button>
              <Button
                variant="secondary"
                onClick={() => void skipForever()}
                disabled={busy}
                title="Never offer this photo again for this output folder (saved to disk)"
              >
                <Ban className="w-4 h-4" /> Never again (X)
              </Button>
              <Button
                variant="secondary"
                onClick={() => void reroll()}
                disabled={busy || candidate.generated_count <= 1}
                title={
                  candidate.generated_count > 1
                    ? "Show a different generation of this photo's captions"
                    : "This photo has only one usable generation"
                }
              >
                <RefreshCw className="w-4 h-4" /> Other generation (N)
              </Button>
              <Button variant="danger" onClick={removeCaption} disabled={busy}>
                <Trash2 className="w-4 h-4" />
                {armDelete
                  ? candidate.caption_lines.length > 1
                    ? "Press again to remove this line from the .txt (photo kept)"
                    : "Press again to remove the .txt (its only line; photo kept)"
                  : "Caption wrong — remove this line (D)"}
              </Button>
              {armDelete && <span className="text-xs text-[var(--color-on-surface-secondary)]">Esc to cancel</span>}
              <Button variant="ghost" onClick={() => void undo()} disabled={busy || !status?.can_undo}>
                <Undo2 className="w-4 h-4" />
                {status?.undo_action ? `Undo ${status.undo_action} (Ctrl+Z)` : "Undo (Ctrl+Z)"}
              </Button>
            </div>
          </div>
        )}

        {active && !candidate && !done && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-on-surface-secondary)]">
            <Loader2 className="w-4 h-4 animate-spin" /> Waiting for the next candidate…
          </div>
        )}

        {active && !candidate && status?.can_undo && (
          <div>
            <Button variant="ghost" onClick={() => void undo()} disabled={busy}>
              <Undo2 className="w-4 h-4" />
              {status.undo_action ? `Undo ${status.undo_action} (Ctrl+Z)` : "Undo (Ctrl+Z)"}
            </Button>
          </div>
        )}

        {active && done && (
          <div className="text-sm text-[var(--color-on-surface)] bg-[var(--color-surface-container)] rounded p-3">
            No more candidates. Generate images for more captions (use the suggester below), then start again — the scan
            is cached, so a re-run only reads new files.
          </div>
        )}

        {active && (
          <details className="rounded border border-[var(--color-border-subtle)] px-3 py-2">
            <summary className="cursor-pointer select-none text-sm font-semibold text-[var(--color-on-surface)]">
              What to generate next
            </summary>
            <div className="flex flex-col gap-2 text-sm pt-2">
              <div className="grid grid-cols-5 gap-3 items-end">
                <div
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      e.stopPropagation();
                      void loadBuckets(keywords);
                      void suggest();
                    }
                  }}
                  onBlur={() => void loadBuckets(keywords)}
                >
                  <FormEntry
                    label="Caption keywords"
                    value={keywords}
                    onChange={(v) => setKeywords(String(v))}
                    placeholder="comma-separated, all required"
                    tooltip="Only captions containing every term (case-insensitive) are offered; bucket counts follow the filter."
                  />
                </div>
                <Select
                  label="Generate at ratio"
                  value={bucket}
                  onChange={setBucket}
                  options={[
                    { label: remaining != null ? `Any (${remaining} photos)` : "Any", value: "" },
                    ...Object.entries(buckets ?? {}).map(([label, info]) => ({
                      label: `${label} — ${info.photos} photo${info.photos === 1 ? "" : "s"} · ${info.generate_size[0]}×${info.generate_size[1]}`,
                      value: label,
                    })),
                  ]}
                  formatLabel={(v) => v}
                />
                <FormEntry
                  label="How many"
                  type="number"
                  value={suggestCount}
                  onChange={(v) => setSuggestCount(Number(v))}
                />
                <div className="col-span-2 flex items-end gap-2">
                  <Button variant="secondary" onClick={() => void suggest()} disabled={suggesting}>
                    {suggesting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Dices className="w-4 h-4" />}
                    {suggestions === undefined ? "Load captions (R)" : "Re-roll (R)"}
                  </Button>
                  {bucket && buckets?.[bucket] && (
                    <span className="text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
                      batch this bucket at{" "}
                      <span className="font-semibold text-[var(--color-on-surface)]">
                        {buckets[bucket].generate_size[0]}×{buckets[bucket].generate_size[1]}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => copy(`${buckets[bucket].generate_size[0]}x${buckets[bucket].generate_size[1]}`)}
                        title="Copy bucket size"
                      >
                        <Copy className="w-3 h-3" />
                      </Button>
                    </span>
                  )}
                </div>
              </div>
              <div className="text-xs text-[var(--color-on-surface-secondary)]">
                Random photos with no usable generation yet, one caption each, and the size to generate at. Sizes use
                only ratios the generator can produce (1:1, 4:3, 3:2, 8:5, 16:9, 21:9 and their portrait inverses),
                picking the one nearest each photo; pick a ratio above to batch a whole group at one size. Photos too
                small for their generation size are never suggested.
              </div>
              <div className="flex items-end gap-2 pt-1">
                <div
                  className="flex-1"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      e.stopPropagation();
                      void lookup();
                    }
                  }}
                >
                  <FormEntry
                    label="Or look up a photo by file name"
                    value={lookupName}
                    onChange={(v) => setLookupName(String(v))}
                    placeholder="name.jpg, name, or sub/folder/name.jpg — searched recursively"
                  />
                </div>
                <Button
                  variant="secondary"
                  onClick={() => void lookup()}
                  disabled={lookingUp || !lookupName.trim()}
                  onKeyDown={(e) => e.stopPropagation()}
                >
                  {lookingUp ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                  Look up
                </Button>
              </div>
              {lookupNote && <div className="text-xs text-[var(--color-on-surface-secondary)]">{lookupNote}</div>}
              {lookupResults && lookupResults.length === 0 && (
                <div className="text-xs text-[var(--color-on-surface-secondary)]">
                  No indexed photo with that name (photos without a caption sidecar are not indexed).
                </div>
              )}
              {lookupResults && lookupResults.length > 0 && (
                <div className="flex flex-col gap-2">
                  {lookupResults.map((m) => (
                    <div
                      key={m.real_path}
                      className="flex flex-col gap-1 bg-[var(--color-surface-container)] rounded p-3"
                    >
                      <div className="flex flex-wrap items-center gap-x-4 text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
                        <span>
                          Generate at{" "}
                          <span className="font-semibold text-[var(--color-on-surface)]">
                            {m.generate_size[0]}×{m.generate_size[1]}
                          </span>{" "}
                          ({m.generate_aspect}, trains in {m.aspect_bucket})
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copy(`${m.generate_size[0]}x${m.generate_size[1]}`)}
                          title="Copy size"
                        >
                          <Copy className="w-3 h-3" />
                        </Button>
                        <span>
                          photo {m.real_size[0]}×{m.real_size[1]}
                        </span>
                        {m.aspect_delta != null && m.aspect_delta > 0.005 && (
                          <span>aspect Δ {(m.aspect_delta * 100).toFixed(0)}%</span>
                        )}
                        {m.paired && <span className="text-[var(--color-success-500)]">already paired</span>}
                        {m.skipped && <span>skipped</span>}
                        {m.has_usable_generation && !m.paired && <span>has a usable generation</span>}
                        {m.too_small_for_suggested_size && (
                          <span className="text-[var(--color-error-500)]">
                            photo is smaller than this size — it can&apos;t be paired at it
                          </span>
                        )}
                      </div>
                      {m.caption_lines.map((line, i) => (
                        <div key={i} className="flex items-start gap-2 text-sm">
                          <span
                            className={
                              m.lines_needing_generation.includes(line)
                                ? "text-[var(--color-on-surface)]"
                                : "text-[var(--color-on-surface-secondary)] line-through"
                            }
                          >
                            {line}
                          </span>
                          <Button variant="ghost" size="sm" onClick={() => copy(line)} title="Copy caption">
                            <Copy className="w-3 h-3" />
                          </Button>
                        </div>
                      ))}
                      <div
                        className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                        title={m.real_path}
                      >
                        {m.real_path}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {suggestions && suggestions.length === 0 && (
                <div className="text-[var(--color-on-surface-secondary)]">
                  {bucket
                    ? `No photo in bucket ${bucket}${keywords.trim() ? " matching those keywords" : ""} still needs a generation.`
                    : keywords.trim()
                      ? "No remaining caption matches those keywords."
                      : "Every remaining caption already has a generation or a pair."}
                </div>
              )}
              {suggestions && suggestions.length > 0 && (
                <div className="flex flex-col gap-2">
                  {suggestions.map((sg) => (
                    <div
                      key={sg.real_path}
                      className="flex flex-col gap-1 bg-[var(--color-surface-container)] rounded p-3"
                    >
                      <div className="flex items-start gap-2">
                        <div className="flex-1 whitespace-pre-wrap text-[var(--color-on-surface)]">{sg.caption}</div>
                        <Button variant="ghost" size="sm" onClick={() => copy(sg.caption)} title="Copy caption">
                          <Copy className="w-4 h-4" />
                        </Button>
                      </div>
                      <div className="flex flex-wrap items-center gap-x-4 text-xs text-[var(--color-on-surface-secondary)] tabular-nums">
                        <span>
                          Generate at{" "}
                          <span className="font-semibold text-[var(--color-on-surface)]">
                            {sg.generate_size[0]}×{sg.generate_size[1]}
                          </span>{" "}
                          ({sg.generate_aspect})
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copy(`${sg.generate_size[0]}x${sg.generate_size[1]}`)}
                          title="Copy size"
                        >
                          <Copy className="w-3 h-3" />
                        </Button>
                        <span>
                          photo {sg.real_size[0]}×{sg.real_size[1]}
                        </span>
                        {sg.aspect_delta != null && sg.aspect_delta > 0.005 && (
                          <span className={sg.within_tolerance ? "" : "text-[var(--color-warning-500,#eab308)]"}>
                            aspect Δ {(sg.aspect_delta * 100).toFixed(0)}%
                            {sg.within_tolerance ? "" : " — over tolerance, pair may be rejected"}
                          </span>
                        )}
                      </div>
                      <div
                        className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                        title={sg.real_path}
                      >
                        {sg.real_path}
                      </div>
                    </div>
                  ))}
                  {remaining != null && (
                    <div className="text-xs text-[var(--color-on-surface-secondary)]">
                      {remaining} photo(s) still without a usable generation across all buckets.
                    </div>
                  )}
                </div>
              )}
            </div>
          </details>
        )}

        {active && (
          <div className="flex justify-end">
            <Button variant="secondary" onClick={() => void finish()}>
              Finish session
            </Button>
          </div>
        )}
      </div>
      {preview && <PreviewOverlay path={preview} onClose={() => setPreview(null)} />}
    </ModalBase>
  );
}
