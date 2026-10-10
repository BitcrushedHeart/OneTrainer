import { AlertCircle, Ban, Check, ChevronsRight, Loader2, RefreshCw, SkipForward, Undo2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { type Character, characterDpoApi, type OffPolicyPickCandidate } from "@/api/characterDpoApi";
import { dpoApi, dpoOffPolicyApi, type OffPolicyCandidate, type OffPolicyStatus } from "@/api/dpoApi";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "../dpo/PreviewOverlay";
import { Bar, baseName, Chip, errorText, HINT, joinPath, type OffPolicyDefaults, SECTION } from "./ui";

const PHASE_LABEL: Record<OffPolicyStatus["phase"], string> = {
  idle: "Idle",
  indexing: "Reading real photos and captions…",
  scanning: "Reading generation metadata…",
  matching: "Matching captions…",
  scoring: "Picking negatives from cached realism scores…",
  ready: "All candidates scored",
  error: "Scan failed",
};

/** Tolerates a backend that does not (yet) send the picker fields. */
function asPick(c: OffPolicyCandidate | OffPolicyPickCandidate): OffPolicyPickCandidate {
  const partial = c as Partial<OffPolicyPickCandidate>;
  const paths = Array.isArray(partial.generated_paths)
    ? partial.generated_paths
    : c.rejected_path
      ? [c.rejected_path]
      : [];
  const idx =
    typeof partial.rejected_index === "number" ? partial.rejected_index : paths.indexOf(c.rejected_path ?? "");
  return { ...c, generated_paths: paths, rejected_index: idx };
}

export interface RejectedPickerProps {
  /** Checked characters that have generations, in list order. */
  characters: Character[];
  defaults: OffPolicyDefaults;
  /** Called after each character's session ends (pairs on disk changed). */
  onCharacterDone: () => void;
}

type Phase = "starting" | "session" | "all-done";

export function RejectedPicker({ characters, defaults, onCharacterDone }: RejectedPickerProps) {
  const [charIdx, setCharIdx] = useState(0);
  const [phase, setPhase] = useState<Phase>(characters.length > 0 ? "starting" : "all-done");
  const [status, setStatus] = useState<OffPolicyStatus | null>(null);
  const [candidate, setCandidate] = useState<OffPolicyPickCandidate | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState<Record<string, number>>({});
  const pollRef = useRef<number | null>(null);
  const fetchingRef = useRef(false);
  const candidateRef = useRef<OffPolicyPickCandidate | null>(null);
  candidateRef.current = candidate;
  const endedRef = useRef(false);
  const charIdxRef = useRef(charIdx);
  charIdxRef.current = charIdx;
  const charactersRef = useRef(characters);
  charactersRef.current = characters;
  const onCharacterDoneRef = useRef(onCharacterDone);
  onCharacterDoneRef.current = onCharacterDone;

  const character = characters[charIdx] ?? null;
  const totalConfirmed = Object.values(confirmed).reduce((a, b) => a + b, 0);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  /** Ends the current session and moves on to the next checked character. */
  const advance = useCallback(async () => {
    if (endedRef.current) return;
    endedRef.current = true;
    clearPoll();
    try {
      await dpoOffPolicyApi.cancel();
    } catch {
      /* best effort */
    }
    setCandidate(null);
    setStatus(null);
    onCharacterDoneRef.current();
    const next = charIdxRef.current + 1;
    if (next >= charactersRef.current.length) {
      setPhase("all-done");
    } else {
      setCharIdx(next);
      setPhase("starting");
    }
  }, [clearPoll]);

  const fetchNext = useCallback(async () => {
    if (fetchingRef.current || endedRef.current) return;
    fetchingRef.current = true;
    try {
      const res = await dpoOffPolicyApi.next();
      if (!res.ok) {
        setError(res.error ?? "Failed to fetch next candidate");
        return;
      }
      if (res.candidate) {
        setCandidate(asPick(res.candidate));
        setPreview(null);
      } else if (res.done) {
        void advance();
      }
      // waiting: the poll calls again once something is ready
    } catch (e) {
      setError(errorText(e));
    } finally {
      fetchingRef.current = false;
    }
  }, [advance]);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      void dpoOffPolicyApi
        .status()
        .then((s) => {
          if (endedRef.current) return;
          setStatus(s);
          if (!s.active) {
            clearPoll();
            return;
          }
          if (s.error) setError(s.error);
          if (s.has_candidate && !candidateRef.current) {
            void dpoOffPolicyApi
              .current()
              .then((c) => {
                if (c.ok && c.candidate && !endedRef.current) setCandidate(asPick(c.candidate));
              })
              .catch(() => {
                /* transient */
              });
            return;
          }
          if (!s.has_candidate && (s.ready > 0 || s.worker_finished)) void fetchNext();
        })
        .catch(() => {
          /* transient */
        });
    }, 700);
  }, [clearPoll, fetchNext]);

  // Start a session for the current character.
  useEffect(() => {
    if (phase !== "starting" || !character) return;
    let cancelled = false;
    endedRef.current = false;
    setError(null);
    setNotice(null);
    setCandidate(null);
    setStatus(null);
    void (async () => {
      try {
        // A stale session (another tool, or this picker before a re-mount) must not answer our polls.
        await dpoOffPolicyApi.cancel().catch(() => undefined);
        const res = await characterDpoApi.offPolicyStart({
          generated_folder: joinPath(character.path, "dpo", "generated"),
          real_folder: joinPath(character.path, "train"),
          output_dir: joinPath(character.path, "dpo", "offpolicy"),
          aspect_tolerance: defaults.tolerance,
          max_upscale: 1 + Math.max(0, Number(defaults.upscalePct) || 0) / 100,
          base_resolution: defaults.baseResolution,
          order: defaults.order,
        });
        if (cancelled) return;
        if (!res.ok) {
          setError(`${character.name}: ${res.error ?? "could not start"}`);
          return;
        }
        const notes: string[] = [];
        if ((res.existing_pairs ?? 0) > 0) notes.push(`${res.existing_pairs} pair(s) already in the output folder.`);
        if ((res.permanently_skipped ?? 0) > 0) notes.push(`${res.permanently_skipped} photo(s) on the never list.`);
        if (notes.length) setNotice(notes.join(" "));
        setPhase("session");
        startPolling();
      } catch (e) {
        if (!cancelled) setError(errorText(e));
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, charIdx]);

  // Leaving the step ends the session.
  useEffect(
    () => () => {
      clearPoll();
      endedRef.current = true;
      void dpoOffPolicyApi.cancel().catch(() => undefined);
    },
    [clearPoll],
  );

  const act = useCallback(
    async (action: () => Promise<{ ok: boolean; error?: string }>, onOk?: () => void) => {
      if (busy || !candidate) return;
      setBusy(true);
      setError(null);
      try {
        const res = await action();
        if (!res.ok) {
          setError(res.error ?? "Action failed");
          return;
        }
        onOk?.();
        setCandidate(null);
        await fetchNext();
      } catch (e) {
        setError(errorText(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, candidate, fetchNext],
  );

  const bump = useCallback(
    (delta: number) => {
      if (!character) return;
      setConfirmed((prev) => ({ ...prev, [character.name]: Math.max(0, (prev[character.name] ?? 0) + delta) }));
    },
    [character],
  );

  const confirm = useCallback(
    () =>
      act(
        () => dpoOffPolicyApi.confirm(),
        () => bump(1),
      ),
    [act, bump],
  );
  const skip = useCallback(() => act(() => dpoOffPolicyApi.skip()), [act]);
  const skipForever = useCallback(() => act(() => dpoOffPolicyApi.skipForever()), [act]);

  const swapCandidate = useCallback(
    async (call: () => Promise<{ ok: boolean; candidate?: OffPolicyCandidate; error?: string }>, fallback: string) => {
      if (busy || !candidate) return;
      setBusy(true);
      setError(null);
      try {
        const res = await call();
        if (!res.ok) {
          setNotice(res.error ?? fallback);
          return;
        }
        if (res.candidate) setCandidate(asPick(res.candidate));
      } catch (e) {
        setError(errorText(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, candidate],
  );

  const reroll = useCallback(() => {
    if (candidate && candidate.generated_count <= 1) return;
    void swapCandidate(() => dpoOffPolicyApi.reroll(), "Could not re-roll");
  }, [candidate, swapCandidate]);

  const choose = useCallback(
    (path: string) => {
      if (!candidate || path === candidate.rejected_path) return;
      void swapCandidate(() => characterDpoApi.offPolicyChoose(path), "Could not choose that generation");
    },
    [candidate, swapCandidate],
  );

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
        setCandidate(asPick(res.candidate));
        if (res.undone === "confirm") bump(-1);
        setNotice(`Undid ${res.undone ?? "last action"}.`);
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }, [busy, bump]);

  // Keyboard: 1–9 pick a tile, Enter confirm, S skip, X never, R re-roll, Ctrl/Cmd+Z undo.
  useEffect(() => {
    if (phase !== "session") return;
    const handler = (e: KeyboardEvent) => {
      if (preview) return;
      const target = e.target as HTMLElement | null;
      if (target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable)) return;
      if (target?.tagName === "BUTTON" && (e.key === "Enter" || e.key === " " || e.key === "Spacebar")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        void undo();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (/^[1-9]$/.test(e.key)) {
        const path = candidate?.generated_paths[Number(e.key) - 1];
        if (path) {
          e.preventDefault();
          choose(path);
        }
      } else if (e.key === "Enter" || key === "y") {
        e.preventDefault();
        void confirm();
      } else if (key === "s") {
        e.preventDefault();
        void skip();
      } else if (key === "x") {
        e.preventDefault();
        void skipForever();
      } else if (key === "r") {
        e.preventDefault();
        reroll();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase, preview, candidate, choose, confirm, skip, skipForever, reroll, undo]);

  const scanning = status ? !status.worker_finished : phase === "starting";

  if (phase === "all-done") {
    return (
      <div className={SECTION}>
        <div className="flex items-center gap-2 text-sm text-[var(--color-on-surface)]">
          <Check className="w-4 h-4 text-[var(--color-success-500)]" />
          {characters.length === 0
            ? "None of the checked characters has generations yet — run step 2 first."
            : `All done — ${totalConfirmed} pair${totalConfirmed === 1 ? "" : "s"} across ${characters.length} character${
                characters.length === 1 ? "" : "s"
              }.`}
        </div>
        {characters.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {characters.map((c) => (
              <Chip key={c.name} tone={(confirmed[c.name] ?? 0) > 0 ? "ok" : "neutral"}>
                {c.name}: {confirmed[c.name] ?? 0}
              </Chip>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`${SECTION} min-h-0 flex-1`}>
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-[var(--color-on-surface)]">
          Character {charIdx + 1}/{characters.length} · {character?.name}
        </span>
        {candidate && (
          <span className={`${HINT} tabular-nums`}>
            photo {candidate.index}/{candidate.total}
          </span>
        )}
        <span className={`${HINT} tabular-nums`}>
          pairs confirmed {confirmed[character?.name ?? ""] ?? 0} (total {totalConfirmed})
        </span>
        {status && !scanning && (
          <span className={`${HINT} tabular-nums`}>
            · skipped {status.skipped} · never {status.skipped_forever} · {status.total_pairs} pairs on disk
          </span>
        )}
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto"
          onClick={() => void advance()}
          title="End this character's session and move on"
        >
          <ChevronsRight className="w-4 h-4" /> Skip character
        </Button>
      </div>

      {scanning && status && (
        <div className="flex flex-col gap-1 bg-[var(--color-surface-container)] rounded p-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> {PHASE_LABEL[status.phase]}
            </span>
            <span className={`${HINT} tabular-nums`}>
              {status.phase === "indexing" && status.progress_total === 0
                ? `listing… ${status.listed.toLocaleString()} files`
                : `${status.progress_done.toLocaleString()} / ${status.progress_total.toLocaleString()}`}
            </span>
          </div>
          <Bar done={status.progress_done} total={status.progress_total} />
        </div>
      )}
      {phase === "starting" && !status && !error && (
        <div className={`flex items-center gap-2 ${HINT}`}>
          <Loader2 className="w-4 h-4 animate-spin" /> Starting the off-policy session for {character?.name}…
        </div>
      )}

      {notice && <div className={HINT}>{notice}</div>}
      {error && (
        <div className="flex items-center gap-2 text-xs text-[var(--color-error-500)]">
          <AlertCircle className="w-4 h-4" /> <span className="whitespace-pre-wrap">{error}</span>
        </div>
      )}
      {status?.warning && <div className="text-xs text-[var(--color-warning-500,#eab308)]">{status.warning}</div>}

      {phase === "session" && candidate && (
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(320px,2fr)_3fr] gap-3 flex-1 min-h-0 h-[52vh]">
            {/* Real photo + caption */}
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
              <div className="text-sm text-[var(--color-on-surface)] whitespace-pre-wrap pt-1 max-h-24 overflow-y-auto">
                {candidate.caption}
              </div>
              {candidate.caption_lines.length > 1 && (
                <div className={HINT}>+ {candidate.caption_lines.length - 1} other caption line(s)</div>
              )}
              <div
                className="text-[10px] text-[var(--color-on-surface-secondary)] truncate"
                title={candidate.real_path}
              >
                {baseName(candidate.real_path)}
              </div>
            </figure>

            {/* Generations */}
            <div className="flex flex-col min-h-0 bg-[var(--color-surface-container)] rounded p-2">
              <div className="text-xs font-semibold mb-1 text-[var(--color-error-500)] flex flex-wrap items-center gap-2">
                <span>
                  REJECTED — {candidate.generated_paths.length} generation
                  {candidate.generated_paths.length === 1 ? "" : "s"}
                  {candidate.rejected_score != null && ` · P(AI) ${candidate.rejected_score.toFixed(2)}`}
                  {candidate.aspect_mismatch != null &&
                    candidate.aspect_mismatch > 0 &&
                    ` · aspect Δ ${(candidate.aspect_mismatch * 100).toFixed(0)}%`}
                </span>
                <span className={`${HINT} font-normal`}>
                  {candidate.scored ? "hardest negative preselected" : "unscored — random preselection"} · click or
                  press 1–9 to choose
                </span>
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto grid grid-cols-2 lg:grid-cols-3 gap-2 content-start">
                {candidate.generated_paths.map((p, i) => {
                  const isRejected = i === candidate.rejected_index || p === candidate.rejected_path;
                  return (
                    <button
                      type="button"
                      key={p}
                      onClick={() => choose(p)}
                      onDoubleClick={() => setPreview(p)}
                      disabled={busy}
                      title={`${baseName(p)} — click to make this the rejected image, double-click to expand`}
                      className={`relative aspect-square rounded overflow-hidden border-2 transition-colors ${
                        isRejected
                          ? "border-[var(--color-error-500)]"
                          : "border-transparent hover:border-[var(--color-cobalt-600)]"
                      }`}
                    >
                      <img
                        src={dpoApi.imageUrl(p)}
                        alt={`generation ${i + 1}`}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute top-1 left-1 px-1 rounded bg-black/70 text-white text-[10px] font-semibold tabular-nums">
                        {i + 1}
                      </span>
                      {isRejected && (
                        <span className="absolute bottom-1 right-1 px-1 rounded bg-[var(--color-error-500)] text-white text-[10px] font-bold tracking-wide">
                          REJECTED
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {candidate.real_score != null &&
            candidate.rejected_score != null &&
            candidate.real_score > candidate.rejected_score && (
              <div className="flex items-center gap-2 text-xs text-[var(--color-warning-500,#eab308)]">
                <AlertCircle className="w-4 h-4" />
                The realism head rates the photo as more AI-looking than the generation — check it really is a photo.
              </div>
            )}

          <div className="flex flex-wrap items-center gap-2">
            <Button variant="primary" size="sm" onClick={() => void confirm()} disabled={busy}>
              <Check className="w-4 h-4" /> Confirm pair (Enter)
            </Button>
            <Button variant="secondary" size="sm" onClick={() => void skip()} disabled={busy}>
              <SkipForward className="w-4 h-4" /> Skip (S)
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void skipForever()}
              disabled={busy}
              title="Never offer this photo again for this character"
            >
              <Ban className="w-4 h-4" /> Never (X)
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={reroll}
              disabled={busy || candidate.generated_count <= 1}
              title="Cycle to another generation of this photo"
            >
              <RefreshCw className="w-4 h-4" /> Re-roll (R)
            </Button>
            <Button variant="ghost" size="sm" onClick={() => void undo()} disabled={busy || !status?.can_undo}>
              <Undo2 className="w-4 h-4" />
              {status?.undo_action ? `Undo ${status.undo_action} (Ctrl+Z)` : "Undo (Ctrl+Z)"}
            </Button>
          </div>
        </div>
      )}

      {phase === "session" && !candidate && !scanning && (
        <div className={`flex items-center gap-2 ${HINT}`}>
          <Loader2 className="w-4 h-4 animate-spin" /> Waiting for the next candidate…
          {status?.can_undo && (
            <Button variant="ghost" size="sm" onClick={() => void undo()} disabled={busy}>
              <Undo2 className="w-4 h-4" /> Undo (Ctrl+Z)
            </Button>
          )}
        </div>
      )}

      {preview && <PreviewOverlay path={preview} onClose={() => setPreview(null)} />}
    </div>
  );
}
