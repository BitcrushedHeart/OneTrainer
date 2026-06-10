import { ArrowLeftRight, CheckCircle2, ChevronDown, CornerUpLeft, SkipForward, Undo2, X, ZoomIn } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { API_BASE } from "@/api/request";
import { Button } from "@/components/shared";

import { PreviewOverlay } from "./PreviewOverlay";
import { PromptExpander } from "./PromptExpander";
import type { TriageStepProps } from "./types";

type Verdict = "good" | "bad" | "skip";

type CardLoc = { type: "pool" } | { type: "slot"; pairIndex: number };

interface PickedCard {
  side: "good" | "bad";
  path: string;
  loc: CardLoc;
}

interface TriagePair {
  chosen: string;
  rejected: string;
}

const GOOD_COLOR = "#22c55e";
const BAD_COLOR = "#ef4444";
const SKIP_COLOR = "#9ca3af";

function basename(p: string): string {
  return p.split(/[/\\]/).pop() ?? p;
}

function imageUrl(path: string): string {
  return `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
}

function aspectRatioStyle(ar: string): React.CSSProperties | undefined {
  const m = /^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/.exec(ar.trim());
  if (!m) return undefined;
  return { aspectRatio: `${m[1]} / ${m[2]}` };
}

function verdictColor(v: Verdict | undefined): string {
  if (v === "good") return GOOD_COLOR;
  if (v === "bad") return BAD_COLOR;
  if (v === "skip") return SKIP_COLOR;
  return "var(--color-border-subtle)";
}

export function TriageStep({ group, pairsDone, onCommitPairs, onSkipGroup, onCancel }: TriageStepProps) {
  const [phase, setPhase] = useState<"voting" | "pairing">("voting");
  // Voting is append-only: verdicts[i] belongs to group.images[i] and the
  // current image index is simply verdicts.length. Undo pops the last vote.
  const [verdicts, setVerdicts] = useState<Verdict[]>([]);
  const [pairs, setPairs] = useState<TriagePair[]>([]);
  const [goodPool, setGoodPool] = useState<string[]>([]);
  const [badPool, setBadPool] = useState<string[]>([]);
  const [picked, setPicked] = useState<PickedCard | null>(null);
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [stripOpen, setStripOpen] = useState(false);
  const [committing, setCommitting] = useState(false);

  const images = group.images;
  const index = Math.min(verdicts.length, images.length - 1);
  const arStyle = aspectRatioStyle(group.aspectratio);

  // A fresh group (auto-advance after commit/skip) starts triage clean.
  useEffect(() => {
    setPhase("voting");
    setVerdicts([]);
    setPairs([]);
    setGoodPool([]);
    setBadPool([]);
    setPicked(null);
    setPreviewPath(null);
    setStripOpen(false);
    setCommitting(false);
  }, [group]);

  // Warm the browser HTTP cache for the next few images so advancing after a
  // vote is instant — the visible <img> then resolves from cache.
  useEffect(() => {
    if (phase !== "voting") return;
    for (let k = 1; k <= 3; k++) {
      const next = images[verdicts.length + k];
      if (next) {
        const im = new Image();
        im.src = imageUrl(next);
      }
    }
  }, [phase, verdicts.length, images]);

  const counts = useMemo(() => {
    let good = 0;
    let bad = 0;
    let skip = 0;
    for (const v of verdicts) {
      if (v === "good") good++;
      else if (v === "bad") bad++;
      else skip++;
    }
    return { good, bad, skip };
  }, [verdicts]);

  const skipped = useMemo(() => images.filter((_, i) => verdicts[i] === "skip"), [images, verdicts]);

  const finishVoting = useCallback(
    (finalVerdicts: Verdict[]) => {
      const good: string[] = [];
      const bad: string[] = [];
      images.forEach((img, i) => {
        if (finalVerdicts[i] === "good") good.push(img);
        else if (finalVerdicts[i] === "bad") bad.push(img);
      });
      const n = Math.min(good.length, bad.length);
      setPairs(good.slice(0, n).map((chosen, i) => ({ chosen, rejected: bad[i] })));
      setGoodPool(good.slice(n));
      setBadPool(bad.slice(n));
      setPicked(null);
      setPhase("pairing");
    },
    [images],
  );

  const vote = useCallback(
    (v: Verdict) => {
      if (verdicts.length >= images.length) return;
      const next = [...verdicts, v];
      setVerdicts(next);
      if (next.length === images.length) finishVoting(next);
    },
    [verdicts, images.length, finishVoting],
  );

  const undo = useCallback(() => {
    setVerdicts((prev) => prev.slice(0, -1));
  }, []);

  // Keyboard: 1/0/Space/Backspace while voting, Escape deselects while pairing.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (previewPath) return;
      if (phase === "voting") {
        if (e.key === "1") {
          e.preventDefault();
          vote("good");
        } else if (e.key === "0") {
          e.preventDefault();
          vote("bad");
        } else if (e.key === " ") {
          // preventDefault stops page scroll; blur stops Space from also
          // "clicking" a previously focused button (Skip Group, etc.).
          e.preventDefault();
          (document.activeElement as HTMLElement | null)?.blur?.();
          vote("skip");
        } else if (e.key === "Backspace") {
          e.preventDefault();
          undo();
        }
      } else if (e.key === "Escape") {
        setPicked(null);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [phase, previewPath, vote, undo]);

  // ---- pairing-phase mutations ----

  const isSameCard = (a: PickedCard, b: PickedCard) => a.path === b.path;

  const handleCardClick = useCallback(
    (card: PickedCard) => {
      if (!picked) {
        setPicked(card);
        return;
      }
      if (isSameCard(picked, card)) {
        setPicked(null);
        return;
      }
      if (picked.side !== card.side) {
        // Re-pick rather than swap across sides — sides hold different roles.
        setPicked(card);
        return;
      }
      // Swap the two images between their locations on this side.
      const side = card.side;
      const replaceInPair = (list: TriagePair[], pairIndex: number, newPath: string) =>
        list.map((p, i) =>
          i === pairIndex ? (side === "good" ? { ...p, chosen: newPath } : { ...p, rejected: newPath }) : p,
        );
      const replaceInPool = (pool: string[], oldPath: string, newPath: string) =>
        pool.map((x) => (x === oldPath ? newPath : x));

      let newPairs = pairs;
      let newGood = goodPool;
      let newBad = badPool;
      const apply = (loc: CardLoc, oldPath: string, newPath: string) => {
        if (loc.type === "slot") {
          newPairs = replaceInPair(newPairs, loc.pairIndex, newPath);
        } else if (side === "good") {
          newGood = replaceInPool(newGood, oldPath, newPath);
        } else {
          newBad = replaceInPool(newBad, oldPath, newPath);
        }
      };
      apply(picked.loc, picked.path, card.path);
      apply(card.loc, card.path, picked.path);
      setPairs(newPairs);
      setGoodPool(newGood);
      setBadPool(newBad);
      setPicked(null);
    },
    [picked, pairs, goodPool, badPool],
  );

  // Move a card to the other side (good↔bad), preserving manual changes:
  // untouched pairs survive verbatim; only the affected pair dissolves or
  // pulls a replacement, then a top-up pass pairs whatever both pools allow.
  const moveToOtherSide = useCallback(
    (card: PickedCard) => {
      const newPairs = [...pairs];
      let newGood = [...goodPool];
      let newBad = [...badPool];

      if (card.side === "good") {
        if (card.loc.type === "pool") {
          newGood = newGood.filter((x) => x !== card.path);
        } else {
          const pr = newPairs[card.loc.pairIndex];
          newBad.push(pr.rejected);
          newPairs.splice(card.loc.pairIndex, 1);
        }
        newBad.push(card.path);
      } else {
        if (card.loc.type === "pool") {
          newBad = newBad.filter((x) => x !== card.path);
        } else {
          const i = card.loc.pairIndex;
          const replacement = newBad.shift();
          if (replacement !== undefined) {
            newPairs[i] = { ...newPairs[i], rejected: replacement };
          } else {
            newGood.push(newPairs[i].chosen);
            newPairs.splice(i, 1);
          }
        }
        newGood.push(card.path);
      }

      while (newGood.length > 0 && newBad.length > 0) {
        newPairs.push({ chosen: newGood.shift() as string, rejected: newBad.shift() as string });
      }

      setPairs(newPairs);
      setGoodPool(newGood);
      setBadPool(newBad);
      setPicked(null);
      // Keep verdicts in sync so "Back to voting" reflects the reassignment.
      setVerdicts((prev) => {
        const idx = images.indexOf(card.path);
        if (idx < 0 || idx >= prev.length) return prev;
        const next = [...prev];
        next[idx] = card.side === "good" ? "bad" : "good";
        return next;
      });
    },
    [pairs, goodPool, badPool, images],
  );

  const backToVoting = useCallback(() => {
    // Drop the last vote so the user lands on a re-votable image and can
    // Backspace further from there.
    setVerdicts((prev) => prev.slice(0, -1));
    setPairs([]);
    setGoodPool([]);
    setBadPool([]);
    setPicked(null);
    setPhase("voting");
  }, []);

  const handleConfirm = useCallback(async () => {
    if (pairs.length === 0 || committing) return;
    setCommitting(true);
    try {
      await onCommitPairs(pairs);
    } finally {
      setCommitting(false);
    }
  }, [pairs, committing, onCommitPairs]);

  // ---- voting-phase mouse input on the big image ----

  const handleImageClick = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    vote("good");
  };

  const handleImageContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    vote("bad");
  };

  const handleImageAuxClick = (e: React.MouseEvent) => {
    if (e.button !== 1) return;
    e.preventDefault();
    vote("skip");
  };

  const handleImageMouseDown = (e: React.MouseEvent) => {
    // Suppress middle-click autoscroll so MMB is a clean "skip".
    if (e.button === 1) e.preventDefault();
  };

  const header = (
    <div className="flex items-center justify-between gap-3">
      <div className="text-xs text-[var(--color-on-surface-secondary)] flex items-center gap-3 flex-wrap">
        <span>
          Group {group.group_index} / {group.total_groups}
        </span>
        <span className="text-[var(--color-border-subtle)]">·</span>
        <span>AR: {group.aspectratio}</span>
        <span className="text-[var(--color-border-subtle)]">·</span>
        {phase === "voting" ? (
          <span className="tabular-nums">
            Image {Math.min(verdicts.length + 1, images.length)} / {images.length}
          </span>
        ) : (
          <span className="tabular-nums">{pairs.length} pair(s) ready</span>
        )}
        <span className="text-[var(--color-border-subtle)]">·</span>
        <span className="tabular-nums">
          <span style={{ color: GOOD_COLOR }}>Good {counts.good}</span>{" "}
          <span style={{ color: BAD_COLOR }}>Bad {counts.bad}</span>{" "}
          <span style={{ color: SKIP_COLOR }}>Skip {counts.skip}</span>
        </span>
        {pairsDone > 0 && (
          <>
            <span className="text-[var(--color-border-subtle)]">·</span>
            <span className="tabular-nums">Saved {pairsDone}</span>
          </>
        )}
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onSkipGroup} style={{ background: "#8B4513", color: "#fff" }}>
          <SkipForward className="w-4 h-4" />
          Skip Group
        </Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          <X className="w-4 h-4" />
          Cancel
        </Button>
      </div>
    </div>
  );

  if (images.length < 2) {
    return (
      <div className="flex flex-col gap-3 h-full min-h-0">
        {header}
        <div className="flex-1 flex items-center justify-center text-sm text-[var(--color-on-surface-secondary)]">
          Not enough images in this group to triage.
        </div>
      </div>
    );
  }

  if (phase === "voting") {
    const current = images[index];
    return (
      <div className="flex flex-col gap-3 h-full min-h-0">
        {header}
        <PromptExpander prompt={group.prompt} />

        {/* Full-preview-size image, same presentation as PreviewOverlay */}
        <div className="relative flex-1 min-h-0 rounded overflow-hidden" style={{ background: "rgba(0, 0, 0, 0.95)" }}>
          <div
            className="w-full h-full flex items-center justify-center overflow-hidden cursor-pointer select-none"
            onClick={handleImageClick}
            onContextMenu={handleImageContextMenu}
            onAuxClick={handleImageAuxClick}
            onMouseDown={handleImageMouseDown}
          >
            <img
              key={current}
              src={imageUrl(current)}
              alt={basename(current)}
              className="w-full h-full object-contain"
              fetchPriority="high"
              draggable={false}
            />
          </div>

          {/* Overview strip: hover the top edge to see the whole stack */}
          <div
            className="absolute top-0 left-0 right-0 z-10"
            onMouseEnter={() => setStripOpen(true)}
            onMouseLeave={() => setStripOpen(false)}
          >
            <div className="h-6 flex items-center justify-center cursor-pointer bg-gradient-to-b from-black/70 to-transparent">
              <ChevronDown className={`w-4 h-4 text-white/60 transition-transform ${stripOpen ? "rotate-180" : ""}`} />
            </div>
            {stripOpen && (
              <div className="flex gap-1.5 overflow-x-auto px-2 py-2 bg-black/85">
                {images.map((img, i) => (
                  <div
                    key={img}
                    className={`relative shrink-0 h-20 rounded border-2 overflow-hidden bg-black/40 ${
                      i === index ? "ring-2 ring-[var(--color-cobalt-600)]" : ""
                    }`}
                    style={{
                      borderColor: verdictColor(verdicts[i]),
                      borderStyle: verdicts[i] === "skip" ? "dashed" : "solid",
                      ...arStyle,
                    }}
                    title={basename(img)}
                  >
                    <img
                      src={imageUrl(img)}
                      alt={basename(img)}
                      loading="lazy"
                      className="w-full h-full object-cover"
                      draggable={false}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Filename, mirroring PreviewOverlay's bottom info bar */}
          <div className="absolute bottom-0 left-0 right-0 px-4 py-2 bg-black/70 text-white text-xs flex items-center justify-between pointer-events-none">
            <span className="font-mono truncate">{basename(current)}</span>
            <span className="text-white/60 tabular-nums">
              {Math.min(verdicts.length + 1, images.length)} / {images.length}
            </span>
          </div>
        </div>

        {/* Control legend — non-focusable so Space never double-fires */}
        <div className="flex items-center justify-center gap-2 flex-wrap text-xs">
          <div
            role="button"
            tabIndex={-1}
            onClick={() => vote("good")}
            className="px-3 py-1.5 rounded cursor-pointer font-semibold"
            style={{ color: GOOD_COLOR, background: "rgba(34,197,94,0.1)", border: `1px solid ${GOOD_COLOR}55` }}
          >
            Good — LMB / 1
          </div>
          <div
            role="button"
            tabIndex={-1}
            onClick={() => vote("bad")}
            className="px-3 py-1.5 rounded cursor-pointer font-semibold"
            style={{ color: BAD_COLOR, background: "rgba(239,68,68,0.1)", border: `1px solid ${BAD_COLOR}55` }}
          >
            Bad — RMB / 0
          </div>
          <div
            role="button"
            tabIndex={-1}
            onClick={() => vote("skip")}
            className="px-3 py-1.5 rounded cursor-pointer font-semibold"
            style={{ color: SKIP_COLOR, background: "rgba(156,163,175,0.1)", border: `1px solid ${SKIP_COLOR}55` }}
          >
            Skip — MMB / Space
          </div>
          <div
            role="button"
            tabIndex={-1}
            onClick={undo}
            className={`px-3 py-1.5 rounded font-semibold flex items-center gap-1 text-[var(--color-on-surface-secondary)] border border-[var(--color-border-subtle)] ${
              verdicts.length === 0 ? "opacity-40 cursor-default" : "cursor-pointer"
            }`}
          >
            <Undo2 className="w-3.5 h-3.5" />
            Undo — Backspace
          </div>
        </div>
      </div>
    );
  }

  // ---- pairing phase ----

  const isPicked = (path: string) => picked?.path === path;

  const renderCard = (card: PickedCard, sizeClass = "h-36") => (
    <div
      key={card.path}
      className={`group/card relative ${sizeClass} rounded border-2 overflow-hidden bg-black/30 cursor-pointer transition-all hover:shadow-lg ${
        isPicked(card.path) ? "ring-2 ring-[var(--color-cobalt-600)] -translate-y-0.5" : ""
      }`}
      style={{
        borderColor: card.side === "good" ? GOOD_COLOR : BAD_COLOR,
        ...arStyle,
      }}
      onClick={() => handleCardClick(card)}
      title={basename(card.path)}
    >
      <img
        src={imageUrl(card.path)}
        alt={basename(card.path)}
        loading="lazy"
        className="w-full h-full object-cover"
        draggable={false}
      />
      <div className="absolute top-1 right-1 flex gap-1 opacity-0 group-hover/card:opacity-100 transition-opacity">
        <div
          role="button"
          tabIndex={-1}
          className="p-1 rounded bg-black/70 text-white hover:bg-black/90"
          title="Preview full size"
          onClick={(e) => {
            e.stopPropagation();
            setPreviewPath(card.path);
          }}
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </div>
        <div
          role="button"
          tabIndex={-1}
          className="p-1 rounded bg-black/70 text-white hover:bg-black/90"
          title={card.side === "good" ? "Move to Bad" : "Move to Good"}
          onClick={(e) => {
            e.stopPropagation();
            moveToOtherSide(card);
          }}
        >
          <ArrowLeftRight className="w-3.5 h-3.5" />
        </div>
      </div>
    </div>
  );

  const connector = (
    <div className="flex items-center w-14 shrink-0" aria-hidden>
      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: "var(--color-cobalt-600)" }} />
      <span className="flex-1 h-0.5" style={{ background: "var(--color-cobalt-600)" }} />
      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: "var(--color-cobalt-600)" }} />
    </div>
  );

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      {header}
      <PromptExpander prompt={group.prompt} />

      <div className="flex-1 min-h-0 overflow-y-auto space-y-4 pr-1">
        {/* Pair rows: good — chain — bad */}
        <div className="space-y-2">
          <div className="flex items-center justify-center gap-2 text-xs font-semibold">
            <span className="flex-1 text-right" style={{ color: GOOD_COLOR }}>
              Chosen (Good)
            </span>
            <span className="w-14 shrink-0" />
            <span className="flex-1 text-left" style={{ color: BAD_COLOR }}>
              Rejected (Bad)
            </span>
          </div>
          {pairs.length === 0 ? (
            <div className="text-center text-sm text-[var(--color-on-surface-secondary)] py-8">
              No pairs possible — {counts.good} good / {counts.bad} bad. Move images between sides or skip this group.
            </div>
          ) : (
            pairs.map((p, i) => (
              <div key={`${p.chosen}::${p.rejected}`} className="flex items-center justify-center gap-2">
                <div className="flex-1 flex justify-end">
                  {renderCard({ side: "good", path: p.chosen, loc: { type: "slot", pairIndex: i } })}
                </div>
                {connector}
                <div className="flex-1 flex justify-start">
                  {renderCard({ side: "bad", path: p.rejected, loc: { type: "slot", pairIndex: i } })}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Surplus pools for mix-and-match */}
        {goodPool.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-xs font-semibold" style={{ color: GOOD_COLOR }}>
              Unpaired Good ({goodPool.length}) — click to swap into a pair
            </div>
            <div className="flex flex-wrap gap-2">
              {goodPool.map((path) => renderCard({ side: "good", path, loc: { type: "pool" } }, "h-28"))}
            </div>
          </div>
        )}
        {badPool.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-xs font-semibold" style={{ color: BAD_COLOR }}>
              Unpaired Bad ({badPool.length}) — click to swap into a pair
            </div>
            <div className="flex flex-wrap gap-2">
              {badPool.map((path) => renderCard({ side: "bad", path, loc: { type: "pool" } }, "h-28"))}
            </div>
          </div>
        )}

        {/* Skipped pile (read-only) */}
        {skipped.length > 0 && (
          <details className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)]">
            <summary className="cursor-pointer select-none px-3 py-1.5 text-xs font-semibold text-[var(--color-on-surface-secondary)]">
              Skipped ({skipped.length})
            </summary>
            <div className="flex flex-wrap gap-2 p-3">
              {skipped.map((path) => (
                <div
                  key={path}
                  className="relative h-24 rounded border-2 border-dashed overflow-hidden bg-black/30"
                  style={{ borderColor: SKIP_COLOR, ...arStyle }}
                  title={basename(path)}
                >
                  <img
                    src={imageUrl(path)}
                    alt={basename(path)}
                    loading="lazy"
                    className="w-full h-full object-cover opacity-70"
                    draggable={false}
                  />
                </div>
              ))}
            </div>
          </details>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={backToVoting}>
            <CornerUpLeft className="w-4 h-4" />
            Back to voting
          </Button>
          <span className="text-[10px] text-[var(--color-on-surface-secondary)]">
            Re-voting rebuilds pairs — manual swaps reset.
          </span>
        </div>
        <div className="flex items-center gap-2">
          {picked && (
            <span className="text-xs text-[var(--color-on-surface-secondary)]">
              Swapping <span className="font-mono">{basename(picked.path)}</span> — click another{" "}
              {picked.side === "good" ? "Good" : "Bad"} card (Esc to cancel)
            </span>
          )}
          <Button
            variant="primary"
            size="md"
            onClick={() => void handleConfirm()}
            disabled={pairs.length === 0 || committing}
          >
            <CheckCircle2 className="w-4 h-4" />
            {committing ? "Saving…" : `Confirm ${pairs.length} Pair${pairs.length === 1 ? "" : "s"}`}
          </Button>
        </div>
      </div>

      {previewPath && (
        <PreviewOverlay path={previewPath} caption={basename(previewPath)} onClose={() => setPreviewPath(null)} />
      )}
    </div>
  );
}
