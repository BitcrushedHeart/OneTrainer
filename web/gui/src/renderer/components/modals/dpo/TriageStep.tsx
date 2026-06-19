import {
  ArrowLeftRight,
  CheckCircle2,
  ChevronDown,
  CornerUpLeft,
  Eye,
  SkipForward,
  ThumbsDown,
  ThumbsUp,
  Undo2,
  X,
  ZoomIn,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/shared";

import { PromptExpander } from "./PromptExpander";
import {
  BAD_COLOR,
  basename,
  GOOD_COLOR,
  imageUrl,
  SKIP_COLOR,
  type Verdict,
  verdictColor,
  verdictLabel,
} from "./triageCommon";
import { TriageReviewOverlay } from "./TriageReviewOverlay";
import type { TriageStepProps } from "./types";

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

interface ReviewState {
  list: string[];
  pos: number;
}

function aspectRatioStyle(ar: string): React.CSSProperties | undefined {
  const m = /^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/.exec(ar.trim());
  if (!m) return undefined;
  return { aspectRatio: `${m[1]} / ${m[2]}` };
}

export function TriageStep({ group, pairsDone, onCommitPairs, onSkipGroup, onCancel }: TriageStepProps) {
  const [phase, setPhase] = useState<"voting" | "pairing">("voting");
  // Verdicts are keyed by image index and sparse: undefined = not yet scored.
  // The cursor moves freely (arrow keys, thumbnail clicks), so earlier votes
  // can be revisited and changed at any time without losing the rest.
  const [verdicts, setVerdicts] = useState<Array<Verdict | undefined>>([]);
  const [cursorRaw, setCursor] = useState(0);
  const [pairs, setPairs] = useState<TriagePair[]>([]);
  const [goodPool, setGoodPool] = useState<string[]>([]);
  const [badPool, setBadPool] = useState<string[]>([]);
  const [picked, setPicked] = useState<PickedCard | null>(null);
  const [review, setReview] = useState<ReviewState | null>(null);
  const [stripOpen, setStripOpen] = useState(false);
  const [committing, setCommitting] = useState(false);

  const images = group.images;
  // When a new group arrives, the reset effect below fires only after the
  // first render — a stale cursor from a larger group would index past the
  // new image list, so clamp at render time.
  const cursor = Math.min(cursorRaw, Math.max(0, images.length - 1));
  const arStyle = aspectRatioStyle(group.aspectratio);

  // A fresh group (auto-advance after commit/skip) starts triage clean.
  useEffect(() => {
    setPhase("voting");
    setVerdicts([]);
    setCursor(0);
    setPairs([]);
    setGoodPool([]);
    setBadPool([]);
    setPicked(null);
    setReview(null);
    setStripOpen(false);
    setCommitting(false);
  }, [group]);

  // Warm the browser HTTP cache for the next few images so advancing after a
  // vote is instant — the visible <img> then resolves from cache.
  useEffect(() => {
    if (phase !== "voting") return;
    for (let k = 1; k <= 3; k++) {
      const next = images[cursor + k];
      if (next) {
        const im = new Image();
        im.src = imageUrl(next);
      }
    }
  }, [phase, cursor, images]);

  const counts = useMemo(() => {
    let good = 0;
    let bad = 0;
    let skip = 0;
    for (const v of verdicts) {
      if (v === "good") good++;
      else if (v === "bad") bad++;
      else if (v === "skip") skip++;
    }
    return { good, bad, skip };
  }, [verdicts]);

  const unscored = images.length - counts.good - counts.bad - counts.skip;

  const skipped = useMemo(() => images.filter((_, i) => verdicts[i] === "skip"), [images, verdicts]);

  const finishVoting = useCallback(
    (finalVerdicts: Array<Verdict | undefined>) => {
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
      const wasUnscored = verdicts[cursor] === undefined;
      const next = [...verdicts];
      next[cursor] = v;
      setVerdicts(next);
      // Jump to the next unscored image (wrapping) — in a fresh pass this is
      // simply cursor+1, after revisits it returns to the frontier.
      for (let k = 1; k < images.length; k++) {
        const i = (cursor + k) % images.length;
        if (next[i] === undefined) {
          setCursor(i);
          return;
        }
      }
      // Everything is scored. Only auto-advance to pairing when this vote
      // completed the set — a pure re-vote stays in voting so several images
      // can be changed before finishing explicitly.
      if (wasUnscored) finishVoting(next);
      else setCursor((c) => Math.min(c + 1, images.length - 1));
    },
    [verdicts, cursor, images.length, finishVoting],
  );

  const undo = useCallback(() => {
    if (cursor === 0) return;
    const next = [...verdicts];
    next[cursor - 1] = undefined;
    setVerdicts(next);
    setCursor(cursor - 1);
  }, [cursor, verdicts]);

  const goPrev = useCallback(() => setCursor((c) => Math.max(0, c - 1)), []);
  const goNext = useCallback(() => setCursor((c) => Math.min(images.length - 1, c + 1)), [images.length]);

  // Finish on demand: anything still unscored becomes an explicit skip, so it
  // lands in the skipped pile and stays rescorable from the pairing screen.
  const finishNow = useCallback(() => {
    const filled = images.map((_, i) => verdicts[i] ?? ("skip" as Verdict));
    setVerdicts(filled);
    finishVoting(filled);
  }, [images, verdicts, finishVoting]);

  // Keyboard: 1/0/Space/Backspace vote, arrows navigate while voting; Escape
  // deselects while pairing. The review overlay handles its own keys in the
  // capture phase, so bail out whenever it is open.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (review) return;
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
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          goPrev();
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          goNext();
        }
      } else if (e.key === "Escape") {
        setPicked(null);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [phase, review, vote, undo, goPrev, goNext]);

  // ---- pairing-phase mutations ----

  // Re-score one image in place, preserving manual arrangements: a pair losing
  // one half pulls a replacement from the matching pool before dissolving,
  // then a top-up pass pairs whatever both pools allow.
  const reassign = useCallback(
    (path: string, v: Verdict) => {
      const idx = images.indexOf(path);
      if (idx < 0 || verdicts[idx] === v) return;
      const newPairs = [...pairs];
      let newGood = [...goodPool];
      let newBad = [...badPool];

      if (verdicts[idx] === "good") {
        const slot = newPairs.findIndex((p) => p.chosen === path);
        if (slot >= 0) {
          const replacement = newGood.shift();
          if (replacement !== undefined) {
            newPairs[slot] = { ...newPairs[slot], chosen: replacement };
          } else {
            newBad.push(newPairs[slot].rejected);
            newPairs.splice(slot, 1);
          }
        } else {
          newGood = newGood.filter((x) => x !== path);
        }
      } else if (verdicts[idx] === "bad") {
        const slot = newPairs.findIndex((p) => p.rejected === path);
        if (slot >= 0) {
          const replacement = newBad.shift();
          if (replacement !== undefined) {
            newPairs[slot] = { ...newPairs[slot], rejected: replacement };
          } else {
            newGood.push(newPairs[slot].chosen);
            newPairs.splice(slot, 1);
          }
        } else {
          newBad = newBad.filter((x) => x !== path);
        }
      }

      if (v === "good") newGood.push(path);
      else if (v === "bad") newBad.push(path);

      while (newGood.length > 0 && newBad.length > 0) {
        newPairs.push({ chosen: newGood.shift() as string, rejected: newBad.shift() as string });
      }

      setPairs(newPairs);
      setGoodPool(newGood);
      setBadPool(newBad);
      setPicked(null);
      setVerdicts((prev) => {
        const next = [...prev];
        next[idx] = v;
        return next;
      });
    },
    [images, verdicts, pairs, goodPool, badPool],
  );

  const moveToOtherSide = useCallback(
    (card: PickedCard) => reassign(card.path, card.side === "good" ? "bad" : "good"),
    [reassign],
  );

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

  const backToVoting = useCallback(() => {
    // Verdicts survive the round-trip — the user lands back in voting with
    // everything scored and can navigate freely to change individual votes.
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

  const openReview = useCallback((list: string[], path?: string) => {
    const pos = path ? Math.max(0, list.indexOf(path)) : 0;
    setReview({ list, pos });
  }, []);

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
            Image {cursor + 1} / {images.length}
          </span>
        ) : (
          <span className="tabular-nums">{pairs.length} pair(s) ready</span>
        )}
        <span className="text-[var(--color-border-subtle)]">·</span>
        <span className="tabular-nums">
          <span style={{ color: GOOD_COLOR }}>Good {counts.good}</span>{" "}
          <span style={{ color: BAD_COLOR }}>Bad {counts.bad}</span>{" "}
          <span style={{ color: SKIP_COLOR }}>Skip {counts.skip}</span>
          {unscored > 0 && <span className="text-[var(--color-on-surface-secondary)]"> Left {unscored}</span>}
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
    const current = images[cursor];
    const currentVerdict = verdicts[cursor];
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

          {/* Existing verdict badge — visible when revisiting a scored image */}
          {currentVerdict && (
            <div
              className="absolute top-8 left-3 px-2.5 py-1 rounded text-xs font-bold pointer-events-none"
              style={{
                background: "rgba(0, 0, 0, 0.7)",
                color: verdictColor(currentVerdict),
                border: `1px solid ${verdictColor(currentVerdict)}`,
              }}
            >
              {verdictLabel(currentVerdict)}
            </div>
          )}

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
                    className={`relative shrink-0 h-20 rounded border-2 overflow-hidden bg-black/40 cursor-pointer ${
                      i === cursor ? "ring-2 ring-[var(--color-cobalt-600)]" : ""
                    }`}
                    style={{
                      borderColor: verdictColor(verdicts[i]),
                      borderStyle: verdicts[i] === "skip" ? "dashed" : "solid",
                      ...arStyle,
                    }}
                    title={basename(img)}
                    onClick={() => setCursor(i)}
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
              {cursor + 1} / {images.length}
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
              cursor === 0 ? "opacity-40 cursor-default" : "cursor-pointer"
            }`}
          >
            <Undo2 className="w-3.5 h-3.5" />
            Undo — Backspace
          </div>
          <div className="px-3 py-1.5 rounded font-semibold text-[var(--color-on-surface-secondary)] border border-[var(--color-border-subtle)]">
            ← / → — browse
          </div>
          <div
            role="button"
            tabIndex={-1}
            onClick={finishNow}
            className="px-3 py-1.5 rounded cursor-pointer font-semibold"
            style={{ color: "#fff", background: "var(--color-cobalt-600)" }}
          >
            {unscored > 0 ? `Finish — skip ${unscored} unscored` : "Finish → pairs"}
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
          title="Inspect & rescore full size"
          onClick={(e) => {
            e.stopPropagation();
            openReview(images, card.path);
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

  const renderSkippedCard = (path: string) => (
    <div
      key={path}
      className="group/card relative h-28 rounded border-2 border-dashed overflow-hidden bg-black/30 cursor-pointer transition-all hover:shadow-lg"
      style={{ borderColor: SKIP_COLOR, ...arStyle }}
      title={basename(path)}
      onClick={() => openReview(images, path)}
    >
      <img
        src={imageUrl(path)}
        alt={basename(path)}
        loading="lazy"
        className="w-full h-full object-cover opacity-70 group-hover/card:opacity-100 transition-opacity"
        draggable={false}
      />
      <div className="absolute top-1 right-1 flex gap-1 opacity-0 group-hover/card:opacity-100 transition-opacity">
        <div
          role="button"
          tabIndex={-1}
          className="p-1 rounded bg-black/70 hover:bg-black/90"
          style={{ color: GOOD_COLOR }}
          title="Mark Good"
          onClick={(e) => {
            e.stopPropagation();
            reassign(path, "good");
          }}
        >
          <ThumbsUp className="w-3.5 h-3.5" />
        </div>
        <div
          role="button"
          tabIndex={-1}
          className="p-1 rounded bg-black/70 hover:bg-black/90"
          style={{ color: BAD_COLOR }}
          title="Mark Bad"
          onClick={(e) => {
            e.stopPropagation();
            reassign(path, "bad");
          }}
        >
          <ThumbsDown className="w-3.5 h-3.5" />
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

        {/* Skipped pile — rescorable in place or via the full-screen pass */}
        {skipped.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold" style={{ color: SKIP_COLOR }}>
                Skipped ({skipped.length}) — click to inspect & rescore
              </span>
              <div
                role="button"
                tabIndex={-1}
                onClick={() => openReview([...skipped])}
                className="px-2 py-0.5 rounded text-xs font-semibold cursor-pointer flex items-center gap-1"
                style={{ color: "#fff", background: "var(--color-cobalt-600)" }}
              >
                <Eye className="w-3.5 h-3.5" />
                Rescore skipped
              </div>
            </div>
            <div className="flex flex-wrap gap-2">{skipped.map(renderSkippedCard)}</div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={backToVoting}>
            <CornerUpLeft className="w-4 h-4" />
            Back to voting
          </Button>
          <Button variant="ghost" size="sm" onClick={() => openReview(images)}>
            <Eye className="w-4 h-4" />
            Review all
          </Button>
          <span className="text-[10px] text-[var(--color-on-surface-secondary)]">
            Re-voting rebuilds pairs — manual swaps reset. Rescoring here keeps them.
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

      {review && (
        <TriageReviewOverlay
          list={review.list}
          pos={review.pos}
          verdictFor={(p) => verdicts[images.indexOf(p)]}
          onVerdict={reassign}
          onNavigate={(pos) => setReview((r) => (r ? { ...r, pos } : r))}
          onClose={() => setReview(null)}
        />
      )}
    </div>
  );
}
