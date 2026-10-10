import { Check, Loader2, SkipForward } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { dpoApi } from "@/api/dpoApi";
import type { TriageItem } from "@/api/dpoTriageFixApi";
import { Button } from "@/components/shared";

import { inSecondRoundWalk, secondKind } from "./triageFix";

const THUMB_KEY = "ot.triageFix.fixedThumbSize";
const THUMB_MIN = 80;
const THUMB_MAX = 480;
const THUMB_DEFAULT = 160;

function readThumbSize(): number {
  try {
    const value = Number(window.localStorage.getItem(THUMB_KEY));
    return value >= THUMB_MIN && value <= THUMB_MAX ? value : THUMB_DEFAULT;
  } catch {
    return THUMB_DEFAULT;
  }
}

/** The grid thumbnail size, remembered across sessions and shared by both triage grids. */
export function useThumbSize(): [number, (value: number) => void] {
  const [thumbSize, setThumbSize] = useState(readThumbSize);
  const changeThumbSize = useCallback((value: number) => {
    setThumbSize(value);
    try {
      window.localStorage.setItem(THUMB_KEY, String(value));
    } catch {
      // Private mode or a full store: the size still applies for this session.
    }
  }, []);
  return [thumbSize, changeThumbSize];
}

export function ThumbSizeSlider({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  return (
    <label className="flex items-center gap-2" title="Thumbnail size">
      Size
      <input
        type="range"
        min={THUMB_MIN}
        max={THUMB_MAX}
        step={20}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-28"
      />
      <span className="tabular-nums w-10">{value}px</span>
    </label>
  );
}

function startOfToday(): number {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.getTime() / 1000;
}

function when(seconds: number | null): string {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export interface SecondRoundGridProps {
  /** Every second-round image the screen's kind and keyword filters let through, handled ones included. */
  pool: TriageItem[];
  busy: boolean;
  /** The session's fix-pair folder; empty when it was started without one. */
  fixPairDir: string;
  /** Show one image in the one-at-a-time view (a handled one too). */
  onOpen: (stem: string) => void;
  /** Queue the selection with one set of fixes (the caller asks which). */
  onConfirm: (stems: string[]) => void;
  /** Pass the selection over for this round. */
  onSkip: (stems: string[]) => Promise<void>;
  /** Write the missing fix pairs of the selection (fixes accepted without a fix-pair folder). */
  onAddFixPairs: (stems: string[]) => Promise<void>;
}

/**
 * The second round as a grid: the same images as the one-at-a-time walk (plus, on request, the ones
 * already handled), newest decision first, for picking many at once.
 */
export function SecondRoundGrid({
  pool,
  busy,
  fixPairDir,
  onOpen,
  onConfirm,
  onSkip,
  onAddFixPairs,
}: SecondRoundGridProps) {
  const [includeHandled, setIncludeHandled] = useState(false);
  const [todayOnly, setTodayOnly] = useState(false);
  const [noPairOnly, setNoPairOnly] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const [thumbSize, changeThumbSize] = useThumbSize();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const anchor = useRef<number | null>(null);

  const visible = useMemo(() => {
    const since = startOfToday();
    return pool
      .filter((item) => includeHandled || inSecondRoundWalk(item))
      .filter((item) => !todayOnly || (item.decided_at ?? 0) >= since)
      .filter((item) => !noPairOnly || (secondKind(item) === "fixed" && !item.has_fix_pair))
      .sort((a, b) => (b.decided_at ?? 0) - (a.decided_at ?? 0));
  }, [pool, includeHandled, todayOnly, noPairOnly]);

  // The selection survives filtering, so a batch can be built from several searches; it only drops
  // images that left the second round's pool.
  useEffect(() => {
    const inPool = new Set(pool.map((item) => item.stem));
    setSelected((current) => {
      const next = new Set([...current].filter((stem) => inPool.has(stem)));
      return next.size === current.size ? current : next;
    });
  }, [pool]);
  useEffect(() => {
    anchor.current = null;
  }, [visible]);

  const byStem = useMemo(() => new Map(pool.map((item) => [item.stem, item])), [pool]);
  const selection = useMemo(() => [...selected].filter((stem) => byStem.has(stem)), [selected, byStem]);
  const hiddenSelected = useMemo(() => {
    const onScreen = new Set(visible.map((item) => item.stem));
    return selection.filter((stem) => !onScreen.has(stem)).length;
  }, [visible, selection]);
  const missingPairs = useMemo(
    () =>
      selection.filter((stem) => {
        const item = byStem.get(stem);
        return item !== undefined && item.status === "fixed" && !item.has_fix_pair;
      }),
    [selection, byStem],
  );

  const toggle = useCallback(
    (index: number, shift: boolean) => {
      setSelected((current) => {
        const next = new Set(current);
        if (shift && anchor.current !== null) {
          const [from, to] = anchor.current < index ? [anchor.current, index] : [index, anchor.current];
          for (let i = from; i <= to; i++) next.add(visible[i].stem);
        } else {
          const stem = visible[index].stem;
          if (next.has(stem)) next.delete(stem);
          else next.add(stem);
        }
        return next;
      });
      anchor.current = index;
    },
    [visible],
  );

  const check = (value: boolean, set: (v: boolean) => void, label: string, title?: string) => (
    <label className="flex items-center gap-1.5 cursor-pointer" title={title}>
      <input type="checkbox" checked={value} onChange={(e) => set(e.target.checked)} />
      {label}
    </label>
  );

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      <div className="flex items-center gap-3 flex-wrap text-xs text-[var(--color-on-surface-secondary)]">
        {check(
          includeHandled,
          setIncludeHandled,
          "Include handled",
          "Also show images already confirmed or skipped this round",
        )}
        {check(todayOnly, setTodayOnly, "Decided today")}
        {check(noPairOnly, setNoPairOnly, "Fixed without a fix pair")}
        {check(showOriginal, setShowOriginal, "Show originals", "Show fixed images as they were before any repair")}
        <ThumbSizeSlider value={thumbSize} onChange={changeThumbSize} />
        <span className="ml-auto flex items-center gap-2 flex-wrap">
          <span className="tabular-nums">
            {visible.length} shown · {selection.length} selected
            {hiddenSelected > 0 && ` (${hiddenSelected} not shown)`}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelected((current) => new Set([...current, ...visible.map((item) => item.stem)]))}
            title="Add every image shown to the selection"
          >
            Select shown
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setSelected(new Set())} disabled={selection.length === 0}>
            Clear
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void onAddFixPairs(missingPairs)}
            disabled={busy || !fixPairDir || missingPairs.length === 0}
            title={
              fixPairDir
                ? `Write (fixed = chosen, original = rejected) pairs into ${fixPairDir} for the selected fixes that have none`
                : "The session was started without a fix pair folder; set one and start again"
            }
          >
            Add {missingPairs.length || ""} to fix dataset
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void onSkip(selection)}
            disabled={busy || selection.length === 0}
            title="Leave them as they are and out of this round"
          >
            <SkipForward className="w-3.5 h-3.5" /> Skip {selection.length || ""}
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => onConfirm(selection)}
            disabled={busy || selection.length === 0}
          >
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
            Confirm {selection.length || ""}…
          </Button>
        </span>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {visible.length === 0 ? (
          <div className="text-sm text-[var(--color-on-surface-secondary)] p-4">Nothing here.</div>
        ) : (
          <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSize}px, 1fr))` }}>
            {visible.map((item, index) => {
              const on = selected.has(item.stem);
              const kind = secondKind(item);
              const waiting = inSecondRoundWalk(item);
              const src = showOriginal && item.original_path ? item.original_path : item.chosen_path;
              return (
                <div
                  key={item.stem}
                  role="button"
                  tabIndex={-1}
                  onClick={(e) => toggle(index, e.shiftKey)}
                  onDoubleClick={() => onOpen(item.stem)}
                  title={`${item.stem} · double-click to open`}
                  className={`relative rounded border-2 overflow-hidden bg-black/40 cursor-pointer select-none ${
                    on ? "border-[var(--color-cobalt-600)]" : "border-transparent"
                  } ${waiting ? "" : "opacity-60"}`}
                >
                  <img
                    src={dpoApi.imageUrl(src)}
                    alt=""
                    loading="lazy"
                    className="w-full object-contain"
                    style={{ height: Math.round(thumbSize * 1.2) }}
                    draggable={false}
                  />
                  <div className="absolute inset-x-0 bottom-0 px-1.5 py-0.5 text-[10px] bg-black/60 text-white flex gap-1.5">
                    <span className="truncate flex-1 font-mono">{item.stem}</span>
                    <span>{when(item.decided_at)}</span>
                  </div>
                  <div className="absolute top-1 left-1 flex flex-wrap gap-1 text-[10px]">
                    {kind && <span className="px-1 rounded bg-black/70 text-white">{kind}</span>}
                    {item.round > 1 && (
                      <span className="px-1 rounded bg-[var(--color-cobalt-600)] text-white">round {item.round}</span>
                    )}
                    {kind === "fixed" && !item.has_fix_pair && (
                      <span className="px-1 rounded bg-black/70 text-white">no fix pair</span>
                    )}
                    {!waiting && (
                      <span className="px-1 rounded bg-black/70 text-white">
                        {item.second_pass === "skipped" ? "skipped this round" : item.status}
                      </span>
                    )}
                    {showOriginal && item.original_path && (
                      <span className="px-1 rounded bg-black/70 text-white">original</span>
                    )}
                  </div>
                  {on && (
                    <div className="absolute top-1 right-1 w-4 h-4 rounded-full bg-[var(--color-cobalt-600)] border-2 border-white" />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
