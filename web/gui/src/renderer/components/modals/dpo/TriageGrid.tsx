import { Check, Loader2, SkipForward } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { dpoApi } from "@/api/dpoApi";
import type { TriageCategory, TriageItem } from "@/api/dpoTriageFixApi";
import { Button } from "@/components/shared";

import { ThumbSizeSlider, useThumbSize } from "./SecondRoundGrid";
import { CATEGORY_OTHER, type TagSet } from "./triageFix";

export interface TriageGridProps {
  /** The triage walk as the keyword filter leaves it, in walk order. */
  order: TriageItem[];
  /** Tags toggled per image this session (detector tags included). */
  tags: Record<string, TagSet>;
  categories: TriageCategory[];
  busy: boolean;
  /** Show one image in the one-at-a-time view. */
  onOpen: (stem: string) => void;
  /** Queue the selection with one set of fixes (the caller asks which). */
  onConfirm: (stems: string[]) => void;
  /** Skip the selection for good. */
  onSkip: (stems: string[]) => Promise<void>;
}

/** First-round triage as a grid: the same images as the one-at-a-time walk, for deciding many at once. */
export function TriageGrid({ order, tags, categories, busy, onOpen, onConfirm, onSkip }: TriageGridProps) {
  const [thumbSize, changeThumbSize] = useThumbSize();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const anchor = useRef<number | null>(null);

  // The selection survives the keyword filter, so a batch can be built from several searches; it
  // only drops images that left triage.
  const byStem = useMemo(() => new Map(order.map((item) => [item.stem, item])), [order]);
  useEffect(() => {
    setSelected((current) => {
      const next = new Set([...current].filter((stem) => byStem.has(stem)));
      return next.size === current.size ? current : next;
    });
    anchor.current = null;
  }, [byStem]);
  const selection = useMemo(() => [...selected].filter((stem) => byStem.has(stem)), [selected, byStem]);

  const labels = useMemo(() => new Map(categories.map((cat) => [cat.id, cat.label])), [categories]);

  const toggle = useCallback(
    (index: number, shift: boolean) => {
      setSelected((current) => {
        const next = new Set(current);
        if (shift && anchor.current !== null) {
          const [from, to] = anchor.current < index ? [anchor.current, index] : [index, anchor.current];
          for (let i = from; i <= to; i++) next.add(order[i].stem);
        } else {
          const stem = order[index].stem;
          if (next.has(stem)) next.delete(stem);
          else next.add(stem);
        }
        return next;
      });
      anchor.current = index;
    },
    [order],
  );

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      <div className="flex items-center gap-3 flex-wrap text-xs text-[var(--color-on-surface-secondary)]">
        <ThumbSizeSlider value={thumbSize} onChange={changeThumbSize} />
        <span>Click selects · Shift+click a range · double-click opens</span>
        <span className="ml-auto flex items-center gap-2 flex-wrap">
          <span className="tabular-nums">
            {order.length} shown · {selection.length} selected
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelected((current) => new Set([...current, ...order.map((item) => item.stem)]))}
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
            onClick={() => void onSkip(selection)}
            disabled={busy || selection.length === 0}
            title="Leave them as they are, for good (Backspace undoes one at a time)"
          >
            <SkipForward className="w-3.5 h-3.5" /> Skip {selection.length || ""} forever
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
        {order.length === 0 ? (
          <div className="text-sm text-[var(--color-on-surface-secondary)] p-4">Nothing left to triage here.</div>
        ) : (
          <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSize}px, 1fr))` }}>
            {order.map((item, index) => {
              const on = selected.has(item.stem);
              const tagSet = tags[item.stem];
              const tagged = tagSet
                ? [...tagSet.ids].map((id) =>
                    id === CATEGORY_OTHER && tagSet.others.length > 0
                      ? tagSet.others.map((o) => o.target).join(", ")
                      : (labels.get(id) ?? id),
                  )
                : [];
              return (
                <div
                  key={item.stem}
                  role="button"
                  tabIndex={-1}
                  onClick={(e) => toggle(index, e.shiftKey)}
                  onDoubleClick={() => onOpen(item.stem)}
                  title={`${item.stem} · double-click to open${item.caption ? `\n${item.caption}` : ""}`}
                  className={`relative rounded border-2 overflow-hidden bg-black/40 cursor-pointer select-none ${
                    on ? "border-[var(--color-cobalt-600)]" : "border-transparent"
                  }`}
                >
                  <img
                    src={dpoApi.imageUrl(item.chosen_path)}
                    alt=""
                    loading="lazy"
                    className="w-full object-contain"
                    style={{ height: Math.round(thumbSize * 1.2) }}
                    draggable={false}
                  />
                  <div className="absolute inset-x-0 bottom-0 px-1.5 py-0.5 text-[10px] bg-black/60 text-white">
                    <span className="block truncate font-mono">{item.stem}</span>
                  </div>
                  <div className="absolute top-1 left-1 right-6 flex flex-wrap gap-1 text-[10px]">
                    {tagged.map((label) => (
                      <span key={label} className="px-1 rounded bg-[#15803d] text-white truncate max-w-full">
                        {label}
                      </span>
                    ))}
                    {item.round > 1 && (
                      <span className="px-1 rounded bg-[var(--color-cobalt-600)] text-white">round {item.round}</span>
                    )}
                    {item.attempt > 0 && item.round <= 1 && (
                      <span className="px-1 rounded bg-black/70 text-white">back from review</span>
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
