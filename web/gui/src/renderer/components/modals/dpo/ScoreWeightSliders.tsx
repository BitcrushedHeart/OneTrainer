import { RotateCcw } from "lucide-react";

import { Button } from "@/components/shared";

import {
  DEFAULT_SCORE_WEIGHTS,
  MAX_SCORE_WEIGHT,
  SCORE_AXES,
  type ScoreAxis,
  type ScoreSortMode,
  type ScoreWeights,
} from "./scoreCombine";

export interface ScoreWeightSlidersProps {
  weights: ScoreWeights;
  onChange: (weights: ScoreWeights) => void;
  sortMode: ScoreSortMode;
  onSortModeChange: (mode: ScoreSortMode) => void;
  /** Axes the current group has no scores for (greyed, still adjustable). */
  unavailable?: Partial<Record<ScoreAxis, boolean>>;
}

const LABEL: Record<ScoreAxis, string> = {
  text: "Text",
  adherence: "Prompt",
  age: "Age",
  realism: "Realism",
  quality: "Quality",
};

const TITLE: Record<ScoreAxis, string> = {
  text: "On-image text — an OCR model reads the image and checks the text the caption asks for (only when it asks for some)",
  adherence: "Prompt adherence — switched off (the SigLIP scorer was too slow and not discriminative enough)",
  age: "Age — Engine's age head against the age the caption claims (only when it claims one)",
  realism: "Realism — how convincing the realism head finds the image (lower P(AI) wins)",
  quality: "Scalar quality — sharpness, contrast, exposure (Studio's formula)",
};

/**
 * Per-prompt weighting of the five scorers. Changing a slider re-sorts the
 * grid and re-scores the Combined line in every tile immediately: the
 * combination is computed client-side from the breakdown the server sent.
 */
export function ScoreWeightSliders({
  weights,
  onChange,
  sortMode,
  onSortModeChange,
  unavailable,
}: ScoreWeightSlidersProps) {
  const isDefault = SCORE_AXES.every((a) => weights[a] === DEFAULT_SCORE_WEIGHTS[a]);
  return (
    <div className="flex items-center gap-4 flex-wrap text-xs text-[var(--color-on-surface-secondary)]">
      <span className="font-semibold text-[var(--color-on-surface)]">Weights (this prompt)</span>
      {SCORE_AXES.map((axis) => {
        const off = weights[axis] === 0;
        const missing = unavailable?.[axis];
        return (
          <label key={axis} className="flex items-center gap-1.5" title={TITLE[axis]}>
            <span className="w-12" style={{ opacity: off || missing ? 0.5 : 1 }}>
              {LABEL[axis]}
              {missing ? " –" : ""}
            </span>
            <input
              type="range"
              min={0}
              max={MAX_SCORE_WEIGHT}
              step={0.1}
              value={weights[axis]}
              onChange={(e) => onChange({ ...weights, [axis]: Number(e.target.value) })}
              className="w-20 h-1.5 cursor-pointer accent-[var(--color-cobalt-600)]"
              aria-label={`${LABEL[axis]} weight`}
            />
            <span className="w-6 font-mono tabular-nums">{weights[axis].toFixed(1)}</span>
          </label>
        );
      })}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onChange({ ...DEFAULT_SCORE_WEIGHTS })}
        disabled={isDefault}
        title="Back to the default weights"
      >
        <RotateCcw className="w-3.5 h-3.5" />
        Reset
      </Button>
      <span className="text-[var(--color-border-subtle)]">·</span>
      <label className="flex items-center gap-1.5">
        <span>Sort</span>
        <select
          value={sortMode}
          onChange={(e) => onSortModeChange(e.target.value as ScoreSortMode)}
          className="bg-[var(--color-surface-container)] border border-[var(--color-border-subtle)] rounded px-1 py-0.5 text-xs text-[var(--color-on-surface)]"
          aria-label="Grid sort"
        >
          <option value="combined">Combined score</option>
          <option value="realism">Realism only</option>
        </select>
      </label>
    </div>
  );
}
