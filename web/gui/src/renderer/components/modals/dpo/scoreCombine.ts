/**
 * Client-side combination of the DPO curation scorers.
 *
 * The server sends every scorer's verdict per image (`ScoreBreakdown`); the
 * per-prompt weight sliders re-combine them here so a slider move re-sorts the
 * grid instantly, without a round trip. This is a port of
 * `modules/util/dpo_combined_score.py::weighted_z` — within-group z-scores,
 * sign-corrected (realism is P(AI), lower is better), weighted, with axes an
 * image lacks dropped from *its* mean. Keep the two in step.
 */

export type ScoreAxis = "text" | "adherence" | "age" | "realism" | "quality";

/** How the Selection grid is ordered: the weighted combination, or realism alone (the pre-breakdown behaviour). */
export type ScoreSortMode = "combined" | "realism";

export type ScoreWeights = Record<ScoreAxis, number>;

/** One image's axes, `null` where a scorer had nothing to say. */
export type ScoreAxes = Record<ScoreAxis, number | null | undefined>;

export const SCORE_AXES: ScoreAxis[] = ["text", "adherence", "age", "realism", "quality"];

/**
 * Mirrors `DEFAULT_WEIGHTS` on the server. Text is the heaviest by default — a
 * wrong word on a chosen image is poison for a text-rendering model — and
 * outweighs the other defaults combined (3.5), so a verbatim sign beats a
 * garbled one even when every other axis prefers the garbled image.
 */
export const DEFAULT_SCORE_WEIGHTS: ScoreWeights = { text: 4, adherence: 1, age: 1, realism: 1, quality: 0.5 };

/** Upper bound of every weight slider. */
export const MAX_SCORE_WEIGHT = 5;

const SIGN: Record<ScoreAxis, 1 | -1> = { text: 1, adherence: 1, age: 1, realism: -1, quality: 1 };

export function weightedZ(images: Record<string, ScoreAxes>, weights: ScoreWeights): Record<string, number> {
  const stats = {} as Record<ScoreAxis, { mean: number; std: number }>;
  for (const axis of SCORE_AXES) {
    const values: number[] = [];
    for (const axes of Object.values(images)) {
      const v = axes[axis];
      if (typeof v === "number" && Number.isFinite(v)) values.push(v);
    }
    if (values.length < 2) {
      stats[axis] = { mean: values[0] ?? 0, std: 0 };
      continue;
    }
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    stats[axis] = { mean, std: Math.sqrt(variance) };
  }

  const out: Record<string, number> = {};
  for (const [path, axes] of Object.entries(images)) {
    let numerator = 0;
    let denominator = 0;
    for (const axis of SCORE_AXES) {
      const weight = weights[axis] ?? 0;
      const value = axes[axis];
      if (weight === 0 || typeof value !== "number" || !Number.isFinite(value)) continue;
      const { mean, std } = stats[axis];
      const z = std > 0 ? (value - mean) / std : 0;
      numerator += weight * SIGN[axis] * z;
      denominator += weight;
    }
    out[path] = denominator > 0 ? numerator / denominator : 0;
  }
  return out;
}

/** Paths best-first under `weightedZ`; unscored paths keep their order at the end. */
export function sortByCombined(paths: string[], images: Record<string, ScoreAxes>, weights: ScoreWeights): string[] {
  const scored: Record<string, ScoreAxes> = {};
  for (const p of paths) if (images[p]) scored[p] = images[p];
  const z = weightedZ(scored, weights);
  return [...paths].sort((a, b) => {
    const za = z[a];
    const zb = z[b];
    if (za === undefined && zb === undefined) return 0;
    if (za === undefined) return 1;
    if (zb === undefined) return -1;
    return zb - za;
  });
}
