import type { ScoreBreakdown, ScoreProposition } from "@/api/dpoApi";

export interface ScoreBreakdownPanelProps {
  breakdown?: ScoreBreakdown;
  /** Combined score under the user's current weights (client-side). */
  combined?: number;
  /** Stack rows vertically beside a portrait image, or lay them out in a strip under a landscape one. */
  layout: "side" | "below";
}

// Shared ramp with the realism badge: green is good, red is bad. Coarse on
// purpose — the ordering is the signal, the number is a hint.
function rampGood(p: number): string {
  if (p >= 0.75) return "#22c55e";
  if (p >= 0.5) return "#eab308";
  if (p >= 0.25) return "#f97316";
  return "#ef4444";
}

function realismColor(pAi: number): string {
  if (pAi < 0.5) return "#22c55e";
  if (pAi < 0.8) return "#eab308";
  if (pAi < 0.95) return "#f97316";
  return "#ef4444";
}

function combinedColor(z: number): string {
  if (z >= 0.5) return "#22c55e";
  if (z >= 0) return "#eab308";
  if (z >= -0.5) return "#f97316";
  return "#ef4444";
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

/** The propositions the image most clearly missed, worst first. */
function weakest(props: ScoreProposition[], n: number): ScoreProposition[] {
  return [...props].sort((a, b) => a.prob - b.prob).slice(0, n);
}

function propLabel(p: ScoreProposition): string {
  if (p.axis === "text_ocr") {
    // Verdict order is verbatim, then absent, then wrong: a misspelled sign
    // is worse than no sign, so say which one this is.
    if (!p.read) return `“${p.label}” not found`;
    if (p.verdict === "verbatim") return `“${p.label}” rendered verbatim`;
    return `“${p.label}” misrendered as “${p.read}”`;
  }
  if (p.axis === "age_model") return `age ≈ ${p.estimated_age?.toFixed(0) ?? "?"} vs ${p.label ?? ""}`;
  if (p.axis && p.label) return p.label.toLowerCase();
  return p.text.replace(/^a photo of (a person )?/i, "");
}

interface RowProps {
  label: string;
  value: string;
  color?: string;
  title?: string;
  muted?: boolean;
}

function Row({ label, value, color, title, muted }: RowProps) {
  return (
    <div
      className="flex items-baseline justify-between gap-2 leading-tight"
      title={title}
      style={{ opacity: muted ? 0.45 : 1 }}
    >
      <span className="text-[var(--color-on-surface-secondary)]">{label}</span>
      <span className="font-mono font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  );
}

/**
 * Per-tile breakdown of every scorer's verdict. Lives in the space a portrait
 * image leaves beside itself (or under a landscape one), so the number that
 * put an image where it is in the grid is visible without opening anything.
 */
export function ScoreBreakdownPanel({ breakdown, combined, layout }: ScoreBreakdownPanelProps) {
  const base =
    layout === "side"
      ? "flex flex-col gap-0.5 w-[8.5rem] shrink-0 px-2 py-1.5 text-[10px] text-left bg-black/40"
      : "flex flex-col gap-0.5 w-full px-2 py-1 text-[10px] text-left bg-black/40";
  if (!breakdown) {
    return (
      <div className={base}>
        <span className="text-[var(--color-on-surface-secondary)] italic">scoring…</span>
      </div>
    );
  }
  const { realism, adherence, age, estimated_age, text, text_min, ocr_text, quality, propositions } = breakdown;
  // A text span that is anything short of verbatim is always worth showing;
  // the other propositions only when they are clearly missed.
  const misses = weakest(propositions, layout === "side" ? 3 : 2).filter(
    (p) => p.prob < 0.6 || (p.axis === "text_ocr" && p.prob < 0.999),
  );
  const worstSpan = text_min === null || text_min === undefined ? "–" : pct(text_min);
  const textTitle =
    text === null || text === undefined
      ? "On-image text: the caption mentions none, so OCR was not run"
      : `OCR match against the text the caption asks for (worst span ${worstSpan}). Read: ${
          ocr_text ? `“${ocr_text.slice(0, 300)}”` : "nothing"
        }`;
  return (
    <div className={base}>
      <Row
        label="Combined"
        value={combined === undefined ? "–" : `${combined >= 0 ? "+" : ""}${combined.toFixed(2)}`}
        color={combined === undefined ? undefined : combinedColor(combined)}
        title={`Weighted mean of the within-group z-scores under the current weights (rank #${breakdown.gate_rank} under gate-then-rank)`}
      />
      <Row
        label="Realism"
        value={realism === null || realism === undefined ? "–" : `${pct(realism)} AI`}
        color={realism === null || realism === undefined ? undefined : realismColor(realism)}
        title="Realism head: how likely the image is AI-generated (lower wins)"
        muted={realism === null || realism === undefined}
      />
      <Row
        label="Text"
        value={text === null || text === undefined ? "–" : pct(text)}
        color={text === null || text === undefined ? undefined : rampGood(text)}
        title={textTitle}
        muted={text === null || text === undefined}
      />
      <Row
        label="Prompt"
        value={adherence === null || adherence === undefined ? "–" : pct(adherence)}
        color={adherence === null || adherence === undefined ? undefined : rampGood(adherence)}
        title="Prompt adherence: switched off (the SigLIP scorer was too slow and not discriminative enough)"
        muted={adherence === null || adherence === undefined}
      />
      <Row
        label="Age"
        value={
          age === null || age === undefined
            ? estimated_age === null || estimated_age === undefined
              ? "–"
              : `≈${estimated_age.toFixed(0)}`
            : `≈${(estimated_age ?? 0).toFixed(0)} · ${pct(age)}`
        }
        color={age === null || age === undefined ? undefined : rampGood(age)}
        title="Engine age head: estimated age, and agreement with the age the caption claims"
        muted={age === null || age === undefined}
      />
      <Row
        label="Quality"
        value={quality === null || quality === undefined ? "–" : quality.toFixed(0)}
        color={quality === null || quality === undefined ? undefined : rampGood(quality / 100)}
        title="Scalar quality 0–100: sharpness, contrast, exposure"
        muted={quality === null || quality === undefined}
      />
      {misses.length > 0 && (
        <div className="mt-0.5 border-t border-white/10 pt-0.5 text-[var(--color-on-surface-secondary)]">
          {misses.map((p) => (
            <div key={p.text} className="flex justify-between gap-1 truncate" title={p.text}>
              <span className="truncate">{propLabel(p)}</span>
              <span className="font-mono tabular-nums shrink-0" style={{ color: rampGood(p.prob) }}>
                {pct(p.prob)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
