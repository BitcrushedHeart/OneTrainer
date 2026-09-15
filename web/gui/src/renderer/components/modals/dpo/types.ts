import type {
  GroupData,
  GroupOrder,
  RealismDevice,
  ReviewPair,
  ScoreBreakdown,
  SessionStatus,
  SwissScores,
} from "@/api/dpoApi";

import type { ScoreSortMode, ScoreWeights } from "./scoreCombine";

export type { GroupData, GroupOrder, RealismDevice, ReviewPair, ScoreBreakdown, SessionStatus, SwissScores };

export type CurationStep = "setup" | "scanning" | "selecting" | "swiss" | "ranking" | "triage" | "review" | "export";

export type CurationMode = "selection" | "swiss" | "triage";

export type SelectionPhase = "best" | "worst";

export interface SessionConfig {
  sourceFolder: string;
  outputDir: string;
  pairsPerGroup: number;
  mode: CurationMode;
  groupOrder: GroupOrder;
  /** Comma-separated caption keywords; all must appear. Empty = no filter. */
  captionFilter: string;
  /** Skip prompts with fewer images than this. 2 (the pair floor) = no filter. */
  minImagesPerGroup: number;
}

export interface ResumeInfo {
  existingPairs: number;
  pruned: number;
}

export interface FinalResult {
  total_pairs: number;
  train_count: number;
  val_count: number;
}

export interface SetupStepProps {
  sourceFolder: string;
  outputDir: string;
  pairsPerGroup: number;
  groupOrder: GroupOrder;
  captionFilter: string;
  minImagesPerGroup: number;
  onSourceChange: (dir: string) => void;
  onOutputChange: (dir: string) => void;
  onPairsPerGroupChange: (n: number) => void;
  onGroupOrderChange: (order: GroupOrder) => void;
  onCaptionFilterChange: (text: string) => void;
  onMinImagesPerGroupChange: (n: number) => void;
  onStart: (config: SessionConfig) => Promise<void>;
  onReview: () => void;
  onClose: () => void;
  resume: ResumeInfo | null;
  onSelectOutput: (dir: string) => void;
  error: string | null;
  // Realism scoring, chosen before the session starts. "off" is the reason this
  // lives on setup at all: the scan orders groups by score, so by the time
  // Selection's toggle is reachable the model has already been asked.
  realismDevice: RealismDevice;
  onSetRealismDevice: (device: RealismDevice) => void;
}

export interface ScanningStepProps {
  scanCount: number;
  scanTotal: number;
  hashCount: number;
  cacheHits: number;
  onCancel: () => void;
}

export interface SelectionStepProps {
  group: GroupData;
  onPick: (path: string) => Promise<void>;
  // discardRest commits the pair and then retires the whole group from the
  // source tree (paired sources → .chosen/.rejected, leftovers → .discard).
  onAcceptPair: (keepScoring: boolean, discardRest?: boolean) => void;
  onDismissPair: () => void;
  onSkipGroup: () => void;
  onDiscardGroup: () => void;
  onCancel: () => void;
  phase: SelectionPhase;
  bestImage: string | null;
  remainingImages: string[];
  pairsDone: number;
  showAcceptDialog: boolean;
  pendingBest: string | null;
  pendingWorst: string | null;
  // P(AI-generated) per path. Drives the grid order (most photorealistic
  // first) and the per-tile badge. Empty while the group is still being scored,
  // and empty for good if the model is unavailable — the grid falls back to
  // source order in both cases.
  realismScores?: Record<string, number>;
  realismError?: string | null;
  // Whole-tree backfill: scores every image in the source folder on every
  // axis once so no group ever loads a model. Optional so the step still
  // renders in modes and tests that do not wire it.
  backfill?: {
    running: boolean;
    done: number;
    total: number;
    errors: Record<string, string>;
    error: string | null;
  };
  onStartBackfill?: () => void;
  onStopBackfill?: () => void;
  // Scoring device. "cpu" keeps the head off the GPU — slower, but it leaves
  // the VRAM to a training or generation run sharing the machine; "off" skips
  // scoring entirely, so no badges and no sort.
  realismDevice?: RealismDevice;
  onSetRealismDevice?: (device: RealismDevice) => void;
  // Every scorer's verdict per path (realism, prompt adherence, age, on-image
  // text, quality) for the per-tile breakdown panel and the combined sort. Empty until the
  // group has been scored; per-scorer failures arrive in the errors map so
  // the panel degrades a row at a time.
  scoreBreakdown?: Record<string, ScoreBreakdown>;
  scoreBreakdownErrors?: Record<string, string>;
  // Per-prompt weights over the five scorers. Changing them re-sorts the grid
  // client-side; nothing is re-scored.
  scoreWeights?: ScoreWeights;
  onSetScoreWeights?: (weights: ScoreWeights) => void;
  scoreSortMode?: ScoreSortMode;
  onSetScoreSortMode?: (mode: ScoreSortMode) => void;
}

export interface TournamentStepProps {
  group: GroupData;
  match: [string, string] | null;
  round: number;
  totalRounds: number;
  matchesPlayed: number;
  matchesTotal: number;
  onVote: (winner: "a" | "b" | "tie") => Promise<void>;
  onFinishEarly: () => Promise<void>;
  onSkipGroup: () => void;
  onDiscardGroup: () => void;
  onCancel: () => void;
  pairsDone: number;
}

export interface RankingStepProps {
  group: GroupData;
  order: string[];
  scores: SwissScores;
  maxPairs: number;
  pairCount: number;
  onPairCountChange: (n: number) => void;
  onSwap: (i: number, j: number) => Promise<void>;
  // discardRest exports the chosen pairs and then retires the group from the
  // source tree, so the mid-ranked leftovers can't reappear in a later session.
  onExport: (discardRest?: boolean) => Promise<void>;
  onSkipGroup: () => void;
  onDiscardGroup: () => void;
  onCancel: () => void;
  pairsDone: number;
}

export interface TriageStepProps {
  group: GroupData;
  pairsDone: number;
  // discardRest also retires the whole group from the source tree (paired
  // sources → .chosen/.rejected, leftovers → .discard) so it can't reappear.
  onCommitPairs: (pairs: Array<{ chosen: string; rejected: string }>, discardRest?: boolean) => Promise<void>;
  onAutoAlign: (
    good: string[],
    bad: string[],
  ) => Promise<{
    pairs: Array<{ chosen: string; rejected: string }>;
    chosenPool: string[];
    rejectedPool: string[];
  } | null>;
  onSkipGroup: () => void;
  onDiscardGroup: () => void;
  onCancel: () => void;
}

export interface ReviewStepProps {
  onBack: () => void;
  onClose: () => void;
}

export interface ExportStepProps {
  totalPairs: number;
  onFinalize: (valPct: number) => Promise<void>;
  onClose: () => void;
  finalResult: FinalResult | null;
}
