import type { GroupData, ReviewPair, SessionStatus, SwissScores } from "@/api/dpoApi";

export type { GroupData, ReviewPair, SessionStatus, SwissScores };

export type CurationStep = "setup" | "scanning" | "selecting" | "swiss" | "ranking" | "triage" | "review" | "export";

export type CurationMode = "selection" | "swiss" | "triage";

export type SelectionPhase = "best" | "worst";

export interface SessionConfig {
  sourceFolder: string;
  outputDir: string;
  pairsPerGroup: number;
  mode: CurationMode;
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
  onSourceChange: (dir: string) => void;
  onOutputChange: (dir: string) => void;
  onPairsPerGroupChange: (n: number) => void;
  onStart: (config: SessionConfig) => Promise<void>;
  onReview: () => void;
  onClose: () => void;
  resume: ResumeInfo | null;
  onSelectOutput: (dir: string) => void;
  error: string | null;
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
  onAcceptPair: (keepScoring: boolean) => void;
  onDismissPair: () => void;
  onSkipGroup: () => void;
  onCancel: () => void;
  phase: SelectionPhase;
  bestImage: string | null;
  remainingImages: string[];
  pairsDone: number;
  showAcceptDialog: boolean;
  pendingBest: string | null;
  pendingWorst: string | null;
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
  onExport: () => Promise<void>;
  onSkipGroup: () => void;
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
