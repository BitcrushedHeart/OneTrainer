import type { EloPairResponse, GroupData, ReviewPair, SessionStatus } from "@/api/dpoApi";

export type { EloPairResponse, GroupData, ReviewPair, SessionStatus };

export type CurationStep = "setup" | "scanning" | "selecting" | "elo" | "review" | "export";

export type CurationMode = "selection" | "elo";

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

export interface EloStepProps {
  group: GroupData;
  pair: [string, string] | null;
  ratings: Record<string, number>;
  done: number;
  suggested: number;
  finished: boolean;
  onVote: (winner: "a" | "b" | "tie") => Promise<void>;
  onAccept: (continueScoring: boolean) => Promise<void>;
  onSkipGroup: () => void;
  onCancel: () => void;
  pairsDone: number;
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
