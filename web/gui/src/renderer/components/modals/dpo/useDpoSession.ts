import { useCallback, useEffect, useRef, useState } from "react";

import type { SwissStateResponse } from "@/api/dpoApi";
import { dpoApi } from "@/api/dpoApi";

import type {
  CurationMode,
  CurationStep,
  FinalResult,
  GroupData,
  ResumeInfo,
  SelectionPhase,
  SessionConfig,
  SwissScores,
} from "./types";

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

export interface UseDpoSessionResult {
  step: CurationStep;
  setStep: (s: CurationStep) => void;
  sourceFolder: string;
  outputDir: string;
  pairsPerGroup: number;
  mode: CurationMode;
  scanCount: number;
  scanTotal: number;
  hashCount: number;
  cacheHits: number;
  group: GroupData | null;
  phase: SelectionPhase;
  bestImage: string | null;
  remainingImages: string[];
  pairsDone: number;
  swissMatch: [string, string] | null;
  swissRound: number;
  swissTotalRounds: number;
  swissMatchesPlayed: number;
  swissMatchesTotal: number;
  rankedOrder: string[];
  rankingScores: SwissScores;
  maxPairs: number;
  pairCount: number;
  showAcceptDialog: boolean;
  pendingBest: string | null;
  pendingWorst: string | null;
  totalPairs: number;
  finalResult: FinalResult | null;
  resume: ResumeInfo | null;
  error: string | null;
  toast: string | null;
  actions: {
    setSourceFolder: (v: string) => void;
    setOutputDir: (v: string) => void;
    setPairsPerGroup: (n: number) => void;
    startSession: (config: SessionConfig) => Promise<void>;
    pickImage: (path: string) => Promise<void>;
    confirmPair: (keepScoring: boolean) => Promise<void>;
    cancelPendingPair: () => void;
    skipGroup: () => Promise<void>;
    discardGroup: () => Promise<void>;
    commitTriagePairs: (pairs: Array<{ chosen: string; rejected: string }>, discardRest?: boolean) => Promise<void>;
    triageAlign: (
      good: string[],
      bad: string[],
    ) => Promise<{
      pairs: Array<{ chosen: string; rejected: string }>;
      chosenPool: string[];
      rejectedPool: string[];
    } | null>;
    cancelSession: () => Promise<void>;
    swissVote: (winner: "a" | "b" | "tie") => Promise<void>;
    swissFinishEarly: () => Promise<void>;
    swapRanked: (i: number, j: number) => Promise<void>;
    setPairCount: (n: number) => void;
    swissExportPairs: () => Promise<void>;
    finalize: (valPct: number) => Promise<void>;
    setError: (msg: string | null) => void;
    setToast: (msg: string | null) => void;
    loadOutputManifest: (outputDir: string) => void;
    goToReview: () => void;
  };
}

export function useDpoSession(): UseDpoSessionResult {
  const [step, setStepState] = useState<CurationStep>("setup");
  const [sourceFolder, setSourceFolder] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [pairsPerGroup, setPairsPerGroup] = useState(1);
  const [mode, setMode] = useState<CurationMode>("selection");
  const [scanCount, setScanCount] = useState(0);
  const [scanTotal, setScanTotal] = useState(0);
  const [hashCount, setHashCount] = useState(0);
  const [cacheHits, setCacheHits] = useState(0);

  const [group, setGroup] = useState<GroupData | null>(null);
  const [phase, setPhase] = useState<SelectionPhase>("best");
  const [bestImage, setBestImage] = useState<string | null>(null);
  const [remainingImages, setRemainingImages] = useState<string[]>([]);
  const [pairsDone, setPairsDone] = useState(0);

  const [swissMatch, setSwissMatch] = useState<[string, string] | null>(null);
  const [swissRound, setSwissRound] = useState(0);
  const [swissTotalRounds, setSwissTotalRounds] = useState(0);
  const [swissMatchesPlayed, setSwissMatchesPlayed] = useState(0);
  const [swissMatchesTotal, setSwissMatchesTotal] = useState(0);
  const [rankedOrder, setRankedOrder] = useState<string[]>([]);
  const [rankingScores, setRankingScores] = useState<SwissScores>({});
  const [maxPairs, setMaxPairs] = useState(0);
  const [pairCount, setPairCount] = useState(1);

  const [showAcceptDialog, setShowAcceptDialog] = useState(false);
  const [pendingBest, setPendingBest] = useState<string | null>(null);
  const [pendingWorst, setPendingWorst] = useState<string | null>(null);

  const [totalPairs, setTotalPairs] = useState(0);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [resume, setResume] = useState<ResumeInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const scanAbortRef = useRef<AbortController | null>(null);
  const modeRef = useRef<CurationMode>("selection");
  modeRef.current = mode;

  const setStep = useCallback((s: CurationStep) => {
    setStepState(s);
    setError(null);
    setToast(null);
  }, []);

  const fetchSwissRanking = useCallback(async () => {
    const res = await dpoApi.swissRanking();
    if (!res.ok) {
      setError(res.error ?? "Ranking fetch failed");
      return;
    }
    setRankedOrder(res.order ?? []);
    setRankingScores(res.scores ?? {});
    setMaxPairs(res.max_pairs ?? 0);
    setPairCount(res.default_pairs ?? 1);
    setStepState("ranking");
  }, []);

  const applySwissState = useCallback(
    async (res: SwissStateResponse) => {
      if (!res.ok) {
        setError(res.error ?? "Tournament state fetch failed");
        return;
      }
      setSwissRound(res.round ?? 0);
      setSwissTotalRounds(res.total_rounds ?? 0);
      setSwissMatchesPlayed(res.matches_played ?? 0);
      setSwissMatchesTotal(res.matches_total ?? 0);
      if (res.finished) {
        setSwissMatch(null);
        await fetchSwissRanking();
      } else {
        setSwissMatch(res.match ?? null);
      }
    },
    [fetchSwissRanking],
  );

  const refreshSwissState = useCallback(async () => {
    await applySwissState(await dpoApi.swissState());
  }, [applySwissState]);

  const fetchNextGroup = useCallback(async () => {
    const signal = scanAbortRef.current?.signal;
    while (!signal?.aborted) {
      const res = await dpoApi.nextGroup();
      if (res.done) {
        setTotalPairs(res.total_pairs ?? 0);
        setStepState("export");
        return;
      }
      if (res.waiting) {
        if (!signal) {
          await new Promise((r) => setTimeout(r, 500));
          continue;
        }
        try {
          await sleep(500, signal);
        } catch {
          return;
        }
        continue;
      }
      if (res.group) {
        setGroup(res.group);
        setRemainingImages(res.group.images);
        setPhase("best");
        setBestImage(null);
        setPendingBest(null);
        setPendingWorst(null);
        setShowAcceptDialog(false);
        setPairsDone(res.group.pairs_done);
        const effectiveMode = res.group.mode ?? modeRef.current;
        if (effectiveMode === "swiss") {
          setRankedOrder([]);
          setRankingScores({});
          setStepState("swiss");
          await refreshSwissState();
        } else if (effectiveMode === "triage") {
          // No server prep needed — group.images is already client-side and
          // all triage voting happens locally until the batch commit.
          setStepState("triage");
        } else {
          setStepState("selecting");
        }
        return;
      }
      return;
    }
  }, [refreshSwissState]);

  const pollScan = useCallback(
    async (signal: AbortSignal) => {
      while (!signal.aborted) {
        try {
          const status = await dpoApi.sessionStatus();
          setScanCount(status.scan_count);
          setScanTotal(status.scan_total ?? 0);
          setHashCount(status.hash_count ?? 0);
          setCacheHits(status.cache_hits ?? 0);
          if (status.groups_queued > 0 || status.worker_finished) {
            await fetchNextGroup();
            return;
          }
          await sleep(500, signal);
        } catch (e) {
          if (e instanceof DOMException && e.name === "AbortError") return;
          setError(String(e));
          return;
        }
      }
    },
    [fetchNextGroup],
  );

  const startSession = useCallback(
    async (config: SessionConfig) => {
      if (!config.sourceFolder || !config.outputDir) {
        setError("Select both source and output folders.");
        return;
      }
      setError(null);
      setToast(null);
      setMode(config.mode);
      setPairsPerGroup(config.pairsPerGroup);
      setScanCount(0);
      setScanTotal(0);
      setHashCount(0);
      setCacheHits(0);

      const res = await dpoApi.startSession(config.sourceFolder, config.outputDir, config.pairsPerGroup, config.mode);
      if (!res.ok) {
        setError(res.error ?? "Failed to start session");
        return;
      }
      setResume({ existingPairs: res.existing_pairs ?? 0, pruned: res.pruned ?? 0 });
      setStepState("scanning");

      scanAbortRef.current?.abort();
      const ctrl = new AbortController();
      scanAbortRef.current = ctrl;
      void pollScan(ctrl.signal);
    },
    [pollScan],
  );

  const loadOutputManifest = useCallback((dir: string) => {
    setOutputDir(dir);
    setResume(null);
  }, []);

  const pickImage = useCallback(
    async (path: string) => {
      if (!group) return;
      if (phase === "best") {
        const res = await dpoApi.selectImage(path);
        if (!res.ok) {
          setError(res.error ?? "Selection failed");
          return;
        }
        if (res.phase === "worst") {
          setPhase("worst");
          setBestImage(res.best ?? path);
          setRemainingImages(res.remaining_images ?? remainingImages.filter((i) => i !== path));
        }
        return;
      }

      // Phase "worst": backend returns `pair_pending` and holds the write so
      // we can offer Confirm / Pick More / Cancel. On the last-possible pair
      // of a group backend auto-commits (pair_created) since Cancel would be
      // meaningless — match Ctk, which skips the dialog in that case too.
      const res = await dpoApi.selectImage(path);
      if (!res.ok) {
        setError(res.error ?? "Selection failed");
        return;
      }
      if (res.pair_pending) {
        setPendingBest(res.chosen ?? bestImage);
        setPendingWorst(res.rejected ?? path);
        setShowAcceptDialog(true);
        return;
      }
      if (res.pair_created) {
        setPairsDone(res.pairs_done ?? pairsDone + 1);
        await fetchNextGroup();
      }
    },
    [group, phase, remainingImages, bestImage, pairsDone, fetchNextGroup],
  );

  const confirmPair = useCallback(
    async (keepScoring: boolean) => {
      const res = await dpoApi.confirmPair(keepScoring);
      if (!res.ok) {
        setError(res.error ?? "Confirm failed");
        return;
      }
      setShowAcceptDialog(false);
      setPendingBest(null);
      setPendingWorst(null);
      setPairsDone(res.pairs_done ?? pairsDone + 1);
      if (res.continue_group) {
        setPhase("best");
        setBestImage(null);
        setRemainingImages(res.remaining_images ?? []);
      } else {
        await fetchNextGroup();
      }
    },
    [fetchNextGroup, pairsDone],
  );

  const cancelPendingPair = useCallback(async () => {
    const res = await dpoApi.cancelPair();
    if (!res.ok) {
      setError(res.error ?? "Cancel failed");
      return;
    }
    setShowAcceptDialog(false);
    setPendingBest(null);
    setPendingWorst(null);
    // Backend kept phase="worst" and preserved _selected_best, so the user
    // returns to the worst-pick UI with the same best image and the original
    // pool minus that best — identical to Ctk's Cancel branch.
    setPhase("worst");
    if (res.best) setBestImage(res.best);
    if (res.remaining_images) setRemainingImages(res.remaining_images);
  }, []);

  const skipGroup = useCallback(async () => {
    await dpoApi.skipGroup();
    await fetchNextGroup();
  }, [fetchNextGroup]);

  const discardGroup = useCallback(async () => {
    const res = await dpoApi.discardGroup();
    if (!res.ok) {
      setError(res.error ?? "Failed to discard group");
      return;
    }
    await fetchNextGroup();
  }, [fetchNextGroup]);

  const commitTriagePairs = useCallback(
    async (pairs: Array<{ chosen: string; rejected: string }>, discardRest = false) => {
      const res = await dpoApi.commitTriagePairs(pairs, discardRest);
      if (!res.ok) {
        setError(res.error ?? "Failed to commit pairs");
        return;
      }
      if (res.pairs_done !== undefined) setPairsDone(res.pairs_done);
      await fetchNextGroup();
    },
    [fetchNextGroup],
  );

  const triageAlign = useCallback(async (good: string[], bad: string[]) => {
    const res = await dpoApi.triageAlign(good, bad);
    if (!res.ok) {
      setError(res.error ?? "Auto-align failed");
      return null;
    }
    return {
      pairs: res.pairs ?? [],
      chosenPool: res.chosen_pool ?? [],
      rejectedPool: res.rejected_pool ?? [],
    };
  }, []);

  const cancelSession = useCallback(async () => {
    scanAbortRef.current?.abort();
    scanAbortRef.current = null;
    await dpoApi.cancelSession();
    setStepState("setup");
    setGroup(null);
    setRemainingImages([]);
    setBestImage(null);
    setPhase("best");
    setPairsDone(0);
    setSwissMatch(null);
    setSwissRound(0);
    setSwissTotalRounds(0);
    setSwissMatchesPlayed(0);
    setSwissMatchesTotal(0);
    setRankedOrder([]);
    setRankingScores({});
    setMaxPairs(0);
    setPairCount(1);
    setShowAcceptDialog(false);
    setPendingBest(null);
    setPendingWorst(null);
    setScanCount(0);
    setScanTotal(0);
    setHashCount(0);
    setCacheHits(0);
    setTotalPairs(0);
    setFinalResult(null);
    setResume(null);
    setError(null);
    setToast(null);
  }, []);

  const swissVote = useCallback(
    async (winner: "a" | "b" | "tie") => {
      if (!swissMatch) return;
      const [a, b] = swissMatch;
      await applySwissState(await dpoApi.swissVote(a, b, winner));
    },
    [swissMatch, applySwissState],
  );

  const swissFinishEarly = useCallback(async () => {
    const res = await dpoApi.swissFinishEarly();
    if (!res.ok) {
      setError(res.error ?? "Finish early failed");
      return;
    }
    setRankedOrder(res.order ?? []);
    setRankingScores(res.scores ?? {});
    setMaxPairs(res.max_pairs ?? 0);
    setPairCount(res.default_pairs ?? 1);
    setStepState("ranking");
  }, []);

  const swapRanked = useCallback(
    async (i: number, j: number) => {
      if (i === j || i < 0 || j < 0 || i >= rankedOrder.length || j >= rankedOrder.length) return;
      const next = [...rankedOrder];
      [next[i], next[j]] = [next[j], next[i]];
      setRankedOrder(next);
      const res = await dpoApi.swissReorder(next);
      if (!res.ok) {
        // Server is authoritative — roll back the optimistic swap.
        setRankedOrder(rankedOrder);
        setError(res.error ?? "Reorder failed");
      }
    },
    [rankedOrder],
  );

  const swissExportPairs = useCallback(async () => {
    const res = await dpoApi.swissExport(pairCount);
    if (!res.ok) {
      setError(res.error ?? "Export failed");
      return;
    }
    if (res.pairs_done !== undefined) setPairsDone(res.pairs_done);
    await fetchNextGroup();
  }, [pairCount, fetchNextGroup]);

  const finalize = useCallback(async (valPct: number) => {
    const res = await dpoApi.finalizeSession(valPct);
    if (!res.ok) {
      setError("Finalize failed");
      return;
    }
    setFinalResult({
      total_pairs: res.total_pairs ?? 0,
      train_count: res.train_count ?? 0,
      val_count: res.val_count ?? 0,
    });
  }, []);

  const goToReview = useCallback(() => {
    setStepState("review");
    setError(null);
    setToast(null);
  }, []);

  useEffect(() => {
    return () => {
      scanAbortRef.current?.abort();
      scanAbortRef.current = null;
    };
  }, []);

  return {
    step,
    setStep,
    sourceFolder,
    outputDir,
    pairsPerGroup,
    mode,
    scanCount,
    scanTotal,
    hashCount,
    cacheHits,
    group,
    phase,
    bestImage,
    remainingImages,
    pairsDone,
    swissMatch,
    swissRound,
    swissTotalRounds,
    swissMatchesPlayed,
    swissMatchesTotal,
    rankedOrder,
    rankingScores,
    maxPairs,
    pairCount,
    showAcceptDialog,
    pendingBest,
    pendingWorst,
    totalPairs,
    finalResult,
    resume,
    error,
    toast,
    actions: {
      setSourceFolder,
      setOutputDir,
      setPairsPerGroup,
      startSession,
      pickImage,
      confirmPair,
      cancelPendingPair,
      skipGroup,
      discardGroup,
      commitTriagePairs,
      triageAlign,
      cancelSession,
      swissVote,
      swissFinishEarly,
      swapRanked,
      setPairCount,
      swissExportPairs,
      finalize,
      setError,
      setToast,
      loadOutputManifest,
      goToReview,
    },
  };
}
