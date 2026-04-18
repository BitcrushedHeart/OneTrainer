import { useCallback, useEffect, useRef, useState } from "react";

import { dpoApi } from "@/api/dpoApi";

import type { CurationStep, FinalResult, GroupData, ResumeInfo, SelectionPhase, SessionConfig } from "./types";

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
  mode: "selection" | "elo";
  scanCount: number;
  group: GroupData | null;
  phase: SelectionPhase;
  bestImage: string | null;
  remainingImages: string[];
  pairsDone: number;
  eloPair: [string, string] | null;
  eloRatings: Record<string, number>;
  eloDone: number;
  eloSuggested: number;
  eloFinished: boolean;
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
    cancelSession: () => Promise<void>;
    eloVote: (winner: "a" | "b" | "tie") => Promise<void>;
    eloAccept: (continueScoring: boolean) => Promise<void>;
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
  const [mode, setMode] = useState<"selection" | "elo">("selection");
  const [scanCount, setScanCount] = useState(0);

  const [group, setGroup] = useState<GroupData | null>(null);
  const [phase, setPhase] = useState<SelectionPhase>("best");
  const [bestImage, setBestImage] = useState<string | null>(null);
  const [remainingImages, setRemainingImages] = useState<string[]>([]);
  const [pairsDone, setPairsDone] = useState(0);

  const [eloPair, setEloPair] = useState<[string, string] | null>(null);
  const [eloRatings, setEloRatings] = useState<Record<string, number>>({});
  const [eloDone, setEloDone] = useState(0);
  const [eloSuggested, setEloSuggested] = useState(0);
  const [eloFinished, setEloFinished] = useState(false);

  const [showAcceptDialog, setShowAcceptDialog] = useState(false);
  const [pendingBest, setPendingBest] = useState<string | null>(null);
  const [pendingWorst, setPendingWorst] = useState<string | null>(null);
  const [pendingRemaining, setPendingRemaining] = useState<string[]>([]);

  const [totalPairs, setTotalPairs] = useState(0);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [resume, setResume] = useState<ResumeInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const scanAbortRef = useRef<AbortController | null>(null);
  const modeRef = useRef<"selection" | "elo">("selection");
  modeRef.current = mode;

  const setStep = useCallback((s: CurationStep) => {
    setStepState(s);
    setError(null);
    setToast(null);
  }, []);

  const refreshEloPair = useCallback(async () => {
    const res = await dpoApi.eloPair();
    if (!res.ok) {
      setError(res.error ?? "ELO pair fetch failed");
      return;
    }
    setEloRatings(res.ratings ?? {});
    setEloDone(res.done ?? 0);
    setEloSuggested(res.suggested ?? 0);
    if (res.finished) {
      setEloFinished(true);
      setEloPair(null);
    } else if (res.pair) {
      setEloFinished(false);
      setEloPair(res.pair);
    }
  }, []);

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
        if (effectiveMode === "elo") {
          setStepState("elo");
          await refreshEloPair();
        } else {
          setStepState("selecting");
        }
        return;
      }
      return;
    }
  }, [refreshEloPair]);

  const pollScan = useCallback(
    async (signal: AbortSignal) => {
      while (!signal.aborted) {
        try {
          const status = await dpoApi.sessionStatus();
          setScanCount(status.scan_count);
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

      // Phase "worst": backend `selectImage` auto-registers the pair.
      // Ctk offers a yes/no/cancel dialog BEFORE registering so users can back out;
      // the web backend has no "unregister" endpoint, so we register first and then
      // let the user decide continuation. Cancel becomes "advance to next group".
      const remainingAfter = remainingImages.filter((i) => i !== path);
      const canContinue = remainingAfter.length >= 2;
      const res = await dpoApi.selectImage(path);
      if (!res.ok) {
        setError(res.error ?? "Selection failed");
        return;
      }
      if (res.pair_created) {
        setPairsDone(res.pairs_done ?? pairsDone + 1);
        setPendingBest(res.chosen ?? bestImage);
        setPendingWorst(res.rejected ?? path);
        setPendingRemaining(res.remaining_images ?? remainingAfter);
        if (canContinue && res.continue_group) {
          setShowAcceptDialog(true);
        } else {
          await fetchNextGroup();
        }
      }
    },
    [group, phase, remainingImages, bestImage, pairsDone, fetchNextGroup],
  );

  const confirmPair = useCallback(
    async (keepScoring: boolean) => {
      setShowAcceptDialog(false);
      setPendingBest(null);
      setPendingWorst(null);
      if (keepScoring) {
        // Backend already advanced internal state (phase=best, selected_best=null)
        // and updated remaining_images when select_image returned continue_group=true.
        // Sync local state — DO NOT call nextGroup, which would advance past this group.
        setPhase("best");
        setBestImage(null);
        setRemainingImages(pendingRemaining);
        setPendingRemaining([]);
      } else {
        setPendingRemaining([]);
        await fetchNextGroup();
      }
    },
    [fetchNextGroup, pendingRemaining],
  );

  const cancelPendingPair = useCallback(() => {
    setShowAcceptDialog(false);
    setPendingBest(null);
    setPendingWorst(null);
    setPendingRemaining([]);
  }, []);

  const skipGroup = useCallback(async () => {
    await dpoApi.skipGroup();
    await fetchNextGroup();
  }, [fetchNextGroup]);

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
    setEloPair(null);
    setEloRatings({});
    setEloDone(0);
    setEloSuggested(0);
    setEloFinished(false);
    setShowAcceptDialog(false);
    setPendingBest(null);
    setPendingWorst(null);
    setPendingRemaining([]);
    setScanCount(0);
    setTotalPairs(0);
    setFinalResult(null);
    setResume(null);
    setError(null);
    setToast(null);
  }, []);

  const eloVote = useCallback(
    async (winner: "a" | "b" | "tie") => {
      if (!eloPair) return;
      const [a, b] = eloPair;
      const res = await dpoApi.eloVote(a, b, winner);
      if (!res.ok) {
        setError(res.error ?? "ELO vote failed");
        return;
      }
      // Backend returns only the current pair's ratings when `finished=false`.
      // Merge into the existing map so the full group's ratings stay visible
      // to the accept-pair panel (which sorts all ratings to pick best/worst).
      setEloRatings((prev) => ({ ...prev, ...(res.ratings ?? {}) }));
      setEloDone(res.done ?? 0);
      setEloSuggested(res.suggested ?? 0);
      if (res.finished) {
        setEloFinished(true);
        setEloPair(null);
      } else if (res.pair) {
        setEloPair(res.pair);
      }
    },
    [eloPair],
  );

  const eloAccept = useCallback(
    async (continueScoring: boolean) => {
      const res = await dpoApi.eloAccept(continueScoring);
      if (!res.ok) {
        setError(res.error ?? "Accept failed");
        return;
      }
      setPairsDone(res.pairs_done ?? pairsDone + 1);
      if (res.continue_group) {
        setEloFinished(false);
        await refreshEloPair();
      } else {
        await fetchNextGroup();
      }
    },
    [pairsDone, refreshEloPair, fetchNextGroup],
  );

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
    group,
    phase,
    bestImage,
    remainingImages,
    pairsDone,
    eloPair,
    eloRatings,
    eloDone,
    eloSuggested,
    eloFinished,
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
      cancelSession,
      eloVote,
      eloAccept,
      finalize,
      setError,
      setToast,
      loadOutputManifest,
      goToReview,
    },
  };
}
