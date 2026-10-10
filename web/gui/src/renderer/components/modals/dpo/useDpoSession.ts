import { useCallback, useEffect, useRef, useState } from "react";

import type { GroupOrder, RealismDevice, ScoreBreakdown, SwissStateResponse } from "@/api/dpoApi";
import { dpoApi } from "@/api/dpoApi";

import { EMPTY_REGEN, mergeAdded, type RegenStartRequest, type RegenUiState } from "./regenFixes";
import { DEFAULT_SCORE_WEIGHTS, type ScoreSortMode, type ScoreWeights } from "./scoreCombine";
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
  groupOrder: GroupOrder;
  captionFilter: string;
  loraFilter: string;
  minImagesPerGroup: number;
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
  /** P(AI-generated) per image path for the current group; empty until scored. */
  realismScores: Record<string, number>;
  realismError: string | null;
  /** Every scorer's verdict per image path for the current group (Selection panel). */
  scoreBreakdown: Record<string, ScoreBreakdown>;
  /** Scorer name -> why it produced nothing, for the current group. */
  scoreBreakdownErrors: Record<string, string>;
  /**
   * Weights over the five scorers for the current prompt. Stored per prompt so
   * revisiting a group restores them; a new prompt starts from the last edit.
   */
  scoreWeights: ScoreWeights;
  scoreSortMode: ScoreSortMode;
  /**
   * Where the realism head runs. Server-owned — this mirrors it so the toggle
   * can render the truth rather than an optimistic guess.
   */
  realismDevice: RealismDevice;
  /**
   * Whole-tree score backfill, every axis. Scoring a group on first view is
   * the default, so this only exists to pay that cost once up front instead
   * of a group at a time — after it runs, every group is a cache hit and no
   * model is loaded while curating.
   */
  backfill: BackfillState;
  /** Images Shift+clicked for Fix & Regenerate in the current group. */
  regenSelection: string[];
  /** The Fix & Regenerate run: per-source status and what it added. */
  regen: RegenUiState;
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
    setGroupOrder: (order: GroupOrder) => void;
    setCaptionFilter: (text: string) => void;
    setLoraFilter: (text: string) => void;
    setMinImagesPerGroup: (n: number) => void;
    startSession: (config: SessionConfig) => Promise<void>;
    pickImage: (path: string) => Promise<void>;
    confirmPair: (keepScoring: boolean, discardRest?: boolean) => Promise<void>;
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
    swissExportPairs: (discardRest?: boolean) => Promise<void>;
    finalize: (valPct: number) => Promise<void>;
    setError: (msg: string | null) => void;
    setToast: (msg: string | null) => void;
    loadOutputManifest: (outputDir: string) => void;
    goToReview: () => void;
    startScoreBackfill: () => Promise<void>;
    stopScoreBackfill: () => Promise<void>;
    setRealismDevice: (device: RealismDevice) => Promise<void>;
    setScoreWeights: (weights: ScoreWeights) => void;
    setScoreSortMode: (mode: ScoreSortMode) => void;
    toggleRegenSelect: (path: string) => void;
    clearRegenSelection: () => void;
    /** Resolves to an error message, or null when the run started. */
    startRegen: (req: RegenStartRequest) => Promise<string | null>;
    cancelRegen: () => Promise<void>;
  };
}

export interface BackfillState {
  running: boolean;
  done: number;
  total: number;
  /** Scorers whose model could not load; the run continued without them. */
  errors: Record<string, string>;
  error: string | null;
}

export function useDpoSession(): UseDpoSessionResult {
  const [step, setStepState] = useState<CurationStep>("setup");
  const [sourceFolder, setSourceFolder] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [pairsPerGroup, setPairsPerGroup] = useState(1);
  const [groupOrder, setGroupOrder] = useState<GroupOrder>("most_realistic");
  const [captionFilter, setCaptionFilter] = useState("");
  const [loraFilter, setLoraFilter] = useState("");
  const [minImagesPerGroup, setMinImagesPerGroup] = useState(2);
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

  // P(AI-generated) per image for the current group. Lives here rather than in
  // SelectionStep because the step components wipe their per-group state on
  // group change, which would throw the scores away and re-request them.
  const [realismScores, setRealismScores] = useState<Record<string, number>>({});
  const [realismError, setRealismError] = useState<string | null>(null);
  // Guards against a slow score response landing after the user has already
  // moved on, which would decorate the wrong group.
  const realismRequestRef = useRef(0);

  // The full breakdown (realism + age + text + quality; adherence is always
  // null with SigLIP off) for the same group. Separate from realismScores
  // because it is a heavier call — the OCR reader loads on first use — and the
  // realism badge must not wait on it.
  const [scoreBreakdown, setScoreBreakdown] = useState<Record<string, ScoreBreakdown>>({});
  const [scoreBreakdownErrors, setScoreBreakdownErrors] = useState<Record<string, string>>({});
  const breakdownRequestRef = useRef(0);

  // Per-prompt scorer weights. The last edit is the starting point for any
  // prompt not seen before, so a user tuning for "age + prompt" keeps that
  // across groups until they change it again.
  const [scoreWeightsByPrompt, setScoreWeightsByPrompt] = useState<Record<string, ScoreWeights>>({});
  const lastWeightsRef = useRef<ScoreWeights>({ ...DEFAULT_SCORE_WEIGHTS });
  const [scoreSortMode, setScoreSortMode] = useState<ScoreSortMode>("combined");

  // Whole-tree backfill. Polled rather than pushed: it is minutes of work on a
  // background thread and a coarse progress read is all the UI needs.
  const [backfill, setBackfill] = useState<BackfillState>({
    running: false,
    done: 0,
    total: 0,
    errors: {},
    error: null,
  });
  const backfillTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fix & Regenerate. Polled like the backfill; results are appended to the
  // group as they land. The group index taken at fetchNextGroup keeps a run
  // that outlives its group from painting spinners on the next one.
  const [regenSelection, setRegenSelection] = useState<string[]>([]);
  const [regen, setRegen] = useState<RegenUiState>(EMPTY_REGEN);
  const regenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const regenGroupIndexRef = useRef(0);
  const appendedRef = useRef<Set<string>>(new Set());

  const pollBackfill = useCallback(async () => {
    const res = await dpoApi.scoreBackfillStatus().catch(() => null);
    if (!res) return;
    setBackfill({
      running: res.running,
      done: res.done ?? 0,
      total: res.total ?? 0,
      errors: res.errors ?? {},
      error: res.error ?? null,
    });
    if (!res.running && backfillTimerRef.current) {
      clearInterval(backfillTimerRef.current);
      backfillTimerRef.current = null;
    }
  }, []);

  const startScoreBackfill = useCallback(async () => {
    const res = await dpoApi.startScoreBackfill().catch(() => null);
    if (!res?.ok) {
      setBackfill((b) => ({ ...b, error: res?.error ?? "Could not start the backfill" }));
      return;
    }
    setBackfill({ running: true, done: 0, total: res.total ?? 0, errors: {}, error: null });
    if (backfillTimerRef.current) clearInterval(backfillTimerRef.current);
    backfillTimerRef.current = setInterval(() => void pollBackfill(), 1000);
  }, [pollBackfill]);

  const stopScoreBackfill = useCallback(async () => {
    await dpoApi.stopScoreBackfill().catch(() => null);
    await pollBackfill();
  }, [pollBackfill]);

  // Scoring device, "off" included. The server is the source of truth (a
  // backfill thread reads it per chunk), so the toggle sends and then adopts
  // whatever comes back rather than assuming the write landed.
  const [realismDevice, setRealismDeviceState] = useState<RealismDevice>("auto");
  // Mirrored into a ref because loadRealismScores is a stable callback: reading
  // the state variable there would pin it to whatever the device was when the
  // session mounted, and the whole point of "off" is that it takes effect now.
  const realismDeviceRef = useRef<RealismDevice>("auto");
  const adoptRealismDevice = useCallback((device: RealismDevice) => {
    realismDeviceRef.current = device;
    setRealismDeviceState(device);
  }, []);

  useEffect(() => {
    void dpoApi
      .realismDevice()
      .then((res) => {
        if (res?.ok && res.device) adoptRealismDevice(res.device);
      })
      .catch(() => undefined);
  }, [adoptRealismDevice]);

  const setRealismDevice = useCallback(
    async (device: RealismDevice) => {
      const res = await dpoApi.setRealismDevice(device).catch(() => null);
      if (res?.device) adoptRealismDevice(res.device);
      // Scores on screen were produced by a setting that no longer holds, and
      // with scoring off nothing will replace them until the next group.
      if (res?.device === "off") {
        realismRequestRef.current += 1;
        setRealismScores({});
        setRealismError(null);
        breakdownRequestRef.current += 1;
        setScoreBreakdown({});
        setScoreBreakdownErrors({});
      }
    },
    [adoptRealismDevice],
  );

  // The interval outlives any single render, so it has to be torn down on
  // unmount or it keeps polling a closed modal for ever.
  useEffect(
    () => () => {
      if (backfillTimerRef.current) clearInterval(backfillTimerRef.current);
      if (regenTimerRef.current) clearInterval(regenTimerRef.current);
    },
    [],
  );

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

  const loadScoreBreakdown = useCallback(async (images: string[]) => {
    const token = ++breakdownRequestRef.current;
    setScoreBreakdown({});
    setScoreBreakdownErrors({});
    if (images.length === 0) return;
    if (realismDeviceRef.current === "off") return;
    try {
      const res = await dpoApi.scoreBreakdown(images);
      if (token !== breakdownRequestRef.current) return; // user already moved on
      if (res.ok && res.scores) {
        setScoreBreakdown(res.scores);
        setScoreBreakdownErrors(res.errors ?? {});
      } else if (res.error) {
        setScoreBreakdownErrors({ breakdown: res.error });
      }
    } catch {
      if (token === breakdownRequestRef.current) {
        setScoreBreakdownErrors({ breakdown: "Score breakdown is unavailable." });
      }
    }
  }, []);

  // Fire-and-forget on purpose: group advance must not wait on a model call.
  // Scores arrive a moment later and the grid re-sorts, which is far better
  // than stalling every group behind inference.
  const loadRealismScores = useCallback(async (images: string[]) => {
    const token = ++realismRequestRef.current;
    setRealismScores({});
    setRealismError(null);
    if (images.length === 0) return;
    // The server refuses too; not asking keeps a disabled session from making a
    // round trip per group, and keeps realismError clear so the UI shows
    // "scoring off" rather than "scoring unavailable".
    if (realismDeviceRef.current === "off") return;
    try {
      const res = await dpoApi.realismScores(images);
      if (token !== realismRequestRef.current) return; // user already moved on
      if (res.ok && res.scores) {
        setRealismScores(res.scores);
      } else if (res.error) {
        setRealismError(res.error);
      }
    } catch {
      if (token === realismRequestRef.current) {
        setRealismError("Realism scoring is unavailable.");
      }
    }
  }, []);

  // Scores for images added mid-group. Merged into the maps rather than
  // replacing them: the loaders above wipe everything and bump the request
  // token, which would throw away the scores already on screen.
  const mergeScoresFor = useCallback(async (images: string[]) => {
    if (images.length === 0 || realismDeviceRef.current === "off") return;
    const realismToken = realismRequestRef.current;
    const breakdownToken = breakdownRequestRef.current;
    void dpoApi
      .realismScores(images)
      .then((res) => {
        if (realismToken !== realismRequestRef.current || !res.ok || !res.scores) return;
        const scores = res.scores;
        setRealismScores((prev) => ({ ...prev, ...scores }));
      })
      .catch(() => undefined);
    void dpoApi
      .scoreBreakdown(images)
      .then((res) => {
        if (breakdownToken !== breakdownRequestRef.current || !res.ok || !res.scores) return;
        const scores = res.scores;
        setScoreBreakdown((prev) => ({ ...prev, ...scores }));
      })
      .catch(() => undefined);
  }, []);

  const stopRegenTimer = useCallback(() => {
    if (regenTimerRef.current) {
      clearInterval(regenTimerRef.current);
      regenTimerRef.current = null;
    }
  }, []);

  const pollRegen = useCallback(async () => {
    const res = await dpoApi.regenStatus().catch(() => null);
    if (!res) return;
    if (res.group_index !== regenGroupIndexRef.current) {
      // A run for a group that is no longer on screen: the server refuses its
      // results; here only the "still running" chip is worth keeping.
      setRegen((r) => ({ ...r, running: res.running }));
      if (!res.running) stopRegenTimer();
      return;
    }
    const fresh = res.added.filter((p) => !appendedRef.current.has(p));
    if (fresh.length > 0) {
      for (const p of fresh) appendedRef.current.add(p);
      setRemainingImages((prev) => mergeAdded(prev, fresh));
      setGroup((g) => (g ? { ...g, images: mergeAdded(g.images, fresh) } : g));
      void mergeScoresFor(fresh);
    }
    setRegen({
      running: res.running,
      phase: res.phase,
      planned: res.planned,
      done: res.done,
      failed: res.failed,
      items: res.items,
      added: res.added,
      dropped: res.dropped,
      error: res.error,
      warning: res.warning,
      groupIndex: res.group_index,
    });
    if (!res.running) {
      stopRegenTimer();
      const skipped = Object.values(res.items).filter((item) => item.status === "skipped").length;
      if (res.error) {
        setError(res.error);
      } else {
        const parts = [`${res.added.length} added`, `${res.failed} failed`];
        if (skipped > 0) parts.push(`${skipped} identical skipped`);
        if (res.dropped.length > 0) parts.push(`${res.dropped.length} finished after the group changed`);
        setToast(`Fix & Regenerate: ${parts.join(", ")}`);
      }
    }
  }, [mergeScoresFor, stopRegenTimer]);

  const startRegen = useCallback(
    async (req: RegenStartRequest): Promise<string | null> => {
      const res = await dpoApi
        .regenStart({
          paths: req.paths,
          fixes: req.fixes,
          segment_params: req.segmentParams,
          model_override: req.modelOverride,
          base_url: req.baseUrl,
          token: req.token,
          generated_folder: req.generatedFolder,
          concurrency: req.concurrency,
        })
        .catch((e: unknown): { ok: boolean; error?: string; planned?: number } => ({ ok: false, error: String(e) }));
      if (!res.ok) return res.error ?? "Could not start the regeneration";
      setRegenSelection([]);
      setRegen({ ...EMPTY_REGEN, running: true, phase: "generating", planned: res.planned ?? req.paths.length });
      stopRegenTimer();
      regenTimerRef.current = setInterval(() => void pollRegen(), 1000);
      void pollRegen();
      return null;
    },
    [pollRegen, stopRegenTimer],
  );

  const cancelRegen = useCallback(async () => {
    await dpoApi.regenCancel().catch(() => null);
    await pollRegen();
  }, [pollRegen]);

  const toggleRegenSelect = useCallback((path: string) => {
    setRegenSelection((prev) => (prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path]));
  }, []);

  const clearRegenSelection = useCallback(() => setRegenSelection([]), []);

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
        regenGroupIndexRef.current = res.group.group_index;
        appendedRef.current = new Set();
        setRegenSelection([]);
        // A run still going belongs to the previous group; keep its chip,
        // drop its per-tile state so nothing here is painted as regenerating.
        setRegen((r) => (r.running ? { ...r, items: {}, added: [], dropped: [] } : EMPTY_REGEN));
        void loadRealismScores(res.group.images);
        void loadScoreBreakdown(res.group.images);
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
  }, [refreshSwissState, loadRealismScores, loadScoreBreakdown]);

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
      setGroupOrder(config.groupOrder);
      setCaptionFilter(config.captionFilter);
      setLoraFilter(config.loraFilter);
      setMinImagesPerGroup(config.minImagesPerGroup);
      setScanCount(0);
      setScanTotal(0);
      setHashCount(0);
      setCacheHits(0);

      const res = await dpoApi.startSession(
        config.sourceFolder,
        config.outputDir,
        config.pairsPerGroup,
        config.mode,
        config.groupOrder,
        config.captionFilter,
        config.minImagesPerGroup,
        config.loraFilter,
      );
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
    async (keepScoring: boolean, discardRest = false) => {
      const res = await dpoApi.confirmPair(keepScoring, discardRest);
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
    if (regenTimerRef.current) {
      clearInterval(regenTimerRef.current);
      regenTimerRef.current = null;
    }
    setRegenSelection([]);
    setRegen(EMPTY_REGEN);
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

  const swissExportPairs = useCallback(
    async (discardRest = false) => {
      const res = await dpoApi.swissExport(pairCount, discardRest);
      if (!res.ok) {
        setError(res.error ?? "Export failed");
        return;
      }
      if (res.pairs_done !== undefined) setPairsDone(res.pairs_done);
      await fetchNextGroup();
    },
    [pairCount, fetchNextGroup],
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

  const scoreWeights = (group && scoreWeightsByPrompt[group.prompt]) ?? lastWeightsRef.current;
  const setScoreWeights = useCallback(
    (weights: ScoreWeights) => {
      lastWeightsRef.current = weights;
      const prompt = group?.prompt;
      if (prompt !== undefined) setScoreWeightsByPrompt((m) => ({ ...m, [prompt]: weights }));
    },
    [group?.prompt],
  );

  return {
    step,
    setStep,
    sourceFolder,
    outputDir,
    pairsPerGroup,
    groupOrder,
    captionFilter,
    loraFilter,
    minImagesPerGroup,
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
    realismScores,
    realismError,
    realismDevice,
    scoreBreakdown,
    scoreBreakdownErrors,
    scoreWeights,
    scoreSortMode,
    backfill,
    regenSelection,
    regen,
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
      setGroupOrder,
      setCaptionFilter,
      setLoraFilter,
      setMinImagesPerGroup,
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
      startScoreBackfill,
      stopScoreBackfill,
      setRealismDevice,
      setScoreWeights,
      setScoreSortMode,
      toggleRegenSelect,
      clearRegenSelection,
      startRegen,
      cancelRegen,
    },
  };
}
