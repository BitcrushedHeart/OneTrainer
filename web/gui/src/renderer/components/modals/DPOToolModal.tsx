import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  Loader2,
  MousePointer2,
  Scissors,
  Search,
  SkipForward,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  dpoApi,
  type GroupData,
  type PairCheckResult,
  type ReviewPair,
} from "@/api/dpoApi";
import { API_BASE } from "@/api/request";
import { Button, FormEntry } from "@/components/shared";
import { ModalBase } from "./ModalBase";

interface Props {
  open: boolean;
  onClose: () => void;
}

// ---------- Sub-components ----------

function CheckResults({ result }: { result: PairCheckResult }) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="p-2 rounded bg-[var(--color-surface-container)]">
          <span className="text-[var(--color-on-surface-secondary)]">Matched: </span>
          <span className="font-medium text-[var(--color-success-500)]">{result.total_matched}</span>
        </div>
        <div className="p-2 rounded bg-[var(--color-surface-container)]">
          <span className="text-[var(--color-on-surface-secondary)]">Chosen strays: </span>
          <span className="font-medium text-[var(--color-warning-500)]">{result.total_chosen_stray}</span>
        </div>
        <div className="p-2 rounded bg-[var(--color-surface-container)]">
          <span className="text-[var(--color-on-surface-secondary)]">Rejected strays: </span>
          <span className="font-medium text-[var(--color-warning-500)]">{result.total_rejected_stray}</span>
        </div>
      </div>
      {result.multiline_captions > 0 && (
        <div className="flex items-center gap-2 text-sm text-[var(--color-warning-500)]">
          <AlertTriangle className="w-4 h-4" />
          {result.multiline_captions} caption(s) contain newlines
        </div>
      )}
      {Object.keys(result.format_stats).length > 0 && (
        <div className="text-xs text-[var(--color-on-surface-secondary)]">
          Formats:{" "}
          {Object.entries(result.format_stats)
            .map(([ext, count]) => `${ext}: ${count}`)
            .join(", ")}
        </div>
      )}
      {result.pairs.map((pair, i) => (
        <div key={i} className="text-xs p-2 rounded bg-[var(--color-surface-container)] space-y-0.5">
          <div className="text-[var(--color-on-surface)]">
            Chosen: <span className="font-mono">{pair.chosen_path}</span>
          </div>
          <div className="text-[var(--color-on-surface)]">
            Rejected: <span className="font-mono">{pair.rejected_path}</span>
          </div>
          <div className="text-[var(--color-on-surface-secondary)]">
            Matched: {pair.matched}, Chosen stray: {pair.chosen_stray}, Rejected stray: {pair.rejected_stray}
          </div>
        </div>
      ))}
    </div>
  );
}

function ReviewPanel({
  pairs,
  onRemove,
}: {
  pairs: ReviewPair[];
  onRemove: (chosen: string, rejected: string) => void;
}) {
  const [index, setIndex] = useState(0);
  if (pairs.length === 0) {
    return <div className="text-sm text-[var(--color-on-surface-secondary)]">No pairs found.</div>;
  }
  const pair = pairs[index];
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-sm">
        <span className="text-[var(--color-on-surface-secondary)]">
          Pair {index + 1} of {pairs.length}
        </span>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0}>
            Prev
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setIndex(Math.min(pairs.length - 1, index + 1))}
            disabled={index === pairs.length - 1}
          >
            Next
          </Button>
          <Button size="sm" variant="danger" onClick={() => onRemove(pair.chosen_path, pair.rejected_path)}>
            <Trash2 className="w-3.5 h-3.5 mr-1" />
            Remove
          </Button>
        </div>
      </div>
      <div className="text-xs text-[var(--color-on-surface-secondary)] font-mono truncate">
        {pair.caption || "(no caption)"}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <div className="text-xs font-medium text-[#22c55e]">Chosen</div>
          <div className="text-xs font-mono text-[var(--color-on-surface-secondary)] truncate">{pair.chosen_path}</div>
        </div>
        <div className="space-y-1">
          <div className="text-xs font-medium text-[#ef4444]">Rejected</div>
          <div className="text-xs font-mono text-[var(--color-on-surface-secondary)] truncate">
            {pair.rejected_path}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- Selection Mode Curation ----------

type CurationStep = "setup" | "scanning" | "selecting" | "export";

function SelectionCuration({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState<CurationStep>("setup");
  const [sourceFolder, setSourceFolder] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [pairsPerGroup, setPairsPerGroup] = useState(1);
  const [mode, setMode] = useState<"selection" | "elo">("selection");
  const [error, setError] = useState<string | null>(null);
  const [scanCount, setScanCount] = useState(0);

  // Selection state
  const [group, setGroup] = useState<GroupData | null>(null);
  const [images, setImages] = useState<string[]>([]);
  const [phase, setPhase] = useState<"best" | "worst">("best");
  const [bestImage, setBestImage] = useState<string | null>(null);
  const [pairsDone, setPairsDone] = useState(0);

  // ELO state
  const [eloPair, setEloPair] = useState<[string, string] | null>(null);
  const [eloRatings, setEloRatings] = useState<Record<string, number>>({});
  const [eloDone, setEloDone] = useState(0);
  const [eloSuggested, setEloSuggested] = useState(0);
  const [eloFinished, setEloFinished] = useState(false);

  // Export state
  const [totalPairs, setTotalPairs] = useState(0);
  const [valPercentage, setValPercentage] = useState(10);
  const [finalResult, setFinalResult] = useState<{
    total_pairs: number;
    train_count: number;
    val_count: number;
  } | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // ---- Setup step ----
  const handleStart = async () => {
    if (!sourceFolder || !outputDir) {
      setError("Select both source and output folders.");
      return;
    }
    setError(null);
    const res = await dpoApi.startSession(sourceFolder, outputDir, pairsPerGroup, mode);
    if (!res.ok) {
      setError(res.error ?? "Failed to start session");
      return;
    }
    setStep("scanning");
    pollRef.current = setInterval(pollScan, 500);
  };

  // ---- ELO helpers ----
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

  const handleEloVote = async (winner: "a" | "b" | "tie") => {
    if (!eloPair) return;
    const [a, b] = eloPair;
    const res = await dpoApi.eloVote(a, b, winner);
    if (!res.ok) {
      setError(res.error ?? "ELO vote failed");
      return;
    }
    setEloRatings(res.ratings ?? {});
    setEloDone(res.done ?? 0);
    setEloSuggested(res.suggested ?? 0);
    if (res.finished) {
      setEloFinished(true);
      setEloPair(null);
    } else if (res.pair) {
      setEloPair(res.pair);
    }
  };

  const handleEloAccept = async (continueScoring: boolean) => {
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
  };

  const pollScan = async () => {
    const status = await dpoApi.sessionStatus();
    setScanCount(status.scan_count);
    if (status.groups_queued > 0 || status.worker_finished) {
      if (pollRef.current) clearInterval(pollRef.current);
      await fetchNextGroup();
    }
  };

  const fetchNextGroup = async () => {
    const res = await dpoApi.nextGroup();
    if (res.done) {
      setTotalPairs(res.total_pairs ?? 0);
      setStep("export");
      return;
    }
    if (res.waiting) {
      // poll again shortly
      setTimeout(fetchNextGroup, 500);
      return;
    }
    if (res.group) {
      setGroup(res.group);
      setImages(res.group.images);
      setPhase("best");
      setBestImage(null);
      setPairsDone(res.group.pairs_done);
      setStep("selecting");
      if (mode === "elo" || res.group.mode === "elo") {
        await refreshEloPair();
      }
    }
  };

  const handleSelect = async (path: string) => {
    const res = await dpoApi.selectImage(path);
    if (!res.ok) {
      setError(res.error ?? "Selection failed");
      return;
    }
    if (res.phase === "worst") {
      // Picked best, now pick worst
      setPhase("worst");
      setBestImage(res.best ?? path);
      setImages(res.remaining_images ?? images.filter((i) => i !== path));
    } else if (res.pair_created) {
      setPairsDone(res.pairs_done ?? pairsDone + 1);
      if (res.continue_group) {
        // More pairs to create in this group
        setPhase("best");
        setBestImage(null);
        setImages(res.remaining_images ?? []);
      } else {
        // Move to next group
        await fetchNextGroup();
      }
    }
  };

  const handleSkipGroup = async () => {
    await dpoApi.skipGroup();
    await fetchNextGroup();
  };

  const handleFinalize = async () => {
    const res = await dpoApi.finalizeSession(valPercentage);
    if (res.ok) {
      setFinalResult({
        total_pairs: res.total_pairs ?? 0,
        train_count: res.train_count ?? 0,
        val_count: res.val_count ?? 0,
      });
    }
  };

  const handleCancel = async () => {
    if (pollRef.current) clearInterval(pollRef.current);
    await dpoApi.cancelSession();
    onDone();
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // ---- Render ----
  if (step === "setup") {
    return (
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-[var(--color-on-surface)]">Start Selection Mode</h3>
        <p className="text-xs text-[var(--color-on-surface-secondary)]">
          Point to a folder of generated images with prompt metadata. Images are grouped by prompt,
          then you pick the best and worst from each group to create chosen/rejected pairs.
        </p>
        <div className="space-y-3">
          <div>
            <label className="block text-sm text-[var(--color-on-surface)] mb-1">Source Folder</label>
            <FormEntry label="" value={sourceFolder} onChange={(v) => setSourceFolder(String(v))} />
          </div>
          <div>
            <label className="block text-sm text-[var(--color-on-surface)] mb-1">Output Folder</label>
            <FormEntry label="" value={outputDir} onChange={(v) => setOutputDir(String(v))} />
          </div>
          <div>
            <label className="block text-sm text-[var(--color-on-surface)] mb-1">Pairs per Group</label>
            <FormEntry
              label=""
              type="number"
              value={pairsPerGroup}
              onChange={(v) => setPairsPerGroup(Math.max(1, Number(v)))}
            />
          </div>
          <div>
            <label className="block text-sm text-[var(--color-on-surface)] mb-1">Mode</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setMode("selection")}
                className={`px-3 py-1.5 text-sm rounded-md border ${
                  mode === "selection"
                    ? "border-[var(--color-cobalt-600)] bg-[var(--color-cobalt-600-alpha-15)] text-[var(--color-on-surface)]"
                    : "border-[var(--color-border-subtle)] text-[var(--color-on-surface-secondary)]"
                }`}
              >
                Selection (best/worst)
              </button>
              <button
                type="button"
                onClick={() => setMode("elo")}
                className={`px-3 py-1.5 text-sm rounded-md border ${
                  mode === "elo"
                    ? "border-[var(--color-cobalt-600)] bg-[var(--color-cobalt-600-alpha-15)] text-[var(--color-on-surface)]"
                    : "border-[var(--color-border-subtle)] text-[var(--color-on-surface-secondary)]"
                }`}
              >
                ELO ranking
              </button>
            </div>
          </div>
        </div>
        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)]">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </div>
        )}
        <div className="flex gap-2">
          <Button variant="primary" onClick={handleStart}>
            <MousePointer2 className="w-4 h-4 mr-1" />
            Start Selection
          </Button>
          <Button variant="ghost" onClick={onDone}>
            Back
          </Button>
        </div>
      </div>
    );
  }

  if (step === "scanning") {
    return (
      <div className="text-center py-8 space-y-4">
        <Loader2 className="w-8 h-8 mx-auto animate-spin text-[var(--color-cobalt-600)]" />
        <p className="text-sm text-[var(--color-on-surface)]">Scanning and deduplicating...</p>
        <p className="text-xs text-[var(--color-on-surface-secondary)]">{scanCount} files scanned</p>
        <Button variant="ghost" size="sm" onClick={handleCancel}>
          Cancel
        </Button>
      </div>
    );
  }

  if (step === "selecting" && group && (mode === "elo" || group.mode === "elo")) {
    const promptDisplay =
      group.prompt === "UNCONDITIONAL"
        ? "UNCONDITIONAL"
        : group.prompt.length > 100
          ? group.prompt.slice(0, 100) + "..."
          : group.prompt;

    return (
      <div className="space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-[var(--color-on-surface-secondary)]">
            Group {group.group_index} / {group.total_groups} &middot; AR: {group.aspectratio} &middot;
            ELO comparisons {eloDone}/{eloSuggested}
          </div>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={handleSkipGroup}>
              <SkipForward className="w-3.5 h-3.5 mr-1" />
              Skip
            </Button>
            <Button size="sm" variant="ghost" onClick={handleCancel}>
              <X className="w-3.5 h-3.5 mr-1" />
              Cancel
            </Button>
          </div>
        </div>

        {/* Prompt */}
        <div className="text-xs font-mono p-2 rounded bg-[var(--color-surface-container)] text-[var(--color-on-surface)]">
          {promptDisplay}
        </div>

        {!eloFinished && eloPair ? (
          <>
            <div className="grid grid-cols-2 gap-3">
              {eloPair.map((path, idx) => (
                <button
                  key={path}
                  className="rounded-lg overflow-hidden border-2 border-[var(--color-border-subtle)] hover:border-[var(--color-cobalt-600)] transition-all"
                  onClick={() => handleEloVote(idx === 0 ? "a" : "b")}
                  title={`${idx === 0 ? "Left" : "Right"} wins`}
                >
                  <img
                    src={`${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`}
                    alt={path.split(/[/\\]/).pop()}
                    className="w-full h-72 object-contain bg-black/30"
                  />
                  <div className="bg-black/60 text-white text-xs px-2 py-1 flex items-center justify-between">
                    <span className="truncate">{path.split(/[/\\]/).pop()}</span>
                    <span className="tabular-nums ml-2">ELO {eloRatings[path]?.toFixed(0) ?? "—"}</span>
                  </div>
                </button>
              ))}
            </div>
            <div className="flex justify-center gap-2">
              <Button variant="secondary" onClick={() => handleEloVote("tie")}>
                Tie / Skip
              </Button>
              <Button
                variant="primary"
                onClick={() => handleEloAccept(false)}
                disabled={eloDone < Math.min(5, eloSuggested)}
                title="Finalize current ELO ranking into a chosen/rejected pair"
              >
                Accept Pair
              </Button>
            </div>
          </>
        ) : (
          <div className="text-center py-6 space-y-3">
            <p className="text-sm text-[var(--color-on-surface)]">
              Suggested comparisons reached. Accept the current ranking?
            </p>
            <div className="flex justify-center gap-2">
              <Button variant="primary" onClick={() => handleEloAccept(false)}>
                Accept &amp; Next Group
              </Button>
              <Button variant="secondary" onClick={() => handleEloAccept(true)}>
                Accept &amp; Keep Scoring
              </Button>
              <Button variant="ghost" onClick={handleSkipGroup}>
                Skip Group
              </Button>
            </div>
          </div>
        )}

        {error && <div className="text-xs text-[var(--color-error-500)]">{error}</div>}
      </div>
    );
  }

  if (step === "selecting" && group) {
    const promptDisplay =
      group.prompt === "UNCONDITIONAL"
        ? "UNCONDITIONAL"
        : group.prompt.length > 100
          ? group.prompt.slice(0, 100) + "..."
          : group.prompt;

    return (
      <div className="space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-[var(--color-on-surface-secondary)]">
            Group {group.group_index} / {group.total_groups} &middot; AR: {group.aspectratio} &middot; Pair{" "}
            {pairsDone + 1}/{group.pairs_target}
          </div>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={handleSkipGroup}>
              <SkipForward className="w-3.5 h-3.5 mr-1" />
              Skip
            </Button>
            <Button size="sm" variant="ghost" onClick={handleCancel}>
              <X className="w-3.5 h-3.5 mr-1" />
              Cancel
            </Button>
          </div>
        </div>

        {/* Prompt */}
        <div className="text-xs font-mono p-2 rounded bg-[var(--color-surface-container)] text-[var(--color-on-surface)]">
          {promptDisplay}
        </div>

        {/* Phase indicator */}
        <div
          className="text-sm font-semibold text-center py-1 rounded"
          style={{
            color: phase === "best" ? "#22c55e" : "#ef4444",
            background: phase === "best" ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
          }}
        >
          {phase === "best" ? (
            "Click an image to select as BEST (chosen)"
          ) : (
            <>
              Best selected: {bestImage ? bestImage.split(/[/\\]/).pop() : ""}.{" "}
              Now click to select WORST (rejected)
            </>
          )}
        </div>

        {/* Image grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 max-h-[400px] overflow-y-auto">
          {images.map((imgPath) => (
            <button
              key={imgPath}
              className="relative rounded-lg overflow-hidden border-2 transition-all hover:shadow-lg"
              style={{
                borderColor:
                  imgPath === bestImage
                    ? "#22c55e"
                    : "var(--color-border-subtle)",
              }}
              onClick={() => handleSelect(imgPath)}
            >
              <img
                src={`${API_BASE}/dpo/session/image?path=${encodeURIComponent(imgPath)}`}
                alt={imgPath.split(/[/\\]/).pop()}
                className="w-full h-40 object-cover"
                loading="lazy"
              />
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] px-1 py-0.5 truncate">
                {imgPath.split(/[/\\]/).pop()}
              </div>
            </button>
          ))}
        </div>

        {error && (
          <div className="text-xs text-[var(--color-error-500)]">{error}</div>
        )}
      </div>
    );
  }

  if (step === "export") {
    return (
      <div className="space-y-4 text-center py-4">
        {finalResult ? (
          <>
            <CheckCircle2 className="w-10 h-10 mx-auto text-[var(--color-success-500)]" />
            <p className="text-lg font-semibold text-[var(--color-on-surface)]">Finalized!</p>
            <p className="text-sm text-[var(--color-on-surface-secondary)]">
              {finalResult.total_pairs} pairs ({finalResult.train_count} train, {finalResult.val_count} val)
            </p>
            <Button variant="primary" onClick={onDone}>
              Done
            </Button>
          </>
        ) : (
          <>
            <CheckCircle2 className="w-10 h-10 mx-auto text-[var(--color-cobalt-600)]" />
            <p className="text-lg font-semibold text-[var(--color-on-surface)]">Scoring Complete</p>
            <p className="text-sm text-[var(--color-on-surface-secondary)]">
              {totalPairs} pairs collected. Set validation split and finalize.
            </p>
            <div className="flex items-center justify-center gap-2">
              <label className="text-sm text-[var(--color-on-surface)]">Validation %:</label>
              <input
                type="number"
                className="w-16 px-2 py-1 rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] text-[var(--color-on-surface)] text-sm"
                value={valPercentage}
                onChange={(e) => setValPercentage(Number(e.target.value))}
                min={0}
                max={100}
              />
            </div>
            <div className="flex gap-2 justify-center">
              <Button variant="primary" onClick={handleFinalize}>
                Finalize
              </Button>
              <Button variant="ghost" onClick={onDone}>
                Close (Pairs Already Saved)
              </Button>
            </div>
          </>
        )}
      </div>
    );
  }

  return null;
}

// ---------- Main Modal ----------

export function DPOToolModal({ open, onClose }: Props) {
  const [loading, setLoading] = useState(false);
  const [checkResult, setCheckResult] = useState<PairCheckResult | null>(null);
  const [reviewPairs, setReviewPairs] = useState<ReviewPair[] | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCuration, setShowCuration] = useState(false);

  const clearState = () => {
    setMessage(null);
    setError(null);
  };

  const handleCheckPairs = useCallback(async () => {
    clearState();
    setLoading(true);
    try {
      const res = await dpoApi.checkPairs();
      if (res.ok && res.result) {
        setCheckResult(res.result);
      } else {
        setError(res.error ?? "Check failed");
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleRemoveStrays = useCallback(async () => {
    clearState();
    setLoading(true);
    try {
      const res = await dpoApi.removeStrays();
      if (res.ok) {
        setMessage(`Removed ${res.removed ?? 0} stray file(s).`);
        const check = await dpoApi.checkPairs();
        if (check.ok && check.result) setCheckResult(check.result);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleReviewPairs = useCallback(async () => {
    clearState();
    setCheckResult(null);
    setLoading(true);
    try {
      const res = await dpoApi.reviewPairs();
      if (res.ok && res.pairs) {
        setReviewPairs(res.pairs);
      } else {
        setError(res.error ?? "Failed to load pairs");
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleRemovePair = useCallback(async (chosen: string, rejected: string) => {
    try {
      await dpoApi.removePair(chosen, rejected);
      setReviewPairs((prev) => prev?.filter((p) => p.chosen_path !== chosen || p.rejected_path !== rejected) ?? null);
      setMessage("Pair removed.");
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const handleFixCaptions = useCallback(async () => {
    clearState();
    setLoading(true);
    try {
      const res = await dpoApi.fixCaptions();
      if (res.ok) {
        setMessage(`Fixed ${res.fixed ?? 0} multiline caption(s).`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  if (showCuration) {
    return (
      <ModalBase open={open} onClose={onClose} title="DPO Pair Tool — Selection Mode" size="2xl">
        <SelectionCuration onDone={() => setShowCuration(false)} />
      </ModalBase>
    );
  }

  return (
    <ModalBase open={open} onClose={onClose} title="DPO Pair Tool" size="xl">
      <div className="space-y-4">
        {/* Actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <Button variant="primary" onClick={() => setShowCuration(true)}>
            <MousePointer2 className="w-4 h-4 mr-1" />
            Selection Mode
          </Button>
          <Button variant="secondary" onClick={handleCheckPairs} disabled={loading}>
            <Search className="w-4 h-4 mr-1" />
            Check Pairs
          </Button>
          <Button variant="secondary" onClick={handleRemoveStrays} disabled={loading}>
            <Scissors className="w-4 h-4 mr-1" />
            Remove Strays
          </Button>
          <Button variant="secondary" onClick={handleReviewPairs} disabled={loading}>
            <Eye className="w-4 h-4 mr-1" />
            Review Pairs
          </Button>
          <Button variant="secondary" onClick={handleFixCaptions} disabled={loading}>
            <Wrench className="w-4 h-4 mr-1" />
            Fix Captions
          </Button>
        </div>

        {/* Messages */}
        {message && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-success-500)]">
            <CheckCircle2 className="w-4 h-4" />
            {message}
          </div>
        )}
        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)]">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </div>
        )}

        {/* Check Results */}
        {checkResult && <CheckResults result={checkResult} />}

        {/* Review Panel */}
        {reviewPairs && <ReviewPanel pairs={reviewPairs} onRemove={handleRemovePair} />}

        {/* Empty state */}
        {!checkResult && !reviewPairs && !loading && !error && (
          <div className="text-center py-8 text-[var(--color-on-surface-secondary)]">
            <MousePointer2 className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Curate and manage DPO chosen/rejected image pairs.</p>
            <p className="text-xs mt-1">
              Use <strong>Selection Mode</strong> to score generated images and create pairs, or use
              the other tools to manage existing pairs.
            </p>
          </div>
        )}
      </div>
    </ModalBase>
  );
}
