import { Loader2, StopCircle, Wand2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { type Character, characterDpoApi, type GenerateJob } from "@/api/characterDpoApi";
import { swarmApi, type SwarmJobSummary, type SwarmStatus } from "@/api/swarmApi";
import { Button } from "@/components/shared";

import { Bar, Chip, errorText, HINT, SECTION, SECTION_TITLE } from "./ui";

/**
 * Runs queued by the character flow carry the character name as `label`;
 * the generic Swarm status type does not know the field.
 */
type LabeledJob = SwarmJobSummary & { label?: string };

const PHASE_LABEL: Record<SwarmStatus["phase"], string> = {
  idle: "Idle",
  starting: "Starting…",
  indexing: "Reading real photos and captions…",
  scanning: "Reading existing generations…",
  planning: "Choosing captions…",
  generating: "Generating in SwarmUI…",
  done: "Finished",
  error: "Failed",
  cancelling: "Stopping…",
  cancelled: "Stopped",
};

export interface GenerateProgressProps {
  /** The checked characters, in list order. */
  characters: Character[];
  root: string;
  modelsDir: string;
  swarmUrl: string;
  swarmToken: string;
  swarmLoraDir: string;
  paramFolder: string;
  gensPerCaption: number;
  /** 0 = every caption without a pair. */
  photosPerCharacter: number;
  /** Called once a run that this panel started has stopped (to refresh the list). */
  onFinished: () => void;
}

export function GenerateProgress({
  characters,
  root,
  modelsDir,
  swarmUrl,
  swarmToken,
  swarmLoraDir,
  paramFolder,
  gensPerCaption,
  photosPerCharacter,
  onFinished,
}: GenerateProgressProps) {
  const [captionCounts, setCaptionCounts] = useState<Record<string, number>>({});
  const [jobs, setJobs] = useState<GenerateJob[] | null>(null);
  const [status, setStatus] = useState<SwarmStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const wasRunningRef = useRef(false);
  const onFinishedRef = useRef(onFinished);
  onFinishedRef.current = onFinished;

  const names = characters.map((c) => c.name);
  const namesKey = names.join("\u0000");

  // Caption totals, fetched lazily and tolerant of a slow or missing route.
  useEffect(() => {
    let cancelled = false;
    const missing = names.filter((n) => captionCounts[n] === undefined);
    if (missing.length === 0) return;
    void Promise.all(
      missing.map((n) =>
        characterDpoApi
          .trainCaptions(n, root)
          .then((r) => [n, Number(r.count) || 0] as const)
          .catch(() => [n, -1] as const),
      ),
    ).then((pairs) => {
      if (cancelled) return;
      setCaptionCounts((prev) => {
        const next = { ...prev };
        for (const [n, c] of pairs) next[n] = c;
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namesKey, root]);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    clearPoll();
    // The run is queued on a worker thread, so the first poll or two may land
    // before `running` flips; only give up after it has stayed idle a moment.
    let idlePolls = 0;
    pollRef.current = window.setInterval(() => {
      void swarmApi
        .status()
        .then((s) => {
          setStatus(s);
          if (s.running) {
            idlePolls = 0;
            wasRunningRef.current = true;
          } else if (++idlePolls >= 3) {
            clearPoll();
            if (wasRunningRef.current) {
              wasRunningRef.current = false;
              onFinishedRef.current();
            }
          }
        })
        .catch(() => {
          /* transient */
        });
    }, 1000);
  }, [clearPoll]);

  // Reflect a run already in flight when the step is (re)entered.
  useEffect(() => {
    let cancelled = false;
    void swarmApi
      .status()
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        if (s.running) {
          wasRunningRef.current = true;
          startPolling();
        }
      })
      .catch(() => {
        /* backend not reachable yet */
      });
    return () => {
      cancelled = true;
      clearPoll();
    };
  }, [startPolling, clearPoll]);

  const generate = useCallback(async () => {
    if (names.length === 0) return;
    setBusy(true);
    setError(null);
    setJobs(null);
    try {
      const res = await characterDpoApi.generate({
        names,
        root,
        models_dir: modelsDir || undefined,
        swarm_url: swarmUrl || undefined,
        swarm_token: swarmToken || undefined,
        swarm_lora_dir: swarmLoraDir,
        param_folder: paramFolder,
        images_per_caption: gensPerCaption,
        captions: photosPerCharacter,
      });
      setJobs(res.jobs ?? []);
      if (!res.ok && res.error) setError(res.error);
      if ((res.jobs ?? []).some((j) => j.ok)) {
        wasRunningRef.current = true;
        startPolling();
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }, [
    names,
    root,
    modelsDir,
    swarmUrl,
    swarmToken,
    swarmLoraDir,
    paramFolder,
    gensPerCaption,
    photosPerCharacter,
    startPolling,
  ]);

  const cancel = useCallback(async () => {
    try {
      await swarmApi.cancel();
    } catch {
      /* best effort */
    }
  }, []);

  const running = status?.running ?? false;
  const known = names.filter((n) => (captionCounts[n] ?? -1) >= 0);
  const captionTotal = known.reduce((sum, n) => sum + captionCounts[n], 0);
  const current = status?.current_job as LabeledJob | null | undefined;
  const queued = (status?.queued ?? []) as LabeledJob[];

  return (
    <div className={`${SECTION} min-h-0 flex-1 overflow-y-auto`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className={SECTION_TITLE}>Generate negatives</span>
        <span className={`${HINT} tabular-nums`}>
          {names.length} character{names.length === 1 ? "" : "s"} · captions:{" "}
          {known.length === names.length ? captionTotal : known.length > 0 ? `${captionTotal}+ (counting…)` : "…"} ·{" "}
          {photosPerCharacter > 0 ? `up to ${photosPerCharacter} photos each` : "all photos"} · {gensPerCaption} image
          {gensPerCaption === 1 ? "" : "s"} per caption
        </span>
        <div className="ml-auto flex gap-2">
          {running && (
            <Button variant="danger" size="sm" onClick={() => void cancel()}>
              <StopCircle className="w-4 h-4" /> Cancel
            </Button>
          )}
          <Button variant="primary" size="sm" onClick={() => void generate()} disabled={busy || names.length === 0}>
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
            {running ? "Queue" : "Generate"} negatives for {names.length} character{names.length === 1 ? "" : "s"}
          </Button>
        </div>
      </div>
      <div className={HINT}>
        One SwarmUI run per character: its published LoRA, every train caption still without a generation, at each
        photo&apos;s bucket size. Images land under &lt;character&gt;/dpo/generated.
      </div>

      {error && <div className="text-xs text-[var(--color-error-500)] whitespace-pre-wrap">{error}</div>}

      {jobs && (
        <div className="flex flex-col gap-1">
          {jobs.map((j) => (
            <div
              key={j.name}
              className="flex flex-wrap items-center gap-2 text-xs bg-[var(--color-surface-container)] rounded px-2 py-1"
            >
              <span className="font-semibold text-[var(--color-on-surface)]">{j.name}</span>
              {j.ok ? (
                <>
                  <Chip tone="ok">queued{j.position != null ? ` #${j.position}` : ""}</Chip>
                  {j.swarm_lora && <Chip title={j.swarm_lora}>LoRA {j.swarm_lora}</Chip>}
                  {captionCounts[j.name] != null && captionCounts[j.name] >= 0 && (
                    <Chip>{captionCounts[j.name]} captions</Chip>
                  )}
                </>
              ) : (
                <span className="text-[var(--color-error-500)]">{j.error ?? "not queued"}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {status && status.phase !== "idle" && (
        <div className="flex flex-col gap-2 bg-[var(--color-surface-container)] rounded p-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            {running && <Loader2 className="w-4 h-4 animate-spin" />}
            <span>{PHASE_LABEL[status.phase]}</span>
            {current?.label && <span className="font-semibold text-[var(--color-on-surface)]">— {current.label}</span>}
            {status.job_total > 1 && (
              <span className={`${HINT} tabular-nums`}>
                run {status.job_index} of {status.job_total}
              </span>
            )}
            <span className={`ml-auto ${HINT} tabular-nums`}>
              {status.done} / {status.planned} · {status.images} images
              {status.failed > 0 ? ` · ${status.failed} failed` : ""}
            </span>
          </div>
          {running && <Bar done={status.done} total={status.planned} />}
          {queued.length > 0 && (
            <div className={HINT}>Waiting: {queued.map((j) => j.label || j.keywords || "(run)").join(", ")}</div>
          )}
          {status.param_source && (
            <div className={HINT}>
              Settings from <span className="text-[var(--color-on-surface)]">{status.param_source}</span>
              {Object.keys(status.params ?? {}).length > 0 && (
                <>
                  {" — "}
                  {Object.entries(status.params)
                    .filter(([key]) => key !== "negativeprompt")
                    .map(([key, value]) => `${key} ${Array.isArray(value) ? value.join("+") : String(value)}`)
                    .join(" · ")}
                </>
              )}
            </div>
          )}
          {status.error && <div className="text-xs text-[var(--color-error-500)]">{status.error}</div>}
        </div>
      )}
    </div>
  );
}
