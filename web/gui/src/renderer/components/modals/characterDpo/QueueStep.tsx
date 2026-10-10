import { Download, Eye, ListPlus, Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  type Character,
  characterDpoApi,
  type PrepareStatus,
  type QueueBuildMode,
  type QueueBuildResponse,
} from "@/api/characterDpoApi";
import { ApiError } from "@/api/request";
import { Button } from "@/components/shared";
import { useQueueStore } from "@/store/queueStore";

import { Bar, Chip, downloadJson, errorText, HINT, SECTION, SECTION_TITLE } from "./ui";

export interface QueueStepProps {
  characters: Character[];
  root: string;
  modelsDir: string;
  valPct: number;
  dpoResolution: string;
  maskModel: string;
  addOnWeight: number;
  dpoBeta: number;
  onNotice: (text: string) => void;
  onError: (text: string) => void;
  /** Called after a prepare run or an add changed what is on disk / in the queue. */
  onChanged: () => void;
}

function waitForPrepare(onTick: (s: PrepareStatus) => void): Promise<PrepareStatus> {
  return new Promise((resolve, reject) => {
    let failures = 0;
    const timer = window.setInterval(() => {
      void characterDpoApi
        .prepareStatus()
        .then((s) => {
          failures = 0;
          onTick(s);
          if (!s.running) {
            window.clearInterval(timer);
            resolve(s);
          }
        })
        .catch((e: unknown) => {
          // A blip is fine; a backend that stays away is not.
          if (++failures < 5) return;
          window.clearInterval(timer);
          reject(e instanceof Error ? e : new Error(String(e)));
        });
    }, 1000);
  });
}

export function QueueStep({
  characters,
  root,
  modelsDir,
  valPct,
  dpoResolution,
  maskModel,
  addOnWeight,
  dpoBeta,
  onNotice,
  onError,
  onChanged,
}: QueueStepProps) {
  const [busy, setBusy] = useState<QueueBuildMode | null>(null);
  const [prepare, setPrepare] = useState<PrepareStatus | null>(null);
  const [buildResult, setBuildResult] = useState<{ mode: QueueBuildMode; res: QueueBuildResponse } | null>(null);
  const [excluded, setExcluded] = useState<string[]>([]);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    // Show a prepare already in flight (another window, or a step change mid-run).
    void characterDpoApi
      .prepareStatus()
      .then((s) => {
        if (mountedRef.current && s.running) setPrepare(s);
      })
      .catch(() => undefined);
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const names = characters.map((c) => c.name);

  const run = useCallback(
    async (mode: QueueBuildMode) => {
      if (names.length === 0) return;
      setBusy(mode);
      setBuildResult(null);
      setExcluded([]);
      try {
        // 1. Finalize everything in one background job.
        let start;
        try {
          start = await characterDpoApi.prepare({
            names,
            root,
            models_dir: modelsDir || undefined,
            val_pct: valPct,
            mask_model: maskModel,
            dpo_resolution: dpoResolution,
          });
        } catch (e) {
          if (e instanceof ApiError && e.status === 409) {
            onError("A finalize is already running — wait for it to finish, then try again.");
            return;
          }
          throw e;
        }
        if (!start.ok) {
          onError(start.error ?? "Could not start the finalize job");
          return;
        }
        setPrepare({
          running: true,
          current: "",
          done: 0,
          total: names.length,
          stage: "starting",
          message: "",
          error: null,
          results: [],
        });
        const final = await waitForPrepare((s) => {
          if (mountedRef.current) setPrepare(s);
        });
        if (!mountedRef.current) return;
        onChanged();
        if (final.error) onError(final.error);

        // 2. Build only the characters whose prepare went through.
        const failed = (final.results ?? []).filter((r) => r.ok === false && !r.skipped).map((r) => r.name);
        const okNames = names.filter((n) => !failed.includes(n));
        setExcluded(failed);
        if (okNames.length === 0) {
          onError("Every selected character failed to finalize — nothing to build.");
          return;
        }
        const res = await characterDpoApi.queueBuild({
          names: okNames,
          root,
          models_dir: modelsDir || undefined,
          add_on_weight: addOnWeight,
          dpo_beta: dpoBeta,
          mode,
        });
        if (!mountedRef.current) return;
        setBuildResult({ mode, res });
        if (mode === "add") {
          onNotice(`Added ${res.added} queue entr${res.added === 1 ? "y" : "ies"}.`);
          await useQueueStore.getState().loadQueue();
          onChanged();
        } else if (mode === "export") {
          if (res.entries.length > 0) {
            downloadJson("character-dpo-queue.json", {
              __version: 1,
              entries: res.entries.map((e) => ({
                name: e.name,
                overrides: e.overrides,
                status: "PENDING",
                included: true,
              })),
            });
            onNotice(`Exported ${res.entries.length} entr${res.entries.length === 1 ? "y" : "ies"}.`);
          } else {
            onError("Nothing to export — every selected character failed to build.");
          }
        }
      } catch (e) {
        onError(errorText(e));
      } finally {
        if (mountedRef.current) setBusy(null);
      }
    },
    [names, root, modelsDir, valPct, maskModel, dpoResolution, addOnWeight, dpoBeta, onNotice, onError, onChanged],
  );

  const disabled = busy !== null || names.length === 0 || prepare?.running === true;

  return (
    <div className={`${SECTION} min-h-0 flex-1 overflow-y-auto`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className={SECTION_TITLE}>Queue</span>
        <span className={HINT}>
          {names.length} checked · finalize first (val {valPct}%, masks {maskModel} — masks can take a minute per
          character), then build one queue entry each
        </span>
        <div className="ml-auto flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" onClick={() => void run("preview")} disabled={disabled}>
            {busy === "preview" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
            Preview
          </Button>
          <Button variant="primary" size="sm" onClick={() => void run("add")} disabled={disabled}>
            {busy === "add" ? <Loader2 className="w-4 h-4 animate-spin" /> : <ListPlus className="w-4 h-4" />}
            Add to queue
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void run("export")}
            disabled={disabled}
            title="Download character-dpo-queue.json for the Queue page's Import"
          >
            {busy === "export" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Export queue file
          </Button>
        </div>
      </div>

      {prepare && (
        <div className="flex flex-col gap-1 text-xs">
          {prepare.running ? (
            <>
              <div className="flex items-center justify-between text-[var(--color-on-surface-secondary)]">
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Finalizing {prepare.current || "…"}
                  {prepare.stage ? ` — ${prepare.stage}` : ""}
                  {prepare.message ? `: ${prepare.message}` : ""}
                </span>
                <span className="tabular-nums">
                  {prepare.done} / {prepare.total}
                </span>
              </div>
              <Bar done={prepare.done} total={prepare.total} />
            </>
          ) : (
            <div className="text-[var(--color-success-500)]">
              Finalize finished{prepare.message ? ` — ${prepare.message}` : "."}
            </div>
          )}
          {prepare.error && !prepare.running && (
            <div className="text-[var(--color-error-500)] whitespace-pre-wrap">{prepare.error}</div>
          )}
          {prepare.results.length > 0 && (
            <div className="flex flex-col gap-1">
              {prepare.results.map((r) => (
                <div
                  key={r.name}
                  className="flex flex-wrap items-center gap-1 bg-[var(--color-surface-container)] rounded px-2 py-1"
                >
                  <span className="font-semibold text-[var(--color-on-surface)]">{r.name}</span>
                  {r.skipped && <Chip tone="info">skipped{r.message ? ` — ${r.message}` : ""}</Chip>}
                  {!r.skipped && r.ok && <Chip tone="ok">finalized</Chip>}
                  {!r.skipped && !r.ok && <Chip tone="bad">failed</Chip>}
                  {r.counts &&
                    Object.entries(r.counts).map(([k, v]) => (
                      <Chip key={k}>
                        {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
                      </Chip>
                    ))}
                  {r.error && <span className="text-[var(--color-error-500)]">{r.error}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {excluded.length > 0 && (
        <div className="text-xs text-[var(--color-warning-500,#eab308)]">
          Left out of the build (finalize failed): {excluded.join(", ")}
        </div>
      )}

      {buildResult && (
        <div className="flex flex-col gap-1 max-h-[36vh] overflow-y-auto">
          {buildResult.res.errors.map((e) => (
            <div key={`err-${e.name}`} className="text-xs text-[var(--color-error-500)]">
              <span className="font-semibold">{e.name}</span>: {e.error}
            </div>
          ))}
          {buildResult.res.entries.map((e) => (
            <details key={e.name} className="text-xs bg-[var(--color-surface-container)] rounded px-2 py-1">
              <summary className="cursor-pointer select-none flex flex-wrap items-center gap-2">
                <span className="font-semibold text-[var(--color-on-surface)]">{e.name}</span>
                <span className={HINT}>{Object.keys(e.overrides).length} override(s)</span>
                {e.warnings.length > 0 && (
                  <Chip tone="warn">
                    {e.warnings.length} warning{e.warnings.length === 1 ? "" : "s"}
                  </Chip>
                )}
                {buildResult.mode === "add" && <Chip tone="ok">added</Chip>}
                {buildResult.mode === "export" && <Chip tone="ok">exported</Chip>}
              </summary>
              {e.warnings.length > 0 && (
                <ul className="list-disc ml-4 mt-1 text-[var(--color-warning-500,#eab308)]">
                  {e.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              )}
              <pre className="mt-1 whitespace-pre-wrap break-all font-mono text-[10px] text-[var(--color-on-surface-secondary)]">
                {JSON.stringify(e.overrides, null, 2)}
              </pre>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
