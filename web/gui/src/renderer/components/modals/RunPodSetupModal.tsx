import { CheckCircle2, Cloud, GitBranch, HardDrive, KeyRound, Loader2, Play, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { cloudApi, type RunpodGitPreflight, type RunpodPreview } from "@/api/cloudApi";
import { ModalBase } from "@/components/modals/ModalBase";
import { Button } from "@/components/shared";
import { useConfigStore } from "@/store/configStore";
import { useTrainingStore } from "@/store/trainingStore";
import { useUiStore } from "@/store/uiStore";
import { INPUT_FULL } from "@/utils/inputStyles";

interface RunPodSetupModalProps {
  open: boolean;
  onClose: () => void;
}

function formatCommit(commit: string): string {
  return commit ? commit.slice(0, 10) : "";
}

function StatusIcon({ ok }: { ok: boolean }) {
  return ok ? <CheckCircle2 className="w-4 h-4 text-[var(--color-success-500)]" /> : <XCircle className="w-4 h-4 text-[var(--color-error-500)]" />;
}

export function RunPodSetupModal({ open, onClose }: RunPodSetupModalProps) {
  const loadConfig = useConfigStore((s) => s.loadConfig);
  const startTraining = useTrainingStore((s) => s.startTraining);
  const setTerminalOpen = useUiStore((s) => s.setTerminalOpen);

  const [preview, setPreview] = useState<RunpodPreview | null>(null);
  const [preflight, setPreflight] = useState<RunpodGitPreflight | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [sshUser, setSshUser] = useState("root");
  const [sshKeyFile, setSshKeyFile] = useState("");
  const [sshPassword, setSshPassword] = useState("");
  const [podName, setPodName] = useState("OneTrainer");
  const [runId, setRunId] = useState("job1");
  const [minDownload, setMinDownload] = useState(0);
  const [liveTestEpochs, setLiveTestEpochs] = useState(3);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [runningPreflight, setRunningPreflight] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startingLiveTest, setStartingLiveTest] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canStart = useMemo(
    () => Boolean(preview && preview.errors.length === 0 && preflight && apiKey.trim()),
    [apiKey, preflight, preview],
  );

  const refreshPreview = async () => {
    setLoadingPreview(true);
    setError(null);
    try {
      const data = await cloudApi.runpodPreview();
      setPreview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingPreview(false);
    }
  };

  useEffect(() => {
    if (open) void refreshPreview();
  }, [open]);

  const runPreflight = async () => {
    setRunningPreflight(true);
    setError(null);
    try {
      const result = await cloudApi.runpodGitPreflight();
      setPreflight(result);
      await refreshPreview();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunningPreflight(false);
    }
  };

  const prepareAndStart = async () => {
    setStarting(true);
    setError(null);
    try {
      await cloudApi.runpodPrepare({
        api_key: apiKey,
        ssh_user: sshUser,
        ssh_key_file: sshKeyFile,
        ssh_password: sshPassword,
        pod_name: podName,
        run_id: runId,
        min_download: minDownload,
        start: false,
      });
      await loadConfig();
      setTerminalOpen(true);
      await startTraining();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  };

  const runLiveTest = async () => {
    setStartingLiveTest(true);
    setError(null);
    try {
      setTerminalOpen(true);
      await cloudApi.runpodLiveTest({
        api_key: apiKey,
        ssh_user: sshUser,
        ssh_key_file: sshKeyFile,
        ssh_password: sshPassword,
        pod_name: `${podName}-live-test`,
        run_id: runId,
        min_download: minDownload,
        start: true,
        epochs: liveTestEpochs,
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStartingLiveTest(false);
    }
  };

  return (
    <ModalBase open={open} onClose={onClose} title="RunPod Sourceless Setup" size="xl" closeOnBackdrop={!starting}>
      <div className="grid grid-cols-[minmax(0,1fr)_320px] gap-5 max-lg:grid-cols-1">
        <div className="flex flex-col gap-4">
          <section className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)]">
            <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[var(--color-border-subtle)]">
              <div className="flex items-center gap-2 text-[var(--color-on-surface)] font-semibold">
                <HardDrive className="w-4 h-4" />
                Storage Preview
              </div>
              <Button variant="secondary" size="sm" onClick={refreshPreview} loading={loadingPreview}>
                <RefreshCw className="w-3.5 h-3.5" /> Refresh
              </Button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-[var(--color-on-surface-secondary)]">
                  <tr className="border-b border-[var(--color-border-subtle)]">
                    <th className="px-4 py-2 font-semibold">Item</th>
                    <th className="px-4 py-2 font-semibold">Size</th>
                    <th className="px-4 py-2 font-semibold">Files</th>
                    <th className="px-4 py-2 font-semibold">Path</th>
                  </tr>
                </thead>
                <tbody>
                  {preview?.entries.map((entry) => (
                    <tr key={entry.label} className="border-b border-[var(--color-border-subtle)] last:border-b-0">
                      <td className="px-4 py-2">
                        <span className="inline-flex items-center gap-2">
                          <StatusIcon ok={entry.exists || !entry.required} />
                          {entry.label.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="px-4 py-2 tabular-nums">{entry.exists ? `${entry.gib} GiB` : "-"}</td>
                      <td className="px-4 py-2 tabular-nums">{entry.exists ? entry.files.toLocaleString() : "-"}</td>
                      <td className="px-4 py-2 text-[var(--color-on-surface-secondary)] max-w-[380px] truncate" title={entry.path}>
                        {entry.path || "(empty)"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {preview && (
              <div className="px-4 py-3 flex flex-wrap gap-3 text-sm border-t border-[var(--color-border-subtle)]">
                <span className="tabular-nums">Total {preview.total_gib} GiB</span>
                <span className="tabular-nums">Buffer {preview.buffer_gib} GiB</span>
                <span className="font-semibold tabular-nums text-[var(--color-cobalt-600)]">RunPod volume {preview.required_gb} GB</span>
              </div>
            )}
          </section>

          <section className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)]">
            <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[var(--color-border-subtle)]">
              <div className="flex items-center gap-2 text-[var(--color-on-surface)] font-semibold">
                <GitBranch className="w-4 h-4" />
                Git Preflight
              </div>
              <Button variant="secondary" size="sm" onClick={runPreflight} loading={runningPreflight}>
                <GitBranch className="w-3.5 h-3.5" /> Commit & Push
              </Button>
            </div>
            <div className="px-4 py-3 grid grid-cols-2 gap-3 text-sm max-sm:grid-cols-1">
              <div className="rounded border border-[var(--color-border-subtle)] px-3 py-2">
                <div className="text-[var(--color-on-surface-secondary)] text-xs uppercase font-semibold">OneTrainer</div>
                <div className="mt-1">{preflight ? `${formatCommit(preflight.main.commit)} -> ${preflight.main.remote}/${preflight.main.branch}` : preview?.git.main_dirty ? "Pending local changes" : "No local changes detected"}</div>
              </div>
              <div className="rounded border border-[var(--color-border-subtle)] px-3 py-2">
                <div className="text-[var(--color-on-surface-secondary)] text-xs uppercase font-semibold">mgds</div>
                <div className="mt-1">{preflight ? `${formatCommit(preflight.mgds.commit)} -> ${preflight.mgds.branch}` : preview?.git.mgds_dirty ? "Pending local changes" : "No local changes detected"}</div>
              </div>
            </div>
          </section>

          {preview?.errors.length ? (
            <div className="rounded-md border border-[var(--color-error-500)] bg-[var(--color-error-500-alpha-08)] px-4 py-3 text-sm text-[var(--color-error-500)]">
              {preview.errors.join(" ")}
            </div>
          ) : null}
          {preview?.warnings.length ? (
            <div className="rounded-md border border-[var(--color-warning-500-alpha-20)] bg-[var(--color-warning-500-alpha-12)] px-4 py-3 text-sm text-[var(--color-warning-500)]">
              {preview.warnings.join(" ")}
            </div>
          ) : null}
          {error && (
            <div className="rounded-md border border-[var(--color-error-500)] bg-[var(--color-error-500-alpha-08)] px-4 py-3 text-sm text-[var(--color-error-500)]">
              {error}
            </div>
          )}
        </div>

        <aside className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] p-4 flex flex-col gap-4 h-fit">
          <div className="flex items-center gap-2 font-semibold">
            <Cloud className="w-4 h-4" />
            RTX 5090 Europe
          </div>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">API Key</span>
            <input className={INPUT_FULL} type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">SSH User</span>
            <input className={INPUT_FULL} value={sshUser} onChange={(e) => setSshUser(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">SSH Key File</span>
            <input className={INPUT_FULL} value={sshKeyFile} onChange={(e) => setSshKeyFile(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">SSH Password</span>
            <input className={INPUT_FULL} type="password" value={sshPassword} onChange={(e) => setSshPassword(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">Pod Name</span>
            <input className={INPUT_FULL} value={podName} onChange={(e) => setPodName(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">Run ID</span>
            <input className={INPUT_FULL} value={runId} onChange={(e) => setRunId(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">Min Download Mbps</span>
            <input className={INPUT_FULL} type="number" value={minDownload} onChange={(e) => setMinDownload(Number(e.target.value) || 0)} />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase font-semibold text-[var(--color-on-surface-secondary)]">Live Test Epochs</span>
            <input
              className={INPUT_FULL}
              type="number"
              min={1}
              max={10}
              value={liveTestEpochs}
              onChange={(e) => setLiveTestEpochs(Math.max(1, Number(e.target.value) || 3))}
            />
          </label>
          <Button
            variant="secondary"
            onClick={runLiveTest}
            disabled={!canStart}
            loading={startingLiveTest}
            className="w-full mt-1"
          >
            {startingLiveTest ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Run Live Test
          </Button>
          <Button onClick={prepareAndStart} disabled={!canStart} loading={starting} className="w-full mt-1">
            {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Prepare & Start
          </Button>
          {!preflight && <div className="text-xs text-[var(--color-on-surface-secondary)]">Git preflight must complete before launch.</div>}
          <div className="flex items-center gap-2 text-xs text-[var(--color-on-surface-secondary)]">
            <KeyRound className="w-3.5 h-3.5" />
            Secrets stay in the local secrets store.
          </div>
        </aside>
      </div>
    </ModalBase>
  );
}
