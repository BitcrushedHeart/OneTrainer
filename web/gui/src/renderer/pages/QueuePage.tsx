import {
    AlertTriangle,
    CheckCircle2,
    ChevronDown,
    ChevronUp,
    Copy,
    Download,
    FilePlus,
    Loader2,
    Pause,
    Play,
    Plus,
    SkipForward,
    Square,
    Trash2,
    Upload,
    XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { queueApi } from "@/api/queueApi";
import { DiffPanel } from "@/components/queue/DiffPanel";
import { Button, Card, FormEntry, Toggle } from "@/components/shared";
import { useQueueStore } from "@/store/queueStore";

const STATUS_STYLES: Record<string, { bg: string; text: string; icon: typeof CheckCircle2 }> = {
    PENDING: { bg: "rgba(148, 163, 184, 0.15)", text: "#94a3b8", icon: Loader2 },
    RUNNING: { bg: "rgba(59, 130, 246, 0.15)", text: "#3b82f6", icon: Play },
    COMPLETED: { bg: "rgba(34, 197, 94, 0.15)", text: "#22c55e", icon: CheckCircle2 },
    FAILED: { bg: "rgba(239, 68, 68, 0.15)", text: "#ef4444", icon: XCircle },
    SKIPPED: { bg: "rgba(234, 179, 8, 0.15)", text: "#eab308", icon: SkipForward },
};

function StatusBadge({ status }: { status: string }) {
    const style = STATUS_STYLES[status] ?? STATUS_STYLES.PENDING;
    const Icon = style.icon;
    return (
        <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
            style={{ background: style.bg, color: style.text }}
        >
            <Icon className="w-3 h-3" />
            {status}
        </span>
    );
}

function EntryList() {
    const entries = useQueueStore((s) => s.entries);
    const selectedEntryId = useQueueStore((s) => s.selectedEntryId);
    const selectEntry = useQueueStore((s) => s.selectEntry);
    const addEntry = useQueueStore((s) => s.addEntry);
    const removeEntry = useQueueStore((s) => s.removeEntry);
    const duplicateEntry = useQueueStore((s) => s.duplicateEntry);
    const reorder = useQueueStore((s) => s.reorder);
    const loadQueue = useQueueStore((s) => s.loadQueue);
    const status = useQueueStore((s) => s.status);
    const isRunning = status === "running" || status === "stopping";
    const fromFileInputRef = useRef<HTMLInputElement>(null);

    const handleAddFromFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        try {
            const nameStem = file.name.replace(/\.json$/i, "");
            await queueApi.entryFromFile(file, nameStem);
            await loadQueue();
        } catch (err) {
            alert(`Add from file failed: ${err instanceof Error ? err.message : String(err)}`);
        }
        e.target.value = "";
    };

    return (
        <div className="flex flex-col h-full">
            <div className="flex items-center gap-1 p-2 border-b border-[var(--color-border-subtle)]">
                <Button size="sm" variant="ghost" onClick={() => addEntry()} disabled={isRunning} title="Add entry">
                    <Plus className="w-4 h-4" />
                </Button>
                <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => fromFileInputRef.current?.click()}
                    disabled={isRunning}
                    title="Add entry from a training-config JSON file"
                >
                    <FilePlus className="w-4 h-4" />
                </Button>
                <input
                    ref={fromFileInputRef}
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleAddFromFile}
                />
                <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => selectedEntryId && duplicateEntry(selectedEntryId)}
                    disabled={!selectedEntryId || isRunning}
                    title="Duplicate"
                >
                    <Copy className="w-4 h-4" />
                </Button>
                <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => selectedEntryId && removeEntry(selectedEntryId)}
                    disabled={!selectedEntryId || isRunning}
                    title="Remove"
                >
                    <Trash2 className="w-4 h-4" />
                </Button>
                <div className="w-px h-4 bg-[var(--color-border-subtle)] mx-1" />
                <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => selectedEntryId && reorder(selectedEntryId, "up")}
                    disabled={!selectedEntryId || isRunning}
                    title="Move up"
                >
                    <ChevronUp className="w-4 h-4" />
                </Button>
                <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => selectedEntryId && reorder(selectedEntryId, "down")}
                    disabled={!selectedEntryId || isRunning}
                    title="Move down"
                >
                    <ChevronDown className="w-4 h-4" />
                </Button>
            </div>

            <div className="flex-1 overflow-y-auto">
                {entries.length === 0 && (
                    <div className="p-4 text-sm text-[var(--color-on-surface-secondary)] text-center">
                        No queue entries. Click + to add one.
                    </div>
                )}
                {entries.map((entry, i) => (
                    <button
                        key={entry.id}
                        className="w-full text-left px-3 py-2 border-b border-[var(--color-border-subtle)] transition-colors"
                        style={{
                            background:
                                entry.id === selectedEntryId ? "var(--color-cobalt-600-alpha-15)" : "transparent",
                        }}
                        onClick={() => selectEntry(entry.id)}
                    >
                        <div className="flex items-center justify-between gap-2">
                            <span className="text-sm font-medium text-[var(--color-on-surface)] truncate">
                                {entry.name || `Entry ${i + 1}`}
                            </span>
                            <StatusBadge status={entry.status} />
                        </div>
                        {Object.keys(entry.overrides).length > 0 && (
                            <div className="text-xs text-[var(--color-on-surface-secondary)] mt-0.5">
                                {Object.keys(entry.overrides).length} override(s)
                            </div>
                        )}
                    </button>
                ))}
            </div>
        </div>
    );
}

function EntryEditor() {
    const entries = useQueueStore((s) => s.entries);
    const selectedEntryId = useQueueStore((s) => s.selectedEntryId);
    const updateEntry = useQueueStore((s) => s.updateEntry);
    const validationResults = useQueueStore((s) => s.validationResults);

    const entry = entries.find((e) => e.id === selectedEntryId);

    const [draftName, setDraftName] = useState("");
    const [draftOverridesJson, setDraftOverridesJson] = useState("");
    const [jsonError, setJsonError] = useState<string | null>(null);
    const nameTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    useEffect(() => {
        if (entry) {
            setDraftName(entry.name);
            setDraftOverridesJson(JSON.stringify(entry.overrides, null, 2));
            setJsonError(null);
        }
    }, [entry?.id]); // eslint-disable-line react-hooks/exhaustive-deps

    const handleNameChange = useCallback(
        (val: string | number) => {
            const name = String(val);
            setDraftName(name);
            if (nameTimerRef.current) clearTimeout(nameTimerRef.current);
            nameTimerRef.current = setTimeout(() => {
                if (selectedEntryId) updateEntry(selectedEntryId, { name });
            }, 500);
        },
        [selectedEntryId, updateEntry],
    );

    const handleOverridesBlur = useCallback(() => {
        if (!selectedEntryId) return;
        try {
            const parsed = JSON.parse(draftOverridesJson);
            setJsonError(null);
            updateEntry(selectedEntryId, { overrides: parsed });
        } catch (e) {
            setJsonError(String(e));
        }
    }, [selectedEntryId, draftOverridesJson, updateEntry]);

    if (!entry) {
        return (
            <div className="flex items-center justify-center h-full text-[var(--color-on-surface-secondary)]">
                Select an entry to edit
            </div>
        );
    }

    const validation = validationResults[entry.id];

    return (
        <div className="p-4 space-y-4 overflow-y-auto h-full">
            <div>
                <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">Name</label>
                <FormEntry label="" value={draftName} onChange={handleNameChange} />
            </div>

            <div>
                <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">
                    Config Overrides (JSON)
                </label>
                <textarea
                    className="w-full h-64 p-2 rounded-md text-sm font-mono border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)] text-[var(--color-on-surface)] resize-y"
                    value={draftOverridesJson}
                    onChange={(e) => setDraftOverridesJson(e.target.value)}
                    onBlur={handleOverridesBlur}
                    spellCheck={false}
                />
                {jsonError && <p className="text-xs text-[var(--color-error-500)] mt-1">{jsonError}</p>}
            </div>

            <DiffPanel entryId={entry.id} />

            {entry.failure_history.length > 0 && (
                <div>
                    <label className="block text-sm font-medium text-[var(--color-on-surface)] mb-1">
                        Failure History
                    </label>
                    <div className="space-y-1">
                        {entry.failure_history.map((f, i) => (
                            <div
                                key={i}
                                className="text-xs p-2 rounded bg-[rgba(239,68,68,0.08)] text-[var(--color-error-500)]"
                            >
                                Step {f.step}: {f.error}
                                <span className="text-[var(--color-on-surface-secondary)] ml-2">
                                    {new Date(f.timestamp).toLocaleTimeString()}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {validation && (
                <div className="space-y-1">
                    {validation.errors.map((err, i) => (
                        <div key={`e${i}`} className="flex items-start gap-2 text-xs text-[var(--color-error-500)]">
                            <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                            {err}
                        </div>
                    ))}
                    {validation.warnings.map((warn, i) => (
                        <div key={`w${i}`} className="flex items-start gap-2 text-xs text-[var(--color-warning-500)]">
                            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                            {warn}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function QueueSettings() {
    const settings = useQueueStore((s) => s.settings);
    const updateSettings = useQueueStore((s) => s.updateSettings);

    return (
        <Card className="space-y-3">
            <h3 className="text-sm font-semibold text-[var(--color-on-surface)]">Queue Settings</h3>
            <div className="grid grid-cols-2 gap-3">
                <div className="flex items-center gap-2">
                    <Toggle value={settings.retry_on_error} onChange={(v) => updateSettings({ retry_on_error: v })} />
                    <span className="text-sm text-[var(--color-on-surface)]">Retry on Error</span>
                </div>
                <div className="flex items-center gap-2">
                    <Toggle
                        value={settings.retry_from_backup}
                        onChange={(v) => updateSettings({ retry_from_backup: v })}
                    />
                    <span className="text-sm text-[var(--color-on-surface)]">Retry from Backup</span>
                </div>
                <div>
                    <label className="block text-xs text-[var(--color-on-surface-secondary)] mb-0.5">Max Retries</label>
                    <FormEntry
                        label=""
                        type="number"
                        value={settings.max_retries}
                        onChange={(v) => updateSettings({ max_retries: Number(v) })}
                    />
                </div>
                <div>
                    <label className="block text-xs text-[var(--color-on-surface-secondary)] mb-0.5">
                        OOM Skip Threshold
                    </label>
                    <FormEntry
                        label=""
                        type="number"
                        value={settings.oom_skip_threshold}
                        onChange={(v) => updateSettings({ oom_skip_threshold: Number(v) })}
                    />
                </div>
            </div>
        </Card>
    );
}

export default function QueuePage() {
    const loadQueue = useQueueStore((s) => s.loadQueue);
    const execute = useQueueStore((s) => s.execute);
    const stopCurrent = useQueueStore((s) => s.stopCurrent);
    const stopAll = useQueueStore((s) => s.stopAll);
    const validate = useQueueStore((s) => s.validate);
    const status = useQueueStore((s) => s.status);
    const runIndex = useQueueStore((s) => s.runIndex);
    const totalEntries = useQueueStore((s) => s.totalEntries);
    const loading = useQueueStore((s) => s.loading);

    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        loadQueue();
    }, [loadQueue]);

    const handleExport = async () => {
        const data = await queueApi.exportQueue();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "queue.json";
        a.click();
        URL.revokeObjectURL(url);
    };

    const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const text = await file.text();
        try {
            const data = JSON.parse(text);
            await queueApi.importQueue(data);
            await loadQueue();
        } catch {
            // toast or alert
        }
        e.target.value = "";
    };

    const isRunning = status === "running" || status === "stopping";

    return (
        <div className="flex flex-col h-full gap-4">
            {/* Action bar */}
            <div className="flex items-center gap-2 flex-wrap">
                {!isRunning ? (
                    <Button variant="primary" onClick={execute} disabled={loading}>
                        <Play className="w-4 h-4 mr-1" />
                        Start Queue
                    </Button>
                ) : (
                    <>
                        <Button variant="secondary" onClick={stopCurrent}>
                            <Pause className="w-4 h-4 mr-1" />
                            Skip Entry
                        </Button>
                        <Button variant="danger" onClick={stopAll}>
                            <Square className="w-4 h-4 mr-1" />
                            Stop Queue
                        </Button>
                    </>
                )}
                <Button variant="secondary" onClick={validate} disabled={isRunning}>
                    <AlertTriangle className="w-4 h-4 mr-1" />
                    Validate
                </Button>
                <div className="flex-1" />
                <Button variant="ghost" onClick={handleExport} title="Export queue">
                    <Download className="w-4 h-4" />
                </Button>
                <Button
                    variant="ghost"
                    onClick={() => fileInputRef.current?.click()}
                    title="Import queue"
                    disabled={isRunning}
                >
                    <Upload className="w-4 h-4" />
                </Button>
                <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />

                {isRunning && (
                    <span className="text-sm text-[var(--color-on-surface-secondary)]">
                        Entry {runIndex} / {totalEntries}
                    </span>
                )}
            </div>

            {/* Main content: sidebar + editor */}
            <div className="flex flex-1 gap-4 min-h-0">
                <Card className="w-72 shrink-0 p-0 flex flex-col overflow-hidden">
                    <EntryList />
                </Card>
                <div className="flex-1 flex flex-col gap-4 min-h-0">
                    <Card className="flex-1 p-0 overflow-hidden">
                        <EntryEditor />
                    </Card>
                    <QueueSettings />
                </div>
            </div>
        </div>
    );
}
