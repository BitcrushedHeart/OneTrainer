import { AlertCircle, Check, Loader2, RefreshCw, SkipForward, Undo2, Wand2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { dpoApi } from "@/api/dpoApi";
import {
  dpoReplaceApi,
  type ReplaceModelGroup,
  type ReplaceOrder,
  type ReplacePair,
  type ReplaceProposal,
  type ReplaceStatus,
} from "@/api/dpoReplaceApi";
import { Button, DirPicker, FormEntry, SectionCard, Select, Toggle } from "@/components/shared";

import { PreviewOverlay } from "./dpo/PreviewOverlay";
import { ModalBase } from "./ModalBase";

export interface ReplacePairsModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.replacePairs";
/** The other DPO tools' settings, reused as defaults. */
const OFFPOLICY_KEY = "ot.offpolicyDpo.folders";
const SWARM_KEY = "ot.swarmGenerate";

interface Settings {
  concept: string;
  generated: string;
  baseUrl: string;
  token: string;
  order: ReplaceOrder;
  allowPromote: boolean;
  count: number;
  imagesPerPair: number;
  concurrency: number;
  autoApply: boolean;
}

const DEFAULTS: Settings = {
  concept: "",
  generated: "",
  baseUrl: "http://localhost:7801",
  token: "",
  order: "stale_first",
  allowPromote: true,
  count: 10,
  imagesPerPair: 2,
  concurrency: 3,
  autoApply: false,
};

function loadSettings(): Settings {
  const settings = { ...DEFAULTS };
  try {
    const shared = window.localStorage.getItem(OFFPOLICY_KEY);
    if (shared) {
      const folders = JSON.parse(shared) as { generated?: string; output?: string };
      if (folders.generated) settings.generated = folders.generated;
      if (folders.output) settings.concept = folders.output;
    }
    const swarm = window.localStorage.getItem(SWARM_KEY);
    if (swarm) {
      const s = JSON.parse(swarm) as { baseUrl?: string; token?: string };
      if (s.baseUrl) settings.baseUrl = s.baseUrl;
      if (s.token) settings.token = s.token;
    }
    const own = window.localStorage.getItem(STORAGE_KEY);
    if (own) Object.assign(settings, JSON.parse(own) as Partial<Settings>);
  } catch {
    /* storage unavailable — defaults are fine */
  }
  return settings;
}

const ORDER_OPTIONS: Array<{ label: string; value: ReplaceOrder }> = [
  { label: "Corrected captions first, then oldest rejected", value: "stale_first" },
  { label: "Oldest rejected generation first", value: "oldest" },
  { label: "Least realistic rejected first (easiest negatives)", value: "least_realistic" },
  { label: "Pair id", value: "path" },
];

function pct(done: number, total: number) {
  return total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
}

function day(stamp: number | null | undefined) {
  if (stamp == null) return "?";
  return new Date(stamp * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function score(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(2);
}

function paramsSummary(params: Record<string, unknown>) {
  return Object.entries(params)
    .filter(([key]) => key !== "negativeprompt" && key !== "prompt")
    .map(([key, value]) => `${key} ${Array.isArray(value) ? value.join("+") : String(value)}`)
    .join(" · ");
}

export function ReplacePairsModal({ open, onClose }: ReplacePairsModalProps) {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [concepts, setConcepts] = useState<string[]>([]);
  const [active, setActive] = useState(false);
  const [status, setStatus] = useState<ReplaceStatus | null>(null);
  const [pairs, setPairs] = useState<ReplacePair[] | null>(null);
  const [models, setModels] = useState<ReplaceModelGroup[]>([]);
  const [params, setParams] = useState<{ params: Record<string, unknown>; model: string; found: boolean } | null>(null);
  const [proposal, setProposal] = useState<ReplaceProposal | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  // The reviewer's ranking of the group on screen.
  const [selChosen, setSelChosen] = useState<string | null>(null);
  const [selRejected, setSelRejected] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const fetchingRef = useRef(false);
  const pairsLoadedRef = useRef(false);
  const proposalRef = useRef<ReplaceProposal | null>(null);
  proposalRef.current = proposal;

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((previous) => {
      const next = { ...previous, ...patch };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
  }, []);

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const loadPairs = useCallback(async () => {
    try {
      const res = await dpoReplaceApi.pairs(0, 100);
      if (res.ok && !res.waiting) {
        setPairs(res.pairs);
        setModels(res.models);
      }
    } catch {
      /* transient */
    }
    try {
      const p = await dpoReplaceApi.params();
      if (p.ok) setParams({ params: p.params ?? {}, model: p.model ?? "", found: Boolean(p.found) });
    } catch {
      /* transient */
    }
  }, []);

  const fetchNext = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    try {
      const res = await dpoReplaceApi.next();
      if (!res.ok) {
        setError(res.error ?? "Failed to fetch the next proposal");
        return;
      }
      if (res.proposal) {
        setProposal(res.proposal);
        setPreview(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      fetchingRef.current = false;
    }
  }, []);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      void dpoReplaceApi
        .status()
        .then((s) => {
          setStatus(s);
          if (!s.active) {
            clearPoll();
            setActive(false);
            return;
          }
          if (s.error) setError(s.error);
          if (s.scan_finished && !pairsLoadedRef.current) {
            pairsLoadedRef.current = true;
            void loadPairs();
          }
          if (s.has_proposal && !proposalRef.current) {
            void dpoReplaceApi
              .current()
              .then((c) => {
                if (c.ok && c.proposal) setProposal(c.proposal);
              })
              .catch(() => {
                /* transient */
              });
            return;
          }
          if (!s.has_proposal && s.ready > 0) void fetchNext();
        })
        .catch(() => {
          /* transient */
        });
    }, 700);
  }, [clearPoll, fetchNext, loadPairs]);

  // Reflect a session that is already running when the modal is reopened.
  useEffect(() => {
    if (!open) {
      clearPoll();
      return;
    }
    let cancelled = false;
    void dpoReplaceApi
      .concepts()
      .then((res) => {
        if (!cancelled && res.ok) setConcepts(res.concepts);
      })
      .catch(() => {
        /* no config — fine */
      });
    void dpoReplaceApi.status().then((s) => {
      if (cancelled) return;
      setStatus(s);
      if (s.active) {
        setActive(true);
        pairsLoadedRef.current = false;
        startPolling();
      }
    });
    return () => {
      cancelled = true;
    };
  }, [open, clearPoll, startPolling]);

  useEffect(() => () => clearPoll(), [clearPoll]);

  const start = useCallback(async () => {
    setError(null);
    setNotice(null);
    setProposal(null);
    setPairs(null);
    setModels([]);
    setParams(null);
    pairsLoadedRef.current = false;
    setBusy(true);
    try {
      const res = await dpoReplaceApi.start(
        settings.concept,
        settings.generated,
        settings.baseUrl,
        settings.token,
        settings.order,
        settings.allowPromote,
      );
      if (!res.ok) {
        setError(res.error ?? "Could not start");
        return;
      }
      setActive(true);
      if ((res.pruned ?? 0) > 0) setNotice(`${res.pruned} manifest entr(y/ies) whose files were gone were pruned.`);
      startPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [settings, startPolling]);

  const finish = useCallback(async () => {
    clearPoll();
    try {
      await dpoReplaceApi.cancel();
    } catch {
      /* best effort */
    }
    setActive(false);
    setProposal(null);
    setPairs(null);
  }, [clearPoll]);

  const generate = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      const res = await dpoReplaceApi.generate(
        Math.max(1, Math.round(Number(settings.count)) || 1),
        Math.max(1, Math.min(8, Math.round(Number(settings.imagesPerPair)) || 1)),
        Math.max(1, Math.min(8, Math.round(Number(settings.concurrency)) || 1)),
        settings.autoApply,
      );
      if (!res.ok) {
        setError(res.error ?? "Could not start generating");
        return;
      }
      setNotice(`Generating for ${res.planned} pair(s) with ${res.model ?? "the last-used model"}.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [settings]);

  const act = useCallback(
    async (action: () => Promise<{ ok: boolean; error?: string }>) => {
      if (busy || !proposal) return;
      setBusy(true);
      setError(null);
      try {
        const res = await action();
        if (!res.ok) {
          setError(res.error ?? "Action failed");
          return;
        }
        setProposal(null);
        // The ranked list and its scores changed underneath.
        pairsLoadedRef.current = false;
        await fetchNext();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, proposal, fetchNext],
  );

  const apply = useCallback(() => {
    if (!proposal || !selChosen || !selRejected) return;
    if (selChosen === proposal.chosen_path && selRejected === proposal.rejected_path) {
      void act(() => dpoReplaceApi.keep());
      return;
    }
    void act(() => dpoReplaceApi.apply(selChosen, selRejected));
  }, [act, proposal, selChosen, selRejected]);
  const keep = useCallback(() => act(() => dpoReplaceApi.keep()), [act]);

  // Right-click an image to rank it: first = best (chosen), second = second-best
  // (rejected). Clicking a ranked image clears that rank. With promotion off
  // chosen side the best is fixed and a click sets the rejected.
  const rank = useCallback(
    (path: string) => {
      if (!proposal) return;
      if (proposal.chosen_locked) {
        if (path === proposal.chosen_path) return;
        setSelRejected((previous) => (previous === path ? null : path));
        return;
      }
      if (path === selChosen) {
        setSelChosen(null);
        return;
      }
      if (path === selRejected) {
        setSelRejected(null);
        return;
      }
      if (!selChosen) {
        setSelChosen(path);
      } else if (!selRejected) {
        setSelRejected(path);
      } else {
        // Both ranked: the click replaces the rejected pick.
        setSelRejected(path);
      }
    },
    [proposal, selChosen, selRejected],
  );

  const resetToSuggestion = useCallback(() => {
    if (!proposal) return;
    setSelChosen(proposal.suggested_chosen);
    setSelRejected(proposal.suggested_rejected);
  }, [proposal]);

  const undo = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await dpoReplaceApi.undo();
      if (!res.ok) {
        setNotice(res.error ?? "Nothing to undo");
        return;
      }
      if (res.proposal) {
        setProposal(res.proposal);
        setNotice(`Undid ${res.undone ?? "last action"}.`);
        pairsLoadedRef.current = false;
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [busy]);

  // A new group on screen starts from the suggestion.
  useEffect(() => {
    if (!proposal) {
      setSelChosen(null);
      setSelRejected(null);
      return;
    }
    setSelChosen(proposal.suggested_chosen);
    setSelRejected(proposal.suggested_rejected);
  }, [proposal]);

  // Keyboard: 1-9 rank an image, Enter/Y apply, S keep, 0 back to suggestion, Ctrl/Cmd+Z undo.
  useEffect(() => {
    if (!open || !active) return;
    const handler = (e: KeyboardEvent) => {
      if (preview) return;
      const target = e.target as HTMLElement | null;
      if (target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable)) return;
      if (target?.tagName === "BUTTON" && (e.key === "Enter" || e.key === " " || e.key === "Spacebar")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        void undo();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (!proposal) return;
      if (e.key === "Enter" || e.key.toLowerCase() === "y") {
        e.preventDefault();
        apply();
      } else if (e.key.toLowerCase() === "s") {
        e.preventDefault();
        void keep();
      } else if (e.key === "0") {
        e.preventDefault();
        resetToSuggestion();
      } else if (/^[1-9]$/.test(e.key)) {
        const image = proposal.images[Number(e.key) - 1];
        if (image) {
          e.preventDefault();
          rank(image.path);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, active, proposal, apply, keep, rank, resetToSuggestion, undo, preview]);

  const selectionIsKeep =
    proposal != null && selChosen === proposal.chosen_path && selRejected === proposal.rejected_path;
  const selectionComplete = Boolean(selChosen && selRejected && selChosen !== selRejected);

  const scanning = status ? !status.scan_finished : false;
  const batchPairs = Math.max(1, Math.round(Number(settings.count)) || 1);
  const batchPerPair = Math.max(1, Math.min(8, Math.round(Number(settings.imagesPerPair)) || 1));
  const batchImages = batchPairs * batchPerPair;
  const running = Boolean(status?.running);
  const batchDone =
    active && !scanning && !running && !proposal && (status?.shown ?? 0) > 0 && (status?.ready ?? 0) === 0;

  return (
    <ModalBase open={open} onClose={onClose} title="Replace Pairs" size="full">
      <div className="flex flex-col gap-4">
        {!active && (
          <>
            <div className="text-sm text-[var(--color-on-surface-secondary)]">
              Regenerates the captions of existing DPO pairs with the checkpoint and adapters you last generated with in
              SwarmUI, scores the fresh images with the realism head, and swaps them in: a fresh image that looks more
              real than the current rejected image becomes the new rejected (a harder, on-policy negative); one that
              beats a <em>generated</em> chosen side becomes the new chosen and the old chosen is demoted to rejected.
              Real-photo chosen sides never move. Pairs whose caption was corrected after export go first. Every
              displaced file is kept under <code>.replaced/</code> and the manifest records which model made each side.
            </div>
            <SectionCard title="Concept and generation">
              <div className="space-y-3">
                {concepts.length > 0 && (
                  <Select
                    label="Configured DPO concepts"
                    value={concepts.includes(settings.concept) ? settings.concept : ""}
                    onChange={(v) => update({ concept: v })}
                    options={[
                      { label: "Pick one, or browse below", value: "" },
                      ...concepts.map((c) => ({ label: c, value: c })),
                    ]}
                    formatLabel={(v) => v}
                  />
                )}
                <DirPicker
                  label="DPO concept folder (manifest.json, chosen/, rejected/)"
                  value={settings.concept}
                  onChange={(v) => update({ concept: v })}
                />
                <DirPicker
                  label="Generated images (SwarmUI raw output root)"
                  value={settings.generated}
                  onChange={(v) => update({ generated: v })}
                />
                <div className="grid grid-cols-3 gap-3">
                  <FormEntry
                    label="SwarmUI URL"
                    value={settings.baseUrl}
                    onChange={(v) => update({ baseUrl: String(v) })}
                  />
                  <FormEntry
                    label="Token (if required)"
                    value={settings.token}
                    onChange={(v) => update({ token: String(v) })}
                  />
                  <Select
                    label="Order"
                    value={settings.order}
                    onChange={(v) => update({ order: v as ReplaceOrder })}
                    options={ORDER_OPTIONS}
                    formatLabel={(v) => v}
                  />
                </div>
                <div className="flex items-center gap-6">
                  <Toggle
                    label="Allow another image to become chosen (off: only the rejected side is replaced)"
                    value={settings.allowPromote}
                    onChange={(v) => update({ allowPromote: v })}
                  />
                  <Button
                    variant="primary"
                    onClick={() => void start()}
                    disabled={busy || !settings.concept || !settings.generated}
                  >
                    {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                    Read concept
                  </Button>
                </div>
              </div>
            </SectionCard>
          </>
        )}

        {active && status && scanning && (
          <div className="flex flex-col gap-2 bg-[var(--color-surface-container)] rounded p-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" /> Reading pairs, captions and generation metadata…
              </span>
              <span className="tabular-nums text-[var(--color-on-surface-secondary)]">
                {status.progress_done.toLocaleString()} / {status.progress_total.toLocaleString()} ·{" "}
                {pct(status.progress_done, status.progress_total)}%
              </span>
            </div>
            <div className="h-2 rounded bg-[var(--color-border-subtle)] overflow-hidden">
              <div
                className="h-full bg-[var(--color-cobalt-600)] transition-all"
                style={{ width: `${pct(status.progress_done, status.progress_total)}%` }}
              />
            </div>
          </div>
        )}

        {active && status && !scanning && (
          <div className="flex flex-col gap-3">
            <div className="bg-[var(--color-surface-container)] rounded px-3 py-2 text-xs tabular-nums flex flex-wrap gap-x-4 text-[var(--color-on-surface-secondary)]">
              <span className="text-[var(--color-on-surface)] font-semibold">{status.pairs} pairs</span>
              <span>{status.stale_captions} with corrected captions</span>
              {models.map((m) => (
                <span key={m.label} title={m.label}>
                  {m.label}: {m.pairs} ({day(m.oldest)} – {day(m.newest)})
                </span>
              ))}
              <span className="ml-auto text-[var(--color-on-surface)]">
                applied {status.applied}
                {status.auto_applied > 0 && ` (${status.auto_applied} auto)`} · promoted {status.promoted} · kept{" "}
                {status.kept}
                {status.unscored > 0 && ` · ${status.unscored} unscored`}
              </span>
            </div>

            <SectionCard title="Next batch">
              <div className="text-xs text-[var(--color-on-surface-secondary)] mb-2">
                {params?.found ? (
                  <>
                    Using your last generation:{" "}
                    <span className="text-[var(--color-on-surface)] font-semibold">{params.model}</span>
                    {Object.keys(params.params).length > 0 && (
                      <>
                        {" "}
                        · <span className="text-[var(--color-on-surface)]">{paramsSummary(params.params)}</span>
                      </>
                    )}
                    . Prompt, size and seed are set per pair.
                  </>
                ) : (
                  <>
                    No previous generation with settings found in the generated folder. Generate once in SwarmUI with
                    the checkpoint and adapters you want, then come back.
                  </>
                )}
              </div>
              <div className="grid grid-cols-5 gap-3 items-end">
                <FormEntry
                  label="Pairs to refresh"
                  type="number"
                  value={settings.count}
                  onChange={(v) => update({ count: Number(v) })}
                  tooltip="Taken from the top of the ranking, skipping pairs already refreshed or awaiting review this session."
                />
                <FormEntry
                  label="Fresh images per pair"
                  type="number"
                  value={settings.imagesPerPair}
                  onChange={(v) => update({ imagesPerPair: Number(v) })}
                  tooltip="Each pair's caption is generated this many times (different seeds). All of them appear in the group to rank."
                />
                <FormEntry
                  label="Requests in flight"
                  type="number"
                  value={settings.concurrency}
                  onChange={(v) => update({ concurrency: Number(v) })}
                  tooltip="Requests kept queued at SwarmUI so the GPU never idles. Does not change how many images are made."
                />
                <Toggle
                  label="Auto-apply rejected-only suggestions"
                  value={settings.autoApply}
                  onChange={(v) => update({ autoApply: v })}
                  tooltip="A scored suggestion that leaves the chosen side alone is applied without review. Anything that would change the chosen side, and unscored groups, always wait for you."
                />
                <Button variant="primary" onClick={() => void generate()} disabled={busy || running || !params?.found}>
                  {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                  {running ? "Generating…" : `Generate ${batchImages} image${batchImages === 1 ? "" : "s"}`}
                </Button>
              </div>
              <div className="text-xs text-[var(--color-on-surface-secondary)] pt-2 tabular-nums">
                {batchPairs} pair{batchPairs === 1 ? "" : "s"} × {batchPerPair} fresh image
                {batchPerPair === 1 ? "" : "s"} each ={" "}
                <span className="text-[var(--color-on-surface)]">{batchImages}</span> images from SwarmUI, one request
                per pair at that pair&apos;s bucket size.
                {status.pairs > 0 &&
                  batchPairs > status.pairs &&
                  ` Only ${status.pairs} pairs exist, so fewer will run.`}
              </div>
              {(running || status.run_phase !== "idle") && (
                <div className="flex flex-wrap gap-x-4 text-xs tabular-nums text-[var(--color-on-surface-secondary)] pt-2">
                  <span>
                    {status.done} / {status.planned} pairs
                  </span>
                  <span>{status.images} images</span>
                  {status.failed > 0 && <span className="text-[var(--color-error-500)]">{status.failed} failed</span>}
                  <span>{status.ready} awaiting review</span>
                  {status.run_error && <span className="text-[var(--color-error-500)]">{status.run_error}</span>}
                </div>
              )}
            </SectionCard>
            {status.warning && <div className="text-xs text-[var(--color-warning-500,#eab308)]">{status.warning}</div>}
          </div>
        )}

        {notice && <div className="text-sm text-[var(--color-on-surface-secondary)]">{notice}</div>}
        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)] bg-[var(--color-surface-container)] rounded p-3">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        {active && proposal && (
          <div className="flex flex-col gap-3">
            <div className="bg-[var(--color-surface-container)] rounded p-3 flex flex-col gap-1">
              <div className="text-xs text-[var(--color-on-surface-secondary)]">
                Pair {proposal.pair_id} · group {proposal.index}
                {proposal.queued > 0 && ` · ${proposal.queued} more waiting`}
                {proposal.replacements > 0 && ` · replaced ${proposal.replacements}× before`}
                {" · "}
                {proposal.chosen_locked
                  ? "promotion is off — right-click the image that should be the rejected side"
                  : "right-click the best image (chosen), then the second-best (rejected); left-click views full size"}
                {proposal.caption_changed && " · caption was corrected after export"}
              </div>
              <div className="text-sm text-[var(--color-on-surface)] whitespace-pre-wrap">{proposal.caption}</div>
              <div className="text-xs text-[var(--color-on-surface-secondary)]">
                {proposal.scored ? `Suggestion: ${proposal.reason}` : proposal.reason}
                {" · fresh images by "}
                {proposal.new_model}
              </div>
            </div>

            <div
              className={`grid gap-3 ${proposal.images.length <= 3 ? "grid-cols-3" : proposal.images.length === 4 ? "grid-cols-4" : "grid-cols-5"}`}
            >
              {proposal.images.map((image, i) => {
                const isChosen = image.path === selChosen;
                const isRejected = image.path === selRejected;
                const locked = proposal.chosen_locked && image.role === "chosen";
                const ring = isChosen
                  ? "ring-4 ring-[var(--color-success-500)]"
                  : isRejected
                    ? "ring-4 ring-[var(--color-error-500)]"
                    : "ring-1 ring-[var(--color-border-subtle)]";
                const roleLabel =
                  image.role === "chosen" ? "current CHOSEN" : image.role === "rejected" ? "current REJECTED" : "FRESH";
                return (
                  <figure
                    key={image.path}
                    className={`flex flex-col min-h-0 bg-[var(--color-surface-container)] rounded p-2 ${ring} cursor-zoom-in`}
                    onClick={() => setPreview(image.path)}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      rank(image.path);
                    }}
                    title={
                      locked
                        ? "Real photo — stays chosen. Left-click to view full size"
                        : "Left-click: view full size · Right-click: rank (1st = chosen, 2nd = rejected)"
                    }
                  >
                    <figcaption className="text-xs font-semibold mb-1 flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-[var(--color-border-subtle)] text-[10px]">
                        {i + 1}
                      </span>
                      <span
                        className={
                          isChosen
                            ? "text-[var(--color-success-500)]"
                            : isRejected
                              ? "text-[var(--color-error-500)]"
                              : "text-[var(--color-on-surface-secondary)]"
                        }
                      >
                        {isChosen ? "→ CHOSEN" : isRejected ? "→ REJECTED" : "—"}
                      </span>
                      <span className="text-[var(--color-on-surface-secondary)] font-normal truncate">{roleLabel}</span>
                    </figcaption>
                    <img
                      src={dpoApi.imageUrl(image.path)}
                      alt={roleLabel}
                      className="h-[40vh] min-h-[220px] object-contain w-full cursor-zoom-in"
                    />
                    <div className="text-[10px] text-[var(--color-on-surface-secondary)] tabular-nums flex flex-wrap gap-x-3">
                      <span>P(AI) {score(image.score)}</span>
                      {image.size && (
                        <span>
                          {image.size[0]}×{image.size[1]}
                        </span>
                      )}
                      {image.role === "rejected" && <span>{proposal.rejected_model}</span>}
                      {image.role === "chosen" && <span>{proposal.chosen_provenance.model || "unknown model"}</span>}
                      <span className="ml-auto">left-click: full size · right-click: rank</span>
                    </div>
                  </figure>
                );
              })}
            </div>

            {!proposal.scored && (
              <div className="rounded px-2 py-1 text-xs font-bold tracking-wide text-center bg-[var(--color-warning-500,#eab308)] text-black">
                UNSCORED — no realism scores for this group; rank by eye
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2">
              <Button variant="primary" onClick={apply} disabled={busy || !selectionComplete}>
                <Check className="w-4 h-4" />
                {selectionIsKeep ? "Keep pair as is (Enter)" : "Apply ranking (Enter)"}
              </Button>
              <Button variant="secondary" onClick={() => void keep()} disabled={busy}>
                <SkipForward className="w-4 h-4" /> Keep pair as is (S)
              </Button>
              <Button variant="secondary" onClick={resetToSuggestion} disabled={busy}>
                <RefreshCw className="w-4 h-4" /> Back to suggestion (0)
              </Button>
              <Button variant="ghost" onClick={() => void undo()} disabled={busy || !status?.can_undo}>
                <Undo2 className="w-4 h-4" />
                {status?.undo_action ? `Undo ${status.undo_action} (Ctrl+Z)` : "Undo (Ctrl+Z)"}
              </Button>
              {!selectionComplete && (
                <span className="text-xs text-[var(--color-on-surface-secondary)]">
                  {proposal.chosen_locked ? "Pick the rejected image." : "Pick a chosen and a rejected image."}
                </span>
              )}
            </div>
          </div>
        )}

        {active && !proposal && running && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-on-surface-secondary)]">
            <Loader2 className="w-4 h-4 animate-spin" /> Waiting for the next proposal…
          </div>
        )}

        {batchDone && (
          <div className="text-sm text-[var(--color-on-surface)] bg-[var(--color-surface-container)] rounded p-3 flex items-center gap-3">
            <span>Batch reviewed. Generate another batch to walk further down the ranking.</span>
            {status?.can_undo && (
              <Button variant="ghost" onClick={() => void undo()} disabled={busy}>
                <Undo2 className="w-4 h-4" /> Undo (Ctrl+Z)
              </Button>
            )}
          </div>
        )}

        {active && pairs && (
          <details className="rounded border border-[var(--color-border-subtle)] px-3 py-2">
            <summary className="cursor-pointer select-none text-sm font-semibold text-[var(--color-on-surface)]">
              Ranking (top {pairs.length} of {status?.pairs ?? pairs.length})
            </summary>
            <div className="overflow-x-auto pt-2">
              <table className="w-full text-xs tabular-nums">
                <thead className="text-[var(--color-on-surface-secondary)] text-left">
                  <tr>
                    <th className="pr-3">#</th>
                    <th className="pr-3">rejected by</th>
                    <th className="pr-3">made</th>
                    <th className="pr-3">P(AI) chosen / rejected</th>
                    <th className="pr-3">caption</th>
                    <th>note</th>
                  </tr>
                </thead>
                <tbody>
                  {pairs.map((p) => (
                    <tr key={p.pair_id} className="border-t border-[var(--color-border-subtle)]">
                      <td className="pr-3 py-1">{p.pair_id}</td>
                      <td className="pr-3 py-1 truncate max-w-[220px]" title={p.rejected_model}>
                        {p.rejected_model}
                      </td>
                      <td className="pr-3 py-1">{day(p.rejected_mtime)}</td>
                      <td className="pr-3 py-1">
                        {score(p.chosen_score)} / {score(p.rejected_score)}
                      </td>
                      <td className="pr-3 py-1 truncate max-w-[420px]" title={p.caption}>
                        {p.caption}
                      </td>
                      <td className="py-1 text-[var(--color-on-surface-secondary)]">
                        {p.stale_caption && (
                          <span className="text-[var(--color-warning-500,#eab308)]">corrected caption · </span>
                        )}
                        {p.replacements > 0 && ` · replaced ${p.replacements}×`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}

        {active && (
          <div className="flex justify-end">
            <Button variant="secondary" onClick={() => void finish()}>
              Finish session
            </Button>
          </div>
        )}
      </div>
      {preview && <PreviewOverlay path={preview} onClose={() => setPreview(null)} />}
    </ModalBase>
  );
}
