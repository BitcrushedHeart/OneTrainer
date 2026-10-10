import { CheckCircle2, Loader2, Pause, Play, Plug, SkipForward, Undo2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  dpoReplaceApi,
  MAX_IMAGES_PER_PAIR,
  type ReplaceGroup,
  type ReplacePair,
  type ReplaceResult,
  type ReplaceSettings,
  type ReplaceStatus,
} from "@/api/dpoReplaceApi";
import { API_BASE } from "@/api/request";
import { swarmApi } from "@/api/swarmApi";
import { Button, DirPicker, FormEntry, SectionCard, Select } from "@/components/shared";

import { PreviewOverlay } from "./dpo/PreviewOverlay";
import { PromptExpander } from "./dpo/PromptExpander";
import { ModalBase } from "./ModalBase";

export interface ReplacePairsModalProps {
  open: boolean;
  onClose: () => void;
}

const STORAGE_KEY = "ot.replacePairs.v2";

/** Selection keys in grid order: 1–9, 0, then - and = (ten fresh images plus the old pair). */
const PICK_KEYS = "1234567890-=";
const PICK_CODES = [
  "Digit1",
  "Digit2",
  "Digit3",
  "Digit4",
  "Digit5",
  "Digit6",
  "Digit7",
  "Digit8",
  "Digit9",
  "Digit0",
  "Minus",
  "Equal",
];

/** Grid position a key selects, by physical key so Shift+1 still means image 1. */
function pickIndex(event: KeyboardEvent): number {
  const byCode = PICK_CODES.indexOf(event.code);
  return byCode >= 0 ? byCode : PICK_KEYS.indexOf(event.key);
}

function validImagesPerPair(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= MAX_IMAGES_PER_PAIR;
}

function loadSettings(): ReplaceSettings {
  const fallback = {
    concept_dir: "",
    generated_folder: "",
    base_url: "http://localhost:7801",
    token: "",
    images_per_pair: 2,
  };
  try {
    const folders = JSON.parse(localStorage.getItem("ot.offpolicyDpo.folders") ?? "{}");
    const swarm = JSON.parse(localStorage.getItem("ot.swarmGenerate") ?? "{}");
    const old = JSON.parse(localStorage.getItem("ot.replacePairs") ?? "{}");
    const own = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
    return {
      concept_dir: own.concept_dir ?? old.concept ?? folders.output ?? "",
      generated_folder: own.generated_folder ?? old.generated ?? swarm.generated ?? folders.generated ?? "",
      base_url: own.base_url ?? old.baseUrl ?? swarm.baseUrl ?? fallback.base_url,
      token: own.token ?? old.token ?? swarm.token ?? "",
      images_per_pair: validImagesPerPair(own.images_per_pair) ? own.images_per_pair : 2,
    };
  } catch {
    return fallback;
  }
}

export function ReplacePairsModal({ open, onClose }: ReplacePairsModalProps) {
  const [settings, setSettings] = useState<ReplaceSettings>(loadSettings);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const [concepts, setConcepts] = useState<string[]>([]);
  const [status, setStatus] = useState<ReplaceStatus>({ ok: true, active: false });
  const [group, setGroup] = useState<ReplaceGroup | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [preview, setPreview] = useState<string | null>(null);
  // Pairs waiting to be saved together (Shift+pick). The group's BEST image
  // (``group.best_id``) is the chosen side the next pick pairs with.
  const [pending, setPending] = useState<ReplacePair[]>([]);
  const [reversal, setReversal] = useState<ReplacePair[] | null>(null);
  const busyRef = useRef(false);
  const epoch = useRef(0);
  const polling = useRef(false);

  const update = (patch: Partial<ReplaceSettings>) => {
    setSettings((previous) => {
      const next = { ...previous, ...patch };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* optional persistence */
      }
      return next;
    });
  };

  const refresh = useCallback(async (force = false) => {
    if (!force && (busyRef.current || polling.current)) return;
    const version = epoch.current;
    polling.current = true;
    try {
      const s = await dpoReplaceApi.status();
      const result = s.active ? await dpoReplaceApi.current() : { ok: true, group: null };
      if (version !== epoch.current) return;
      setStatus(s);
      setGroup(result.group ?? null);
      if (s.error) setError(s.error);
      if (!result.ok) setError(result.error ?? "Could not load review group");
    } finally {
      polling.current = false;
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    let disposed = false;
    setLoading(true);
    const initialize = async () => {
      try {
        const available = await dpoReplaceApi.concepts();
        if (disposed) return;
        setConcepts(available.concepts);
        if (available.warning) setError(available.warning);
        const live = await dpoReplaceApi.status();
        if (disposed) return;
        if (!live.active && settingsRef.current.concept_dir) {
          const restored = await dpoReplaceApi.restore(settingsRef.current);
          if (!restored.ok && !disposed) setError(restored.error ?? "Could not restore session");
        }
        if (!disposed) await refresh();
      } catch (e) {
        if (!disposed) setError(String(e));
      } finally {
        if (!disposed) setLoading(false);
      }
    };
    void initialize();
    const timer = window.setInterval(() => {
      if (!disposed) void refresh().catch((e) => setError(String(e)));
    }, 800);
    return () => {
      disposed = true;
      epoch.current += 1;
      window.clearInterval(timer);
    };
  }, [open, refresh]);

  useEffect(() => {
    if (!open || status.active || !settings.generated_folder) return;
    let disposed = false;
    void swarmApi
      .lastParams(settings.generated_folder)
      .then((result) => {
        if (!disposed) setParams(result.params ?? {});
      })
      .catch(() => {
        if (!disposed) setParams({});
      });
    return () => {
      disposed = true;
    };
  }, [open, status.active, settings.generated_folder]);

  useEffect(() => {
    setPreview(null);
    setReversal(null);
    setPending([]);
  }, [group?.group_id]);

  const act = useCallback(
    async (operation: () => Promise<ReplaceResult | ReplaceStatus>) => {
      if (busyRef.current) return;
      busyRef.current = true;
      epoch.current += 1;
      setBusy(true);
      setError(null);
      try {
        const result = await operation();
        if (!result.ok) setError(result.error ?? "Action failed");
        await refresh(true);
      } catch (e) {
        setError(String(e));
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [refresh],
  );

  const save = useCallback(
    (pairs: ReplacePair[]) => {
      if (!group || pairs.length === 0) return;
      void act(async () => {
        const result = await dpoReplaceApi.applyPairs(group, pairs);
        if (result.requires_reversal_confirmation) {
          setReversal(pairs);
          return { ok: true };
        }
        return result;
      });
    },
    [group, act],
  );

  /** Right-click or number key. With no BEST image yet it picks one; with a
   *  BEST it pairs the image as worse and saves, or with ``add`` (Shift)
   *  keeps the pair and waits for more. Picking the BEST image again frees it
   *  so another BEST can be chosen; picking an earlier BEST resumes it. */
  const pick = useCallback(
    (id: string, add = false) => {
      if (!group || reversal) return;
      setPreview(null);
      const best = group.best_id;
      const isChosen = pending.some((pair) => pair.chosen_id === id);
      const isRejected = pending.some((pair) => pair.rejected_id === id);
      const label = `Image ${group.images.findIndex((image) => image.id === id) + 1}`;
      if (!best || id === best || isChosen) {
        if (best && id === best) {
          void act(() => dpoReplaceApi.best(group, null));
          return;
        }
        if (isRejected) {
          setError(`${label} is already a worse image in a pair; an image cannot be on both sides`);
          return;
        }
        void act(() => dpoReplaceApi.best(group, id));
        return;
      }
      const exists = pending.some((pair) => pair.chosen_id === best && pair.rejected_id === id);
      if (add) {
        setError(null);
        setPending((current) =>
          exists
            ? current.filter((pair) => !(pair.chosen_id === best && pair.rejected_id === id))
            : [...current, { chosen_id: best, rejected_id: id }],
        );
        return;
      }
      save(exists ? pending : [...pending, { chosen_id: best, rejected_id: id }]);
    },
    [group, reversal, pending, act, save],
  );

  useEffect(() => {
    if (!open || !status.active) return;
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target?.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName)) return;
      if (reversal || preview || busyRef.current) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && status.can_undo) {
        event.preventDefault();
        void act(() => dpoReplaceApi.undo(status.session_id ?? ""));
      } else if (event.key === "Enter" && pending.length > 0) {
        event.preventDefault();
        save(pending);
      } else if (!event.ctrlKey && !event.metaKey && !event.altKey && pickIndex(event) >= 0) {
        const candidate = group?.images[pickIndex(event)];
        if (candidate) {
          event.preventDefault();
          pick(candidate.id, event.shiftKey);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, status, group, reversal, preview, pending, act, pick, save]);

  const counts = status.counts;
  const previewImage = group?.images.find((image) => image.id === preview);
  const aspect = group ? group.size[0] / group.size[1] : 1;
  const baseColumns = aspect >= 1.7 ? 2 : aspect >= 1.15 ? 3 : aspect >= 0.85 ? 4 : 5;
  // Up to twelve images: one more column past eight, and shorter rows past two.
  const columns = group ? Math.min(group.images.length, baseColumns + (group.images.length > 8 ? 1 : 0)) : 3;
  const rows = group ? Math.ceil(group.images.length / columns) : 1;
  const imageHeight = rows <= 2 ? "48vh" : `${Math.max(22, Math.floor(84 / rows))}vh`;
  const number = (id: string) => (group ? group.images.findIndex((image) => image.id === id) + 1 : 0);
  const chosenIds = new Set(pending.map((pair) => pair.chosen_id));
  const rejectedOf = (id: string) => pending.filter((pair) => pair.rejected_id === id).map((pair) => pair.chosen_id);

  return (
    <ModalBase
      open={open}
      onClose={onClose}
      title="Replace Pairs"
      size={status.active ? "full" : "xl"}
      closeOnEscape={!preview && !reversal}
    >
      <div className="flex flex-col gap-4">
        {loading && (
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading session…
          </div>
        )}
        {error && (
          <div role="alert" className="rounded border border-red-500/40 bg-red-500/10 p-3 text-sm">
            {error}
          </div>
        )}
        {!status.active && !loading && (
          <>
            <p className="text-sm text-[var(--color-on-surface-secondary)]">
              Generate alternatives from each chosen image’s on-disk caption, then blindly select the best and worst
              images for the pair.
            </p>
            <SectionCard title="SwarmUI">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
                <FormEntry
                  label="Server URL"
                  value={settings.base_url}
                  onChange={(value) => update({ base_url: String(value) })}
                />
                <FormEntry
                  label="Auth token (optional)"
                  value={settings.token}
                  onChange={(value) => update({ token: String(value) })}
                />
                <div className="md:col-span-2 flex items-center gap-2">
                  <Button
                    variant="secondary"
                    disabled={busy}
                    onClick={() =>
                      void act(async () => {
                        setConnection(null);
                        const result = await swarmApi.check(settings.base_url, settings.token);
                        if (result.ok)
                          setConnection(`SwarmUI ${result.version ?? ""} · ${result.user_id ?? "Connected"}`);
                        return result;
                      })
                    }
                  >
                    <Plug className="h-4 w-4" /> Check connection
                  </Button>
                  {connection && (
                    <span className="text-xs text-green-500 flex gap-1 items-center">
                      <CheckCircle2 className="h-4 w-4" />
                      {connection}
                    </span>
                  )}
                </div>
              </div>
              <div className="pt-3 space-y-2">
                <DirPicker
                  label="Generated images (SwarmUI output)"
                  value={settings.generated_folder}
                  onChange={(value) => update({ generated_folder: value })}
                />
                <p className="text-xs text-[var(--color-on-surface-secondary)]">
                  Reuse my last generation · refreshed at start and resume. Prompt, size and seed are set for each
                  image.
                </p>
                <p className="text-xs break-words">
                  {Object.keys(params).length
                    ? Object.entries(params)
                        .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(", ") : String(value)}`)
                        .join(" · ")
                    : "No previous generation found in this folder. Generate once in SwarmUI first."}
                </p>
              </div>
            </SectionCard>
            <SectionCard title="DPO concept">
              <div className="grid grid-cols-1 md:grid-cols-[1fr_180px] gap-3">
                <Select
                  label="Select DPO Concept"
                  value={settings.concept_dir}
                  onChange={(value) => update({ concept_dir: value })}
                  options={[
                    { label: "Select a concept…", value: "" },
                    ...concepts.map((value) => ({ label: value, value })),
                  ]}
                />
                <Select
                  label="Images per pair"
                  value={String(settings.images_per_pair)}
                  onChange={(value) => update({ images_per_pair: Number(value) })}
                  options={Array.from({ length: MAX_IMAGES_PER_PAIR }, (_, i) => String(i + 1))}
                />
              </div>
              {!concepts.length && (
                <p className="text-sm pt-2">
                  No exported DPO concepts found. Configure a concept with chosen/rejected folders and a manifest first.
                </p>
              )}
            </SectionCard>
            <div className="flex justify-end">
              <Button
                variant="primary"
                disabled={busy || !concepts.includes(settings.concept_dir) || !settings.generated_folder}
                onClick={() => void act(() => dpoReplaceApi.start(settings))}
              >
                <Play className="h-4 w-4" /> Start / resume pass
              </Button>
            </div>
          </>
        )}
        {status.active && counts && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <SectionCard title="Queue 1 · Generation">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm space-y-1">
                    <p>
                      {status.running
                        ? status.paused
                          ? "Pausing after current image…"
                          : "Generating…"
                        : counts.queued
                          ? "Paused"
                          : "Generation finished"}
                    </p>
                    <p className="text-xs text-[var(--color-on-surface-secondary)]">
                      {status.images} alternatives · {counts.queued + counts.generating} pairs waiting · {counts.failed}{" "}
                      need attention
                    </p>
                  </div>
                  <Button
                    variant="secondary"
                    disabled={
                      busy ||
                      status.closing ||
                      (status.running && status.paused) ||
                      (!status.running && counts.queued === 0)
                    }
                    onClick={() =>
                      void act(() =>
                        status.running
                          ? dpoReplaceApi.pause(status.session_id ?? "")
                          : dpoReplaceApi.resume(status.session_id ?? "", settings.token),
                      )
                    }
                  >
                    {status.running ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {status.running ? "Pause" : "Resume"}
                  </Button>
                </div>
              </SectionCard>
              <SectionCard title="Queue 2 · Selection">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm space-y-1">
                    <p>{counts.ready} pairs ready to review</p>
                    <p className="text-xs text-[var(--color-on-surface-secondary)]">
                      {counts.applied} saved · {counts.skipped} skipped · {status.total} total
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    disabled={busy || !status.can_undo}
                    onClick={() => void act(() => dpoReplaceApi.undo(status.session_id ?? ""))}
                  >
                    <Undo2 className="h-4 w-4" /> Undo
                  </Button>
                </div>
              </SectionCard>
            </div>
            {group && (
              <>
                <PromptExpander prompt={group.caption} />
                <div
                  className={`rounded py-2 px-3 text-center font-bold text-sm ${group.best_id ? "bg-red-500/10 text-red-400" : "bg-green-500/10 text-green-400"}`}
                >
                  {group.best_id
                    ? "Select the WORST image"
                    : pending.length > 0
                      ? `Select another BEST image, or press Enter to save ${pending.length} pair${pending.length === 1 ? "" : "s"}`
                      : "Select the BEST image"}
                  <span className="block text-xs font-normal pt-1">
                    Click to preview · Right-click or press {PICK_KEYS[0]}–{PICK_KEYS[group.images.length - 1]} to
                    select
                    {group.best_id &&
                      " · Shift+select adds a worse image without saving, so one pick can penalise several (the old chosen too) · select the BEST again to pick another BEST"}
                  </span>
                  {group.best_id && group.images.some((image) => image.current_rejected) && (
                    <span className="block text-xs font-normal pt-1">
                      Picking the image marked CURRENT REJECTED keeps the old negative; pick a fresh image to refresh
                      it.
                    </span>
                  )}
                </div>
                <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
                  {group.images.map((image, index) => (
                    <button
                      key={image.id}
                      type="button"
                      aria-label={`Preview Image ${index + 1}`}
                      disabled={busy || !!reversal}
                      onClick={() => setPreview(image.id)}
                      onContextMenu={(event) => {
                        event.preventDefault();
                        pick(image.id, event.shiftKey);
                      }}
                      className={`relative rounded border-2 overflow-hidden bg-black/30 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-[var(--color-cobalt-600)] ${group.best_id === image.id ? "border-green-500 ring-2 ring-green-500" : chosenIds.has(image.id) ? "border-green-500/60" : rejectedOf(image.id).length > 0 || image.current_rejected ? "border-red-500/60" : "border-[var(--color-border-subtle)]"}`}
                    >
                      <img
                        src={`${API_BASE}${image.url}`}
                        alt={`Image ${index + 1}`}
                        draggable={false}
                        className="w-full object-contain"
                        style={{ aspectRatio: `${group.size[0]} / ${group.size[1]}`, maxHeight: imageHeight }}
                      />
                      {image.current_rejected && (
                        <span className="absolute top-2 left-2 rounded bg-red-600/90 px-2 py-0.5 text-xs font-bold text-white">
                          CURRENT REJECTED
                        </span>
                      )}
                      {rejectedOf(image.id).length > 0 && (
                        <span className="absolute top-2 right-2 rounded bg-red-700/90 px-2 py-0.5 text-xs font-bold text-white">
                          WORSE than {rejectedOf(image.id).map(number).join(", ")}
                        </span>
                      )}
                      <span className="block p-2 text-xs">
                        Image {index + 1}
                        {image.current_rejected ? " · current rejected" : ""}
                        {group.best_id === image.id || chosenIds.has(image.id) ? " · BEST" : ""}
                      </span>
                    </button>
                  ))}
                </div>
                {pending.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-[var(--color-on-surface-secondary)]">Pairs to save:</span>
                    {pending.map((pair, index) => (
                      <span
                        key={`${pair.chosen_id}-${pair.rejected_id}`}
                        className="flex items-center gap-1 rounded border border-[var(--color-border-subtle)] px-2 py-0.5"
                        title={index === 0 ? "Replaces the pair under review" : "Added to the dataset as a new pair"}
                      >
                        Image {number(pair.chosen_id)} over Image {number(pair.rejected_id)}
                        {index === 0 ? " (this pair)" : " (new)"}
                        <button
                          type="button"
                          aria-label={`Remove pair ${index + 1}`}
                          className="opacity-70 hover:opacity-100"
                          disabled={busy}
                          onClick={() => setPending((current) => current.filter((_, i) => i !== index))}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                <div className="flex gap-2">
                  {pending.length > 0 && (
                    <Button variant="primary" disabled={busy || !!reversal} onClick={() => save(pending)}>
                      Save {pending.length} pair{pending.length === 1 ? "" : "s"} (Enter)
                    </Button>
                  )}
                  {(group.best_id || pending.length > 0) && (
                    <Button
                      variant="secondary"
                      disabled={busy || !!reversal}
                      onClick={() => {
                        setPending([]);
                        if (group.best_id) void act(() => dpoReplaceApi.best(group, null));
                      }}
                    >
                      {pending.length > 0 ? "Clear" : "Change best"}
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    disabled={busy || !!reversal}
                    onClick={() => void act(() => dpoReplaceApi.skip(group.session_id, group.group_id))}
                  >
                    <SkipForward className="h-4 w-4" /> Skip pair
                  </Button>
                </div>
              </>
            )}
            {!group && !status.complete && (
              <div className="p-8 text-center text-sm text-[var(--color-on-surface-secondary)]">
                {status.running
                  ? "Waiting for the next group of images…"
                  : "Resume generation to prepare more pairs for review."}
              </div>
            )}
            {!!status.errors?.length && (
              <details className="rounded border border-[var(--color-border-subtle)] p-3 text-sm">
                <summary>{status.errors.length} pairs need attention</summary>
                <div className="space-y-2 pt-2">
                  {status.errors.map((item) => (
                    <div key={item.group_id} className="flex items-center gap-2">
                      <span className="flex-1">
                        Pair {item.pair_id}: {item.error}
                      </span>
                      <Button
                        variant="secondary"
                        disabled={busy}
                        onClick={() => void act(() => dpoReplaceApi.retry(status.session_id ?? "", item.group_id))}
                      >
                        Retry
                      </Button>
                      <Button
                        variant="ghost"
                        disabled={busy}
                        onClick={() => void act(() => dpoReplaceApi.skip(status.session_id ?? "", item.group_id))}
                      >
                        Skip
                      </Button>
                    </div>
                  ))}
                </div>
              </details>
            )}
            {status.complete && (
              <SectionCard title="Pass complete">
                <div className="flex items-center justify-between">
                  <p className="text-sm">Every pair has been reviewed or skipped.</p>
                  <Button
                    variant="primary"
                    disabled={busy}
                    onClick={() =>
                      void act(async () => {
                        const closed = await dpoReplaceApi.close(status.session_id ?? "");
                        if (!closed.ok) return closed;
                        return dpoReplaceApi.start(
                          {
                            ...settings,
                            concept_dir: status.concept_dir ?? settings.concept_dir,
                            generated_folder: status.generated_folder ?? settings.generated_folder,
                            base_url: status.base_url ?? settings.base_url,
                          },
                          true,
                        );
                      })
                    }
                  >
                    Start new pass
                  </Button>
                </div>
              </SectionCard>
            )}
            <div className="flex items-center justify-between text-xs text-[var(--color-on-surface-secondary)]">
              <span>Closing this window keeps generation running. Queues are saved automatically.</span>
              <Button
                variant="secondary"
                disabled={busy || status.closing}
                onClick={() => void act(() => dpoReplaceApi.close(status.session_id ?? ""))}
              >
                {status.closing ? "Saving after current image…" : "Save and close session"}
              </Button>
            </div>
          </>
        )}
      </div>
      {previewImage && group && (
        <PreviewOverlay
          path={previewImage.id}
          imageUrl={`${API_BASE}${previewImage.url}`}
          caption={`Image ${group.images.indexOf(previewImage) + 1}${previewImage.current_rejected ? " · current rejected" : ""}`}
          onClose={() => setPreview(null)}
          onPick={() => pick(previewImage.id)}
        />
      )}
      {reversal && group && (
        <ModalBase
          open
          onClose={() => setReversal(null)}
          title="Reverse the previous pair?"
          size="sm"
          closeOnBackdrop={false}
        >
          <p className="text-sm pb-4">
            {reversal.length > 1 ? "One of these pairs puts" : "You selected"} the previous rejected image as BEST and
            the previous chosen image as WORST. This may be valid. Replace the pair anyway?
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" disabled={busy} onClick={() => setReversal(null)}>
              Back
            </Button>
            <Button
              variant="primary"
              disabled={busy}
              onClick={() =>
                void act(async () => {
                  const result = await dpoReplaceApi.applyPairs(group, reversal, true);
                  if (result.ok) setReversal(null);
                  return result;
                })
              }
            >
              Replace anyway
            </Button>
          </div>
        </ModalBase>
      )}
    </ModalBase>
  );
}
