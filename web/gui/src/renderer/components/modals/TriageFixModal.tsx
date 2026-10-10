import {
  AlertCircle,
  Brush,
  Check,
  ChevronDown,
  Hand,
  Loader2,
  Pause,
  Plug,
  Plus,
  RefreshCw,
  Send,
  SkipForward,
  Trash2,
  Undo2,
  Wand2,
  X,
} from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { dpoApi } from "@/api/dpoApi";
import {
  MAIN_TRIAGE_SESSION,
  type ReadyItem,
  type TriageCategory,
  triageFixApi,
  type TriageFixSpec,
  type TriageFixStatus,
  type TriageItem,
  triageSessionsApi,
} from "@/api/dpoTriageFixApi";
import { swarmApi } from "@/api/swarmApi";
import { Button, DirPicker, FormEntry, SectionCard, Select, Toggle } from "@/components/shared";
import { INPUT_FLEX, INPUT_FULL, PLACEHOLDER, SELECT_BASE } from "@/utils/inputStyles";

import { PreviewOverlay } from "./dpo/PreviewOverlay";
import { PromptExpander } from "./dpo/PromptExpander";
import { QuickMaskCanvas, type QuickMaskCanvasHandle } from "./dpo/QuickMaskCanvas";
import { SecondRoundGrid } from "./dpo/SecondRoundGrid";
import {
  captionFilterActive,
  categoriesFromDetected,
  CATEGORY_MASK,
  CATEGORY_OTHER,
  CATEGORY_PHOTO,
  cleanColorCorrection,
  cleanCropReference,
  COLOR_CORRECTIONS,
  type ColorCorrection,
  cursorForStem,
  emptyTags,
  filterByCaption,
  filterByCaptionOrStem,
  KEYWORD_FILTER_KEY,
  loadTriageFixSettings,
  loadTriageSessions,
  newOtherFix,
  type OtherFix,
  parseCaptionFilter,
  parseSegmentString,
  pushRecent,
  saveTriageFixSettings,
  saveTriageSessions,
  SECOND_KINDS,
  type SecondKind,
  secondKind,
  secondRoundHandled,
  secondRoundOrder,
  segmentParamsFromForm,
  sessionIdFor,
  setOthers,
  type TagSet,
  tagsFromFixes,
  tagsToFixes,
  toggleTag,
  type TriageFixSettings,
  triageOrder,
  type TriageSessionTab,
  YOLO_THRESHOLD,
} from "./dpo/triageFix";
import { TriageGrid } from "./dpo/TriageGrid";
import { ModalBase } from "./ModalBase";

export interface TriageFixModalProps {
  open: boolean;
  onClose: () => void;
}

type Screen = "setup" | "triage" | "triage2" | "compare";
type TriageView = "walk" | "grid";
const FIRST_VIEW_KEY = "ot.triageFix.firstRoundView";
const SECOND_VIEW_KEY = "ot.triageFix.secondRoundView";

function readView(key: string): TriageView {
  try {
    return window.localStorage.getItem(key) === "grid" ? "grid" : "walk";
  } catch {
    return "walk";
  }
}

function storeView(key: string, view: TriageView): void {
  try {
    window.localStorage.setItem(key, view);
  } catch {
    // Private mode or a full store: the view still applies for this session.
  }
}

/** SwarmUI's "Segment Refining" group, by API id. Blank = SwarmUI's default. */
const SEGMENT_FIELDS: Array<{
  key: string;
  label: string;
  kind: "number" | "text" | "select" | "bool";
  hint: string;
  options?: string[];
}> = [
  { key: "segmentmaskblur", label: "Mask blur", kind: "number", hint: "0–64. SwarmUI default 10." },
  { key: "segmentmaskgrow", label: "Mask grow (px)", kind: "number", hint: "0–512. SwarmUI default 16." },
  {
    key: "segmentmaskoversize",
    label: "Mask oversize (px)",
    kind: "number",
    hint: "0–512, default 16. Larger includes more context; smaller gets closer for finer detail.",
  },
  { key: "segmentthresholdmax", label: "Threshold max", kind: "number", hint: "0–1, default 1." },
  { key: "segmentsteps", label: "Segment steps", kind: "number", hint: "Blank = the request's step count." },
  { key: "segmentcfgscale", label: "Segment CFG", kind: "number", hint: "Blank = the request's CFG." },
  {
    key: "segmenttargetresolution",
    label: "Target resolution",
    kind: "text",
    hint: "e.g. 1024x1024. The size each region is regenerated at.",
  },
  {
    key: "segmentsortorder",
    label: "Sort order",
    kind: "select",
    hint: "Affects yolo-…-N indexed targets only. largest-smallest makes -1 the subject.",
    options: ["", "largest-smallest", "smallest-largest", "left-right", "right-left", "top-bottom", "bottom-top"],
  },
  {
    key: "segmentapplyafter",
    label: "Apply after",
    kind: "select",
    hint: "Refiner (default) or Base.",
    options: ["", "Refiner", "Base"],
  },
  { key: "savesegmentmask", label: "Save segment mask", kind: "bool", hint: "Also write the mask SwarmUI used." },
];

/** What a second-round confirm makes of an image locally, until the next items load. */
function takenUp(item: TriageItem, second: boolean): Partial<TriageItem> {
  return second ? { second_pass: "open", second_from: secondKind(item) } : {};
}

function basename(path: string): string {
  return path.split(/[/\\]/).pop() ?? path;
}

function numberOrNull(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

/** Images a bulk confirm left alone because another session has fixes queued for the same area. */
function conflictMessage(conflicts: Record<string, string>): string {
  const stems = Object.keys(conflicts);
  const first = conflicts[stems[0]];
  if (stems.length === 1) return first;
  return `Cannot proceed on ${stems.length} images (${stems.slice(0, 5).join(", ")}${stems.length > 5 ? ", …" : ""}): another session already has fixes queued for the same area. First: ${first}`;
}

function keyboardTargetIsInput(e: KeyboardEvent): boolean {
  const target = e.target as HTMLElement | null;
  if (!target) return false;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable) return true;
  return target.tagName === "BUTTON" && (e.key === "Enter" || e.key === " " || e.key === "Spacebar");
}

// --------------------------------------------------------------------------- //
// Small portal dialogs
// --------------------------------------------------------------------------- //

function Dialog({
  title,
  onClose,
  children,
  wide,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  /** Room for the Fix Other rows. */
  wide?: boolean;
}) {
  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", handle, true);
    return () => window.removeEventListener("keydown", handle, true);
  }, [onClose]);
  return createPortal(
    <div
      className="fixed inset-0 z-[56] flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.6)" }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={`bg-[var(--color-surface)] rounded-lg shadow-xl ${
          wide ? "w-[min(760px,calc(100vw-32px))]" : "w-[min(560px,calc(100vw-32px))]"
        } p-4 flex flex-col gap-3 border border-[var(--color-border-subtle)]`}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[var(--color-on-surface)]">{title}</h3>
          <button type="button" onClick={onClose} className="text-[var(--color-on-surface-secondary)]">
            <X className="w-4 h-4" />
          </button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  );
}

/** SwarmUI's detection models, for the Fix Other target dropdown (empty until SwarmUI answers). */
const YoloModelsContext = createContext<string[]>([]);

/** The Fix Other rows: one segment target each, with its own strengths. */
function OthersEditor({
  rows,
  categories,
  autoFocus,
  onChange,
  onEnter,
}: {
  rows: OtherFix[];
  categories: TriageCategory[];
  autoFocus?: boolean;
  onChange: (rows: OtherFix[]) => void;
  onEnter?: () => void;
}) {
  const yoloModels = useContext(YoloModelsContext);
  const lastTargetRef = useRef<HTMLInputElement>(null);
  const focusLast = useCallback(() => {
    window.setTimeout(() => lastTargetRef.current?.focus(), 60);
  }, []);
  useEffect(() => {
    if (autoFocus) focusLast();
  }, [autoFocus, focusLast]);
  const update = (index: number, patch: Partial<OtherFix>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  const add = () => {
    onChange([...rows, newOtherFix(categories)]);
    focusLast();
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    // Ctrl+Enter adds a row; plain Enter is the dialog's confirm.
    if (e.ctrlKey || e.metaKey) add();
    else onEnter?.();
  };
  return (
    <div className="flex flex-col gap-2">
      {rows.length > 0 && (
        <div className="grid grid-cols-[1.5fr_1.4fr_64px_64px_20px] gap-2 text-[10px] uppercase tracking-wide text-[var(--color-on-surface-secondary)]">
          <span>Target</span>
          <span>Instruction (optional)</span>
          <span>Creativity</span>
          <span>Threshold</span>
          <span />
        </div>
      )}
      {rows.map((row, index) => (
        <div key={index} className="grid grid-cols-[1.5fr_1.4fr_64px_64px_20px] gap-2 items-center">
          <div className="flex items-center gap-1 min-w-0">
            <input
              ref={index === rows.length - 1 ? lastTargetRef : undefined}
              className={`${INPUT_FLEX} ${PLACEHOLDER} min-w-[7rem]`}
              value={row.target}
              placeholder="newspaper (any phrase)"
              onChange={(e) => update(index, { target: e.target.value })}
              onKeyDown={onKeyDown}
            />
            <select
              className={`${SELECT_BASE} w-[5.5rem] shrink-0`}
              value=""
              disabled={yoloModels.length === 0}
              title={
                yoloModels.length > 0
                  ? "Use one of SwarmUI's YOLO detection models as the target. The threshold becomes its confidence (0.25)."
                  : "No YOLO models listed. Check the SwarmUI connection on the setup screen."
              }
              onChange={(e) => {
                if (e.target.value) update(index, { target: `yolo-${e.target.value}`, threshold: YOLO_THRESHOLD });
              }}
            >
              <option value="">YOLO…</option>
              {yoloModels.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <input
            className={`${INPUT_FULL} ${PLACEHOLDER}`}
            value={row.instruction}
            placeholder="Make the headline legible"
            onChange={(e) => update(index, { instruction: e.target.value })}
            onKeyDown={onKeyDown}
          />
          <input
            className={INPUT_FULL}
            type="number"
            step="0.05"
            min={0}
            max={1}
            value={row.creativity}
            title="How much of the region is regenerated (0 to 1)."
            onChange={(e) => update(index, { creativity: Number(e.target.value) })}
            onKeyDown={onKeyDown}
          />
          <input
            className={INPUT_FULL}
            type="number"
            step="0.05"
            min={-1}
            max={1}
            value={row.threshold}
            title="Mask cutoff. Negative inverts the mask."
            onChange={(e) => update(index, { threshold: Number(e.target.value) })}
            onKeyDown={onKeyDown}
          />
          <button
            type="button"
            title="Remove this selection"
            onClick={() => onChange(rows.filter((_, i) => i !== index))}
            className="text-[var(--color-on-surface-secondary)] hover:text-[var(--color-on-surface)]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      ))}
      <div>
        <Button variant="ghost" size="sm" onClick={add}>
          + Add selection (Ctrl+Enter)
        </Button>
      </div>
    </div>
  );
}

function OtherDialog({
  others,
  categories,
  onSubmit,
  onClose,
}: {
  others: OtherFix[];
  categories: TriageCategory[];
  onSubmit: (others: OtherFix[]) => void;
  onClose: () => void;
}) {
  const [rows, setRows] = useState<OtherFix[]>(() => (others.length > 0 ? others : [newOtherFix(categories)]));
  const submit = () => onSubmit(rows);
  return (
    <Dialog title="Fix Other" onClose={onClose} wide>
      <div className="text-xs text-[var(--color-on-surface-secondary)]">
        One row per thing to segment (e.g. newspaper, watch, tattoo), each with its own creativity and threshold. Type a
        word for CLIPSeg, or pick one of SwarmUI&apos;s detection models from the YOLO dropdown. New rows start from the
        Fix Other category&apos;s values. A blank instruction becomes &quot;Restore the &lt;target&gt; so it looks
        natural …&quot; The instructions are also the prompt if segmentation finds nothing and Klein edits the whole
        image. Rows with a blank target are dropped; none left removes the tag.
      </div>
      <OthersEditor rows={rows} categories={categories} autoFocus onChange={setRows} onEnter={submit} />
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="primary" size="sm" onClick={submit}>
          Set (Enter)
        </Button>
      </div>
    </Dialog>
  );
}

function TagDialog({
  title,
  initial,
  categories,
  note,
  confirmLabel,
  onSubmit,
  onClose,
}: {
  title: string;
  initial: TagSet;
  categories: TriageCategory[];
  note?: string;
  confirmLabel: string;
  onSubmit: (tags: TagSet) => void;
  onClose: () => void;
}) {
  const [tags, setTags] = useState<TagSet>(initial);
  const fixes = tagsToFixes(tags, categories);
  return (
    <Dialog title={title} onClose={onClose} wide={tags.ids.has(CATEGORY_OTHER)}>
      {note && <div className="text-xs text-[var(--color-on-surface-secondary)]">{note}</div>}
      <div className="flex flex-col gap-2">
        {categories
          .filter((cat) => cat.id !== CATEGORY_MASK)
          .map((cat) => (
            <div key={cat.id} className="flex items-center gap-3">
              <Toggle
                value={tags.ids.has(cat.id)}
                onChange={() =>
                  setTags((t) => {
                    const next = toggleTag(t, cat.id);
                    // Turning Other on with nothing in it opens a first row to type into.
                    if (cat.id === CATEGORY_OTHER && next.ids.has(cat.id) && next.others.length === 0) {
                      return { ...next, others: [newOtherFix(categories)] };
                    }
                    return next;
                  })
                }
              />
              <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-[var(--color-border-subtle)] text-[10px] font-bold">
                {cat.key}
              </span>
              <span className="text-sm text-[var(--color-on-surface)]">{cat.label}</span>
            </div>
          ))}
        {tags.ids.has(CATEGORY_OTHER) && (
          <div className="pl-8">
            {/* Raw rows while typing; tagsToFixes drops the blank ones on send. */}
            <OthersEditor
              rows={tags.others}
              categories={categories}
              onChange={(rows) => setTags((t) => ({ ...t, others: rows }))}
            />
          </div>
        )}
      </div>
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="primary" size="sm" disabled={fixes.length === 0} onClick={() => onSubmit(tags)}>
          {confirmLabel}
        </Button>
      </div>
    </Dialog>
  );
}

/** What a Quick Mask send is made with; the next dialog starts from the last one sent. */
interface QuickMaskSettings {
  instruction: string;
  creativity: number;
  radius: number;
  colorCorrection: ColorCorrection;
  cropReference: number;
}

const QUICK_MASK_RADIUS = 24;

function QuickMaskDialog({
  stem,
  path,
  width,
  height,
  initial,
  defaults,
  onSend,
  onClose,
}: {
  stem: string;
  path: string;
  width: number;
  height: number;
  /** Where the dialog starts: the last send this session, or the defaults. */
  initial: QuickMaskSettings;
  /** The Quick Mask category's own instruction and creativity. */
  defaults: QuickMaskSettings;
  onSend: (mask: string, settings: QuickMaskSettings) => Promise<void>;
  onClose: () => void;
}) {
  const [radius, setRadius] = useState(initial.radius);
  const [erase, setErase] = useState(false);
  const [hasPaint, setHasPaint] = useState(false);
  const [instruction, setInstruction] = useState(initial.instruction);
  const [creativity, setCreativity] = useState(initial.creativity);
  const [colorCorrection, setColorCorrection] = useState<ColorCorrection>(initial.colorCorrection);
  const [cropReference, setCropReference] = useState(initial.cropReference);
  const customised =
    instruction !== defaults.instruction ||
    creativity !== defaults.creativity ||
    radius !== defaults.radius ||
    colorCorrection !== defaults.colorCorrection ||
    cropReference !== defaults.cropReference;
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const handleRef = useRef<QuickMaskCanvasHandle | null>(null);
  const onReady = useCallback((handle: QuickMaskCanvasHandle) => {
    handleRef.current = handle;
  }, []);

  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", handle, true);
    return () => window.removeEventListener("keydown", handle, true);
  }, [onClose]);

  const send = async () => {
    const mask = handleRef.current?.exportMask() ?? null;
    if (!mask) {
      setError("Paint the area to repair first");
      return;
    }
    setSending(true);
    setError(null);
    try {
      await onSend(mask, {
        instruction,
        creativity,
        radius,
        colorCorrection,
        cropReference: cleanCropReference(cropReference),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSending(false);
    }
  };

  return createPortal(
    <div
      className="fixed inset-0 z-[56] flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.7)" }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[var(--color-surface)] rounded-lg shadow-xl w-[calc(100vw-48px)] h-[calc(100vh-48px)] p-4 flex flex-col gap-3 border border-[var(--color-border-subtle)]">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-[var(--color-on-surface)]">
            Quick Mask · {stem}{" "}
            <span className="font-normal text-[var(--color-on-surface-secondary)]">
              paint the area to repaint; Klein re-samples just that crop
            </span>
          </h3>
          <button type="button" onClick={onClose} className="text-[var(--color-on-surface-secondary)]">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 min-h-0 rounded overflow-hidden" style={{ background: "rgba(0,0,0,0.95)" }}>
          <QuickMaskCanvas
            src={dpoApi.imageUrl(path)}
            width={width}
            height={height}
            brushRadius={radius}
            erase={erase}
            onReady={onReady}
            onChange={setHasPaint}
          />
        </div>
        <div className="flex items-center gap-3 flex-wrap text-xs">
          <label className="flex items-center gap-2 text-[var(--color-on-surface-secondary)]">
            Brush {radius}px
            <input type="range" min={4} max={120} value={radius} onChange={(e) => setRadius(Number(e.target.value))} />
          </label>
          <Toggle label="Erase" value={erase} onChange={setErase} />
          <Button variant="ghost" size="sm" onClick={() => handleRef.current?.clear()}>
            Clear
          </Button>
          <input
            className={`${INPUT_FULL} ${PLACEHOLDER} flex-1 min-w-[16rem]`}
            value={instruction}
            placeholder={defaults.instruction}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void send();
              }
            }}
          />
          <label className="flex items-center gap-2 text-[var(--color-on-surface-secondary)]">
            Creativity
            <input
              className={`${INPUT_FULL} w-20`}
              type="number"
              step="0.05"
              min={0}
              max={1}
              value={creativity}
              onChange={(e) => setCreativity(Number(e.target.value))}
            />
          </label>
          <label
            className="flex items-center gap-2 text-[var(--color-on-surface-secondary)]"
            title="Match the repaired crop's colours to the image around it. None when the colour itself is the fix."
          >
            Colour
            <select
              className={`${SELECT_BASE} w-[6.5rem]`}
              value={colorCorrection}
              onChange={(e) => setColorCorrection(cleanColorCorrection(e.target.value))}
            >
              {COLOR_CORRECTIONS.map((choice) => (
                <option key={choice} value={choice}>
                  {choice}
                </option>
              ))}
            </select>
          </label>
          <label
            className="flex items-center gap-2 text-[var(--color-on-surface-secondary)]"
            title="Flux.2 / Klein: how much of the sampling (from the end) sees the original crop as a reference. Lower frees the composition: 1 = always, 0.5 = second half only, 0 = never."
          >
            Reference
            <input
              className={`${INPUT_FULL} w-20`}
              type="number"
              step="0.05"
              min={0}
              max={1}
              value={cropReference}
              onChange={(e) => setCropReference(Number(e.target.value))}
            />
          </label>
          {customised && (
            <Button
              variant="ghost"
              size="sm"
              title={`Back to the defaults: creativity ${defaults.creativity}, the category's instruction, brush ${defaults.radius}px, colour ${defaults.colorCorrection}`}
              onClick={() => {
                setInstruction(defaults.instruction);
                setCreativity(defaults.creativity);
                setRadius(defaults.radius);
                setColorCorrection(defaults.colorCorrection);
                setCropReference(defaults.cropReference);
              }}
            >
              Reset to defaults
            </Button>
          )}
          <Button variant="primary" size="sm" onClick={() => void send()} loading={sending} disabled={!hasPaint}>
            <Brush className="w-3.5 h-3.5" /> Send (Enter)
          </Button>
          {error && <span className="text-[var(--color-error-500)]">{error}</span>}
        </div>
      </div>
    </div>,
    document.body,
  );
}

function DeleteDialog({ stem, onConfirm, onClose }: { stem: string; onConfirm: () => void; onClose: () => void }) {
  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        onConfirm();
      }
    };
    window.addEventListener("keydown", handle, true);
    return () => window.removeEventListener("keydown", handle, true);
  }, [onConfirm]);
  return (
    <Dialog title={`Delete pair ${stem}?`} onClose={onClose}>
      <div className="text-sm text-[var(--color-on-surface)]">
        Delete this pair from the dataset? Files move to .triage_fix/deleted and the manifest entry is removed. Undo is
        available on the Review screen (Ctrl+Z).
      </div>
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onClose}>
          Cancel (Esc)
        </Button>
        <Button variant="danger" size="sm" onClick={onConfirm}>
          <Trash2 className="w-3.5 h-3.5" /> Delete (Enter)
        </Button>
      </div>
    </Dialog>
  );
}

// --------------------------------------------------------------------------- //
// Modal
// --------------------------------------------------------------------------- //

interface TriageFixSessionProps {
  /** The modal is open: the session polls and resumes. */
  open: boolean;
  session: string;
  name: string;
  /** Its tab is showing: only then does it take keys. */
  visible: boolean;
  onActiveChange: (session: string, active: boolean) => void;
}

/** One Triage & Fix session: its own fixer model, settings, queue and hold. */
function TriageFixSession({ open, session, name, visible, onActiveChange }: TriageFixSessionProps) {
  // Fixed for the life of the component: each tab is keyed by its session.
  const api = useMemo(() => triageFixApi(session), [session]);
  const [settings, setSettings] = useState<TriageFixSettings>(() => loadTriageFixSettings(session));
  const [screen, setScreen] = useState<Screen>("setup");
  const [concepts, setConcepts] = useState<string[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [availableLoras, setAvailableLoras] = useState<string[]>([]);
  const [yoloModels, setYoloModels] = useState<string[]>([]);
  const [loraToAdd, setLoraToAdd] = useState("");
  const [loraWeight, setLoraWeight] = useState(1);
  const [connection, setConnection] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [inspect, setInspect] = useState<{
    hasLedger: boolean;
    total: number;
    undecided: number;
    ready: number;
  } | null>(null);
  const [advanced, setAdvanced] = useState(false);

  const [active, setActive] = useState(false);
  const [status, setStatus] = useState<TriageFixStatus | null>(null);
  const [items, setItems] = useState<TriageItem[]>([]);
  const [categories, setCategories] = useState<TriageCategory[]>(settings.categories);
  const [seed, setSeed] = useState(0);
  // One walk position per triage screen; ``setCursor`` follows the screen on show.
  const [cursor1, setCursor1] = useState(0);
  const [cursor2, setCursor2] = useState(0);
  // Caption keywords narrowing the triage walk (e.g. "wide shot, full body"). View only: the
  // ledger and the backend never see it, so clearing the box brings every image back.
  const [keywordQuery, setKeywordQuery] = useState<string>(() => {
    try {
      return window.localStorage.getItem(KEYWORD_FILTER_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [tags, setTags] = useState<Record<string, TagSet>>({});
  const [lastTags, setLastTags] = useState<TagSet | null>(null);
  const [triageUndo, setTriageUndo] = useState<Array<{ stem: string; action: "confirm" | "skip" }>>([]);
  const [secondUndo, setSecondUndo] = useState<string[]>([]);
  // The last Quick Mask send this session; the next dialog starts from it.
  const [lastMask, setLastMask] = useState<QuickMaskSettings | null>(null);
  const [firstView, setFirstView] = useState<TriageView>(() => readView(FIRST_VIEW_KEY));
  const [secondView, setSecondView] = useState<TriageView>(() => readView(SECOND_VIEW_KEY));
  // Second round: a handled image opened from the grid, shown ahead of the walk until decided.
  const [pinned2, setPinned2] = useState<string | null>(null);
  const [secondKinds, setSecondKinds] = useState<Set<SecondKind>>(() => new Set(SECOND_KINDS));
  // Grid view: the selection waiting for its fixes in the confirm dialog.
  const [bulkStems, setBulkStems] = useState<string[] | null>(null);
  const [clearSecondOpen, setClearSecondOpen] = useState(false);
  const [otherOpen, setOtherOpen] = useState(false);
  const [confirmAllOpen, setConfirmAllOpen] = useState(false);
  const [requeueOpen, setRequeueOpen] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [stripOpen, setStripOpen] = useState(false);
  // Stems accepted this session, newest first, for "fix more" (round two).
  const [fixedHistory, setFixedHistory] = useState<string[]>([]);
  const lastFixed = fixedHistory[0] ?? null;

  const [ready, setReady] = useState<ReadyItem | null>(null);
  const [compareState, setCompareState] = useState<"idle" | "waiting" | "done">("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);
  const fetchingRef = useRef(false);
  const readyRef = useRef<ReadyItem | null>(null);
  readyRef.current = ready;
  const screenRef = useRef<Screen>("setup");
  screenRef.current = screen;
  // The walk position of whichever triage screen is showing.
  const setCursor = useCallback((value: number | ((c: number) => number)) => {
    if (screenRef.current === "triage2") setCursor2(value);
    else setCursor1(value);
  }, []);

  const update = useCallback(
    (patch: Partial<TriageFixSettings>) => {
      setSettings((previous) => {
        const next = { ...previous, ...patch };
        saveTriageFixSettings(next, session);
        return next;
      });
    },
    [session],
  );

  const updateCategory = useCallback(
    (id: string, patch: Partial<TriageCategory>) => {
      update({ categories: settings.categories.map((cat) => (cat.id === id ? { ...cat, ...patch } : cat)) });
    },
    [settings.categories, update],
  );

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // ---- data loading --------------------------------------------------------

  const loadItems = useCallback(async () => {
    try {
      const res = await api.items();
      if (!res.ok || !res.items) return null;
      setItems(res.items);
      if (res.categories && res.categories.length > 0) setCategories(res.categories);
      // Seed the per-image tag sets: queued fixes or detector pre-tags.
      setTags((previous) => {
        const next = { ...previous };
        for (const item of res.items ?? []) {
          // Stored fixes (a retriaged or resumed image) always win; otherwise
          // keep what the user toggled, then fall back to the detector.
          if (item.fixes.length > 0 && (item.status === "tagged" || (!next[item.stem] && item.status !== "fixed"))) {
            next[item.stem] = tagsFromFixes(item.fixes);
            continue;
          }
          // A later round starts from a clean slate on the repaired image.
          if (item.status === "tagged" && item.round > 1 && item.fixes.length === 0) {
            next[item.stem] = emptyTags();
            continue;
          }
          if (next[item.stem]) continue;
          else if (item.detected) {
            next[item.stem] = categoriesFromDetected(parseSegmentString(item.detected), res.categories ?? categories);
          }
        }
        return next;
      });
      return res.items;
    } catch {
      /* transient */
      return null;
    }
  }, [api, categories]);

  const fetchReady = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    try {
      const res = await api.nextReady();
      if (!res.ok) {
        setError(res.error ?? "Could not fetch the next result");
        return;
      }
      if (res.item) {
        setReady(res.item);
        setCompareState("idle");
        setPreview(null);
      } else {
        setReady(null);
        setCompareState(res.waiting ? "waiting" : "done");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      fetchingRef.current = false;
    }
  }, [api]);

  const startPolling = useCallback(() => {
    clearPoll();
    pollRef.current = window.setInterval(() => {
      void api
        .status()
        .then((s) => {
          setStatus(s);
          if (!s.active) {
            clearPoll();
            setActive(false);
            if (s.error) setError(s.error);
            return;
          }
          if (screenRef.current === "compare" && !readyRef.current) {
            if (s.counts.ready > 0) void fetchReady();
            else setCompareState(s.counts.queued + s.counts.generating > 0 ? "waiting" : "done");
          }
        })
        .catch(() => {
          /* transient */
        });
    }, 800);
  }, [api, clearPoll, fetchReady]);

  // Reflect a session already running in this process when the modal opens.
  // Runs on open only: the callbacks it needs change identity as state
  // updates, and re-running it would drag the user back to triage.
  const openHandlersRef = useRef({ clearPoll, startPolling, loadItems });
  openHandlersRef.current = { clearPoll, startPolling, loadItems };
  useEffect(() => {
    const handlers = openHandlersRef.current;
    if (!open) {
      handlers.clearPoll();
      return;
    }
    let cancelled = false;
    void api
      .concepts()
      .then((res) => {
        if (!cancelled && res.ok) setConcepts(res.concepts);
      })
      .catch(() => {
        /* no config — fine */
      });
    void api
      .status()
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        if (s.active) {
          setActive(true);
          setSeed(s.shuffle_seed ?? 0);
          setScreen((current) => (current === "setup" ? "triage" : current));
          void handlers.loadItems();
          handlers.startPolling();
        }
      })
      .catch(() => {
        /* backend not up yet */
      });
    return () => {
      cancelled = true;
    };
  }, [api, open]);

  useEffect(() => () => clearPoll(), [clearPoll]);

  // The host keeps Escape from closing the window while any session runs.
  useEffect(() => onActiveChange(session, active), [onActiveChange, session, active]);

  // Resume banner for the chosen folder.
  useEffect(() => {
    if (!open || active || !settings.conceptDir) {
      setInspect(null);
      return;
    }
    let cancelled = false;
    void api
      .inspect(settings.conceptDir)
      .then((res) => {
        if (cancelled) return;
        if (!res.ok || !res.counts) {
          setInspect(null);
          return;
        }
        setInspect({
          hasLedger: Boolean(res.has_ledger),
          total: res.total ?? 0,
          undecided: res.counts.undecided,
          ready: res.counts.ready,
        });
      })
      .catch(() => {
        if (!cancelled) setInspect(null);
      });
    return () => {
      cancelled = true;
    };
  }, [open, active, settings.conceptDir, api]);

  // ---- setup actions -------------------------------------------------------

  const check = useCallback(async () => {
    setChecking(true);
    setConnection(null);
    try {
      const res = await swarmApi.check(settings.baseUrl, settings.token);
      if (!res.ok) {
        setConnection(res.error ?? "SwarmUI did not answer");
        return;
      }
      setModels(res.models ?? []);
      setAvailableLoras(res.loras ?? []);
      setYoloModels(res.yolo ?? []);
      setConnection(`Connected${res.version ? ` · SwarmUI ${res.version}` : ""} · ${res.models?.length ?? 0} models`);
      if (!settings.model && res.loaded && res.loaded.length > 0) update({ model: res.loaded[0] });
    } catch (e) {
      setConnection(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  }, [settings.baseUrl, settings.token, settings.model, update]);

  // A resumed session skips the setup screen's Check, so the Fix Other dropdown asks SwarmUI itself.
  useEffect(() => {
    if (!open || !active || yoloModels.length > 0) return;
    let cancelled = false;
    swarmApi
      .check(settings.baseUrl, settings.token)
      .then((res) => {
        if (cancelled || !res.ok) return;
        setYoloModels(res.yolo ?? []);
        setAvailableLoras((current) => (current.length > 0 ? current : (res.loras ?? [])));
        setModels((current) => (current.length > 0 ? current : (res.models ?? [])));
      })
      .catch(() => {
        /* the dropdown just stays empty */
      });
    return () => {
      cancelled = true;
    };
  }, [open, active, yoloModels.length, settings.baseUrl, settings.token]);

  const addLora = useCallback(() => {
    const name = loraToAdd.trim();
    if (!name || settings.loras.some((entry) => entry.name === name)) return;
    update({
      loras: [...settings.loras, { name, weight: Number.isFinite(Number(loraWeight)) ? Number(loraWeight) : 1 }],
    });
    setLoraToAdd("");
  }, [loraToAdd, loraWeight, settings.loras, update]);

  const start = useCallback(async () => {
    if (busy) return;
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      const res = await api.start({
        concept_dir: settings.conceptDir,
        base_url: settings.baseUrl,
        token: settings.token,
        model: settings.model,
        steps: Math.max(1, Math.round(Number(settings.steps)) || 4),
        cfgscale: Number(settings.cfgscale) || 1,
        sampler: settings.sampler,
        scheduler: settings.scheduler,
        sigmashift: numberOrNull(settings.sigmashift),
        color_correction: settings.colorCorrection,
        crop_reference: cleanCropReference(settings.cropReference),
        loras: settings.loras,
        options_per_image: Math.max(1, Math.min(4, Math.round(Number(settings.optionsPerImage)) || 2)),
        concurrency: Math.max(1, Math.min(4, Math.round(Number(settings.concurrency)) || 2)),
        segment_params: segmentParamsFromForm(settings.segment),
        mask_params: settings.maskParams,
        categories: settings.categories,
        prompt_template: settings.promptTemplate,
        fix_pair_dir: settings.fixPairDir,
        shuffle_seed: settings.shuffleSeed,
        detector: "",
        hold: settings.holdSending,
        name,
      });
      if (!res.ok) {
        setError(res.error ?? "Could not start");
        return;
      }
      setActive(true);
      setSeed(res.shuffle_seed ?? 0);
      setItems([]);
      setTags({});
      setTriageUndo([]);
      setLastMask(null);
      setReady(null);
      setCursor(0);
      setScreen("triage");
      setNotice(
        res.resumed
          ? `Resumed: ${res.undecided} to triage, ${res.ready} ready to review`
          : `${res.total} images, seed ${res.shuffle_seed}`,
      );
      if (res.held) setNotice((n) => `${n} · sending held: confirms queue up until you press Send`);
      await loadItems();
      startPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [
    busy,
    api,
    settings.conceptDir,
    settings.baseUrl,
    settings.token,
    settings.model,
    settings.steps,
    settings.cfgscale,
    settings.sampler,
    settings.scheduler,
    settings.sigmashift,
    settings.colorCorrection,
    settings.cropReference,
    settings.loras,
    settings.optionsPerImage,
    settings.concurrency,
    settings.segment,
    settings.maskParams,
    settings.categories,
    settings.promptTemplate,
    settings.fixPairDir,
    settings.shuffleSeed,
    settings.holdSending,
    name,
    setCursor,
    loadItems,
    startPolling,
  ]);

  const pause = useCallback(async () => {
    try {
      await api.cancel();
    } catch {
      /* the poll notices */
    }
    clearPoll();
    setActive(false);
    setReady(null);
    setScreen("setup");
    setNotice("Paused. Start again on the same folder to resume where you left off.");
  }, [api, clearPoll]);

  const held = Boolean(status?.held);

  const toggleHold = useCallback(async () => {
    const next = !held;
    setError(null);
    try {
      const res = await api.hold(next);
      if (!res.ok) {
        setError(res.error ?? "Could not change sending");
        return;
      }
      update({ holdSending: next });
      setStatus((s) => (s ? { ...s, held: next } : s));
      if (next) {
        const back = res.withdrawn ? ` · ${res.withdrawn} not yet started went back to the queue` : "";
        setNotice(`Sending held: confirms queue up; images SwarmUI already has will finish${back}`);
      } else {
        setNotice("Sending: the queue goes to SwarmUI in the order it was confirmed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api, held, update]);

  // ---- triage --------------------------------------------------------------

  const captionFilter = useMemo(() => parseCaptionFilter(keywordQuery), [keywordQuery]);
  const filtering = captionFilterActive(captionFilter);
  const unfilteredOrder = useMemo(() => triageOrder(items, seed), [items, seed]);
  // Bulk actions run on the whole walk, so they wait while the keyword filter narrows it.
  const narrowed = filtering;
  const isSecond = screen === "triage2";
  const isWalk = (screen === "triage" && firstView === "walk") || (isSecond && secondView === "walk");
  const unfilteredOrder2 = useMemo(() => secondRoundOrder(items, seed), [items, seed]);
  const secondCounts = useMemo(() => {
    const out: Record<SecondKind, number> = { fixed: 0, skipped: 0, kept: 0 };
    for (const item of unfilteredOrder2) {
      const kind = secondKind(item);
      if (kind) out[kind] += 1;
    }
    return out;
  }, [unfilteredOrder2]);
  const secondWalk = useCallback(
    (list: TriageItem[]) =>
      filterByCaptionOrStem(
        list.filter((item) => {
          const kind = secondKind(item);
          return kind !== null && secondKinds.has(kind);
        }),
        captionFilter,
      ),
    [secondKinds, captionFilter],
  );
  const secondHandled = useMemo(() => items.filter(secondRoundHandled).length, [items]);
  // Grid view: every second-round image the filters let through, handled ones included.
  const secondPool = useMemo(() => secondWalk(items), [secondWalk, items]);
  const order = useMemo(() => {
    if (isSecond) {
      const walk = secondWalk(unfilteredOrder2);
      const pinnedItem = pinned2 ? items.find((item) => item.stem === pinned2) : undefined;
      return pinnedItem && !walk.some((item) => item.stem === pinned2) ? [pinnedItem, ...walk] : walk;
    }
    return filtering ? filterByCaption(unfilteredOrder, captionFilter) : unfilteredOrder;
  }, [isSecond, secondWalk, unfilteredOrder2, pinned2, items, unfilteredOrder, captionFilter, filtering]);
  const cursor = isSecond ? cursor2 : cursor1;
  const changeKeywordQuery = useCallback((value: string) => {
    setKeywordQuery(value);
    setCursor1(0);
    setCursor2(0);
    try {
      window.localStorage.setItem(KEYWORD_FILTER_KEY, value);
    } catch {
      // Private mode or a full store: the filter still works for this session.
    }
  }, []);
  const safeCursor = order.length === 0 ? 0 : Math.min(cursor, order.length - 1);
  const current = order[safeCursor];
  const currentTags = current ? (tags[current.stem] ?? emptyTags()) : emptyTags();
  const anyDetected = useMemo(() => items.some((item) => Boolean(item.detected)), [items]);

  const setCurrentTags = useCallback(
    (next: TagSet) => {
      if (!current) return;
      setTags((previous) => ({ ...previous, [current.stem]: next }));
    },
    [current],
  );

  const markLocal = useCallback(
    (stem: string, statusValue: TriageItem["status"], fixes?: TriageFixSpec[], patch?: Partial<TriageItem>) => {
      setItems((previous) =>
        previous.map((item) =>
          item.stem === stem ? { ...item, status: statusValue, fixes: fixes ?? item.fixes, ...patch } : item,
        ),
      );
    },
    [],
  );

  const confirmCurrent = useCallback(async () => {
    if (!current || busy) return;
    const fixes = tagsToFixes(currentTags, categories);
    if (fixes.length === 0) {
      setNotice("Toggle at least one fix (1–6) before confirming, or press S to skip");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await api.confirm(current.stem, fixes, isSecond);
      if (!res.ok) {
        setError(res.error ?? "Could not confirm");
        return;
      }
      setLastTags(currentTags);
      if (isSecond) setSecondUndo((u) => [...u, current.stem]);
      else setTriageUndo((u) => [...u, { stem: current.stem, action: "confirm" }]);
      if (current.stem === pinned2) setPinned2(null);
      markLocal(current.stem, "queued", fixes, takenUp(current, isSecond));
      setNotice(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [current, busy, currentTags, categories, api, isSecond, pinned2, markLocal]);

  const skipCurrent = useCallback(async () => {
    if (!current || busy) return;
    setBusy(true);
    try {
      if (isSecond) {
        const res = await api.skipSecond([current.stem]);
        if (!res.ok || !res.skipped?.length) {
          setError(res.error ?? "Could not skip");
          return;
        }
        setSecondUndo((u) => [...u, current.stem]);
        if (current.stem === pinned2) setPinned2(null);
        // A taken-up image goes back to what it was.
        const was = secondKind(current);
        markLocal(current.stem, was, undefined, { second_pass: "skipped" });
        return;
      }
      const res = await api.skip(current.stem);
      if (!res.ok) {
        setError(res.error ?? "Could not skip");
        return;
      }
      setTriageUndo((u) => [...u, { stem: current.stem, action: "skip" }]);
      markLocal(current.stem, "skipped");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [current, busy, isSecond, api, markLocal, pinned2]);

  const undoSecond = useCallback(async () => {
    const stem = secondUndo[secondUndo.length - 1];
    if (!stem || busy) return;
    setBusy(true);
    try {
      const res = await api.undoSecond(stem);
      setSecondUndo((u) => u.slice(0, -1));
      if (!res.ok) {
        setNotice(res.error ?? "Could not undo");
        return;
      }
      const fresh = await loadItems();
      const index = cursorForStem(secondWalk(secondRoundOrder(fresh ?? items, seed)), stem);
      if (index >= 0) setCursor2(index);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [secondUndo, busy, api, loadItems, secondWalk, items, seed]);

  const undoTriage = useCallback(async () => {
    const last = triageUndo[triageUndo.length - 1];
    if (!last || busy) return;
    setBusy(true);
    try {
      const res = await api.undoTriage(last.stem);
      setTriageUndo((u) => u.slice(0, -1));
      if (!res.ok) {
        setNotice(res.error ?? "Could not undo");
        return;
      }
      markLocal(last.stem, res.status ?? null);
      await loadItems();
      const index = filterByCaption(
        triageOrder(
          items.map((i) => (i.stem === last.stem ? { ...i, status: null } : i)),
          seed,
        ),
        captionFilter,
      ).findIndex((i) => i.stem === last.stem);
      if (index >= 0) setCursor(index);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [triageUndo, busy, api, markLocal, loadItems, items, seed, captionFilter, setCursor]);

  const confirmAll = useCallback(
    async (mode: "default" | "detected", tagSet?: TagSet) => {
      setBusy(true);
      setError(null);
      try {
        const fixes = tagSet ? tagsToFixes(tagSet, categories) : [];
        const res = await api.confirmAll(mode, fixes);
        if (!res.ok) {
          setError(res.error ?? "Could not confirm");
          return;
        }
        setNotice(`${res.queued ?? 0} images ${held ? "queued (sending held)" : "sent for fixing"}`);
        if (Object.keys(res.conflicts ?? {}).length > 0) setError(conflictMessage(res.conflicts ?? {}));
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, categories, held, loadItems],
  );

  const maskCategory = categories.find((cat) => cat.id === CATEGORY_MASK);
  const otherCategory = categories.find((cat) => cat.id === CATEGORY_OTHER);
  const maskKey = maskCategory?.key ?? "9";
  const otherKey = otherCategory?.key ?? "8";
  const maskTarget = useMemo(() => {
    if (isWalk && current) {
      return { stem: current.stem, path: current.chosen_path, width: current.width, height: current.height };
    }
    if (screen === "compare" && ready) {
      return { stem: ready.stem, path: ready.original_path, width: ready.width, height: ready.height };
    }
    return null;
  }, [screen, isWalk, current, ready]);

  const maskDefaults = useMemo<QuickMaskSettings>(
    () => ({
      instruction: maskCategory?.instruction ?? "Fix this area of the image.",
      creativity: maskCategory?.creativity ?? 1,
      radius: QUICK_MASK_RADIUS,
      colorCorrection: settings.colorCorrection,
      cropReference: cleanCropReference(settings.cropReference),
    }),
    [maskCategory?.instruction, maskCategory?.creativity, settings.colorCorrection, settings.cropReference],
  );

  const sendMask = useCallback(
    async (mask: string, sent: QuickMaskSettings) => {
      const target = maskTarget;
      if (!target) return;
      const { instruction, creativity, colorCorrection, cropReference } = sent;
      const res = await api.confirmMask(
        target.stem,
        mask,
        instruction,
        creativity,
        isSecond,
        colorCorrection,
        cropReference,
      );
      if (!res.ok) throw new Error(res.error ?? "Could not send the mask");
      setLastMask(sent);
      setMaskOpen(false);
      setNotice(`${target.stem}: masked repair ${held ? "queued (sending held)" : "sent"}`);
      if (isWalk) {
        if (isSecond) setSecondUndo((u) => [...u, target.stem]);
        else setTriageUndo((u) => [...u, { stem: target.stem, action: "confirm" }]);
        const item = items.find((i) => i.stem === target.stem);
        markLocal(
          target.stem,
          "queued",
          [{ id: CATEGORY_MASK, target: "", instruction }],
          item ? takenUp(item, isSecond) : undefined,
        );
      } else {
        setReady(null);
        await fetchReady();
        await loadItems();
      }
    },
    [maskTarget, api, isSecond, held, isWalk, items, markLocal, fetchReady, loadItems],
  );

  const goCompare = useCallback(() => {
    setScreen("compare");
    setReady(null);
    void fetchReady();
  }, [fetchReady]);

  const goTriage = useCallback(() => {
    setScreen("triage");
    void loadItems();
  }, [loadItems]);

  const goSecond = useCallback(() => {
    setScreen("triage2");
    void loadItems();
  }, [loadItems]);

  const changeFirstView = useCallback((view: TriageView) => {
    setFirstView(view);
    storeView(FIRST_VIEW_KEY, view);
  }, []);

  const changeSecondView = useCallback((view: TriageView) => {
    setSecondView(view);
    storeView(SECOND_VIEW_KEY, view);
    if (view === "grid") setPinned2(null);
  }, []);

  const toggleSecondKind = useCallback((kind: SecondKind) => {
    setSecondKinds((current) => {
      const next = new Set(current);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });
    setCursor2(0);
  }, []);

  // ---- compare -------------------------------------------------------------

  const pick = useCallback(
    async (k: number) => {
      const item = readyRef.current;
      if (!item || busy) return;
      const candidate = item.candidates.find((c) => c.k === k);
      if (!candidate || candidate.error || !candidate.path) return;
      setBusy(true);
      setError(null);
      try {
        const res = await api.pick(item.stem, k);
        if (!res.ok) {
          setError(res.error ?? "Could not apply");
          return;
        }
        setNotice(
          `${item.stem}: candidate ${k + 1} is now the chosen image${
            res.fix_pair_id != null ? ` · fix pair ${res.fix_pair_id} written` : ""
          }`,
        );
        setFixedHistory((h) => pushRecent(h, item.stem));
        setReady(null);
        await fetchReady();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, busy, fetchReady],
  );

  const keep = useCallback(async () => {
    const item = readyRef.current;
    if (!item || busy) return;
    setBusy(true);
    try {
      const res = await api.keep(item.stem);
      if (!res.ok) {
        setError(res.error ?? "Could not keep");
        return;
      }
      setNotice(`${item.stem}: original kept`);
      setReady(null);
      await fetchReady();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api, busy, fetchReady]);

  const requeue = useCallback(
    async (stem: string, fixes?: TriageFixSpec[], lower = false) => {
      if (busy) return;
      setBusy(true);
      try {
        const res = await api.requeue(stem, fixes, lower);
        if (!res.ok) {
          setError(res.error ?? "Could not re-queue");
          return;
        }
        setNotice(`${stem}: re-queued with a fresh seed${lower ? " and lower creativity" : ""}`);
        if (readyRef.current?.stem === stem) {
          setReady(null);
          await fetchReady();
        }
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, busy, fetchReady, loadItems],
  );

  const undoDecision = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    try {
      const res = await api.undoDecision();
      if (!res.ok) {
        setNotice(res.error ?? "Nothing to undo");
        return;
      }
      setNotice(`Undid: ${res.undone} ${res.stem}`);
      setReady(null);
      await fetchReady();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api, busy, fetchReady]);

  const deleteTarget = useMemo(() => {
    if (isWalk && current) return current.stem;
    if (screen === "compare" && ready) return ready.stem;
    return null;
  }, [screen, isWalk, current, ready]);

  const deletePair = useCallback(
    async (stem: string) => {
      if (busy) return;
      setBusy(true);
      setDeleteOpen(false);
      try {
        const res = await api.deletePair(stem);
        if (!res.ok) {
          setError(res.error ?? "Could not delete the pair");
          return;
        }
        setNotice(`${stem}: pair removed from the dataset (Ctrl+Z on Review undoes it)`);
        markLocal(stem, "deleted");
        if (readyRef.current?.stem === stem) {
          setReady(null);
          await fetchReady();
        }
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, api, markLocal, loadItems, fetchReady],
  );

  const retriage = useCallback(
    async (stem: string) => {
      if (busy) return;
      setBusy(true);
      try {
        const res = await api.retriage(stem);
        if (!res.ok) {
          setError(res.error ?? "Could not send back to triage");
          return;
        }
        setNotice(`${stem}: back in triage with its fixes`);
        if (readyRef.current?.stem === stem) {
          setReady(null);
          await fetchReady();
        }
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, busy, fetchReady, loadItems],
  );

  /** Round two: send an accepted fix back to triage and jump to it. */
  const fixMore = useCallback(
    async (stem: string) => {
      if (busy) return;
      setBusy(true);
      setError(null);
      try {
        const res = await api.retriage(stem);
        if (!res.ok) {
          setError(res.error ?? "Could not start another round");
          return;
        }
        setFixedHistory((h) => h.filter((s) => s !== stem));
        setNotice(`Round ${res.round ?? "?"} on ${stem}`);
        const fresh = await loadItems();
        setScreen("triage2");
        changeSecondView("walk");
        const index = cursorForStem(secondWalk(secondRoundOrder(fresh ?? [], seed)), stem);
        if (index >= 0) setCursor2(index);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, api, loadItems, changeSecondView, secondWalk, seed],
  );

  /** Triage, grid view: open one image in the one-at-a-time walk. */
  const openFirst = useCallback(
    (stem: string) => {
      const index = cursorForStem(order, stem);
      if (index < 0) {
        setNotice(`${stem} is no longer in triage`);
        return;
      }
      setCursor1(index);
      changeFirstView("walk");
    },
    [order, changeFirstView],
  );

  /** Second round, grid view: open one image in the one-at-a-time walk. A handled image is not in
   * the walk, so it is shown ahead of it until it is confirmed or skipped. */
  const openSecond = useCallback(
    (stem: string) => {
      const index = cursorForStem(secondWalk(unfilteredOrder2), stem);
      setPinned2(index < 0 ? stem : null);
      setCursor2(Math.max(0, index));
      changeSecondView("walk");
    },
    [secondWalk, unfilteredOrder2, changeSecondView],
  );

  const confirmMany = useCallback(
    async (stems: string[], tagSet: TagSet) => {
      const fixes = tagsToFixes(tagSet, categories);
      if (fixes.length === 0) return;
      setBulkStems(null);
      setBusy(true);
      setError(null);
      try {
        const res = isSecond ? await api.confirmMany(stems, fixes) : await api.confirmManyFirst(stems, fixes);
        if (!res.ok) {
          setError(res.error ?? "Could not confirm");
          return;
        }
        setLastTags(tagSet);
        const queued = res.queued ?? [];
        if (isSecond) setSecondUndo((u) => [...u, ...queued]);
        else setTriageUndo((u) => [...u, ...queued.map((stem) => ({ stem, action: "confirm" as const }))]);
        const overlapping = Object.keys(res.conflicts ?? {}).length;
        const refused = (res.refused?.length ?? 0) - overlapping;
        const refusedWhy = isSecond ? "already in progress" : "no longer in triage";
        setNotice(
          `${queued.length} ${held ? "queued (sending held)" : "sent for fixing"}${refused ? ` · ${refused} ${refusedWhy}` : ""}`,
        );
        if (overlapping) setError(conflictMessage(res.conflicts ?? {}));
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [categories, isSecond, api, held, loadItems],
  );

  const clearSecond = useCallback(async () => {
    setClearSecondOpen(false);
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.clearSecond();
      if (!res.ok) {
        setError(res.error ?? "Could not clear the second round");
        return;
      }
      // Their undo entries point at markers that are gone.
      setSecondUndo([]);
      setCursor2(0);
      setNotice(`${res.cleared ?? 0} image(s) back in the second round`);
      await loadItems();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api, busy, loadItems]);

  const skipMany = useCallback(
    async (stems: string[]) => {
      if (busy || stems.length === 0) return;
      setBusy(true);
      setError(null);
      try {
        const res = await api.skipSecond(stems);
        if (!res.ok) {
          setError(res.error ?? "Could not skip");
          return;
        }
        setSecondUndo((u) => [...u, ...(res.skipped ?? [])]);
        setNotice(`${res.skipped?.length ?? 0} skipped this round`);
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, busy, loadItems],
  );

  /** Triage, grid view: skip the selection for good. */
  const skipManyFirst = useCallback(
    async (stems: string[]) => {
      if (busy || stems.length === 0) return;
      setBusy(true);
      setError(null);
      try {
        const res = await api.skipMany(stems);
        if (!res.ok) {
          setError(res.error ?? "Could not skip");
          return;
        }
        const skipped = res.skipped ?? [];
        setTriageUndo((u) => [...u, ...skipped.map((stem) => ({ stem, action: "skip" as const }))]);
        const refused = res.refused?.length ?? 0;
        setNotice(`${skipped.length} skipped for good${refused ? ` · ${refused} no longer in triage` : ""}`);
        await loadItems();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, busy, loadItems],
  );

  const addFixPairs = useCallback(
    async (stems: string[]) => {
      if (busy) return;
      setBusy(true);
      setError(null);
      try {
        const res = await api.addFixPairs(stems);
        if (!res.ok) {
          setError(res.error ?? "Could not write the fix pairs");
          return;
        }
        await loadItems();
        const errors = res.errors ?? [];
        setNotice(`${res.added?.length ?? 0} fix pair(s) written`);
        if (errors.length > 0) {
          setError(`${errors.length} failed: ${errors.slice(0, 3).join("; ")}${errors.length > 3 ? " …" : ""}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [api, busy, loadItems],
  );

  // ---- keyboard ------------------------------------------------------------

  const dialogOpen =
    otherOpen ||
    confirmAllOpen ||
    requeueOpen ||
    maskOpen ||
    deleteOpen ||
    bulkStems !== null ||
    clearSecondOpen ||
    Boolean(preview);

  useEffect(() => {
    if (!open || !visible || !active || screen === "setup") return;
    const handler = (e: KeyboardEvent) => {
      if (dialogOpen || keyboardTargetIsInput(e)) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        if (screen === "compare") {
          e.preventDefault();
          void undoDecision();
        }
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const key = e.key;
      const lower = key.toLowerCase();
      if (key === maskKey) {
        if (maskTarget && maskTarget.width && maskTarget.height) {
          e.preventDefault();
          setMaskOpen(true);
        }
        return;
      }
      if (lower === "d") {
        if (deleteTarget) {
          e.preventDefault();
          setDeleteOpen(true);
        }
        return;
      }
      if (key === "T") {
        e.preventDefault();
        goSecond();
        return;
      }
      if (lower === "t") {
        e.preventDefault();
        goTriage();
        return;
      }
      if ((screen === "triage" || screen === "triage2") && lower === "g") {
        e.preventDefault();
        if (isSecond) changeSecondView(secondView === "grid" ? "walk" : "grid");
        else changeFirstView(firstView === "grid" ? "walk" : "grid");
        return;
      }
      if ((screen === "triage" || screen === "triage2") && !isWalk && key === "Backspace") {
        e.preventDefault();
        void (isSecond ? undoSecond() : undoTriage());
        return;
      }
      if (lower === "c") {
        e.preventDefault();
        goCompare();
        return;
      }
      if (isWalk) {
        if (!current) return;
        const toggled = categories.find((c) => c.key === key && c.id !== CATEGORY_OTHER && c.id !== CATEGORY_MASK);
        if (toggled) {
          e.preventDefault();
          setCurrentTags(toggleTag(currentTags, toggled.id));
        } else if (key === otherKey) {
          e.preventDefault();
          setOtherOpen(true);
        } else if (key === "Enter") {
          e.preventDefault();
          void confirmCurrent();
        } else if (lower === "s") {
          e.preventDefault();
          void skipCurrent();
        } else if (key === "Backspace") {
          e.preventDefault();
          void (isSecond ? undoSecond() : undoTriage());
        } else if (key === "ArrowLeft") {
          e.preventDefault();
          setCursor((c) => Math.max(0, Math.min(c, order.length - 1) - 1));
        } else if (key === "ArrowRight") {
          e.preventDefault();
          setCursor((c) => Math.min(order.length - 1, c + 1));
        } else if (key === " " || key === "Spacebar") {
          e.preventDefault();
          (document.activeElement as HTMLElement | null)?.blur?.();
          setPreview(current.chosen_path);
        }
        return;
      }
      if (screen === "compare") {
        if (lower === "f") {
          if (lastFixed) {
            e.preventDefault();
            void fixMore(lastFixed);
          }
          return;
        }
        const item = readyRef.current;
        if (!item) return;
        if (/^[1-9]$/.test(key)) {
          e.preventDefault();
          void pick(Number(key) - 1);
        } else if (key === "0") {
          e.preventDefault();
          void keep();
        } else if (key === "R") {
          e.preventDefault();
          setRequeueOpen(true);
        } else if (lower === "r") {
          e.preventDefault();
          void requeue(item.stem);
        } else if (lower === "s") {
          e.preventDefault();
          void retriage(item.stem);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    open,
    visible,
    setCursor,
    active,
    screen,
    isWalk,
    isSecond,
    secondView,
    changeSecondView,
    firstView,
    changeFirstView,
    goSecond,
    undoSecond,
    dialogOpen,
    current,
    currentTags,
    categories,
    order.length,
    maskTarget,
    maskKey,
    otherKey,
    deleteTarget,
    setCurrentTags,
    confirmCurrent,
    skipCurrent,
    undoTriage,
    goTriage,
    goCompare,
    pick,
    keep,
    requeue,
    retriage,
    fixMore,
    lastFixed,
    undoDecision,
  ]);

  // Warm the next few images.
  useEffect(() => {
    if (!isWalk) return;
    for (const item of order.slice(safeCursor + 1, safeCursor + 4)) {
      const img = new Image();
      img.src = dpoApi.imageUrl(item.chosen_path);
    }
  }, [isWalk, order, safeCursor]);

  // ---- render ---------------------------------------------------------------

  const counts = status?.counts;
  const failedItems = useMemo(() => items.filter((item) => item.status === "failed"), [items]);

  const header = active && (
    <div className="flex items-center gap-2 flex-wrap text-xs">
      <div className="flex rounded overflow-hidden border border-[var(--color-border-subtle)]">
        <button
          type="button"
          onClick={goTriage}
          className={`px-3 py-1.5 font-semibold ${screen === "triage" ? "bg-[var(--color-cobalt-600)] text-white" : "text-[var(--color-on-surface-secondary)]"}`}
        >
          Triage ({unfilteredOrder.length} left) · T
        </button>
        <button
          type="button"
          onClick={goSecond}
          title="Fixed, skipped and kept images, for another round"
          className={`px-3 py-1.5 font-semibold ${screen === "triage2" ? "bg-[var(--color-cobalt-600)] text-white" : "text-[var(--color-on-surface-secondary)]"}`}
        >
          Triage (second round) ({unfilteredOrder2.length}) · Shift+T
        </button>
        <button
          type="button"
          onClick={goCompare}
          className={`px-3 py-1.5 font-semibold ${screen === "compare" ? "bg-[var(--color-cobalt-600)] text-white" : "text-[var(--color-on-surface-secondary)]"}`}
        >
          Review ({counts?.ready ?? 0} ready) · C
        </button>
      </div>
      {counts && (
        <span className="tabular-nums text-[var(--color-on-surface-secondary)] flex flex-wrap gap-x-3">
          <span>{counts.total} images</span>
          <span className={held ? "text-[var(--color-warning-500,#eab308)] font-semibold" : undefined}>
            {counts.queued} queued{held ? " · held" : ""}
          </span>
          <span className="flex items-center gap-1">
            {counts.generating > 0 && <Loader2 className="w-3 h-3 animate-spin" />}
            {counts.generating} generating
          </span>
          <span>{counts.fixed} fixed</span>
          <span>{counts.kept} kept</span>
          <span>{counts.skipped} skipped</span>
          {counts.deleted > 0 && <span>{counts.deleted} deleted</span>}
          {counts.failed > 0 && <span className="text-[var(--color-error-500)]">{counts.failed} failed</span>}
          {status && status.waiting > 0 && (
            <span
              className="text-[var(--color-warning-500,#eab308)]"
              title="Queued here, but another session has the same image queued first or is repairing it. It goes out once that session's result is decided, and repairs the image as it stands then."
            >
              {status.waiting} waiting on {status.peers.join(", ")}
            </span>
          )}
        </span>
      )}
      <span className="ml-auto flex items-center gap-2">
        {status?.fix_pair_dir && (
          <span className="text-[var(--color-on-surface-secondary)]" title={status.fix_pair_dir}>
            fix pairs → {basename(status.fix_pair_dir)}
          </span>
        )}
        {held ? (
          <Button
            variant="primary"
            size="sm"
            onClick={() => void toggleHold()}
            title="Send the queued fixes to SwarmUI, in the order they were confirmed"
          >
            <Send className="w-3.5 h-3.5" /> Send to SwarmUI ({counts?.queued ?? 0})
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void toggleHold()}
            title="Keep triaging but stop sending: confirms queue up until you press Send. Images SwarmUI already has finish."
          >
            <Hand className="w-3.5 h-3.5" /> Hold sending
          </Button>
        )}
        <Button variant="ghost" size="sm" onClick={() => void pause()}>
          <Pause className="w-3.5 h-3.5" /> Pause
        </Button>
      </span>
    </div>
  );

  const messages = (
    <>
      {status?.warning && <div className="text-xs text-[var(--color-warning-500,#eab308)]">{status.warning}</div>}
      {notice && <div className="text-sm text-[var(--color-on-surface-secondary)]">{notice}</div>}
      {isWalk && current && current.claimed_by.length > 0 && (
        <div className="text-xs text-[var(--color-warning-500,#eab308)]">
          Also queued in {current.claimed_by.map((peer) => `"${peer}"`).join(", ")}
          {current.waiting_on
            ? `: it is repaired there first, then here on the result`
            : `: fixes for the same area are refused until it is decided there`}
        </div>
      )}
      {error && (
        <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)] bg-[var(--color-surface-container)] rounded p-3">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </>
  );

  const chip = (cat: TriageCategory, on: boolean, onClick: () => void, extra?: string) => (
    <div
      key={cat.id}
      role="button"
      tabIndex={-1}
      onClick={onClick}
      className={`px-4 py-2 rounded cursor-pointer font-semibold flex items-center gap-2 border text-sm ${
        on
          ? "bg-[var(--color-cobalt-600)] text-white border-[var(--color-cobalt-600)]"
          : "text-[var(--color-on-surface-secondary)] border-[var(--color-border-subtle)]"
      }`}
    >
      <span
        className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs ${on ? "bg-white/25" : "bg-[var(--color-border-subtle)]"}`}
      >
        {cat.key}
      </span>
      {cat.label.replace(/^Fix /, "")}
      {extra && <span className="font-normal opacity-80">{extra}</span>}
    </div>
  );

  return (
    <YoloModelsContext.Provider value={yoloModels}>
      <div className={visible ? "flex-1 min-h-0 flex flex-col" : "hidden"}>
        <div className={`flex flex-col gap-3 ${screen === "setup" ? "" : "h-full min-h-0"}`}>
          {header}
          {messages}

          {/* ---------------------------------------------------------------- setup */}
          {screen === "setup" && (
            <div className="flex flex-col gap-4">
              <SectionCard title="Dataset">
                <div className="flex flex-col gap-3">
                  {concepts.length > 0 && (
                    <Select
                      label="Configured DPO dataset"
                      options={["", ...concepts]}
                      value={concepts.includes(settings.conceptDir) ? settings.conceptDir : ""}
                      onChange={(value) => update({ conceptDir: value })}
                      formatLabel={(value) => value || "— pick —"}
                    />
                  )}
                  <DirPicker
                    label="Dataset folder (manifest.json, chosen/, rejected/)"
                    value={settings.conceptDir}
                    onChange={(value) => update({ conceptDir: value })}
                  />
                  {inspect && (
                    <div className="text-xs bg-[var(--color-surface-container)] rounded p-2 text-[var(--color-on-surface-secondary)]">
                      {inspect.hasLedger ? (
                        <>
                          <span className="text-[var(--color-on-surface)] font-semibold">Resume:</span> {inspect.total}{" "}
                          images · {inspect.undecided} still to triage · {inspect.ready} ready to review. Decided images
                          will not come back.
                        </>
                      ) : (
                        <>{inspect.total} pairs. No previous Triage & Fix session on this folder.</>
                      )}
                    </div>
                  )}
                  <DirPicker
                    label="Fix pair folder (optional): fixed = chosen, original = rejected. Leave blank to skip."
                    value={settings.fixPairDir}
                    onChange={(value) => update({ fixPairDir: value })}
                  />
                  <FormEntry
                    label="Shuffle seed (blank = random; a resumed session keeps its own)"
                    value={settings.shuffleSeed ?? ""}
                    onChange={(value) => update({ shuffleSeed: numberOrNull(String(value)) })}
                  />
                </div>
              </SectionCard>

              <SectionCard title="SwarmUI and Flux.2 Klein">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <FormEntry
                    label="SwarmUI URL"
                    value={settings.baseUrl}
                    onChange={(v) => update({ baseUrl: String(v) })}
                  />
                  <FormEntry label="Token" value={settings.token} onChange={(v) => update({ token: String(v) })} />
                  <div className="flex items-end gap-2">
                    <Button variant="secondary" size="sm" onClick={() => void check()} loading={checking}>
                      <Plug className="w-3.5 h-3.5" /> Check
                    </Button>
                    {connection && (
                      <span className="text-xs text-[var(--color-on-surface-secondary)]">{connection}</span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
                  {models.length > 0 ? (
                    <Select
                      label="Fixer model"
                      options={
                        models.includes(settings.model) || !settings.model ? models : [settings.model, ...models]
                      }
                      value={settings.model}
                      onChange={(value) => update({ model: value })}
                      formatLabel={(value) => value}
                    />
                  ) : (
                    <FormEntry
                      label="Fixer model (as SwarmUI names it)"
                      value={settings.model}
                      onChange={(v) => update({ model: String(v) })}
                      placeholder="Flux/flux2-klein-9b"
                    />
                  )}
                  <FormEntry
                    label="Steps (Klein: 4 · Z-Image Turbo: 8)"
                    type="number"
                    value={settings.steps}
                    onChange={(v) => update({ steps: Number(v) })}
                  />
                  <FormEntry
                    label="CFG (Klein / Turbo: 1)"
                    type="number"
                    value={settings.cfgscale}
                    onChange={(v) => update({ cfgscale: Number(v) })}
                  />
                  <FormEntry
                    label="Sampler (blank = Swarm default)"
                    value={settings.sampler}
                    onChange={(v) => update({ sampler: String(v) })}
                  />
                  <FormEntry
                    label="Scheduler (blank = Swarm default)"
                    value={settings.scheduler}
                    onChange={(v) => update({ scheduler: String(v) })}
                  />
                  <FormEntry
                    label="Sigma shift (blank = Swarm default)"
                    value={settings.sigmashift}
                    onChange={(v) => update({ sigmashift: String(v) })}
                  />
                  <Select
                    label="Colour correction (pasted-back crop)"
                    value={settings.colorCorrection}
                    onChange={(v) => update({ colorCorrection: cleanColorCorrection(v) })}
                    options={[...COLOR_CORRECTIONS]}
                    tooltip="SwarmUI Color Correction Behavior: matches the repaired crop's colours to the image around it, so a pair differs only in the repair. Linear2 suits the Flux / SDXL VAE colour drift; None when the colour itself is the fix. Quick Mask can override it per send."
                  />
                  <FormEntry
                    label="Klein crop reference (0–1)"
                    type="number"
                    value={settings.cropReference}
                    onChange={(v) => update({ cropReference: cleanCropReference(v) })}
                    tooltip="Needs the patched SwarmUI (Crop Reference Strength). Flux.2 / Klein copy the original crop they are given as a reference, which can keep the old composition even at creativity 1. This is the fraction of the sampling, from the end, that sees it: 1 = always (default), 0.5 = second half only, 0 = never. Quick Mask can override it per send."
                  />
                  <FormEntry
                    label="Options per image (1–4)"
                    type="number"
                    value={settings.optionsPerImage}
                    onChange={(v) => update({ optionsPerImage: Number(v) })}
                  />
                  <FormEntry
                    label="Concurrency (1–4)"
                    type="number"
                    value={settings.concurrency}
                    onChange={(v) => update({ concurrency: Number(v) })}
                  />
                  <FormEntry
                    label="Sub-prompt template"
                    value={settings.promptTemplate}
                    onChange={(v) => update({ promptTemplate: String(v) })}
                    tooltip="{instruction} and {caption} are filled per segment"
                  />
                </div>
                <div className="mt-4 flex flex-col gap-2">
                  <div className="text-xs font-semibold text-[var(--color-on-surface)]">
                    LoRAs / adapters on the fixer model
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
                    <div className="md:col-span-2">
                      {availableLoras.length > 0 ? (
                        <Select
                          label="Add a LoRA"
                          value={loraToAdd}
                          onChange={setLoraToAdd}
                          options={["", ...availableLoras]}
                          formatLabel={(value) => value || "Pick a LoRA…"}
                        />
                      ) : (
                        <FormEntry
                          label="Add a LoRA (as SwarmUI names it; Check lists them)"
                          value={loraToAdd}
                          onChange={(v) => setLoraToAdd(String(v))}
                        />
                      )}
                    </div>
                    <FormEntry
                      label="Weight"
                      type="number"
                      value={loraWeight}
                      onChange={(v) => setLoraWeight(Number(v))}
                    />
                    <Button variant="secondary" onClick={addLora} disabled={!loraToAdd.trim()}>
                      <Plus className="w-4 h-4" /> Add
                    </Button>
                  </div>
                  {settings.loras.map((entry) => (
                    <div
                      key={entry.name}
                      className="flex items-center gap-2 text-sm bg-[var(--color-surface-container)] rounded px-3 py-1"
                    >
                      <span className="flex-1 truncate" title={entry.name}>
                        {entry.name}
                      </span>
                      <input
                        className={`${INPUT_FULL} w-20`}
                        type="number"
                        step="0.05"
                        value={entry.weight}
                        title="Weight"
                        onChange={(e) =>
                          update({
                            loras: settings.loras.map((row) =>
                              row.name === entry.name ? { ...row, weight: Number(e.target.value) } : row,
                            ),
                          })
                        }
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Remove"
                        onClick={() => update({ loras: settings.loras.filter((row) => row.name !== entry.name) })}
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                  <div className="text-xs text-[var(--color-on-surface-secondary)]">
                    {settings.loras.length === 0
                      ? "None: the fixer model runs on its own. Add your finetune's adapter here to inpaint with it."
                      : "Applied to every request: segment passes, Quick Mask and whole-image edits."}
                  </div>
                </div>
              </SectionCard>

              <SectionCard title="Fix categories (number keys)">
                <div className="flex flex-col gap-2">
                  <div className="grid grid-cols-[2rem_9rem_1fr_1.4fr_5rem_5rem_5rem_8rem_4.5rem] gap-2 text-[10px] uppercase tracking-wide text-[var(--color-on-surface-secondary)]">
                    <span>Key</span>
                    <span>Fix</span>
                    <span>Segment target</span>
                    <span>Instruction</span>
                    <span>Caption</span>
                    <span>Creativity</span>
                    <span>Threshold</span>
                    <span title="Retried when the YOLO target finds nothing: first at the low threshold, then as this CLIPSeg word">
                      Fallback (CLIPSeg)
                    </span>
                    <span title="Retried when the YOLO target finds nothing: first at the low threshold, then as this CLIPSeg word">
                      Low thr.
                    </span>
                  </div>
                  {settings.categories.map((cat) => (
                    <div
                      key={cat.id}
                      className="grid grid-cols-[2rem_9rem_1fr_1.4fr_5rem_5rem_5rem_8rem_4.5rem] gap-2 items-center"
                    >
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-border-subtle)] text-xs font-bold">
                        {cat.key}
                      </span>
                      <span className="text-sm text-[var(--color-on-surface)]">{cat.label}</span>
                      <input
                        className={`${INPUT_FULL} ${PLACEHOLDER}`}
                        value={cat.target}
                        disabled={cat.id === CATEGORY_OTHER || cat.id === CATEGORY_MASK || cat.id === CATEGORY_PHOTO}
                        placeholder={
                          cat.id === CATEGORY_OTHER
                            ? "typed per image"
                            : cat.id === CATEGORY_MASK
                              ? "painted per image"
                              : cat.id === CATEGORY_PHOTO
                                ? "whole image (Klein edit)"
                                : "face | yolo-face_yolov8m-seg_60.pt"
                        }
                        onChange={(e) => updateCategory(cat.id, { target: e.target.value })}
                      />
                      <input
                        className={`${INPUT_FULL} ${PLACEHOLDER}`}
                        value={cat.instruction}
                        onChange={(e) => updateCategory(cat.id, { instruction: e.target.value })}
                      />
                      <Toggle
                        value={cat.include_caption}
                        onChange={(value) => updateCategory(cat.id, { include_caption: value })}
                      />
                      <input
                        className={INPUT_FULL}
                        type="number"
                        step="0.05"
                        min={0}
                        max={1}
                        value={cat.creativity}
                        title={
                          cat.id === CATEGORY_PHOTO
                            ? "1 = redraw from noise with the image as reference. Lower starts from the image itself; at 4 steps only 0.75 / 0.5 / 0.25 differ."
                            : undefined
                        }
                        onChange={(e) => updateCategory(cat.id, { creativity: Number(e.target.value) })}
                      />
                      <input
                        className={INPUT_FULL}
                        type="number"
                        step="0.05"
                        min={-1}
                        max={1}
                        value={cat.threshold}
                        disabled={cat.id === CATEGORY_PHOTO}
                        title={
                          cat.id === "background"
                            ? "negative = inverted mask (everything except the match)"
                            : "Mask cutoff. Negative inverts the mask."
                        }
                        onChange={(e) => updateCategory(cat.id, { threshold: Number(e.target.value) })}
                      />
                      <input
                        className={`${INPUT_FULL} ${PLACEHOLDER}`}
                        value={cat.fallback_target}
                        disabled={cat.id === CATEGORY_OTHER || cat.id === CATEGORY_MASK || cat.id === CATEGORY_PHOTO}
                        placeholder={
                          cat.id === CATEGORY_OTHER || cat.id === CATEGORY_MASK || cat.id === CATEGORY_PHOTO
                            ? "—"
                            : "none"
                        }
                        title="Retried when the YOLO target finds nothing: first at the low threshold, then as this CLIPSeg word"
                        onChange={(e) => updateCategory(cat.id, { fallback_target: e.target.value })}
                      />
                      <input
                        className={INPUT_FULL}
                        type="number"
                        step="0.05"
                        min={0}
                        max={1}
                        value={cat.low_threshold}
                        disabled={cat.id === CATEGORY_OTHER || cat.id === CATEGORY_MASK || cat.id === CATEGORY_PHOTO}
                        title="Retried when the YOLO target finds nothing: first at the low threshold, then as this CLIPSeg word"
                        onChange={(e) => updateCategory(cat.id, { low_threshold: Number(e.target.value) })}
                      />
                      {cat.id === "background" && (
                        <div className="col-span-full text-[10px] text-[var(--color-on-surface-secondary)] -mt-1 pl-[11rem]">
                          negative = inverted mask (everything except the match)
                        </div>
                      )}
                    </div>
                  ))}
                  <div className="text-xs text-[var(--color-on-surface-secondary)]">
                    Each segment is refined with its instruction; with Caption on, the pair caption follows it through
                    the template above. Klein runs as the request model with the image as init image at creativity 0, so
                    only the segmented regions change.
                  </div>
                </div>
              </SectionCard>

              <div>
                <button
                  type="button"
                  className="text-xs flex items-center gap-1 text-[var(--color-on-surface-secondary)]"
                  onClick={() => setAdvanced((v) => !v)}
                >
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${advanced ? "rotate-180" : ""}`} />
                  Segment refining parameters
                </button>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                  <FormEntry
                    label="Mask context margin (px)"
                    type="number"
                    value={settings.maskParams.maskshrinkgrow ?? 32}
                    onChange={(v) => update({ maskParams: { ...settings.maskParams, maskshrinkgrow: Number(v) } })}
                    tooltip="Quick Mask: how much of the surrounding image is cropped and sampled with the painted area (SwarmUI maskshrinkgrow)"
                  />
                  <FormEntry
                    label="Mask edge blur"
                    type="number"
                    value={settings.maskParams.maskblur ?? 8}
                    onChange={(v) => update({ maskParams: { ...settings.maskParams, maskblur: Number(v) } })}
                    tooltip="Quick Mask: softens the paste-back edge (SwarmUI maskblur)"
                  />
                </div>
                {advanced && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                    {SEGMENT_FIELDS.map((field) => {
                      const value = settings.segment[field.key];
                      const setValue = (next: string | boolean) =>
                        update({ segment: { ...settings.segment, [field.key]: next } });
                      return (
                        <div key={field.key} className="flex flex-col gap-1">
                          <label className="text-xs text-[var(--color-on-surface-secondary)]" title={field.hint}>
                            {field.label}
                          </label>
                          {field.kind === "bool" ? (
                            <Toggle value={Boolean(value)} onChange={(v) => setValue(v)} />
                          ) : field.kind === "select" ? (
                            <select
                              className={INPUT_FULL}
                              value={String(value ?? "")}
                              onChange={(e) => setValue(e.target.value)}
                            >
                              {(field.options ?? []).map((option) => (
                                <option key={option} value={option}>
                                  {option || "— default —"}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              className={`${INPUT_FULL} ${PLACEHOLDER}`}
                              value={String(value ?? "")}
                              placeholder={field.hint}
                              onChange={(e) => setValue(e.target.value)}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-3">
                <Button
                  variant="primary"
                  onClick={() => void start()}
                  loading={busy}
                  disabled={!settings.conceptDir || !settings.model}
                >
                  <Wand2 className="w-4 h-4" /> {inspect?.hasLedger ? "Resume" : "Start"}
                </Button>
                <Toggle
                  label="Hold sending (queue fixes, send later)"
                  value={settings.holdSending}
                  onChange={(v) => update({ holdSending: Boolean(v) })}
                  tooltip="Triage and confirm as usual, but nothing goes to SwarmUI until you press Send in the header (for when SwarmUI is busy with other work)"
                />
                {!settings.model && (
                  <span className="text-xs text-[var(--color-on-surface-secondary)]">
                    Check SwarmUI and pick the fixer model
                  </span>
                )}
              </div>
            </div>
          )}

          {/* --------------------------------------------------------------- triage */}
          {(screen === "triage" || screen === "triage2") && active && (
            <div className="flex flex-col gap-3 flex-1 min-h-0">
              <div className="flex items-center gap-2 text-xs text-[var(--color-on-surface-secondary)]">
                <span className="shrink-0">Caption keywords</span>
                <input
                  type="text"
                  className={`${INPUT_FULL} ${PLACEHOLDER} flex-1`}
                  placeholder="wide shot, full body, -close-up   (comma = or, leading - = exclude)"
                  value={keywordQuery}
                  onChange={(e) => changeKeywordQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape" || e.key === "Enter") e.currentTarget.blur();
                  }}
                  spellCheck={false}
                />
                {filtering && !isSecond && (
                  <>
                    <span className="shrink-0">
                      {order.length} of {unfilteredOrder.length} match
                    </span>
                    <Button variant="ghost" size="sm" onClick={() => changeKeywordQuery("")}>
                      Clear
                    </Button>
                  </>
                )}
                {isSecond && filtering && (
                  <Button variant="ghost" size="sm" onClick={() => changeKeywordQuery("")}>
                    Clear
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap text-xs">
                <div className="flex rounded overflow-hidden border border-[var(--color-border-subtle)]">
                  {(["walk", "grid"] as TriageView[]).map((view) => (
                    <button
                      key={view}
                      type="button"
                      onClick={() => (isSecond ? changeSecondView(view) : changeFirstView(view))}
                      className={`px-3 py-1.5 font-semibold ${
                        (isSecond ? secondView : firstView) === view
                          ? "bg-[var(--color-cobalt-600)] text-white"
                          : "text-[var(--color-on-surface-secondary)]"
                      }`}
                    >
                      {view === "walk" ? "One at a time" : "Grid"}
                    </button>
                  ))}
                </div>
                <span className="text-[var(--color-on-surface-secondary)]">G switches</span>
                {isSecond && (
                  <>
                    <span className="text-[var(--color-on-surface-secondary)]">·</span>
                    {SECOND_KINDS.map((kind) => (
                      <label
                        key={kind}
                        className="flex items-center gap-1.5 cursor-pointer text-[var(--color-on-surface-secondary)]"
                      >
                        <input
                          type="checkbox"
                          checked={secondKinds.has(kind)}
                          onChange={() => toggleSecondKind(kind)}
                        />
                        {kind} ({secondCounts[kind]})
                      </label>
                    ))}
                    {secondView === "walk" && (
                      <span className="text-[var(--color-on-surface-secondary)]">
                        · {order.length} of {unfilteredOrder2.length} waiting
                      </span>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="ml-auto"
                      onClick={() => setClearSecondOpen(true)}
                      disabled={busy || secondHandled === 0}
                      title="Put every image this round has handled (skipped, or confirmed and decided since) back in the queue"
                    >
                      <RefreshCw className="w-3.5 h-3.5" /> Clear second round ({secondHandled})
                    </Button>
                  </>
                )}
              </div>
              {isSecond && secondView === "grid" ? (
                <SecondRoundGrid
                  pool={secondPool}
                  busy={busy}
                  fixPairDir={status?.fix_pair_dir ?? ""}
                  onOpen={openSecond}
                  onConfirm={(stems) => stems.length > 0 && setBulkStems(stems)}
                  onSkip={skipMany}
                  onAddFixPairs={addFixPairs}
                />
              ) : !isSecond && firstView === "grid" ? (
                <TriageGrid
                  order={order}
                  tags={tags}
                  categories={categories}
                  busy={busy}
                  onOpen={openFirst}
                  onConfirm={(stems) => stems.length > 0 && setBulkStems(stems)}
                  onSkip={skipManyFirst}
                />
              ) : current ? (
                <>
                  <div className="flex items-center justify-between gap-3 text-xs text-[var(--color-on-surface-secondary)]">
                    <span>
                      <span className="text-[var(--color-on-surface)] font-semibold">{current.stem}</span> ·{" "}
                      {safeCursor + 1} / {order.length} {isSecond ? "in the second round" : "to triage"}
                      {current.width && current.height && ` · ${current.width}×${current.height}`}
                      {current.detected && " · detector pre-tagged"}
                      {isSecond && current.stem === pinned2 && (
                        <span className="ml-2 px-1.5 rounded border border-[var(--color-border-subtle)]">
                          already handled · opened from the grid
                        </span>
                      )}
                      {isSecond && secondKind(current) && (
                        <span className="ml-2 px-1.5 rounded border border-[var(--color-border-subtle)]">
                          {secondKind(current)}
                        </span>
                      )}
                      {current.round > 1 && (
                        <span className="ml-2 px-1.5 rounded bg-[var(--color-cobalt-600)] text-white">
                          round {current.round}
                        </span>
                      )}
                      {!isSecond && current.attempt > 0 && current.round <= 1 && (
                        <span className="ml-2 px-1.5 rounded border border-[var(--color-border-subtle)]">
                          back from review
                        </span>
                      )}
                    </span>
                    <span className="flex items-center gap-2">
                      {!isSecond && (
                        <>
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => setConfirmAllOpen(true)}
                            disabled={busy || narrowed}
                            title={
                              narrowed
                                ? "Confirm all sends every image still in triage, not just the ones shown. Clear the keyword filter and phase-two toggle first."
                                : undefined
                            }
                          >
                            Confirm all ({unfilteredOrder.length})…
                          </Button>
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => void confirmAll("detected")}
                            disabled={busy || !anyDetected || narrowed}
                            title={
                              narrowed
                                ? "Runs on every image, not just the ones shown. Clear the keyword filter and phase-two toggle first."
                                : anyDetected
                                  ? "Queue every image with detector tags"
                                  : "No detector output on this dataset"
                            }
                          >
                            Confirm all detected
                          </Button>
                        </>
                      )}
                      <Button variant="primary" size="sm" onClick={goCompare}>
                        Finish triage → Review fixes
                      </Button>
                    </span>
                  </div>
                  {current.caption && <PromptExpander prompt={current.caption} />}

                  <div
                    className="relative flex-1 min-h-0 rounded overflow-hidden"
                    style={{ background: "rgba(0, 0, 0, 0.95)" }}
                  >
                    <div
                      className="w-full h-full flex items-center justify-center overflow-hidden cursor-zoom-in select-none"
                      onClick={() => setPreview(current.chosen_path)}
                    >
                      <img
                        key={current.stem}
                        src={dpoApi.imageUrl(current.chosen_path)}
                        alt={current.stem}
                        className="w-full h-full object-contain"
                        fetchPriority="high"
                        draggable={false}
                      />
                    </div>
                    <div
                      className="absolute top-0 left-0 right-0 z-10"
                      onMouseEnter={() => setStripOpen(true)}
                      onMouseLeave={() => setStripOpen(false)}
                    >
                      <div className="h-6 flex items-center justify-center cursor-pointer bg-gradient-to-b from-black/70 to-transparent">
                        <ChevronDown
                          className={`w-4 h-4 text-white/60 transition-transform ${stripOpen ? "rotate-180" : ""}`}
                        />
                      </div>
                      {stripOpen && (
                        <div className="flex gap-1.5 overflow-x-auto px-2 py-2 bg-black/85">
                          {order.slice(safeCursor, safeCursor + 14).map((item, i) => (
                            <div
                              key={item.stem}
                              className={`relative shrink-0 h-20 w-20 rounded border-2 overflow-hidden bg-black/40 cursor-pointer ${
                                i === 0 ? "ring-2 ring-[var(--color-cobalt-600)]" : ""
                              }`}
                              style={{
                                borderColor:
                                  (tags[item.stem]?.ids.size ?? 0) > 0 ? "#22c55e" : "var(--color-border-subtle)",
                              }}
                              title={item.stem}
                              onClick={() => setCursor(safeCursor + i)}
                            >
                              <img
                                src={dpoApi.imageUrl(item.chosen_path)}
                                alt={item.stem}
                                loading="lazy"
                                className="w-full h-full object-cover"
                                draggable={false}
                              />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 px-4 py-2 bg-black/70 text-white text-xs flex items-center justify-between pointer-events-none">
                      <span className="font-mono truncate">{basename(current.chosen_path)}</span>
                      <span className="text-white/60 tabular-nums">
                        {safeCursor + 1} / {order.length}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-center gap-2 flex-wrap text-sm shrink-0">
                    {categories.map((cat) =>
                      cat.id === CATEGORY_MASK
                        ? chip(cat, false, () => setMaskOpen(true), "paint")
                        : cat.id === CATEGORY_OTHER
                          ? chip(
                              cat,
                              currentTags.ids.has(cat.id),
                              () => setOtherOpen(true),
                              currentTags.others.length > 0
                                ? currentTags.others.map((o) => `“${o.target}”`).join(", ")
                                : undefined,
                            )
                          : chip(cat, currentTags.ids.has(cat.id), () =>
                              setCurrentTags(toggleTag(currentTags, cat.id)),
                            ),
                    )}
                  </div>
                  <div className="flex items-center justify-center gap-2 flex-wrap text-sm shrink-0">
                    <div
                      role="button"
                      tabIndex={-1}
                      onClick={() => void confirmCurrent()}
                      className="px-4 py-2 rounded cursor-pointer font-semibold flex items-center gap-1"
                      style={{ color: "#fff", background: "var(--color-cobalt-600)" }}
                    >
                      <Check className="w-4 h-4" /> Confirm &amp; {held ? "queue" : "send"} — Enter
                    </div>
                    <div
                      role="button"
                      tabIndex={-1}
                      onClick={() => void skipCurrent()}
                      className="px-4 py-2 rounded cursor-pointer font-semibold flex items-center gap-1 text-[var(--color-on-surface-secondary)] border border-[var(--color-border-subtle)]"
                    >
                      <SkipForward className="w-4 h-4" /> {isSecond ? "Skip this round" : "Skip forever"} — S
                    </div>
                    <div
                      role="button"
                      tabIndex={-1}
                      onClick={() => setDeleteOpen(true)}
                      className="px-4 py-2 rounded cursor-pointer font-semibold flex items-center gap-1 text-[var(--color-error-500)] border border-[var(--color-error-500)]"
                    >
                      <Trash2 className="w-4 h-4" /> Delete pair — D
                    </div>
                    <div
                      role="button"
                      tabIndex={-1}
                      onClick={() => void (isSecond ? undoSecond() : undoTriage())}
                      className={`px-4 py-2 rounded font-semibold flex items-center gap-1 text-[var(--color-on-surface-secondary)] border border-[var(--color-border-subtle)] ${
                        (isSecond ? secondUndo : triageUndo).length === 0
                          ? "opacity-40 cursor-default"
                          : "cursor-pointer"
                      }`}
                    >
                      <Undo2 className="w-4 h-4" /> Undo — Backspace
                    </div>
                    <div className="px-4 py-2 rounded font-semibold text-[var(--color-on-surface-secondary)] border border-[var(--color-border-subtle)]">
                      ← / → browse · Space preview · {otherKey} Other · {maskKey} Quick Mask
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center gap-3 text-sm text-[var(--color-on-surface-secondary)]">
                  <span>
                    {isSecond
                      ? unfilteredOrder2.length > 0
                        ? `Nothing waiting matches the filters (${unfilteredOrder2.length} waiting in the second round).`
                        : "Every fixed, skipped and kept image has been through the second round."
                      : filtering && unfilteredOrder.length > 0
                        ? `No image left to triage matches these keywords (${unfilteredOrder.length} without the filter).`
                        : "Every image has been triaged."}
                  </span>
                  <Button variant="primary" onClick={goCompare}>
                    Review fixes ({counts?.ready ?? 0} ready
                    {counts &&
                      counts.queued + counts.generating > 0 &&
                      `, ${counts.queued + counts.generating} generating`}
                    )
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* -------------------------------------------------------------- compare */}
          {screen === "compare" && active && (
            <div className="flex flex-col gap-3 flex-1 min-h-0">
              {ready ? (
                <>
                  <div className="bg-[var(--color-surface-container)] rounded p-3 flex flex-col gap-1">
                    <div className="text-xs text-[var(--color-on-surface-secondary)]">
                      <span className="text-[var(--color-on-surface)] font-semibold">{ready.stem}</span>
                      {ready.attempt > 1 && ` · attempt ${ready.attempt}`}
                      {" · fixes: "}
                      {ready.fixes.map((f) => (f.id === CATEGORY_OTHER ? `other (${f.target})` : f.id)).join(", ")}
                      {` · 1–9 pick a candidate · 0 keep the original · R re-queue · Shift+R re-queue with other fixes · ${maskKey} Quick Mask · S back to triage · D delete pair · Ctrl+Z undo`}
                    </div>
                    {ready.caption && (
                      <div className="text-sm text-[var(--color-on-surface)] whitespace-pre-wrap">{ready.caption}</div>
                    )}
                  </div>
                  {ready.all_no_change && (
                    <div className="rounded px-2 py-1 text-xs font-bold tracking-wide text-center bg-[var(--color-warning-500,#eab308)] text-black">
                      NO CHANGE — segmentation found nothing to fix in every candidate; re-queue with a different target
                      or keep the original
                    </div>
                  )}
                  <div
                    className={`grid gap-3 ${ready.candidates.length + 1 <= 3 ? "grid-cols-3" : ready.candidates.length + 1 === 4 ? "grid-cols-4" : "grid-cols-5"}`}
                  >
                    <figure
                      className="flex flex-col min-h-0 bg-[var(--color-surface-container)] rounded p-2 ring-1 ring-[var(--color-border-subtle)] cursor-zoom-in"
                      onClick={() => setPreview(ready.original_path)}
                    >
                      <figcaption className="text-xs font-semibold mb-1 flex items-center gap-2">
                        <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-[var(--color-border-subtle)] text-[10px]">
                          0
                        </span>
                        <span className="text-[var(--color-on-surface-secondary)]">ORIGINAL</span>
                        {ready.mask && (
                          <span
                            role="button"
                            tabIndex={-1}
                            onClick={(e) => {
                              e.stopPropagation();
                              setShowMask((v) => !v);
                            }}
                            className={`ml-auto px-1.5 rounded text-[10px] border cursor-pointer ${
                              showMask
                                ? "bg-[var(--color-cobalt-600)] text-white border-[var(--color-cobalt-600)]"
                                : "text-[var(--color-on-surface-secondary)] border-[var(--color-border-subtle)]"
                            }`}
                          >
                            mask
                          </span>
                        )}
                      </figcaption>
                      <div className="relative">
                        <img
                          src={dpoApi.imageUrl(ready.original_path)}
                          alt="original"
                          className="h-[48vh] min-h-[220px] object-contain w-full"
                        />
                        {ready.mask && showMask && (
                          <img
                            src={dpoApi.imageUrl(ready.mask)}
                            alt="mask"
                            className="absolute inset-0 h-[48vh] min-h-[220px] object-contain w-full pointer-events-none"
                            style={{ opacity: 0.45, mixBlendMode: "screen" }}
                          />
                        )}
                      </div>
                      <div className="text-[10px] text-[var(--color-on-surface-secondary)] mt-1">press 0 to keep</div>
                    </figure>
                    {ready.candidates.map((candidate) => {
                      const usable = !candidate.error && Boolean(candidate.path);
                      return (
                        <figure
                          key={candidate.k}
                          className={`flex flex-col min-h-0 rounded p-2 ring-1 ${
                            candidate.no_change
                              ? "bg-[color-mix(in_srgb,var(--color-surface-container)_80%,#eab308_20%)]"
                              : "bg-[var(--color-surface-container)]"
                          } ${usable ? "ring-[var(--color-border-subtle)] cursor-zoom-in" : "ring-[var(--color-error-500)]"}`}
                          onClick={() => {
                            if (candidate.path) setPreview(candidate.path);
                          }}
                          onContextMenu={(e) => {
                            e.preventDefault();
                            if (usable) void pick(candidate.k);
                          }}
                          title={
                            usable
                              ? "Left-click: full size · Right-click or number key: make this the chosen"
                              : (candidate.error ?? "")
                          }
                        >
                          <figcaption className="text-xs font-semibold mb-1 flex items-center gap-2">
                            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-[var(--color-cobalt-600)] text-white text-[10px]">
                              {candidate.k + 1}
                            </span>
                            <span className="text-[var(--color-on-surface)]">
                              v{candidate.k + 1} · {candidate.mode ?? "—"}
                              {candidate.fallback ? ` · via ${candidate.fallback}` : ""}
                            </span>
                            {candidate.no_change && (
                              <span className="text-[10px] font-bold text-black bg-[var(--color-warning-500,#eab308)] rounded px-1">
                                no change
                              </span>
                            )}
                          </figcaption>
                          {candidate.path ? (
                            <img
                              src={dpoApi.imageUrl(candidate.path)}
                              alt={`candidate ${candidate.k + 1}`}
                              className="h-[48vh] min-h-[220px] object-contain w-full"
                            />
                          ) : (
                            <div className="h-[48vh] min-h-[220px] flex items-center justify-center text-xs text-[var(--color-error-500)] px-2 text-center">
                              {candidate.error ?? "failed"}
                            </div>
                          )}
                          <div className="text-[10px] text-[var(--color-on-surface-secondary)] tabular-nums flex gap-x-3 mt-1">
                            {candidate.seed != null && <span>seed {candidate.seed}</span>}
                            {candidate.diff != null && (
                              <span
                                title={
                                  candidate.changed != null
                                    ? `${(candidate.changed * 100).toFixed(1)}% of pixels changed`
                                    : undefined
                                }
                              >
                                Δ {candidate.diff}
                                {candidate.changed != null && ` · ${(candidate.changed * 100).toFixed(1)}%`}
                              </span>
                            )}
                            {candidate.error && (
                              <span className="text-[var(--color-error-500)] truncate">{candidate.error}</span>
                            )}
                          </div>
                        </figure>
                      );
                    })}
                  </div>
                  <div className="flex items-center justify-center gap-2 flex-wrap text-xs">
                    <Button variant="secondary" size="sm" onClick={() => void keep()} disabled={busy}>
                      Keep original — 0
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={(e) => void requeue(ready.stem, undefined, e.shiftKey)}
                      disabled={busy}
                      title="Click: new seed only. Shift+click: new seed and creativity down one step."
                    >
                      <RefreshCw className="w-3.5 h-3.5" /> Re-roll (new seed) — R · Shift+click: lower creativity
                    </Button>
                    <Button variant="secondary" size="sm" onClick={() => setRequeueOpen(true)} disabled={busy}>
                      Re-queue with other fixes — Shift+R
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setMaskOpen(true)}
                      disabled={busy || !ready.width || !ready.height}
                    >
                      <Brush className="w-3.5 h-3.5" /> Quick Mask — {maskKey}
                    </Button>
                    <Button variant="secondary" size="sm" onClick={() => void retriage(ready.stem)} disabled={busy}>
                      <Undo2 className="w-3.5 h-3.5" /> Back to triage — S
                    </Button>
                    <Button variant="danger" size="sm" onClick={() => setDeleteOpen(true)} disabled={busy}>
                      <Trash2 className="w-3.5 h-3.5" /> Delete pair — D
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => lastFixed && void fixMore(lastFixed)}
                      disabled={busy || !lastFixed}
                      title={lastFixed ? `Another round on ${lastFixed}` : "Accept a fix first"}
                    >
                      <Wand2 className="w-3.5 h-3.5" /> Fix more on last fixed — F
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void undoDecision()}
                      disabled={busy || !status?.can_undo}
                      title={status?.undo_label ?? ""}
                    >
                      <Undo2 className="w-4 h-4" /> Undo last decision — Ctrl+Z
                    </Button>
                  </div>
                </>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center gap-3 text-sm text-[var(--color-on-surface-secondary)]">
                  {compareState === "waiting" && held && (counts?.generating ?? 0) === 0 ? (
                    <>
                      <span>{counts?.queued ?? 0} queued · sending is held until SwarmUI is free.</span>
                      <Button variant="primary" onClick={() => void toggleHold()}>
                        <Send className="w-4 h-4" /> Send to SwarmUI
                      </Button>
                    </>
                  ) : compareState === "waiting" ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {(counts?.queued ?? 0) + (counts?.generating ?? 0)} still generating… results appear here as they
                      land
                    </span>
                  ) : compareState === "done" ? (
                    <>
                      <span>Nothing left to review.</span>
                      {order.length > 0 && (
                        <Button variant="secondary" onClick={goTriage}>
                          Back to triage ({unfilteredOrder.length} left)
                        </Button>
                      )}
                    </>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" /> Loading…
                    </span>
                  )}
                  {status?.can_undo && (
                    <Button variant="ghost" size="sm" onClick={() => void undoDecision()} disabled={busy}>
                      <Undo2 className="w-4 h-4" /> Undo last decision — Ctrl+Z
                    </Button>
                  )}
                </div>
              )}

              {fixedHistory.length > 0 && (
                <div className="bg-[var(--color-surface-container)] rounded p-2 text-xs flex flex-col gap-1">
                  <span className="font-semibold text-[var(--color-on-surface)]">
                    Recently fixed · F = fix more on the newest
                  </span>
                  {fixedHistory.map((stem) => (
                    <div key={stem} className="flex items-center gap-2">
                      <span className="font-mono flex-1">{stem}</span>
                      <Button variant="ghost" size="sm" onClick={() => void fixMore(stem)} disabled={busy}>
                        Fix more
                      </Button>
                    </div>
                  ))}
                </div>
              )}

              {failedItems.length > 0 && (
                <div className="bg-[var(--color-surface-container)] rounded p-2 text-xs flex flex-col gap-1">
                  <span className="font-semibold text-[var(--color-error-500)]">{failedItems.length} failed</span>
                  {failedItems.slice(0, 8).map((item) => (
                    <div key={item.stem} className="flex items-center gap-2">
                      <span className="font-mono">{item.stem}</span>
                      <span className="text-[var(--color-on-surface-secondary)] truncate flex-1">{item.error}</span>
                      <Button variant="ghost" size="sm" onClick={() => void requeue(item.stem)} disabled={busy}>
                        Re-queue
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => void retriage(item.stem)} disabled={busy}>
                        Back to triage
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => void deletePair(item.stem)} disabled={busy}>
                        Delete pair
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {preview && <PreviewOverlay path={preview} onClose={() => setPreview(null)} />}
        {otherOpen && current && (
          <OtherDialog
            others={currentTags.others}
            categories={categories}
            onClose={() => setOtherOpen(false)}
            onSubmit={(rows) => {
              setCurrentTags(setOthers(currentTags, rows));
              setOtherOpen(false);
            }}
          />
        )}
        {confirmAllOpen && (
          <TagDialog
            title={`Confirm all ${unfilteredOrder.length} remaining images`}
            initial={currentTags.ids.size > 0 ? currentTags : (lastTags ?? emptyTags())}
            categories={categories}
            note="Every image still in triage is sent for fixing with these tags. Images with detector tags keep them if you use “Confirm all detected” instead."
            confirmLabel={`Send ${unfilteredOrder.length} images`}
            onClose={() => setConfirmAllOpen(false)}
            onSubmit={(t) => {
              setConfirmAllOpen(false);
              void confirmAll("default", t);
            }}
          />
        )}
        {maskOpen && maskTarget && maskTarget.width && maskTarget.height && (
          <QuickMaskDialog
            stem={maskTarget.stem}
            path={maskTarget.path}
            width={maskTarget.width}
            height={maskTarget.height}
            initial={lastMask ?? maskDefaults}
            defaults={maskDefaults}
            onSend={sendMask}
            onClose={() => setMaskOpen(false)}
          />
        )}
        {deleteOpen && deleteTarget && (
          <DeleteDialog
            stem={deleteTarget}
            onConfirm={() => void deletePair(deleteTarget)}
            onClose={() => setDeleteOpen(false)}
          />
        )}
        {clearSecondOpen && (
          <Dialog title="Clear the second round?" onClose={() => setClearSecondOpen(false)}>
            <div className="text-sm text-[var(--color-on-surface-secondary)] flex flex-col gap-2">
              <span>
                {secondHandled} image{secondHandled === 1 ? "" : "s"} the second round has handled (skipped there, or
                confirmed there and decided since) go back into its queue.
              </span>
              <span>
                Nothing else changes: repairs, fix pairs and the chosen images stay as they are, and images still queued
                or awaiting review are left alone. Second-round undo history is cleared.
              </span>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => setClearSecondOpen(false)}>
                Cancel
              </Button>
              <Button variant="primary" size="sm" onClick={() => void clearSecond()}>
                Clear second round
              </Button>
            </div>
          </Dialog>
        )}
        {bulkStems !== null && (
          <TagDialog
            title={`Confirm ${bulkStems.length} image${bulkStems.length === 1 ? "" : "s"} with…`}
            initial={lastTags ?? emptyTags()}
            categories={categories}
            confirmLabel="Confirm & send"
            onClose={() => setBulkStems(null)}
            onSubmit={(t) => void confirmMany(bulkStems, t)}
          />
        )}
        {requeueOpen && ready && (
          <TagDialog
            title={`Re-queue ${ready.stem} with other fixes`}
            initial={tagsFromFixes(ready.fixes)}
            categories={categories}
            confirmLabel="Re-queue"
            onClose={() => setRequeueOpen(false)}
            onSubmit={(t) => {
              setRequeueOpen(false);
              void requeue(ready.stem, tagsToFixes(t, categories));
            }}
          />
        )}
      </div>
    </YoloModelsContext.Provider>
  );
}

/** Triage & Fix: one tab per session, so fixers with different jobs (an
 *  anatomy model, Klein for the rest) each keep their own queue. Every tab
 *  stays mounted while the window is open, so a hidden session keeps polling
 *  and sending. */
export function TriageFixModal({ open, onClose }: TriageFixModalProps) {
  const [tabs, setTabs] = useState<TriageSessionTab[]>(loadTriageSessions);
  const [shown, setShown] = useState<string>(MAIN_TRIAGE_SESSION);
  const [running, setRunning] = useState<Record<string, boolean>>({});
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState("");

  const changeTabs = useCallback((update: (previous: TriageSessionTab[]) => TriageSessionTab[]) => {
    setTabs((previous) => {
      const next = update(previous);
      saveTriageSessions(next);
      return next;
    });
  }, []);

  // Sessions the backend is running that this window has no tab for (another window, a reload).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void triageSessionsApi
      .list()
      .then((res) => {
        if (cancelled || !res.ok) return;
        const found = res.sessions.filter((s) => s.active);
        changeTabs((previous) => {
          const known = new Set(previous.map((tab) => tab.id));
          const extra = found.filter((s) => !known.has(s.id)).map((s) => ({ id: s.id, name: s.name || s.id }));
          return extra.length > 0 ? [...previous, ...extra] : previous;
        });
      })
      .catch(() => {
        /* backend not up yet */
      });
    return () => {
      cancelled = true;
    };
  }, [open, changeTabs]);

  const onActiveChange = useCallback((session: string, active: boolean) => {
    setRunning((previous) => (previous[session] === active ? previous : { ...previous, [session]: active }));
  }, []);

  const addTab = useCallback(() => {
    const name = newName.trim();
    if (!name) return;
    const id = sessionIdFor(
      name,
      tabs.map((tab) => tab.id),
    );
    changeTabs((previous) => [...previous, { id, name }]);
    setShown(id);
    setAdding(false);
    setNewName("");
  }, [newName, tabs, changeTabs]);

  const removeTab = useCallback(
    (id: string) => {
      if (id === MAIN_TRIAGE_SESSION || running[id]) return;
      changeTabs((previous) => previous.filter((tab) => tab.id !== id));
      setShown((current) => (current === id ? MAIN_TRIAGE_SESSION : current));
    },
    [running, changeTabs],
  );

  const anyRunning = Object.values(running).some(Boolean);
  const visible = tabs.some((tab) => tab.id === shown) ? shown : MAIN_TRIAGE_SESSION;

  return (
    <ModalBase open={open} onClose={onClose} title="Triage & Fix" size="full" closeOnEscape={!anyRunning}>
      <div className="flex flex-col gap-3 h-full min-h-0">
        <div className="flex items-center gap-1 flex-wrap text-xs border-b border-[var(--color-border-subtle)] pb-2">
          {tabs.map((tab) => {
            const on = tab.id === visible;
            return (
              <div
                key={tab.id}
                className={`flex items-center rounded border ${
                  on
                    ? "bg-[var(--color-cobalt-600)] text-white border-[var(--color-cobalt-600)]"
                    : "text-[var(--color-on-surface-secondary)] border-[var(--color-border-subtle)]"
                }`}
              >
                <button
                  type="button"
                  onClick={() => setShown(tab.id)}
                  className="px-3 py-1.5 font-semibold flex items-center gap-1.5"
                  title={running[tab.id] ? "Running" : "Not running"}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${running[tab.id] ? "bg-[var(--color-success-500,#22c55e)]" : "bg-[var(--color-border-subtle)]"}`}
                  />
                  {tab.name}
                </button>
                {tab.id !== MAIN_TRIAGE_SESSION && !running[tab.id] && (
                  <button
                    type="button"
                    onClick={() => removeTab(tab.id)}
                    className="pr-2 opacity-70 hover:opacity-100"
                    title="Close this tab. Its ledger stays in the dataset; a tab with the same name picks it up again."
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            );
          })}
          {adding ? (
            <span className="flex items-center gap-1">
              <input
                autoFocus
                className={`${INPUT_FLEX} w-48`}
                placeholder="Session name, e.g. Anatomy"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") addTab();
                  if (e.key === "Escape") {
                    e.stopPropagation();
                    setAdding(false);
                  }
                }}
              />
              <Button variant="primary" size="sm" onClick={addTab} disabled={!newName.trim()}>
                Add
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setAdding(false)}>
                Cancel
              </Button>
            </span>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAdding(true)}
              title="Another session with its own fixer model, settings and queue. An image queued in two sessions is repaired in the one that queued it first; the other takes the accepted result."
            >
              <Plus className="w-3.5 h-3.5" /> New session
            </Button>
          )}
        </div>
        {tabs.map((tab) => (
          <TriageFixSession
            key={tab.id}
            open={open}
            session={tab.id}
            name={tab.name}
            visible={tab.id === visible}
            onActiveChange={onActiveChange}
          />
        ))}
      </div>
    </ModalBase>
  );
}
