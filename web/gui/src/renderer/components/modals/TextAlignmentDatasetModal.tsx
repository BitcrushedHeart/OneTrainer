import { useEffect, useRef, useState } from "react";

import { textAlignmentApi, type TextAlignmentCandidate, type TextAlignmentStatus } from "@/api/textAlignmentApi";
import { Button, DirPicker, FormEntry, ProgressBar, Toggle } from "@/components/shared";

import { ModalBase } from "./ModalBase";

interface Props {
  open: boolean;
  onClose: () => void;
}

const REASON_LABELS: Record<string, string> = {
  keep: "Kept (rendered but wrong)",
  verbatim: "Skipped: text already verbatim",
  absent: "Skipped: text not rendered",
  no_boxes: "Skipped: detector boxed no target word",
  no_text_span: "Skipped: caption asks for no text",
  no_metadata: "Skipped: no SwarmUI prompt in metadata",
  duplicate: "Skipped: duplicate image",
  already_in_dataset: "Skipped: already in dataset",
  unreadable: "Skipped: OCR could not read",
};

const DEFAULT_WILDCARD_DIR = "E:\\AI\\Data\\Packages\\SwarmUI\\Data\\Wildcards";

export function TextAlignmentDatasetModal({ open, onClose }: Props) {
  const [sourceFolder, setSourceFolder] = useState("");
  const [outputFolder, setOutputFolder] = useState("");
  const [recursive, setRecursive] = useState(true);
  const [keepAbsent, setKeepAbsent] = useState(false);
  const [keepVerbatim, setKeepVerbatim] = useState(false);
  const [minBoxedWords, setMinBoxedWords] = useState<string | number>(1);
  const [minAgreement, setMinAgreement] = useState<string | number>(0.8);
  const [limit, setLimit] = useState<string | number>(0);
  const [status, setStatus] = useState<TextAlignmentStatus | null>(null);
  const [items, setItems] = useState<TextAlignmentCandidate[]>([]);
  const [message, setMessage] = useState<{ kind: "info" | "error" | "success"; text: string }>({
    kind: "info",
    text: "Scan a SwarmUI output folder to preview which generations qualify.",
  });

  const [wcCount, setWcCount] = useState<string | number>(1000);
  const [wcSeed, setWcSeed] = useState<string | number>(0);
  const [wcName, setWcName] = useState("text_alignment");
  const [wcDir, setWcDir] = useState(DEFAULT_WILDCARD_DIR);
  const [wcMinWords, setWcMinWords] = useState<string | number>(1);
  const [wcMaxWords, setWcMaxWords] = useState<string | number>(8);
  const [wcMessage, setWcMessage] = useState<string>("");

  const pollRef = useRef<number | null>(null);
  const running = status?.running ?? false;

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const refreshResults = async () => {
    try {
      const res = await textAlignmentApi.results(0, 500, true);
      setItems(res.items);
    } catch {
      setItems([]);
    }
  };

  const poll = async () => {
    try {
      const s = await textAlignmentApi.status();
      setStatus(s);
      if (!s.running) {
        stopPolling();
        if (s.phase === "scanned") {
          setMessage({ kind: "success", text: `Scanned ${s.scanned} image(s); ${s.kept} qualify.` });
          await refreshResults();
        } else if (s.phase === "built" && s.build) {
          setMessage({
            kind: "success",
            text: `Wrote ${s.build.written} sample(s) to ${s.output_folder ?? "the dataset folder"} (${s.build.skipped} skipped).`,
          });
        } else if (s.phase === "error") {
          setMessage({ kind: "error", text: s.error ?? "Failed." });
        } else if (s.phase === "cancelled") {
          setMessage({ kind: "info", text: "Cancelled." });
          await refreshResults();
        }
      }
    } catch (err) {
      stopPolling();
      setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
    }
  };

  const startPolling = () => {
    stopPolling();
    pollRef.current = window.setInterval(() => void poll(), 700);
    void poll();
  };

  useEffect(() => {
    if (open) {
      void poll();
    } else {
      stopPolling();
    }
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const scan = async () => {
    if (!sourceFolder.trim() || !outputFolder.trim()) {
      setMessage({ kind: "error", text: "Source and output folders are required." });
      return;
    }
    setItems([]);
    setMessage({ kind: "info", text: "Scanning (the OCR reads every candidate; this takes a while)..." });
    try {
      const res = await textAlignmentApi.scan({
        source_folder: sourceFolder,
        output_folder: outputFolder,
        recursive,
        keep_absent: keepAbsent,
        keep_verbatim: keepVerbatim,
        min_boxed_words: Number(minBoxedWords) || 0,
        min_agreement: Number(minAgreement) || 0,
        limit: Number(limit) || 0,
        ocr_device: "auto",
        detector_device: "cpu",
      });
      if (!res.ok) {
        setMessage({ kind: "error", text: res.error ?? "Scan failed to start." });
        return;
      }
      startPolling();
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
    }
  };

  const build = async () => {
    setMessage({ kind: "info", text: "Building dataset..." });
    try {
      const res = await textAlignmentApi.build();
      if (!res.ok) {
        setMessage({ kind: "error", text: res.error ?? "Build failed to start." });
        return;
      }
      startPolling();
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof Error ? err.message : String(err) });
    }
  };

  const cancel = async () => {
    try {
      await textAlignmentApi.cancel();
    } catch {
      /* status poll will surface the outcome */
    }
  };

  const writeWildcard = async () => {
    setWcMessage("Writing...");
    try {
      const res = await textAlignmentApi.wildcard({
        count: Number(wcCount) || 1000,
        seed: Number(wcSeed) || 0,
        name: wcName.trim() || "text_alignment",
        directory: wcDir.trim() || DEFAULT_WILDCARD_DIR,
        min_words: Number(wcMinWords) || 1,
        max_words: Number(wcMaxWords) || 8,
      });
      if (!res.ok) {
        setWcMessage(res.error ?? "Failed.");
        return;
      }
      setWcMessage(
        `Wrote ${res.count} prompts to ${res.path}. Use <wildcard:${wcName.trim() || "text_alignment"}> in SwarmUI.`,
      );
    } catch (err) {
      setWcMessage(err instanceof Error ? err.message : String(err));
    }
  };

  const messageColor =
    message.kind === "error"
      ? "var(--color-error-500)"
      : message.kind === "success"
        ? "var(--color-success-500)"
        : "var(--color-on-surface-secondary)";

  const progress = status && status.progress_total > 0 ? (100 * status.progress_done) / status.progress_total : 0;
  const counts = status?.counts ?? {};
  const countRows = Object.entries(counts).filter(([, n]) => n > 0);
  // A cancelled scan still holds a usable plan: build what it classified so far.
  const canBuild =
    !running && (status?.phase === "scanned" || status?.phase === "cancelled") && (status?.kept ?? 0) > 0;

  return (
    <ModalBase open={open} onClose={onClose} title="Text Alignment Dataset" size="2xl" closeOnBackdrop={!running}>
      <div className="flex flex-col gap-5">
        <p className="text-sm text-[var(--color-on-surface-secondary)]">
          Reads the prompt SwarmUI stamped into each generation, checks the on-image text with the curation OCR, and
          keeps only images whose text is rendered but wrong: the ones the offline text reward can learn from. Verbatim
          renderings and missing text are skipped.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DirPicker label="SwarmUI Output Folder" value={sourceFolder} onChange={setSourceFolder} disabled={running} />
          <DirPicker label="Dataset Output Folder" value={outputFolder} onChange={setOutputFolder} disabled={running} />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 items-end">
          <Toggle label="Recursive" value={recursive} onChange={setRecursive} disabled={running} />
          <Toggle
            label="Keep verbatim"
            tooltip="Also admit images whose text is already correct (nothing to correct; off by default)."
            value={keepVerbatim}
            onChange={setKeepVerbatim}
            disabled={running}
          />
          <Toggle
            label="Keep absent"
            tooltip="Also admit images where the text was not rendered at all (no box, no gradient; off by default)."
            value={keepAbsent}
            onChange={setKeepAbsent}
            disabled={running}
          />
          <FormEntry
            label="Min boxed words"
            type="number"
            tooltip="Target words the training-time detector must box for the image to qualify."
            value={minBoxedWords}
            onChange={setMinBoxedWords}
            disabled={running}
          />
          <FormEntry
            label="Min reader agreement"
            type="number"
            tooltip="Minimum agreement (0-1) between the curation OCR (PaddleOCR-VL) and the training-time reader (PARSeq) on the boxed words. Below it the image is skipped as reader_disagree so the reward gradient never points at a word the small reader misread. 0 = off."
            value={minAgreement}
            onChange={setMinAgreement}
            disabled={running}
          />
          <FormEntry
            label="Limit (0 = all)"
            type="number"
            tooltip="Only scan the first N images, for a quick preview."
            value={limit}
            onChange={setLimit}
            disabled={running}
          />
        </div>

        <div className="flex items-center justify-between gap-3">
          <span className="text-sm" style={{ color: messageColor }}>
            {message.text}
          </span>
          <div className="flex gap-2">
            {running && (
              <Button variant="secondary" onClick={cancel}>
                Cancel
              </Button>
            )}
            <Button variant="secondary" onClick={scan} disabled={running}>
              Scan (dry run)
            </Button>
            <Button onClick={build} disabled={!canBuild}>
              Build Dataset
            </Button>
          </div>
        </div>

        {running && (
          <ProgressBar
            value={progress}
            label={`${status?.phase === "building" ? "Writing" : "Scanning"} ${status?.progress_done ?? 0} / ${status?.progress_total ?? 0}`}
          />
        )}

        {countRows.length > 0 && (
          <div className="rounded-md border border-[var(--color-border-subtle)] p-3 text-sm text-[var(--color-on-surface)]">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-1">
              {countRows.map(([reason, n]) => (
                <div key={reason} className={reason === "keep" ? "font-semibold" : ""}>
                  {REASON_LABELS[reason] ?? reason}: {n}
                </div>
              ))}
            </div>
          </div>
        )}

        {items.length > 0 && (
          <div className="max-h-[360px] overflow-auto rounded-md border border-[var(--color-border-subtle)]">
            <table className="w-full text-xs text-[var(--color-on-surface)]">
              <thead className="sticky top-0 bg-[var(--color-surface)]">
                <tr className="text-left">
                  <th className="p-2">Image</th>
                  <th className="p-2">Target text</th>
                  <th className="p-2">OCR read</th>
                  <th className="p-2">Reward read</th>
                  <th className="p-2">Agree</th>
                  <th className="p-2">Score</th>
                  <th className="p-2">Boxed</th>
                </tr>
              </thead>
              <tbody>
                {items.map((c) => (
                  <tr key={c.sha256} className="border-t border-[var(--color-border-subtle)] align-top">
                    <td className="p-2">
                      <img
                        src={textAlignmentApi.imageUrl(c.source_image)}
                        alt=""
                        className="w-20 h-20 object-cover rounded"
                        loading="lazy"
                        title={c.prompt}
                      />
                    </td>
                    <td className="p-2 whitespace-pre-wrap">{c.text}</td>
                    <td className="p-2 whitespace-pre-wrap">{c.ocr_reading ?? ""}</td>
                    <td className="p-2 whitespace-pre-wrap">{c.reward_reading ?? ""}</td>
                    <td className="p-2">
                      {c.reader_agreement === null || c.reader_agreement === undefined
                        ? ""
                        : c.reader_agreement.toFixed(2)}
                    </td>
                    <td className="p-2">{c.text_score === null ? "" : c.text_score.toFixed(2)}</td>
                    <td className="p-2">
                      {c.boxed_words} / {c.words.length}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="rounded-md border border-[var(--color-border-subtle)] p-3 flex flex-col gap-3">
          <div className="text-sm font-semibold text-[var(--color-on-surface)]">Write Wildcard for SwarmUI</div>
          <p className="text-xs text-[var(--color-on-surface-secondary)]">
            Writes reward prompts (mixed-case words, prices, dates, times, URLs, abbreviations, numbers, non-words) one
            per line into a SwarmUI wildcard file, so the offline set is drawn from the same distribution as the online
            reward.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 items-end">
            <FormEntry label="Count" type="number" value={wcCount} onChange={setWcCount} />
            <FormEntry label="Seed" type="number" value={wcSeed} onChange={setWcSeed} />
            <FormEntry label="Min words" type="number" value={wcMinWords} onChange={setWcMinWords} />
            <FormEntry label="Max words" type="number" value={wcMaxWords} onChange={setWcMaxWords} />
            <FormEntry label="Name" value={wcName} onChange={(v) => setWcName(String(v))} />
            <Button variant="secondary" onClick={writeWildcard}>
              Write Wildcard
            </Button>
          </div>
          <DirPicker label="Wildcards Folder" value={wcDir} onChange={setWcDir} />
          {wcMessage && <span className="text-xs text-[var(--color-on-surface-secondary)]">{wcMessage}</span>}
        </div>
      </div>
    </ModalBase>
  );
}
