import { AlertCircle, Loader2, Play } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { type BucketAnalysisResult, dpoApi } from "@/api/dpoApi";
import { Button, DirPicker, FormEntry, Select } from "@/components/shared";
import { useConfigField } from "@/hooks/useConfigField";

import { ModalBase } from "./ModalBase";

export interface BucketAnalysisModalProps {
  open: boolean;
  onClose: () => void;
}

// Mirrors MODEL_QUANTIZATION in modules/util/dpo_bucket_analysis_util.py.
// Checked against modules/dataLoader/*BaseDataLoader.py so the default matches
// what the data pipeline uses at train time.
const MODEL_QUANTIZATION: Record<string, number> = {
  STABLE_DIFFUSION_15: 8,
  STABLE_DIFFUSION_15_INPAINTING: 8,
  STABLE_DIFFUSION_20: 8,
  STABLE_DIFFUSION_20_BASE: 8,
  STABLE_DIFFUSION_20_INPAINTING: 8,
  STABLE_DIFFUSION_20_DEPTH: 8,
  STABLE_DIFFUSION_21: 8,
  STABLE_DIFFUSION_21_BASE: 8,
  STABLE_DIFFUSION_3: 64,
  STABLE_DIFFUSION_35: 64,
  STABLE_DIFFUSION_XL_10_BASE: 64,
  STABLE_DIFFUSION_XL_10_BASE_INPAINTING: 64,
  WUERSTCHEN_2: 128,
  STABLE_CASCADE_1: 128,
  PIXART_ALPHA: 16,
  PIXART_SIGMA: 16,
  FLUX_DEV_1: 64,
  FLUX_FILL_DEV_1: 64,
  FLUX_2: 64,
  SANA: 32,
  HUNYUAN_VIDEO: 64,
  HI_DREAM_FULL: 64,
  CHROMA_1: 64,
  QWEN: 64,
  Z_IMAGE: 64,
};

const QUANT_OPTIONS = ["8", "16", "32", "64", "128"];

function parseResolutions(resolution: string): number[] {
  if (!resolution) return [];
  const out: number[] = [];
  for (const token of resolution.split(",")) {
    const t = token.trim();
    if (!t) continue;
    if (t.toLowerCase().includes("x")) continue;
    const n = Number.parseInt(t, 10);
    if (!Number.isNaN(n) && n > 0) out.push(n);
  }
  return out;
}

export function BucketAnalysisModal({ open, onClose }: BucketAnalysisModalProps) {
  const [modelType] = useConfigField<string>("model_type");
  const [configResolution] = useConfigField<string>("resolution");
  const [configBatchSize] = useConfigField<number>("batch_size");

  const defaultQuant = String(MODEL_QUANTIZATION[modelType ?? ""] ?? 64);
  const parsed = useMemo(() => parseResolutions(configResolution ?? ""), [configResolution]);

  const targetOptions = useMemo(() => {
    const set = new Set<number>(parsed);
    set.add(512);
    set.add(768);
    set.add(1024);
    return Array.from(set)
      .sort((a, b) => a - b)
      .map(String);
  }, [parsed]);

  const [conceptPath, setConceptPath] = useState<string>("");
  const [batchSize, setBatchSize] = useState<number>(Math.max(1, configBatchSize ?? 2));
  const [target, setTarget] = useState<string>(parsed.length > 0 ? String(Math.min(...parsed)) : "512");
  const [quant, setQuant] = useState<string>(defaultQuant);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BucketAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(async () => {
    if (!conceptPath) {
      setError("Select a concept folder first.");
      return;
    }
    const bs = Number(batchSize);
    const tgt = Number.parseInt(target, 10);
    const q = Number.parseInt(quant, 10);
    if (!Number.isFinite(bs) || bs < 1) {
      setError("Batch size must be a positive integer.");
      return;
    }
    if (!Number.isFinite(tgt) || tgt < 1) {
      setError("Invalid target resolution.");
      return;
    }
    if (!Number.isFinite(q) || q < 1) {
      setError("Invalid quantization.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await dpoApi.bucketAnalysis(conceptPath, bs, [tgt], q);
      if (!res.ok || !res.result) {
        setError(res.error ?? "Analysis failed");
      } else {
        setResult(res.result);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [conceptPath, batchSize, target, quant]);

  const firstTarget = result?.targets[0];
  const statusLine = result
    ? firstTarget
      ? `${firstTarget.total_pairs} pairs across ${firstTarget.buckets.length} buckets — add ${firstTarget.total_add} OR remove ${firstTarget.total_remove} for bs=${result.batch_size} clean (target ${firstTarget.target}).`
      : "No target resolutions analyzed."
    : "";

  return (
    <ModalBase open={open} onClose={onClose} title="DPO Bucket / Batch-Size Analyzer" size="xl">
      <div className="flex flex-col gap-4">
        <div className="text-sm text-[var(--color-on-surface-secondary)]">
          Select a chosen-side concept folder. Shows how many pairs each aspect bucket holds and what to add or remove
          for clean batches at your batch size.
        </div>

        <div className="flex flex-col gap-3">
          <DirPicker label="Chosen-side concept folder" value={conceptPath} onChange={setConceptPath} />

          <div className="grid grid-cols-4 gap-3">
            <FormEntry label="Batch size" type="number" value={batchSize} onChange={(v) => setBatchSize(Number(v))} />
            <Select label="Target resolution" options={targetOptions} value={target} onChange={setTarget} />
            <Select label="Quantization" options={QUANT_OPTIONS} value={quant} onChange={setQuant} />
            <div className="flex items-end">
              <Button variant="primary" onClick={() => void runAnalysis()} disabled={loading}>
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                Analyze
              </Button>
            </div>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-error-500)] bg-[var(--color-surface-container)] rounded p-3">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        {result && (
          <>
            <div className="text-sm text-[var(--color-on-surface)]">
              Scanned {result.scanned} image(s)
              {result.unreadable > 0 && `, ${result.unreadable} unreadable`} — batch_size={result.batch_size},
              quantization={result.quantization}
            </div>

            {result.targets.map((td) => (
              <div
                key={td.target}
                className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface-container)]"
              >
                <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border-subtle)]">
                  <div className="text-sm font-semibold text-[var(--color-on-surface)]">
                    Target resolution: {td.target}
                  </div>
                  <div className="text-xs text-[var(--color-on-surface-secondary)]">
                    Pairs {td.total_pairs} &middot; Drops {td.total_drops} &middot; Add {td.total_add} &middot; Remove{" "}
                    {td.total_remove}
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase text-[var(--color-on-surface-secondary)]">
                        <th className="px-4 py-2 font-medium">Aspect</th>
                        <th className="px-4 py-2 font-medium">H &times; W</th>
                        <th className="px-4 py-2 font-medium text-right">Count</th>
                        <th className="px-4 py-2 font-medium text-right">Drops</th>
                        <th className="px-4 py-2 font-medium text-right">Add for 0-drop</th>
                        <th className="px-4 py-2 font-medium text-right">Remove for 0-drop</th>
                      </tr>
                    </thead>
                    <tbody>
                      {td.buckets.map((b) => (
                        <tr key={`${b.h}x${b.w}`} className="border-t border-[var(--color-border-subtle)]">
                          <td className="px-4 py-2">{b.aspect_label}</td>
                          <td className="px-4 py-2 font-mono text-xs text-[var(--color-on-surface-secondary)]">
                            {b.h} &times; {b.w}
                          </td>
                          <td className="px-4 py-2 text-right">{b.count}</td>
                          <td
                            className={`px-4 py-2 text-right ${
                              b.drops > 0 ? "text-[var(--color-error-500)] font-semibold" : ""
                            }`}
                          >
                            {b.drops}
                          </td>
                          <td className="px-4 py-2 text-right">{b.add}</td>
                          <td className="px-4 py-2 text-right">{b.remove}</td>
                        </tr>
                      ))}
                      <tr className="border-t border-[var(--color-border-subtle)] font-semibold bg-[var(--color-surface)]">
                        <td className="px-4 py-2">TOTAL</td>
                        <td className="px-4 py-2" />
                        <td className="px-4 py-2 text-right">{td.total_pairs}</td>
                        <td className="px-4 py-2 text-right">{td.total_drops}</td>
                        <td className="px-4 py-2 text-right">{td.total_add}</td>
                        <td className="px-4 py-2 text-right">{td.total_remove}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ))}

            {statusLine && (
              <div className="text-xs text-[var(--color-on-surface-secondary)] pt-2 border-t border-[var(--color-border-subtle)]">
                {statusLine}
              </div>
            )}
          </>
        )}
      </div>
    </ModalBase>
  );
}
