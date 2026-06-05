import { Loader2 } from "lucide-react";

import { Button } from "@/components/shared";

import type { ScanningStepProps } from "./types";

export function ScanningStep({ scanCount, scanTotal, hashCount, cacheHits, onCancel }: ScanningStepProps) {
  return (
    <div className="text-center py-12 space-y-4">
      <Loader2 className="w-10 h-10 mx-auto animate-spin text-[var(--color-cobalt-600)]" />
      <p className="text-base font-semibold text-[var(--color-on-surface)]">Scanning and deduplicating...</p>
      <p className="text-sm text-[var(--color-on-surface-secondary)]">
        {scanCount}
        {scanTotal > 0 ? ` / ${scanTotal}` : ""} files scanned
        {hashCount > 0 ? ` · ${hashCount} hashed` : ""}
        {cacheHits > 0 ? ` · ${cacheHits} from cache` : ""}
      </p>
      <div className="flex justify-center">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
