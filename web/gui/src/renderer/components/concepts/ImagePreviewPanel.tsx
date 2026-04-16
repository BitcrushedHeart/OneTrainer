import { ChevronLeft, ChevronRight, ImageIcon } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { configApi } from "@/api/configApi";
import { Button } from "@/components/shared";
import type { ConceptTextConfig } from "@/types/generated/config";

export interface ImagePreviewPanelProps {
  conceptPath: string;
  includeSubdirectories: boolean;
  textConfig: ConceptTextConfig;
}

interface ImageEntry {
  filename: string;
  path: string;
  caption: string | null;
}

export function ImagePreviewPanel({ conceptPath, includeSubdirectories, textConfig }: ImagePreviewPanelProps) {
  const [images, setImages] = useState<ImageEntry[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [conceptCaption, setConceptCaption] = useState<string | null>(null);

  // Load image list when concept path changes
  useEffect(() => {
    if (!conceptPath) {
      setImages([]);
      setCurrentIndex(0);
      return;
    }

    let cancelled = false;
    setLoading(true);

    configApi
      .conceptImages(conceptPath, includeSubdirectories)
      .then((result) => {
        if (cancelled) return;
        setImages(result.images);
        setCurrentIndex(0);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setImages([]);
        setCurrentIndex(0);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [conceptPath, includeSubdirectories]);

  // Load concept-level caption file when prompt_source is "concept"
  useEffect(() => {
    if (textConfig.prompt_source !== "concept" || !textConfig.prompt_path) {
      setConceptCaption(null);
      return;
    }

    let cancelled = false;
    configApi
      .conceptTextFile(textConfig.prompt_path)
      .then((result) => {
        if (!cancelled) setConceptCaption(result.content);
      })
      .catch(() => {
        if (!cancelled) setConceptCaption(null);
      });

    return () => {
      cancelled = true;
    };
  }, [textConfig.prompt_source, textConfig.prompt_path]);

  const handlePrev = useCallback(() => {
    setCurrentIndex((i) => Math.max(0, i - 1));
  }, []);

  const handleNext = useCallback(() => {
    setCurrentIndex((i) => Math.min(images.length - 1, i + 1));
  }, [images.length]);

  const currentImage = images[currentIndex] ?? null;

  // Determine caption to display
  let displayCaption = "";
  if (currentImage) {
    if (textConfig.prompt_source === "filename") {
      // Show filename without extension
      const stem = currentImage.filename.replace(/\.[^.]+$/, "");
      displayCaption = stem || "[Empty prompt]";
    } else if (textConfig.prompt_source === "concept") {
      displayCaption = conceptCaption ?? "[No concept file loaded]";
    } else {
      // "sample" — use companion .txt caption
      displayCaption = currentImage.caption ?? "[No caption file]";
    }
  }

  const imageUrl = currentImage ? configApi.conceptImageUrl(currentImage.path) : null;

  return (
    <div className="w-[300px] flex-shrink-0 flex flex-col gap-3">
      <div className="w-[300px] h-[300px] rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] flex items-center justify-center overflow-hidden">
        {loading ? (
          <span className="text-xs text-[var(--color-on-surface-secondary)]">Loading...</span>
        ) : imageUrl ? (
          <img src={imageUrl} alt={currentImage?.filename ?? ""} className="max-w-full max-h-full object-contain" />
        ) : (
          <ImageIcon className="w-16 h-16 text-[var(--color-on-surface-secondary)]" strokeWidth={1} />
        )}
      </div>

      <div className="flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={handlePrev} disabled={currentIndex <= 0 || images.length === 0}>
          <ChevronLeft className="w-4 h-4" />
        </Button>
        <span className="flex-1 text-center text-xs text-[var(--color-on-surface-secondary)]">
          {images.length > 0 ? `${currentIndex + 1} / ${images.length}` : "No images"}
        </span>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleNext}
          disabled={currentIndex >= images.length - 1 || images.length === 0}
        >
          <ChevronRight className="w-4 h-4" />
        </Button>
      </div>

      {currentImage && (
        <p className="text-xs text-[var(--color-on-surface-secondary)] break-all leading-snug px-1">
          {currentImage.filename}
        </p>
      )}

      <textarea
        readOnly
        value={displayCaption}
        className="w-full h-[150px] text-xs text-[var(--color-on-surface)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)] rounded-[var(--radius-sm)] p-2 resize-none"
        style={{ wordWrap: "break-word" }}
      />
    </div>
  );
}
