import { useEffect } from "react";
import { createPortal } from "react-dom";

import { API_BASE } from "@/api/request";

export interface PreviewOverlayProps {
  path: string;
  onClose: () => void;
  onPick?: () => void;
  caption?: string;
}

export function PreviewOverlay({ path, onClose, onPick, caption }: PreviewOverlayProps) {
  useEffect(() => {
    // Capture phase + stopPropagation so the enclosing ModalBase's
    // document-level Escape listener doesn't also fire and close the
    // whole DPO modal. Only the preview should dismiss on Esc.
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", handleKey, true);
    return () => window.removeEventListener("keydown", handleKey, true);
  }, [onClose]);

  const src = `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
  const fileName = caption ?? path.split(/[/\\]/).pop() ?? path;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  // The img element fills the container (so small images upscale to fit the
  // display via object-contain), which means letterbox bars are part of the
  // img itself. Hit-test against the painted image rect so clicking the bars
  // still closes the preview, matching the backdrop behavior.
  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (!img.naturalWidth || !img.naturalHeight) return;
    const rect = img.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const paintedWidth = img.naturalWidth * scale;
    const paintedHeight = img.naturalHeight * scale;
    const left = rect.left + (rect.width - paintedWidth) / 2;
    const top = rect.top + (rect.height - paintedHeight) / 2;
    if (e.clientX < left || e.clientX > left + paintedWidth || e.clientY < top || e.clientY > top + paintedHeight) {
      onClose();
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    if (onPick) {
      // Consumer's onPick is responsible for closing; we don't call onClose here
      // to avoid a double state-set if the consumer also invokes setPreviewPath(null).
      onPick();
    }
  };

  return createPortal(
    <div
      className="fixed inset-0 z-[60] flex flex-col items-center justify-center"
      style={{ background: "rgba(0, 0, 0, 0.95)" }}
      onClick={handleBackdropClick}
      onContextMenu={handleContextMenu}
    >
      <div
        className="flex-1 flex items-center justify-center w-full overflow-hidden px-6 py-6"
        onClick={handleBackdropClick}
      >
        <img
          src={src}
          alt={fileName}
          className="w-full h-full object-contain"
          onClick={handleImageClick}
          onContextMenu={handleContextMenu}
          draggable={false}
        />
      </div>
      <div className="w-full px-6 py-3 bg-black/70 text-white text-xs flex items-center justify-between">
        <span className="font-mono truncate">{fileName}</span>
        <span className="text-white/60">{onPick ? "Right-click to pick" : "Click backdrop or press Esc to close"}</span>
      </div>
    </div>,
    document.body,
  );
}
