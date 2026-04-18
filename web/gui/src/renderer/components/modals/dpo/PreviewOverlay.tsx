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
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const src = `${API_BASE}/dpo/session/image?path=${encodeURIComponent(path)}`;
  const fileName = caption ?? path.split(/[/\\]/).pop() ?? path;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
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
      <div className="flex-1 flex items-center justify-center w-full overflow-hidden px-6 py-6">
        <img
          src={src}
          alt={fileName}
          className="max-w-full max-h-full object-contain"
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
