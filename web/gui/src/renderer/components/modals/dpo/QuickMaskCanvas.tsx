import { useCallback, useEffect, useRef, useState } from "react";

export interface QuickMaskCanvasHandle {
  /** The mask as a PNG data URI at the image's native size; null when empty. */
  exportMask: () => string | null;
  clear: () => void;
}

export interface QuickMaskCanvasProps {
  src: string;
  /** Native size of the image; the export is rendered at exactly this size. */
  width: number;
  height: number;
  brushRadius: number;
  erase: boolean;
  onReady?: (handle: QuickMaskCanvasHandle) => void;
  /** Fires after every stroke so the parent can enable Send. */
  onChange?: (hasPaint: boolean) => void;
}

/**
 * A minimal painter: the image with a transparent canvas on top, sized to the
 * displayed image but backed by a native-resolution mask canvas. Strokes are
 * painted white on the native canvas and mirrored onto the display canvas,
 * so the export never depends on the on-screen scale.
 */
export function QuickMaskCanvas({ src, width, height, brushRadius, erase, onReady, onChange }: QuickMaskCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const displayRef = useRef<HTMLCanvasElement>(null);
  const nativeRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const paintedRef = useRef(false);
  const [box, setBox] = useState({ w: 0, h: 0 });

  // Fit the image into the container, keeping its aspect.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const rect = el.getBoundingClientRect();
      if (!width || !height || rect.width === 0 || rect.height === 0) return;
      const scale = Math.min(rect.width / width, rect.height / height);
      setBox({ w: Math.max(1, Math.floor(width * scale)), h: Math.max(1, Math.floor(height * scale)) });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [width, height]);

  // The native mask canvas lives off-screen for the dialog's lifetime.
  useEffect(() => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    nativeRef.current = canvas;
    paintedRef.current = false;
    return () => {
      nativeRef.current = null;
    };
  }, [width, height]);

  const redraw = useCallback(() => {
    const display = displayRef.current;
    const native = nativeRef.current;
    if (!display || !native) return;
    const ctx = display.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, display.width, display.height);
    ctx.drawImage(native, 0, 0, display.width, display.height);
  }, []);

  // Keep the display canvas crisp on high-DPI screens.
  useEffect(() => {
    const display = displayRef.current;
    if (!display || box.w === 0) return;
    const dpr = window.devicePixelRatio || 1;
    display.width = Math.round(box.w * dpr);
    display.height = Math.round(box.h * dpr);
    redraw();
  }, [box, redraw]);

  const clear = useCallback(() => {
    const native = nativeRef.current;
    if (native) native.getContext("2d")?.clearRect(0, 0, native.width, native.height);
    paintedRef.current = false;
    redraw();
    onChange?.(false);
  }, [redraw, onChange]);

  const exportMask = useCallback((): string | null => {
    const native = nativeRef.current;
    if (!native || !paintedRef.current) return null;
    return native.toDataURL("image/png");
  }, []);

  useEffect(() => {
    onReady?.({ exportMask, clear });
  }, [onReady, exportMask, clear]);

  const toNative = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * width,
      y: ((e.clientY - rect.top) / rect.height) * height,
    };
  };

  const stroke = (from: { x: number; y: number } | null, to: { x: number; y: number }) => {
    const native = nativeRef.current;
    const ctx = native?.getContext("2d");
    if (!native || !ctx) return;
    // Brush radius is in display pixels; scale it to native.
    const scale = box.w > 0 ? width / box.w : 1;
    const radius = Math.max(1, brushRadius * scale);
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ffffff";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = radius * 2;
    ctx.beginPath();
    if (from) {
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    } else {
      ctx.arc(to.x, to.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    if (!erase) paintedRef.current = true;
    redraw();
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drawingRef.current = true;
    const point = toNative(e);
    lastRef.current = point;
    stroke(null, point);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    e.preventDefault();
    const point = toNative(e);
    stroke(lastRef.current, point);
    lastRef.current = point;
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    onChange?.(paintedRef.current);
  };

  return (
    <div ref={containerRef} className="w-full h-full flex items-center justify-center overflow-hidden">
      <div className="relative" style={{ width: box.w, height: box.h }}>
        <img src={src} alt="" className="absolute inset-0 w-full h-full select-none" draggable={false} />
        <canvas
          ref={displayRef}
          className="absolute inset-0 w-full h-full cursor-crosshair"
          style={{ opacity: 0.55, touchAction: "none" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={onPointerUp}
        />
      </div>
    </div>
  );
}
