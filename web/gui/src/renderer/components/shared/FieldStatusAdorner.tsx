import { RotateCcw } from "lucide-react";

import { useFieldStatus } from "@/hooks/fieldBinding";

export interface FieldStatusAdornerProps {
  path?: string;
  /** Render the dot only (no reset button). */
  dotOnly?: boolean;
}

/**
 * Small adorner shown next to overridable form fields. In global (default)
 * mode this renders nothing. In override mode it shows a quiet cobalt dot
 * when the field is overridden, plus an inline reset arrow that clears the
 * override on click.
 *
 * The dot is sized at 6px and uses a soft halo so it reads as a deliberate
 * status mark — not a notification badge — at a glance.
 */
export function FieldStatusAdorner({ path, dotOnly }: FieldStatusAdornerProps) {
  const status = useFieldStatus(path);
  if (!status || !status.isOverridden) return null;
  const { reset } = status;
  return (
    <span className="inline-flex items-center gap-1.5 leading-none animate-[rowFade_180ms_ease-out]">
      <span
        aria-label="Overridden for this entry"
        title="Overridden for this entry"
        className="block w-1.5 h-1.5 rounded-full bg-[var(--color-cobalt-600)]
          shadow-[0_0_0_3px_var(--color-cobalt-600-alpha-15),0_0_8px_var(--color-cobalt-600-alpha-30)]"
      />
      {!dotOnly && (
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            reset();
          }}
          className="inline-flex items-center justify-center w-4 h-4 rounded-[3px]
            text-[var(--color-cobalt-600)] hover:text-[var(--color-azure-500)]
            hover:bg-[var(--color-cobalt-600-alpha-12)]
            transition-colors duration-150 cursor-pointer
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-cobalt-600-alpha-30)]"
          aria-label="Reset to global value"
          title="Reset to global value"
        >
          <RotateCcw className="w-3 h-3" />
        </button>
      )}
    </span>
  );
}
