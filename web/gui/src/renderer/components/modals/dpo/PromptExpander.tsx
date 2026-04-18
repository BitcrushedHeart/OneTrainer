import { useState } from "react";

export interface PromptExpanderProps {
  prompt: string;
}

export function PromptExpander({ prompt }: PromptExpanderProps) {
  const [expanded, setExpanded] = useState(false);

  if (prompt === "UNCONDITIONAL") {
    return (
      <div className="flex items-center">
        <span className="font-bold text-sm rounded px-2 py-0.5" style={{ color: "#FFD700", background: "#3A3000" }}>
          UNCONDITIONAL
        </span>
      </div>
    );
  }

  const isLong = prompt.length > 100;
  const truncated = isLong ? `${prompt.slice(0, 100)}...` : prompt;

  return (
    <div className="flex items-start gap-2">
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="shrink-0 px-2 py-1 text-xs rounded bg-[var(--color-surface-container)] text-[var(--color-on-surface)] border border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-container-high)]"
        >
          Prompt {expanded ? "[-]" : "[+]"}
        </button>
      )}
      <div
        className="text-xs font-mono text-[var(--color-on-surface)] p-2 rounded bg-[var(--color-surface-container)] flex-1"
        style={{ wordBreak: "break-word", whiteSpace: "pre-wrap" }}
      >
        {expanded || !isLong ? prompt : truncated}
      </div>
    </div>
  );
}
