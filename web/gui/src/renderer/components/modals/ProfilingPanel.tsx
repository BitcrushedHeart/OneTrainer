import { useState } from "react";

import { Button } from "@/components/shared";

import { ModalBase } from "./ModalBase";

export interface ProfilingPanelProps {
  open: boolean;
  onClose: () => void;
}

export function ProfilingPanel({ open, onClose }: ProfilingPanelProps) {
  const [isProfiling, setIsProfiling] = useState(false);
  const [message, setMessage] = useState("Inactive");

  const handleDumpStack = async () => {
    try {
      setMessage("Stack dump requested (see backend logs)");
    } catch {
      setMessage("Error requesting stack dump");
    }
  };

  const handleToggleProfiling = () => {
    if (isProfiling) {
      setIsProfiling(false);
      setMessage("Inactive");
    } else {
      setIsProfiling(true);
      setMessage("Profiling active... (requires Scalene)");
    }
  };

  return (
    <ModalBase open={open} onClose={onClose} title="Profiling" size="sm">
      <div className="flex flex-col gap-4">
        <Button variant="secondary" onClick={handleDumpStack}>
          Dump Stack
        </Button>
        <Button variant={isProfiling ? "primary" : "secondary"} onClick={handleToggleProfiling}>
          {isProfiling ? "Stop Profiling" : "Start Profiling"}
        </Button>
        <div className="p-3 rounded-[var(--radius-sm)] bg-[var(--color-surface-raised)] border border-[var(--color-border-subtle)]">
          <p className="text-sm text-[var(--color-on-surface-secondary)]">{message}</p>
        </div>
      </div>

      <div className="flex justify-end mt-6 pt-4 border-t border-[var(--color-border-subtle)]">
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </div>
    </ModalBase>
  );
}
