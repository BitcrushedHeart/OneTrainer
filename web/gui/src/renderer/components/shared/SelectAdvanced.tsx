import { MoreHorizontal } from "lucide-react";
import { type ChangeEvent } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { enumLabel } from "@/utils/enumLabels";
import { SELECT_FLEX, SIDE_BUTTON } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface SelectAdvancedProps {
  label: string;
  options: string[];
  configPath?: string;
  value?: string;
  onChange?: (value: string) => void;
  onAdvancedClick?: () => void;
  tooltip?: string;
  disabled?: boolean;
  /** Custom label formatter for option display. Defaults to `enumLabel`. */
  formatLabel?: (value: string) => string;
}

export function SelectAdvanced({
  label,
  options,
  configPath,
  value: controlledValue,
  onChange,
  onAdvancedClick,
  tooltip,
  disabled,
  formatLabel = enumLabel,
}: SelectAdvancedProps) {
  const [configValue, setConfigValue] = useConfigField<string>(configPath);

  const currentValue = configPath ? (configValue ?? "") : (controlledValue ?? "");

  const handleChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (configPath && setConfigValue) setConfigValue(val);
    if (onChange) onChange(val);
  };

  return (
    <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath}>
      <div className="flex gap-1">
        <select value={currentValue} onChange={handleChange} disabled={disabled} className={SELECT_FLEX}>
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {formatLabel(opt)}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={onAdvancedClick}
          disabled={disabled}
          className={SIDE_BUTTON}
          aria-label="Advanced settings"
        >
          <MoreHorizontal className="w-4 h-4" />
        </button>
      </div>
    </FormFieldWrapper>
  );
}
