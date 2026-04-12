import { type ChangeEvent, forwardRef } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { enumLabel } from "@/utils/enumLabels";
import { SELECT_FULL } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface SelectProps {
  label: string;
  options: string[];
  configPath?: string;
  value?: string;
  onChange?: (value: string) => void;
  tooltip?: string;
  disabled?: boolean;
  /** Custom label formatter for option display. Defaults to `enumLabel`. */
  formatLabel?: (value: string) => string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    { label, options, configPath, value: controlledValue, onChange, tooltip, disabled, formatLabel = enumLabel },
    ref,
  ) => {
    const [configValue, setConfigValue] = useConfigField<string>(configPath);

    const currentValue = configPath ? (configValue ?? "") : (controlledValue ?? "");

    const handleChange = (e: ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      if (configPath && setConfigValue) setConfigValue(val);
      if (onChange) onChange(val);
    };

    return (
      <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath}>
        <select ref={ref} value={currentValue} onChange={handleChange} disabled={disabled} className={SELECT_FULL}>
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {formatLabel(opt)}
            </option>
          ))}
        </select>
      </FormFieldWrapper>
    );
  },
);
Select.displayName = "Select";
