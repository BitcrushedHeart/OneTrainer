import { type ChangeEvent } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { SELECT_FULL } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface SelectKVOption {
  label: string;
  value: string;
}

export interface SelectKVProps {
  label: string;
  options: SelectKVOption[];
  configPath?: string;
  value?: string;
  onChange?: (value: string) => void;
  tooltip?: string;
  disabled?: boolean;
}

export function SelectKV({
  label,
  options,
  configPath,
  value: controlledValue,
  onChange,
  tooltip,
  disabled,
}: SelectKVProps) {
  const [configValue, setConfigValue] = useConfigField<string>(configPath);

  const currentValue = configPath ? (configValue ?? "") : (controlledValue ?? "");

  const handleChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (configPath && setConfigValue) setConfigValue(val);
    if (onChange) onChange(val);
  };

  return (
    <FormFieldWrapper label={label} tooltip={tooltip} configPath={configPath}>
      <select value={currentValue} onChange={handleChange} disabled={disabled} className={SELECT_FULL}>
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </FormFieldWrapper>
  );
}
