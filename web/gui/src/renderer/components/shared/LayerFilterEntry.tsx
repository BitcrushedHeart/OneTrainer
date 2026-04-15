import { type ChangeEvent, useEffect, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { INPUT_FULL, PLACEHOLDER, SELECT_BASE } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";
import { Toggle } from "./Toggle";

export interface LayerFilterEntryProps {
  filterPath: string;
  presetPath: string;
  regexPath: string;
  presetOptions?: Array<{ label: string; value: string }>;
  tooltip?: string;
}

export function LayerFilterEntry({
  filterPath,
  presetPath,
  regexPath,
  presetOptions = [],
  tooltip,
}: LayerFilterEntryProps) {
  const [filterValue, setFilterValue] = useConfigField<string>(filterPath);
  const [presetValue, setPresetValue] = useConfigField<string>(presetPath);
  const [localFilter, setLocalFilter] = useState(filterValue ?? "");

  useEffect(() => {
    setLocalFilter(filterValue ?? "");
  }, [filterValue]);

  const handlePresetChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setPresetValue(e.target.value);
  };

  const handleFilterChange = (e: ChangeEvent<HTMLInputElement>) => {
    setLocalFilter(e.target.value);
    setFilterValue(e.target.value);
  };

  const allOptions = [{ label: "Full", value: "full" }, ...presetOptions, { label: "Custom", value: "custom" }];

  return (
    <FormFieldWrapper label="Layer Filter" tooltip={tooltip} configPath={presetPath} className="flex flex-col gap-2">
      <div className="flex gap-2 items-center">
        <select value={presetValue ?? "full"} onChange={handlePresetChange} className={SELECT_BASE}>
          {allOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <Toggle configPath={regexPath} label="Regex" labelPosition="right" />
      </div>
      <input
        type="text"
        value={localFilter}
        onChange={handleFilterChange}
        placeholder="Comma-separated layer names or regex..."
        className={`${INPUT_FULL} ${PLACEHOLDER}`}
      />
    </FormFieldWrapper>
  );
}
