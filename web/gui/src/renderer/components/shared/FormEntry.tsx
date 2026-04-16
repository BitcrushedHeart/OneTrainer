import { type ChangeEvent, forwardRef, useEffect, useState } from "react";

import { useConfigField } from "@/hooks/useConfigField";
import { INPUT_FULL, PLACEHOLDER } from "@/utils/inputStyles";

import { FormFieldWrapper } from "./FormFieldWrapper";

export interface FormEntryProps {
    label: string;
    configPath?: string;
    value?: string | number | null;
    onChange?: (value: string | number) => void;
    type?: "text" | "number";
    placeholder?: string;
    tooltip?: string;
    disabled?: boolean;
    nullable?: boolean;
    width?: string;
}

export const FormEntry = forwardRef<HTMLInputElement, FormEntryProps>(
    (
        {
            label,
            configPath,
            value: controlledValue,
            onChange,
            type = "text",
            placeholder,
            tooltip,
            disabled,
            nullable,
            width,
        },
        ref,
    ) => {
        const [configValue, setConfigValue] = useConfigField<string | number | null>(configPath);

        const externalValue = configPath ? configValue : controlledValue;
        const [localValue, setLocalValue] = useState<string>(externalValue != null ? String(externalValue) : "");

        useEffect(() => {
            setLocalValue(externalValue != null ? String(externalValue) : "");
        }, [externalValue]);

        const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
            const raw = e.target.value;
            setLocalValue(raw);
            if (raw === "" && nullable) {
                if (configPath && setConfigValue) setConfigValue(null as unknown as string | number);
                return;
            }
            if (type === "number") {
                // Don't push an empty field as 0 — let the user keep typing.
                // The config will be updated once a valid number is entered.
                if (raw === "" || raw === "-") return;
                const num = Number(raw);
                if (isNaN(num)) return;
                if (configPath && setConfigValue) setConfigValue(num);
                if (onChange) onChange(num);
            } else {
                if (configPath && setConfigValue) setConfigValue(raw);
                if (onChange) onChange(raw);
            }
        };

        return (
            <FormFieldWrapper
                label={label}
                tooltip={tooltip}
                configPath={configPath}
                style={width ? { width } : undefined}
            >
                <input
                    ref={ref}
                    type={type}
                    value={localValue}
                    onChange={handleChange}
                    placeholder={placeholder}
                    disabled={disabled}
                    className={`${INPUT_FULL} ${PLACEHOLDER}`}
                />
            </FormFieldWrapper>
        );
    },
);
FormEntry.displayName = "FormEntry";
