import { Moon, Save, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { configApi, type PresetInfo } from "@/api/configApi";
import { SavePresetModal } from "@/components/modals/SavePresetModal";
import { useConfigField } from "@/hooks/useConfigField";
import { useConfigStore } from "@/store/configStore";
import { useUiStore } from "@/store/uiStore";
import type { ModelType, TrainingMethod } from "@/types/generated/enums";
import { ModelTypeValues } from "@/types/generated/enums";
import { TRAINING_METHODS_BY_MODEL } from "@/types/generated/modelTypeInfo";
import { enumLabel } from "@/utils/enumLabels";

export default function TopBar() {
  const { theme, toggleTheme, backendConnected } = useUiStore();
  const loadPreset = useConfigStore((s) => s.loadPreset);
  const savePreset = useConfigStore((s) => s.savePreset);
  const loadedPresetName = useConfigStore((s) => s.loadedPresetName);
  const [modelType, setModelType] = useConfigField<ModelType>("model_type");
  const [trainingMethod, setTrainingMethod] = useConfigField<TrainingMethod>("training_method");
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [showSaveModal, setShowSaveModal] = useState(false);

  useEffect(() => {
    if (backendConnected) {
      configApi
        .listPresets()
        .then(setPresets)
        .catch((err) => {
          console.error("[TopBar] Failed to fetch presets:", err);
        });
    }
  }, [backendConnected]);

  const availableMethods = TRAINING_METHODS_BY_MODEL[modelType ?? "STABLE_DIFFUSION_15"];

  const handlePresetLoad = (path: string) => {
    const preset = presets.find((p) => p.path === path);
    loadPreset(path, preset?.name);
  };
  const handleSavePreset = () => {
    setShowSaveModal(true);
  };
  const handleSavePresetConfirm = async (name: string) => {
    await savePreset(name);
    setShowSaveModal(false);
    try {
      const updated = await configApi.listPresets();
      setPresets(updated);
    } catch {
      /* best-effort refresh */
    }
  };
  const handleModelChange = (val: string) => {
    setModelType(val as ModelType);
    // If current training method is not compatible with new model, reset to FINE_TUNE
    const methods = TRAINING_METHODS_BY_MODEL[val as ModelType];
    if (trainingMethod && !methods.includes(trainingMethod)) {
      setTrainingMethod("FINE_TUNE");
    }
  };

  return (
    <header className="top-bar">
      <div className="top-bar-left">
        <h1 className="top-bar-title">OneTrainer</h1>
        {new URLSearchParams(window.location.search).get("dev") === "1" && (
          <span className={`connection-status ${backendConnected ? "connected" : "disconnected"}`}>
            {backendConnected ? "Connected" : "Disconnected"}
          </span>
        )}
      </div>
      <div className="top-bar-right">
        <select
          value=""
          onChange={(e) => {
            if (e.target.value) handlePresetLoad(e.target.value);
          }}
          className="top-bar-select"
        >
          <option value="" disabled>
            {loadedPresetName ? `Preset: ${loadedPresetName.replace(/^#/, "")}` : "Load Preset..."}
          </option>
          {presets.filter((p) => p.is_builtin).length > 0 && (
            <optgroup label="Built-in">
              {presets
                .filter((p) => p.is_builtin)
                .map((p) => (
                  <option key={p.path} value={p.path}>
                    {p.name}
                  </option>
                ))}
            </optgroup>
          )}
          {presets.filter((p) => !p.is_builtin).length > 0 && (
            <optgroup label="User">
              {presets
                .filter((p) => !p.is_builtin)
                .map((p) => (
                  <option key={p.path} value={p.path}>
                    {p.name}
                  </option>
                ))}
            </optgroup>
          )}
        </select>
        <button onClick={handleSavePreset} className="theme-toggle" aria-label="Save preset" title="Save preset">
          <Save className="w-4 h-4" />
        </button>

        <div className="top-bar-separator" aria-hidden="true" />

        <select
          value={modelType ?? "STABLE_DIFFUSION_15"}
          onChange={(e) => handleModelChange(e.target.value)}
          className="top-bar-select"
        >
          {ModelTypeValues.map((mt) => (
            <option key={mt} value={mt}>
              {enumLabel(mt)}
            </option>
          ))}
        </select>

        <select
          value={trainingMethod ?? "FINE_TUNE"}
          onChange={(e) => setTrainingMethod(e.target.value as TrainingMethod)}
          className="top-bar-select"
        >
          {availableMethods.map((m) => (
            <option key={m} value={m}>
              {enumLabel(m)}
            </option>
          ))}
        </select>

        <div className="top-bar-separator" aria-hidden="true" />

        <button
          onClick={toggleTheme}
          className="theme-toggle"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
      <SavePresetModal
        isOpen={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        onSave={handleSavePresetConfirm}
      />
    </header>
  );
}
