import os

from modules.util.config.TrainConfig import TrainConfig
from modules.util.queue.QueueExecutor import QueueExecutor
from modules.util.queue.QueueManager import QueueManager


class QueueValidator:
    @staticmethod
    def validate_entry(
        global_config: TrainConfig,
        overrides: dict,
    ) -> tuple[list[str], list[str]]:
        errors = []
        warnings = []
        try:
            merged = QueueExecutor.merge_config(global_config, overrides)
        except Exception as e:
            errors.append(f"Failed to merge config: {e}")
            return errors, warnings

        if merged.base_model_name and not _path_or_hf_exists(merged.base_model_name):
            errors.append(f"Base model not found: {merged.base_model_name}")

        if merged.concept_file_name and not os.path.isfile(merged.concept_file_name):
            errors.append(f"Concept file not found: {merged.concept_file_name}")

        if merged.workspace_dir:
            parent = os.path.dirname(os.path.abspath(merged.workspace_dir))
            if not os.path.isdir(parent):
                errors.append(f"Workspace parent directory does not exist: {parent}")

        if merged.output_model_destination:
            parent = os.path.dirname(os.path.abspath(merged.output_model_destination))
            if parent and not os.path.isdir(parent):
                warnings.append(f"Output model parent directory does not exist: {parent}")

        if getattr(merged, "patience", False) and not merged.validation:
            errors.append("Patience is enabled but validation is disabled")

        if hasattr(merged, "continue_last_backup") and merged.continue_last_backup:
            backup_path = merged.get_last_backup_path()
            if backup_path is None:
                warnings.append("Continue from backup is enabled but no backup found")

        return errors, warnings

    @staticmethod
    def validate_queue(
        queue_manager: QueueManager,
        global_config: TrainConfig,
    ) -> dict[str, tuple[list[str], list[str]]]:
        results = {}
        for entry in queue_manager.entries:
            if entry.status.value in ("COMPLETED", "SKIPPED"):
                continue
            errors, warnings = QueueValidator.validate_entry(global_config, entry.overrides)
            if queue_manager.settings.retry_from_backup and global_config.backup_after == 0:
                warnings.append("Retry from backup is enabled but no backup interval is configured")
            if errors or warnings:
                results[entry.id] = (errors, warnings)
        return results

    @staticmethod
    def has_critical_errors(
        validation_results: dict[str, tuple[list[str], list[str]]],
    ) -> bool:
        return any(errors for errors, _warnings in validation_results.values())


def _path_or_hf_exists(path: str) -> bool:
    if os.path.exists(path):
        return True
    parts = path.split("/")
    if len(parts) != 2:
        return False
    return all(
        p and not p.startswith(".")
        and not p.endswith((".safetensors", ".ckpt", ".bin", ".pt"))
        for p in parts
    )
