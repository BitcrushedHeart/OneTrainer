"""Tool window for baking an OFT / DoRA-OFT adapter into a base checkpoint.

The user only picks the adapter + output destination (+ dtype/format/strength).
``base_model_name`` / ``transformer_model_name`` / ``vae_model_name`` come
from the parent Train UI's TrainConfig -- whatever the Model tab is set to.
"""

import threading
import traceback

from modules.util.args.MergeOFTArgs import MergeOFTArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PathIOType import PathIOType
from modules.util.oft_merge import merge_oft_adapter
from modules.util.oft_verify import MergeVerificationError
from modules.util.torch_util import torch_gc
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class MergeOFTUI(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        # The merge inherits base_model_name / transformer_model_name from
        # the parent TrainUI's live config. If parent doesn't have one (e.g.,
        # standalone testing), we'll error in the worker.
        self.ui_train_config = getattr(parent, "train_config", None)

        self.merge_args = MergeOFTArgs.default_values()
        self.ui_state = UIState(self, self.merge_args)
        self.button = None
        self.status_label = None
        self.inherited_label = None

        self.title("Merge OFT into base")
        self.geometry("700x420")
        self.resizable(True, True)

        self.frame = ctk.CTkFrame(self, width=700, height=420)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)

        self.main_frame(self.frame)
        self.frame.pack(fill="both", expand=True)

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def main_frame(self, master):
        components.label(
            master,
            0,
            0,
            "Input Model",
            tooltip=(
                "Transformer checkpoint to merge into (acts as the transformer override). "
                "Architecture / base model / VAE are inherited from the parent Model tab. "
                "Swap this between runs to merge into the base then the distilled variant without closing the window."
            ),
        )
        components.path_entry(master, 0, 1, self.ui_state, "input_model_path", mode="file")

        components.label(
            master,
            1,
            0,
            "Output Model Location",
            tooltip="Path for the merged checkpoint. Must differ from the Input Model.",
        )
        components.path_entry(master, 1, 1, self.ui_state, "output_path", mode="file", io_type=PathIOType.MODEL)

        components.label(
            master,
            2,
            0,
            "OFT Adapter",
            tooltip="Trained OFT / DoRA-OFT adapter (safetensors). Training-time config (model_type, layer_filter, oft_block_size, ...) is read from its ot_config metadata.",
        )
        components.path_entry(master, 2, 1, self.ui_state, "oft_adapter_path", mode="file")

        components.label(master, 3, 0, "Output Data Type")
        components.options_kv(
            master,
            3,
            1,
            [
                ("float32", DataType.FLOAT_32),
                ("float16", DataType.FLOAT_16),
                ("bfloat16", DataType.BFLOAT_16),
            ],
            self.ui_state,
            "output_dtype",
        )

        components.label(master, 4, 0, "Output Format")
        components.options_kv(
            master,
            4,
            1,
            [
                ("Safetensors", ModelFormat.SAFETENSORS),
                ("Diffusers", ModelFormat.DIFFUSERS),
            ],
            self.ui_state,
            "output_model_format",
        )

        components.label(
            master,
            5,
            0,
            "Strength",
            tooltip=(
                "Adapter strength to bake. 1.0 = full merge (same as training output). "
                "0.7 reproduces the ComfyUI 'LoRA weight = 0.7' interpolation. "
                "DoRA row-norm invariant check is only run at strength=1.0."
            ),
        )
        components.entry(master, 5, 1, self.ui_state, "strength")

        self.button = components.button(master, 6, 1, "Merge", self.merge)

        # Inherited model config summary (architecture / base / VAE come from parent UI).
        inherited = self._inherited_summary()
        components.label(
            master,
            7,
            0,
            "Inherited from Model tab",
            tooltip="model_type / base_model / VAE override come from the active Model tab. Edit them there before merging.",
        )
        self.inherited_label = ctk.CTkLabel(master, text=inherited, justify="left", anchor="w")
        self.inherited_label.grid(row=7, column=1, sticky="ew", padx=8, pady=4)

        self.status_label = ctk.CTkLabel(master, text="", wraplength=600, justify="left")
        self.status_label.grid(row=8, column=0, columnspan=2, sticky="w", padx=8, pady=8)

    def _inherited_summary(self) -> str:
        if self.ui_train_config is None:
            return "(no parent TrainConfig found)"
        cfg = self.ui_train_config
        lines = [
            f"model_type:      {cfg.model_type.name if cfg.model_type else '(unset)'}",
            f"base_model_name: {cfg.base_model_name or '(unset)'}",
            f"vae_override:    {cfg.vae.model_name or '(stock)'}",
        ]
        return "\n".join(lines)

    def _set_status(self, text: str, error: bool = False):
        color = "#cc4444" if error else "#dddddd"
        self.status_label.configure(text=text, text_color=color)

    def merge(self):
        self.button.configure(state="disabled")
        self._set_status("Merging...")

        thread = threading.Thread(target=self._merge_worker, daemon=True)
        thread.start()

    def _merge_worker(self):
        # Snapshot parent's transformer.model_name and override with our Input Model
        # for the duration of the merge so the parent Model tab isn't permanently mutated.
        # Always restored in finally, even on error.
        saved_transformer = None
        transformer_overridden = False
        try:
            if self.ui_train_config is None:
                raise RuntimeError(
                    "Merge tool launched without a parent TrainConfig. Open from the main Train UI's Tools tab."
                )

            saved_transformer = self.ui_train_config.transformer.model_name
            self.ui_train_config.transformer.model_name = self.merge_args.input_model_path
            transformer_overridden = True

            report = merge_oft_adapter(
                oft_adapter_path=self.merge_args.oft_adapter_path,
                output_path=self.merge_args.output_path,
                output_dtype=self.merge_args.output_dtype,
                output_format=self.merge_args.output_model_format,
                ui_train_config=self.ui_train_config,
                strength=float(self.merge_args.strength),
            )
            keys_merged = len(report.post_zero_rows)
            self.after(
                0,
                lambda: self._set_status(
                    f"Merge OK -- {keys_merged} module(s) baked. Output: {self.merge_args.output_path}"
                ),
            )
        except MergeVerificationError as e:
            traceback.print_exc()
            self.after(0, lambda e=e: self._set_status(str(e), error=True))
        except Exception as e:
            traceback.print_exc()
            self.after(0, lambda e=e: self._set_status(f"Error: {e}", error=True))
        finally:
            if transformer_overridden:
                self.ui_train_config.transformer.model_name = saved_transformer
            torch_gc()
            self.after(0, lambda: self.button.configure(state="normal"))
