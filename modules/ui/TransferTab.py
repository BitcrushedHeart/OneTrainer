from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class TransferTab:
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.scroll_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, minsize=50)
        self.scroll_frame.grid_columnconfigure(3, weight=0)
        self.scroll_frame.grid_columnconfigure(4, weight=1)

        components.label(
            self.scroll_frame,
            0,
            0,
            "Enable Transfer",
            tooltip="Knowledge-transfer mode: match the student's late-block activations to Base + the "
            "original adapter's education. Load the conjugated warm-start adapter in 'base adapter' on "
            "the LoRA tab, and pick the variant checkpoint (Turbo/Lightning) as the base model. "
            "Only late-family adapter parameters train; the model's distillation is never touched.",
        )
        components.switch(self.scroll_frame, 0, 1, self.ui_state, "transfer_enabled")

        components.label(
            self.scroll_frame,
            1,
            0,
            "Base Checkpoint",
            tooltip="The SoReal! Base checkpoint -- the frame the teacher adapter was trained in. Its late "
            "tensors plus the teacher adapter's education form the in-memory teacher (no merged "
            "checkpoint is written).",
        )
        components.path_entry(self.scroll_frame, 1, 1, self.ui_state, "transfer_base_model_name")

        components.label(
            self.scroll_frame,
            1,
            3,
            "Teacher Adapter",
            tooltip="The ORIGINAL Base-trained OFT / DoRA-OFT adapter (e.g. V40). NOT the conjugated "
            "warm-start file -- that one goes into the LoRA tab's base adapter field.",
        )
        components.path_entry(self.scroll_frame, 1, 4, self.ui_state, "transfer_teacher_adapter_name")

        components.label(
            self.scroll_frame,
            2,
            0,
            "Late Start Layer",
            tooltip="First transformer layer of the late band. The forward forks here: everything before "
            "runs once (teacher and student are identical), the late stack runs twice.",
        )
        components.entry(self.scroll_frame, 2, 1, self.ui_state, "transfer_late_start")

        components.label(
            self.scroll_frame,
            2,
            3,
            "Families",
            tooltip="Comma-separated late families to TRAIN: qkv (to_q/to_k/to_v) and/or mlp (w1/w2/w3). "
            "The teacher always carries the education on every late adapted module regardless.",
        )
        components.entry(self.scroll_frame, 2, 4, self.ui_state, "transfer_families")

        components.label(
            self.scroll_frame,
            3,
            0,
            "Normalize Block Loss",
            tooltip="Divide each block's MSE by the teacher output power so every late block contributes "
            "on the same scale.",
        )
        components.switch(self.scroll_frame, 3, 1, self.ui_state, "transfer_block_loss_normalize")

        components.label(
            self.scroll_frame,
            3,
            3,
            "Mask Padded Tokens",
            tooltip="Restrict the activation loss to valid sequence tokens when batches are padded.",
        )
        components.switch(self.scroll_frame, 3, 4, self.ui_state, "transfer_mask_padded_tokens")

        components.label(
            self.scroll_frame,
            4,
            0,
            "Validation",
            tooltip="Deterministic teacher-match residual on cached training batches (fixed seeds), used "
            "for convergence detection and best-checkpoint tracking. Runs on the normal validation cadence.",
        )
        components.switch(self.scroll_frame, 4, 1, self.ui_state, "transfer_validation")

        components.label(
            self.scroll_frame,
            4,
            3,
            "Validation Batches",
            tooltip="Number of training batches cached for the residual validation.",
        )
        components.entry(self.scroll_frame, 4, 4, self.ui_state, "transfer_validation_batches")

        components.label(
            self.scroll_frame,
            5,
            0,
            "Auto-stop (Patience)",
            tooltip="Stop when the residual stops improving meaningfully. The final save always restores "
            "the best-seen adapter, never the last -- a run that drifts late still delivers its best state.",
        )
        components.switch(self.scroll_frame, 5, 1, self.ui_state, "transfer_patience_enabled")

        components.label(
            self.scroll_frame,
            5,
            3,
            "Patience Checks",
            tooltip="Consecutive validations without meaningful improvement before stopping.",
        )
        components.entry(self.scroll_frame, 5, 4, self.ui_state, "transfer_patience_checks")

        components.label(
            self.scroll_frame,
            6,
            0,
            "Min Improvement",
            tooltip="Relative improvement over the best residual that counts as meaningful "
            "(0.005 = 0.5%). Smaller improvements still update the best checkpoint but do not reset patience.",
        )
        components.entry(self.scroll_frame, 6, 1, self.ui_state, "transfer_min_relative_improvement")

        components.label(
            self.scroll_frame,
            6,
            3,
            "Target Residual",
            tooltip="Optional absolute stop: end the run as soon as the validation residual reaches this "
            "value. 0 disables.",
        )
        components.entry(self.scroll_frame, 6, 4, self.ui_state, "transfer_target_residual")
