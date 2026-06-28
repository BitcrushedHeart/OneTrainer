from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class DistillTab:
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
            "Enable Distill",
            tooltip="Turns on DMD2-style image-grounded distillation. Distill and RLHF cannot run together.",
        )
        components.switch(self.scroll_frame, 0, 1, self.ui_state, "distill_enabled")

        components.label(
            self.scroll_frame,
            0,
            3,
            "Target Steps",
            tooltip="Number of student sampling steps to train toward. Use 4 for the first pass.",
        )
        components.entry(self.scroll_frame, 0, 4, self.ui_state, "distill_target_steps")

        components.label(
            self.scroll_frame,
            1,
            0,
            "Teacher Steps",
            tooltip="Teacher trajectory step count. Auto uses metadata/config defaults; override only when needed.",
        )
        components.entry(self.scroll_frame, 1, 1, self.ui_state, "distill_teacher_steps")

        components.label(
            self.scroll_frame,
            1,
            3,
            "Auto Teacher Steps",
            tooltip="Prefer per-image metadata or model defaults for teacher steps when available.",
        )
        components.switch(self.scroll_frame, 1, 4, self.ui_state, "distill_teacher_steps_auto")

        components.label(
            self.scroll_frame,
            2,
            0,
            "Timestep Grid",
            tooltip="AUTO uses a monotonic grid from the teacher schedule to the configured target step count.",
        )
        components.options(self.scroll_frame, 2, 1, ["AUTO", "TRAILING", "UNIFORM"], self.ui_state, "distill_timestep_grid_mode")

        components.label(
            self.scroll_frame,
            2,
            3,
            "VRAM Mode",
            tooltip="SPLIT runs teacher/fake/student forwards sequentially. CONCURRENT is reserved for higher VRAM setups.",
        )
        components.options(self.scroll_frame, 2, 4, ["SPLIT", "CONCURRENT"], self.ui_state, "distill_vram_mode")

        components.label(
            self.scroll_frame,
            3,
            0,
            "KL Weight",
            tooltip="Weight for the DMD2 pseudo-KL direction from fake and real score predictions.",
        )
        components.entry(self.scroll_frame, 3, 1, self.ui_state, "distill_kl_loss_weight")

        components.label(
            self.scroll_frame,
            3,
            3,
            "Endpoint Weight",
            tooltip="Weight for regression from the generated terminal latent to the actual image latent.",
        )
        components.entry(self.scroll_frame, 3, 4, self.ui_state, "distill_endpoint_loss_weight")

        components.label(
            self.scroll_frame,
            4,
            0,
            "TTUR Ratio",
            tooltip="Number of fake-score updates per student update after warmup.",
        )
        components.entry(self.scroll_frame, 4, 1, self.ui_state, "distill_ttur_ratio")

        components.label(
            self.scroll_frame,
            4,
            3,
            "Fake Warmup Steps",
            tooltip="Initial optimizer steps that train only the fake-score adapter.",
        )
        components.entry(self.scroll_frame, 4, 4, self.ui_state, "distill_fake_warmup_steps")

        components.label(
            self.scroll_frame,
            5,
            0,
            "CFG Mode",
            tooltip="AUTO reads CFG from metadata; CFG=1 uses one teacher pass, CFG>1 uses guided teacher mixing.",
        )
        components.options(self.scroll_frame, 5, 1, ["AUTO", "FORCE_1", "METADATA"], self.ui_state, "distill_cfg_mode")

        components.label(
            self.scroll_frame,
            5,
            3,
            "Fake Adapter Type",
            tooltip="LoRA is the stable default. DoRA and OFT are experimental for the fake-score adapter.",
        )
        components.options(self.scroll_frame, 5, 4, ["LORA", "DORA", "OFT"], self.ui_state, "distill_fake_adapter_type")

        components.label(
            self.scroll_frame,
            6,
            0,
            "Fake Adapter Rank",
            tooltip="Rank/block-size used by the fake-score adapter.",
        )
        components.entry(self.scroll_frame, 6, 1, self.ui_state, "distill_fake_adapter_rank")
