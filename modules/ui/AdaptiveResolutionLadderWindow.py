from modules.util.ui.UIState import UIState

from mgds.pipelineModules.ResolutionPolicy import (
    DEFAULT_ADAPTIVE_RESOLUTION_LADDER,
    adaptive_resolution_tiers,
    parse_adaptive_resolution_ladder,
)

import customtkinter as ctk


class AdaptiveResolutionLadderWindow(ctk.CTkToplevel):
    """Compact editor for the scalar tiers used by adaptive resolution."""

    PRESETS = {
        "Standard": DEFAULT_ADAPTIVE_RESOLUTION_LADDER,
        "Power of two": (1024, 512, 256, 128),
        "Fine steps": (1024, 896, 768, 640, 512, 384, 256, 128),
    }

    def __init__(self, parent, ui_state: UIState):
        super().__init__(parent)
        self.ui_state = ui_state
        self.title("Adaptive Resolution Ladder")
        self.geometry("620x330")
        self.minsize(520, 300)
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            self,
            text="RESOLUTION SNAP LADDER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray45", "gray65"),
        ).grid(row=0, column=0, padx=24, pady=(22, 4), sticky="w")
        ctk.CTkLabel(
            self,
            text=(
                "Enter scalar tiers in descending order. The training resolution is always the ceiling; "
                "the smallest active tier is the fallback for very small images."
            ),
            justify="left",
            wraplength=560,
        ).grid(row=1, column=0, padx=24, pady=(0, 14), sticky="ew")

        self.value = ctk.StringVar(value=str(self.ui_state.get_var("adaptive_resolution_ladder").get()))
        self.entry = ctk.CTkEntry(self, textvariable=self.value, placeholder_text="1024,768,512,256,128")
        self.entry.grid(row=2, column=0, padx=24, pady=0, sticky="ew")

        preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        preset_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        for column, (name, values) in enumerate(self.PRESETS.items()):
            preset_frame.grid_columnconfigure(column, weight=1)
            ctk.CTkButton(
                preset_frame,
                text=name,
                height=28,
                fg_color="transparent",
                border_width=1,
                command=lambda selected=values: self.value.set(",".join(str(value) for value in selected)),
            ).grid(row=0, column=column, padx=4, sticky="ew")

        self.preview = ctk.CTkLabel(self, text="", justify="left", anchor="nw", wraplength=560)
        self.preview.grid(row=4, column=0, padx=24, pady=(4, 10), sticky="nsew")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="e")
        ctk.CTkButton(actions, text="Cancel", width=90, fg_color="transparent", command=self.destroy).grid(
            row=0, column=0, padx=4
        )
        self.save_button = ctk.CTkButton(actions, text="Save Ladder", width=120, command=self._save)
        self.save_button.grid(row=0, column=1, padx=4)

        self.value.trace_add("write", self._refresh_preview)
        self._refresh_preview()

    def _refresh_preview(self, *_args):
        try:
            values = parse_adaptive_resolution_ladder(self.value.get())
            resolution = str(self.ui_state.get_var("resolution").get())
            tiers = adaptive_resolution_tiers(resolution, values)
            if not tiers:
                self.preview.configure(
                    text="Fixed WxH resolution detected — adaptive snapping is bypassed.",
                    text_color=("gray45", "gray65"),
                )
            else:
                chain = "  →  ".join(str(value) for value in reversed(tiers))
                self.preview.configure(
                    text=f"Active ladder\n{chain}\n\nFallback: {min(tiers)} px",
                    text_color=("gray20", "gray85"),
                )
            self.save_button.configure(state="normal")
        except ValueError as exc:
            self.preview.configure(text=str(exc), text_color=("#a11d33", "#ff6b7d"))
            self.save_button.configure(state="disabled")

    def _save(self):
        values = parse_adaptive_resolution_ladder(self.value.get())
        canonical = ",".join(str(value) for value in reversed(values))
        self.ui_state.get_var("adaptive_resolution_ladder").set(canonical)
        self.destroy()
