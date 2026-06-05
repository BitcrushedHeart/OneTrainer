import os
import queue
import threading
from tkinter import messagebox

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.ui.ui_utils import set_window_icon
from modules.util.validation_checker_util import (
    BasicScanResult,
    CaptionMatch,
    ValidationCheckerConcept,
    ValidationCheckerMatch,
    collect_validation_checker_concepts,
    move_validation_image_to_train,
    open_in_explorer,
    remove_validation_image,
    scan_basic_validation_matches,
    scan_clip_similarity_matches,
)

import customtkinter as ctk
from PIL import Image, ImageOps


class ValidationCheckerWindow(ctk.CTkToplevel):
    def __init__(self, parent, concepts: list[ConceptConfig], training_is_active, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.concepts = concepts
        self.training_is_active = training_is_active

        self.title("Validation Checker Results")
        self.geometry("1360x920")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._event_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._poll_after_id = None
        self._basic_scan_thread = None
        self._deep_scan_thread = None

        self._scan_result: BasicScanResult | None = None
        self._exact_matches: list[ValidationCheckerMatch] = []
        self._perceptual_matches: list[ValidationCheckerMatch] = []
        self._clip_matches: list[ValidationCheckerMatch] = []
        self._caption_matches: list[CaptionMatch] = []
        self._dismissed_matches: set[tuple[str, str, str]] = set()
        self._resolved_validation_paths: set[str] = set()
        self._thumbnail_refs: list[ctk.CTkImage] = []
        self._show_all_captions = False

        self._train_concepts, self._val_concepts = collect_validation_checker_concepts(concepts)
        self._local_train_concepts = [concept for concept in self._train_concepts if concept.is_local]
        self._has_remote_validation_concepts = any(not concept.is_local for concept in self._val_concepts)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_ui()
        self._start_basic_scan()
        self._poll_queue()

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title_frame, text="Validation Checker Results", font=("", 26, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.summary_label = ctk.CTkLabel(title_frame, text="Scanning validation concepts...", anchor="w")
        self.summary_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.progress_label = ctk.CTkLabel(header, text="Preparing scan...", anchor="e")
        self.progress_label.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.progress_bar = ctk.CTkProgressBar(header, width=280)
        self.progress_bar.grid(row=1, column=1, sticky="e", padx=(10, 0))
        self.progress_bar.set(0)

        self.content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        self.content.grid_columnconfigure(0, weight=1)

        if self._has_remote_validation_concepts:
            warning = ctk.CTkLabel(
                self.content,
                text="Cleanup actions are disabled for non-local validation concepts resolved from cache or Hugging Face datasets.",
                text_color="#c26d00",
                anchor="w",
                wraplength=1200,
            )
            warning.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            start_row = 1
        else:
            start_row = 0

        self.exact_card, self.exact_body, self.exact_heading = self._create_section(
            self.content, start_row, "Exact Duplicates"
        )
        self.perceptual_card, self.perceptual_body, self.perceptual_heading = self._create_section(
            self.content, start_row + 1, "Perceptual Matches"
        )
        self.deep_card, self.deep_body, self.deep_heading = self._create_section(
            self.content, start_row + 2, "Deep Scan (CLIP)"
        )
        self.caption_card, self.caption_body, self.caption_heading = self._create_section(
            self.content, start_row + 3, "Caption Matches"
        )

        self._build_deep_scan_controls()

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        footer.grid_columnconfigure(0, weight=1)

        self.footer_status = ctk.CTkLabel(footer, text="", anchor="w")
        self.footer_status.grid(row=0, column=0, sticky="w")

        self.remove_all_button = ctk.CTkButton(footer, text="Remove All Flagged", command=self._remove_all_flagged)
        self.remove_all_button.grid(row=0, column=1, padx=(8, 8))
        self.close_button = ctk.CTkButton(footer, text="Close", fg_color="gray40", command=self._on_close)
        self.close_button.grid(row=0, column=2)

        self._render_all_sections()

    def _create_section(self, master, row: int, title: str):
        card = ctk.CTkFrame(master)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)

        heading = ctk.CTkLabel(card, text=title, anchor="w", font=("", 18, "bold"))
        heading.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))

        body = ctk.CTkFrame(card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        body.grid_columnconfigure(0, weight=1)
        return card, body, heading

    def _build_deep_scan_controls(self):
        controls = ctk.CTkFrame(self.deep_body, fg_color="transparent")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.grid_columnconfigure(1, weight=1)

        self.deep_scan_button = ctk.CTkButton(
            controls,
            text="Deep Scan (CLIP) — ~30 seconds for 1000 images",
            command=self._start_deep_scan,
        )
        self.deep_scan_button.grid(row=0, column=0, padx=(0, 10))

        self.threshold_value_label = ctk.CTkLabel(controls, text="Threshold: 0.93")
        self.threshold_value_label.grid(row=0, column=1, sticky="e")

        self.threshold_slider = ctk.CTkSlider(
            controls, from_=0.80, to=0.99, number_of_steps=190, command=self._update_threshold_label
        )
        self.threshold_slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.threshold_slider.set(0.93)

        self.deep_scan_progress_label = ctk.CTkLabel(
            self.deep_body,
            text="How similar two images need to be to flag as a potential leak. Lower values catch more but may flag genuinely different images. 0.93 works well for most cases.",
            anchor="w",
            justify="left",
            wraplength=1200,
        )
        self.deep_scan_progress_label.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.deep_scan_results = ctk.CTkFrame(self.deep_body, fg_color="transparent")
        self.deep_scan_results.grid(row=2, column=0, sticky="ew")
        self.deep_scan_results.grid_columnconfigure(0, weight=1)

    def _start_basic_scan(self):
        self._basic_scan_thread = threading.Thread(target=self._run_basic_scan, daemon=True)
        self._basic_scan_thread.start()

    def _run_basic_scan(self):
        try:
            result = scan_basic_validation_matches(
                self.concepts,
                progress_callback=self._queue_progress,
                match_callback=self._queue_match,
                stop_event=self._stop_event,
            )
            self._event_queue.put(("basic_done", result))
        except RuntimeError as error:
            if str(error) != "Validation checker scan cancelled.":
                self._event_queue.put(("error", str(error)))
        except Exception as error:
            self._event_queue.put(("error", str(error)))

    def _start_deep_scan(self):
        if self.training_is_active():
            messagebox.showwarning("Training Active", "Stop training before running Deep Scan.")
            return
        if self._scan_result is None:
            messagebox.showwarning("Still Scanning", "Wait for the initial exact/perceptual scan to finish first.")
            return
        if self._deep_scan_thread is not None and self._deep_scan_thread.is_alive():
            return

        self._clip_matches = []
        self._render_clip_section()
        self.deep_scan_button.configure(state="disabled")
        self.deep_scan_progress_label.configure(text="Loading CLIP model...")
        self._deep_scan_thread = threading.Thread(target=self._run_deep_scan, daemon=True)
        self._deep_scan_thread.start()

    def _run_deep_scan(self):
        try:
            matches = scan_clip_similarity_matches(
                self._scan_result.train_images,
                self._scan_result.val_images,
                threshold=float(self.threshold_slider.get()),
                progress_callback=self._queue_deep_progress,
                stop_event=self._stop_event,
            )
            self._event_queue.put(("deep_done", matches))
        except RuntimeError as error:
            if str(error) != "Validation checker scan cancelled.":
                self._event_queue.put(("deep_error", str(error)))
        except Exception as error:
            self._event_queue.put(("deep_error", str(error)))

    def _queue_progress(self, label: str, current: int, total: int):
        self._event_queue.put(("progress", label, current, total))

    def _queue_deep_progress(self, label: str, current: int, total: int):
        self._event_queue.put(("deep_progress", label, current, total))

    def _queue_match(self, match: ValidationCheckerMatch):
        self._event_queue.put(("match", match))

    def _poll_queue(self):
        if self._stop_event.is_set():
            return

        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

        self._poll_after_id = self.after(100, self._poll_queue)

    def _handle_event(self, event):
        event_type = event[0]

        if event_type == "progress":
            _event_type, label, current, total = event
            self.progress_label.configure(text=f"{label}: {current}/{total}")
            self.progress_bar.set(current / max(total, 1))
        elif event_type == "match":
            match = event[1]
            if match.kind == "exact":
                self._exact_matches.append(match)
                self._render_match_section(self.exact_body, self.exact_heading, "Exact Duplicates", self._exact_matches)
            elif match.kind == "perceptual":
                self._perceptual_matches.append(match)
                self._render_match_section(
                    self.perceptual_body, self.perceptual_heading, "Perceptual Matches", self._perceptual_matches
                )
            self._update_summary_labels()
        elif event_type == "basic_done":
            self._scan_result = event[1]
            self._caption_matches = self._scan_result.caption_matches
            self.progress_label.configure(text="Initial scan complete")
            self.progress_bar.set(1)
            self._render_all_sections()
            self._update_summary_labels()
        elif event_type == "deep_progress":
            _event_type, label, current, total = event
            self.deep_scan_progress_label.configure(text=f"{label}: {current}/{total}")
        elif event_type == "deep_done":
            raw_matches = event[1]
            existing_keys = {self._match_pair_key(match) for match in self._exact_matches + self._perceptual_matches}
            self._clip_matches = [match for match in raw_matches if self._match_pair_key(match) not in existing_keys]
            self.deep_scan_button.configure(state="normal")
            self.deep_scan_progress_label.configure(text="Deep Scan complete.")
            self._render_clip_section()
            self._update_summary_labels()
        elif event_type == "deep_error":
            self.deep_scan_button.configure(state="normal")
            self.deep_scan_progress_label.configure(text=str(event[1]))
        elif event_type == "error":
            self.progress_label.configure(text=str(event[1]))
            self.footer_status.configure(text=str(event[1]))

    def _update_threshold_label(self, value):
        self.threshold_value_label.configure(text=f"Threshold: {value:.2f}")

    def _render_all_sections(self):
        self._render_match_section(self.exact_body, self.exact_heading, "Exact Duplicates", self._exact_matches)
        self._render_match_section(
            self.perceptual_body, self.perceptual_heading, "Perceptual Matches", self._perceptual_matches
        )
        self._render_clip_section()
        self._render_caption_section()
        self._update_summary_labels()

    def _render_clip_section(self):
        for child in self.deep_scan_results.winfo_children():
            child.destroy()
        self._render_match_section(
            self.deep_scan_results,
            self.deep_heading,
            "Deep Scan Matches",
            self._clip_matches,
            no_results_text="No CLIP matches yet.",
        )

    def _render_match_section(
        self,
        body,
        heading,
        title: str,
        matches: list[ValidationCheckerMatch],
        no_results_text: str = "No matches found.",
    ):
        for child in body.winfo_children():
            if child is self.deep_scan_results and body is self.deep_body:
                continue
            child.destroy()

        visible_matches = [match for match in matches if self._match_visible(match)]
        heading.configure(text=f"{title}: {len(visible_matches)} found")

        if not visible_matches:
            label = ctk.CTkLabel(body, text=no_results_text, anchor="w")
            label.grid(row=0, column=0, sticky="ew")
            return

        self._thumbnail_refs = []
        for row_index, match in enumerate(visible_matches):
            row = ctk.CTkFrame(body)
            row.grid(row=row_index, column=0, sticky="ew", pady=(0, 8))
            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=1)
            row.grid_columnconfigure(2, weight=0)

            self._create_image_panel(row, match.train_image, 0)
            self._create_image_panel(row, match.val_image, 1)

            controls = ctk.CTkFrame(row, fg_color="transparent")
            controls.grid(row=0, column=2, sticky="ne", padx=(12, 0), pady=8)

            detail = ctk.CTkLabel(
                controls,
                text=match.detail if match.score is None else f"{match.detail}",
                anchor="w",
                justify="left",
                wraplength=260,
            )
            detail.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

            remove_button = ctk.CTkButton(
                controls,
                text="Remove from Val",
                command=lambda current_match=match: self._remove_match(current_match),
                state="normal" if match.val_image.concept.is_local else "disabled",
            )
            remove_button.grid(row=1, column=0, sticky="ew", pady=(0, 6))

            move_enabled = match.val_image.concept.is_local and bool(self._local_train_concepts)
            move_button = ctk.CTkButton(
                controls,
                text="Move to Train",
                command=lambda current_match=match: self._move_match(current_match),
                state="normal" if move_enabled else "disabled",
            )
            move_button.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(0, 6))

            explorer_button = ctk.CTkButton(
                controls,
                text="Open in Explorer",
                command=lambda current_match=match: self._open_match_in_explorer(current_match),
            )
            explorer_button.grid(row=2, column=0, sticky="ew")

            skip_button = ctk.CTkButton(
                controls,
                text="Skip",
                fg_color="gray40",
                command=lambda current_match=match: self._skip_match(current_match),
            )
            skip_button.grid(row=2, column=1, sticky="ew", padx=(8, 0))

    def _render_caption_section(self):
        for child in self.caption_body.winfo_children():
            child.destroy()

        visible_matches = [match for match in self._caption_matches if match.val_images]
        self.caption_heading.configure(
            text=f"Caption Matches: {len(visible_matches)} prompts appear in both train and val"
        )

        if not visible_matches:
            ctk.CTkLabel(self.caption_body, text="No exact caption matches found.", anchor="w").grid(
                row=0, column=0, sticky="ew"
            )
            return

        matches_to_show = visible_matches if self._show_all_captions else visible_matches[:5]
        for row_index, match in enumerate(matches_to_show):
            preview = match.caption if len(match.caption) <= 110 else match.caption[:107] + "..."
            count_text = f'"{preview}" — {len(match.train_images)} train, {len(match.val_images)} val'
            ctk.CTkLabel(self.caption_body, text=count_text, anchor="w", justify="left", wraplength=1180).grid(
                row=row_index,
                column=0,
                sticky="ew",
                pady=(0, 4),
            )

        if len(visible_matches) > 5:
            toggle_text = "Show Less" if self._show_all_captions else "Show All"
            ctk.CTkButton(
                self.caption_body,
                text=toggle_text,
                fg_color="gray40",
                command=self._toggle_caption_visibility,
                width=120,
            ).grid(row=len(matches_to_show), column=0, sticky="w", pady=(6, 0))

    def _create_image_panel(self, master, image, column: int):
        frame = ctk.CTkFrame(master)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 8), pady=8)
        frame.grid_columnconfigure(0, weight=1)

        thumbnail = self._load_thumbnail(image.path)
        image_widget = ctk.CTkLabel(frame, text="", image=thumbnail)
        image_widget.image = thumbnail
        image_widget.grid(row=0, column=0, sticky="n", padx=8, pady=(8, 6))

        name_widget = ctk.CTkLabel(frame, text=image.display_name, wraplength=250, justify="left")
        name_widget.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        self._thumbnail_refs.append(thumbnail)

    def _load_thumbnail(self, path: str, width: int = 220, height: int = 220) -> ctk.CTkImage:
        try:
            with Image.open(path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
                thumb = image.copy()
                image.close()
        except Exception:
            thumb = Image.new("RGB", (width, height), (48, 48, 48))
        return ctk.CTkImage(light_image=thumb, size=thumb.size)

    def _match_visible(self, match: ValidationCheckerMatch) -> bool:
        if match.val_image.path in self._resolved_validation_paths:
            return False
        return self._match_key(match) not in self._dismissed_matches

    def _match_key(self, match: ValidationCheckerMatch) -> tuple[str, str, str]:
        return match.kind, match.train_image.path, match.val_image.path

    def _match_pair_key(self, match: ValidationCheckerMatch) -> tuple[str, str]:
        return match.train_image.path, match.val_image.path

    def _remove_match(self, match: ValidationCheckerMatch):
        if not messagebox.askyesno(
            "Remove from Validation", f"Remove {os.path.basename(match.val_image.path)} from validation?"
        ):
            return
        try:
            remove_validation_image(match.val_image)
        except Exception as error:
            messagebox.showerror("Remove Failed", str(error))
            return
        self._resolve_validation_path(match.val_image.path)

    def _move_match(self, match: ValidationCheckerMatch):
        target_concept = self._choose_train_concept(match.val_image.concept)
        if target_concept is None:
            return

        original_path = match.val_image.path
        try:
            move_validation_image_to_train(match.val_image, target_concept)
        except Exception as error:
            messagebox.showerror("Move Failed", str(error))
            return
        self._resolve_validation_path(original_path)

    def _open_match_in_explorer(self, match: ValidationCheckerMatch):
        try:
            open_in_explorer(match.val_image.path)
        except Exception as error:
            messagebox.showerror("Open in Explorer Failed", str(error))

    def _skip_match(self, match: ValidationCheckerMatch):
        self._dismissed_matches.add(self._match_key(match))
        self._render_all_sections()

    def _remove_all_flagged(self):
        removable_matches = [
            match
            for match in (self._exact_matches + self._perceptual_matches + self._clip_matches)
            if self._match_visible(match) and match.val_image.concept.is_local
        ]
        unique_paths = sorted({match.val_image.path for match in removable_matches})
        if not unique_paths:
            messagebox.showinfo("Nothing to Remove", "There are no local flagged validation images left to remove.")
            return
        if not messagebox.askyesno(
            "Remove All Flagged", f"Remove {len(unique_paths)} flagged validation image(s) and their .txt sidecars?"
        ):
            return

        failures = []
        for path in unique_paths:
            match = next(match for match in removable_matches if match.val_image.path == path)
            try:
                remove_validation_image(match.val_image)
                self._resolve_validation_path(path)
            except Exception as error:
                failures.append(f"{os.path.basename(path)}: {error}")

        if failures:
            messagebox.showerror("Bulk Remove Incomplete", "\n".join(failures))

    def _resolve_validation_path(self, path: str):
        self._resolved_validation_paths.add(path)
        for caption_match in self._caption_matches:
            caption_match.val_images = [image for image in caption_match.val_images if image.path != path]
        self._caption_matches = [match for match in self._caption_matches if match.val_images]
        self._render_all_sections()

    def _toggle_caption_visibility(self):
        self._show_all_captions = not self._show_all_captions
        self._render_caption_section()

    def _choose_train_concept(self, validation_concept: ValidationCheckerConcept) -> ValidationCheckerConcept | None:
        if not self._local_train_concepts:
            messagebox.showwarning(
                "No Local Training Concepts", "Move to Train requires at least one local training concept."
            )
            return None
        if len(self._local_train_concepts) == 1:
            return self._local_train_concepts[0]

        dialog = _TrainConceptChooser(self, self._local_train_concepts, validation_concept)
        self.wait_window(dialog)
        return dialog.selected_concept

    def _update_summary_labels(self):
        visible_exact = sum(1 for match in self._exact_matches if self._match_visible(match))
        visible_perceptual = sum(1 for match in self._perceptual_matches if self._match_visible(match))
        visible_clip = sum(1 for match in self._clip_matches if self._match_visible(match))
        visible_captions = len([match for match in self._caption_matches if match.val_images])
        leak_count = visible_exact + visible_perceptual + visible_clip

        self.summary_label.configure(
            text=f"Summary: {leak_count} potential leaks found. {visible_captions} caption-match warnings."
        )
        self.footer_status.configure(
            text=f"Exact: {visible_exact}  Perceptual: {visible_perceptual}  Deep Scan: {visible_clip}  Captions: {visible_captions}"
        )
        self.remove_all_button.configure(state="normal" if leak_count else "disabled")

    def _on_close(self):
        self._stop_event.set()
        if self._poll_after_id:
            self.after_cancel(self._poll_after_id)
        self.grab_release()
        self.destroy()


class _TrainConceptChooser(ctk.CTkToplevel):
    def __init__(self, parent, concepts: list[ValidationCheckerConcept], validation_concept: ValidationCheckerConcept):
        super().__init__(parent)

        self.selected_concept: ValidationCheckerConcept | None = None
        self._concepts = concepts

        self.title("Move to Train")
        self.geometry("520x180")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text=f"Choose a training concept for {validation_concept.name or os.path.basename(validation_concept.configured_path)}",
            wraplength=460,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))

        best_match = self._best_match_index(validation_concept)
        self._selected_name = ctk.StringVar(value=self._display_name(concepts[best_match]))
        self._options = {self._display_name(concept): concept for concept in concepts}
        option_menu = ctk.CTkOptionMenu(self, values=list(self._options.keys()), variable=self._selected_name)
        option_menu.grid(row=1, column=0, sticky="ew", padx=20)

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.grid(row=2, column=0, sticky="e", padx=20, pady=20)
        ctk.CTkButton(button_row, text="Cancel", fg_color="gray40", command=self._cancel).grid(
            row=0, column=0, padx=(0, 8)
        )
        ctk.CTkButton(button_row, text="Move", command=self._confirm).grid(row=0, column=1)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()

    def _display_name(self, concept: ValidationCheckerConcept) -> str:
        return concept.name or os.path.basename(concept.configured_path) or concept.configured_path

    def _best_match_index(self, validation_concept: ValidationCheckerConcept) -> int:
        validation_name = (validation_concept.name or os.path.basename(validation_concept.configured_path)).lower()
        for index, concept in enumerate(self._concepts):
            candidate = (concept.name or os.path.basename(concept.configured_path)).lower()
            if candidate == validation_name:
                return index
        return 0

    def _confirm(self):
        self.selected_concept = self._options[self._selected_name.get()]
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.selected_concept = None
        self.grab_release()
        self.destroy()
