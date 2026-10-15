import json
import os
from tkinter import messagebox

from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_curation_util import (
    check_dpo_pairs,
    correct_all_captions_to_chosen,
    find_caption_mismatches,
    fix_multiline_captions,
    remove_finalized_pair,
    repair_rejected_pairs,
)
from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.RLHFAddonSchedule import RLHFAddonSchedule
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TextRewardSource import TextRewardSource
from modules.util.text_reward_util import load_word_list
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class RLHFTab:
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

        training_type = (
            "DPO (Existing Adapter)"
            if self.train_config.effective_dpo_ref_mode() == DPORefMode.EXISTING_ADAPTER
            else "DPO (New Adapter)"
        )

        mode_options = [
            ("DPO", RLHFMode.DPO),
            ("SFT + DPO", RLHFMode.SFT_DPO),
        ]
        components.label(
            self.scroll_frame,
            0,
            0,
            "RLHF Mode",
            tooltip="DPO: the chosen/rejected pairs drive the run; standard concepts become a weighted SFT anchor. "
            "SFT + DPO: your standard concepts drive the run at full weight exactly like a normal finetune, and "
            "a DPO pair is interleaved every N micro-batches as a weighted add-on. Set Repeats on the DPO concept "
            "to control how much preference signal enters each epoch. KTO concepts (single images labelled Liked / "
            "Disliked) are a third loader in both modes: added to every pair step in DPO mode, interleaved like the "
            "pairs in SFT + DPO mode, and the main loader of a DPO-mode run that has no pairs.",
        )
        components.options_kv(self.scroll_frame, 0, 1, mode_options, self.ui_state, "rlhf_mode")

        ref_mode_options = [
            ("New Adapter (base model)", DPORefMode.NEW_ADAPTER),
            ("Existing Adapter (snapshot)", DPORefMode.EXISTING_ADAPTER),
        ]
        components.label(
            self.scroll_frame,
            0,
            3,
            "Reference",
            tooltip="What DPO measures preferences against. New Adapter: the base model with the adapter switched "
            "off. Existing Adapter: the adapter as it stands at the first DPO step, snapshotted once and saved "
            "with backups - pick this when resuming a finetune from a backup so DPO refines the current weights "
            "instead of the bare base. A loaded adapter file always acts as the reference.",
        )
        components.options_kv(self.scroll_frame, 0, 4, ref_mode_options, self.ui_state, "rlhf_dpo_ref_mode")

        components.label(
            self.scroll_frame,
            1,
            0,
            "Beta",
            tooltip="How strongly the model follows your preferences. Higher values make changes more conservative, "
            "lower values make them more aggressive. Start with the default unless you have a reason to change it.",
        )
        components.entry(self.scroll_frame, 1, 1, self.ui_state, "rlhf_dpo_beta")

        components.label(
            self.scroll_frame,
            1,
            3,
            "Enable RLHF",
            tooltip="Turns on preference training. Use this with LoRA training and chosen/rejected concept pairs.",
        )
        components.switch(self.scroll_frame, 1, 4, self.ui_state, "rlhf_enabled")

        components.label(
            self.scroll_frame,
            2,
            0,
            "Label Smoothing",
            tooltip="Helps when your choices were a bit uncertain. Increase it slightly if you were not fully confident in some of the chosen/rejected picks.",
        )
        components.entry(self.scroll_frame, 2, 1, self.ui_state, "rlhf_dpo_label_smoothing")

        components.label(
            self.scroll_frame,
            2,
            3,
            "DPO Validation",
            tooltip="Checks DPO on a held-out set of chosen/rejected pairs so you can see whether preference training is generalizing.",
        )
        components.switch(self.scroll_frame, 2, 4, self.ui_state, "rlhf_dpo_validation")

        components.label(
            self.scroll_frame,
            3,
            0,
            "Supervised Mix",
            tooltip="Blends an NLL term on the CHOSEN images of each DPO pair into the loss. "
            "Anchors the policy to its own chosen samples.",
        )
        components.entry(self.scroll_frame, 3, 1, self.ui_state, "rlhf_supervised_mix")

        objective_options = [
            ("DPO (sigmoid)", DPOObjective.SIGMOID),
            ("IPO", DPOObjective.IPO),
            ("KTO", DPOObjective.KTO),
        ]
        components.label(
            self.scroll_frame,
            4,
            0,
            "Objective",
            tooltip="DPO (sigmoid): the standard preference loss, scaled by Beta — pushes the preference margin "
            "ever higher, which can reward-hack on long runs. "
            "IPO: regresses the margin toward the fixed target 1/(2*Tau) instead — bounded by construction, "
            "more resistant to reward hacking. Beta and Label Smoothing do not apply to IPO. "
            "KTO: scores each half of the pair on its own against the reference (chosen = desirable, rejected = "
            "undesirable) through a saturating sigmoid utility scaled by Beta. The chosen half must actually rise "
            "above the reference, so it resists the chosen-likelihood drift DPO allows. Label Smoothing and "
            "Adaptive Beta do not apply.",
        )
        components.options_kv(self.scroll_frame, 4, 1, objective_options, self.ui_state, "rlhf_dpo_objective")

        components.label(
            self.scroll_frame,
            5,
            0,
            "IPO Tau",
            tooltip="IPO regularization strength; the target preference margin is 1/(2*Tau). Diffusion DPO margins "
            "are tiny (around 1e-3), so Tau in the hundreds to thousands gives realistic targets. "
            "Larger Tau = smaller target = gentler training.",
        )
        components.entry(self.scroll_frame, 5, 1, self.ui_state, "rlhf_dpo_ipo_tau")

        components.label(
            self.scroll_frame,
            6,
            0,
            "KTO Weights",
            tooltip="KTO only. Desirable / undesirable weights (lambda_D / lambda_U): how hard the chosen half is "
            "pulled above the reference and the rejected half pushed below it. Paired data has equal counts, so "
            "1 / 1 is the paper's balance; raise one side to favour it. The KTO reference point is fixed at 0 "
            "(no mismatched-pair KL estimate; there is none to take at batch size 1).",
        )
        components.entry(self.scroll_frame, 6, 1, self.ui_state, "rlhf_kto_desirable_weight")
        components.entry(self.scroll_frame, 6, 2, self.ui_state, "rlhf_kto_undesirable_weight")

        components.label(
            self.scroll_frame,
            7,
            0,
            "Adaptive Beta",
            tooltip="Adjusts Beta per optimizer step from the smoothed reward margin (beta-DPO): informative "
            "batches with above-average margins raise Beta, uninformative ones lower it, bounded to "
            "[Beta/4, Beta*4]. Beta is the starting point. Only applies to the DPO (sigmoid) objective. "
            "Note that Beta and learning rate are coupled - if you double Beta, halve the learning rate.",
        )
        components.switch(self.scroll_frame, 7, 1, self.ui_state, "rlhf_dpo_adaptive_beta")

        components.label(
            self.scroll_frame,
            8,
            0,
            "SFT Anchor Weight",
            tooltip="Weight on a supervised loss computed from your STANDARD concepts during a DPO run. "
            "When >0 and standard concepts are present, the loader runs them in parallel and adds "
            "weight * sft_loss to the DPO loss. Set to 0 to silently drop standard concepts (legacy behavior).",
        )
        components.entry(self.scroll_frame, 8, 1, self.ui_state, "rlhf_sft_anchor_weight")

        components.label(
            self.scroll_frame,
            3,
            3,
            "Validation %",
            tooltip="How many prompt groups the DPO Pair Tool leaves out for validation.",
        )
        components.entry(self.scroll_frame, 3, 4, self.ui_state, "rlhf_dpo_validation_percentage")

        components.label(
            self.scroll_frame,
            4,
            3,
            "Early Stopping",
            tooltip="Stops DPO training when the validation signal stops improving.",
        )
        components.switch(self.scroll_frame, 4, 4, self.ui_state, "rlhf_dpo_patience_enabled")

        components.label(
            self.scroll_frame,
            5,
            3,
            "Patience",
            tooltip="How many validation checks can pass without improvement before training stops.",
        )
        components.entry(self.scroll_frame, 5, 4, self.ui_state, "rlhf_dpo_patience_value")

        components.label(
            self.scroll_frame,
            7,
            3,
            "Save Best",
            tooltip="Saves a checkpoint when validation accuracy hits a new high. "
            "The best checkpoint is restored at the end of training.",
        )
        components.switch(self.scroll_frame, 7, 4, self.ui_state, "rlhf_dpo_save_best")

        components.label(
            self.scroll_frame,
            8,
            3,
            "Timestep Margins",
            tooltip="Logs the reward margin bucketed by timestep quartile (dpo/margin_by_t/q1..q4) to TensorBoard. "
            "Useful to verify pairs are compared evenly across noise levels.",
        )
        components.switch(self.scroll_frame, 8, 4, self.ui_state, "rlhf_dpo_timestep_margin_logging")

        components.label(
            self.scroll_frame,
            9,
            0,
            "Per-Concept Metrics",
            tooltip="Logs margin, accuracy, rewards, applied beta and gradient weight for each DPO concept "
            "separately (dpo/concept/<folder>/...) to TensorBoard, instead of only the blend of all of them.",
        )
        components.switch(self.scroll_frame, 9, 1, self.ui_state, "rlhf_dpo_concept_logging")

        components.label(
            self.scroll_frame,
            9,
            3,
            "Live Saturation",
            tooltip="Tracks every DPO pair while training and writes <workspace>/dpo_live/: report.md (per concept: "
            "pairs done / learning / violating, and where each beta sits relative to its push peak), pairs.csv "
            "(one row per pair) and ledger.json. Margins are relative to this run's DPO reference; a rebase "
            "archives the history and starts over.",
        )
        components.switch(self.scroll_frame, 9, 4, self.ui_state, "rlhf_dpo_live_saturation")

        components.label(
            self.scroll_frame,
            11,
            0,
            "DPO Add-on Weight",
            tooltip="SFT + DPO mode only. Multiplier on the DPO loss when a pair is interleaved into the standard "
            "run; the standard loss always stays at full weight. 1.0 adds the DPO term as-is. Also scales the "
            "interleaved KTO batches of KTO concepts (Liked / Disliked singles).",
        )
        components.entry(self.scroll_frame, 11, 1, self.ui_state, "rlhf_dpo_addon_weight")

        components.label(
            self.scroll_frame,
            11,
            3,
            "DPO Add-on Interval",
            tooltip="SFT + DPO mode, Piggyback schedule only. Interleave one DPO pair every N standard micro-batches. "
            "0 = automatic: derived each epoch from the dataset sizes (including the DPO concept's Repeats) so "
            "both finish an epoch together. Ignored by the Independent schedule.",
        )
        components.entry(self.scroll_frame, 11, 4, self.ui_state, "rlhf_dpo_addon_interval")

        schedule_options = [
            ("Piggyback (add-on rides on standard steps)", RLHFAddonSchedule.PIGGYBACK),
            ("Independent (own micro-step per batch)", RLHFAddonSchedule.INDEPENDENT),
        ]
        components.label(
            self.scroll_frame,
            6,
            3,
            "Add-on Schedule",
            tooltip="How DPO pairs, KTO batches and text reward rollouts are placed among the standard micro-steps "
            "(SFT + DPO mode and the text reward). Piggyback: every micro-step is one standard batch and an add-on "
            "term rides along every N of them (the interval on the right; lowest peak memory, but a text reward "
            "is never allowed on a DPO step and starves when the interval reaches 1, and an interval above 1 "
            "leaves part of the DPO data unvisited). Independent: every standard batch, DPO pair batch, KTO batch "
            "and text rollout is its own micro-step, interleaved evenly through the epoch, so every source is "
            "consumed at its full length and Repeats; the epoch has as many micro-steps as all sources together "
            "and gradient accumulation counts them all. The interval is ignored. Each term keeps its weight; peak "
            "memory is one term's graph.",
        )
        components.options_kv(self.scroll_frame, 6, 4, schedule_options, self.ui_state, "rlhf_addon_schedule")

        components.label(
            self.scroll_frame,
            10,
            3,
            "DPO Add-on Graph Reuse",
            tooltip="SFT + DPO mode only. Run the rejected image's forward with gradients first and keep its graph "
            "until the chosen image has been backwarded, instead of an extra no-grad forward just to read its loss. "
            "Same gradients, one model forward fewer per DPO step, about 1 GB more VRAM at 1024px while both graphs "
            "are held. Ignored (the extra forward is kept) when gradient checkpointing is off or CPU offloaded with "
            "activation or layer offloading.",
        )
        components.switch(self.scroll_frame, 10, 4, self.ui_state, "rlhf_dpo_light_reuse_rejected_graph")

        components.label(
            self.scroll_frame,
            12,
            0,
            "Rebase Reference on Start",
            tooltip="Existing Adapter reference only. When resuming from a backup, discard the reference snapshot "
            "saved with it and take a fresh one from the adapter as it stands at the first DPO step of this run. "
            "Use this when the pairs have stopped contributing: once the policy has drifted far from the reference "
            "the reward margins saturate and the DPO term goes flat; rebasing resets every margin to zero so the "
            "same pairs bite again. Applies on every start while enabled - turn it off after the resume.",
        )
        components.switch(self.scroll_frame, 12, 1, self.ui_state, "rlhf_dpo_ref_rebase")

        components.label(
            self.scroll_frame,
            12,
            3,
            "Rebase Reference Every",
            tooltip="Existing Adapter reference only. Re-snapshot the reference from the current adapter every N "
            "steps (global step, as shown in TensorBoard), right after an optimizer step. 0 = never. A resumed run "
            "waits for the next multiple of N rather than rebasing immediately.",
        )
        components.entry(self.scroll_frame, 12, 4, self.ui_state, "rlhf_dpo_ref_rebase_interval")

        # --- on-image text reward (frozen OCR, CTC through the VAE) ---
        components.label(
            self.scroll_frame,
            13,
            0,
            "Text Reward",
            tooltip="Add a DRaFT-1 style on-image text reward to the standard finetune (plain or SFT + DPO). Each "
            "reward step rolls the adapter out on a synthetic prompt such as 'Neon text reading \"Open 24h\"', "
            "decodes the text region, and backpropagates a frozen OCR's CTC loss into the adapter. Never runs on a "
            "micro-batch that already carries a DPO pair. Needs the python-doctr package.",
        )
        components.switch(self.scroll_frame, 13, 1, self.ui_state, "text_reward_enabled")
        components.label(
            self.scroll_frame,
            13,
            3,
            "Text Reward Weight",
            tooltip="Multiplier on the CTC loss when a reward rollout is interleaved. The standard loss stays at full "
            "weight. Scalar 'loss/text_reward' in TensorBoard already includes this weight.",
        )
        components.entry(self.scroll_frame, 13, 4, self.ui_state, "text_reward_weight")

        components.label(
            self.scroll_frame,
            14,
            0,
            "Text Samples / Epoch",
            tooltip="How many reward rollouts to spread over one epoch of the standard data. The interval between "
            "them is derived each epoch from the primary loader length, like the DPO add-on interval. Prompts are "
            "redrawn and pre-encoded at every epoch start.",
        )
        components.entry(self.scroll_frame, 14, 1, self.ui_state, "text_reward_samples_per_epoch")
        components.label(
            self.scroll_frame,
            14,
            3,
            "Rollout Steps",
            tooltip="Sampling steps of the no-grad rollout prefix. Only the final grad-bearing step (or a random "
            "step in the second half, see Random Grad Step) is differentiated, so this mostly sets rollout cost.",
        )
        components.entry(self.scroll_frame, 14, 4, self.ui_state, "text_reward_rollout_steps")

        components.label(
            self.scroll_frame,
            15,
            0,
            "Random Grad Step",
            tooltip="Take the grad-bearing one-step x0 prediction at a random step in the second half of the rollout "
            "instead of always the last step. Spreads the signal across noise levels where glyph structure is decided.",
        )
        components.switch(self.scroll_frame, 15, 1, self.ui_state, "text_reward_random_grad_step")
        components.label(
            self.scroll_frame,
            15,
            3,
            "Rollout CFG",
            tooltip="Classifier-free guidance scale for the no-grad rollout prefix (1.0 = off, as for Turbo-like "
            "adapters). Above 1.0 the prefix costs two forwards per step and the negative prompt below is encoded.",
        )
        components.entry(self.scroll_frame, 15, 4, self.ui_state, "text_reward_cfg_scale")

        components.label(
            self.scroll_frame,
            16,
            0,
            "Rollout Negative",
            tooltip="Negative prompt for the rollout prefix when Rollout CFG is above 1.0.",
        )
        components.entry(self.scroll_frame, 16, 1, self.ui_state, "text_reward_negative_prompt")
        components.label(
            self.scroll_frame,
            16,
            3,
            "Rollout Resolution",
            tooltip="Square edge of the reward rollout in pixels (multiple of 16).",
        )
        components.entry(self.scroll_frame, 16, 4, self.ui_state, "text_reward_resolution")

        components.label(
            self.scroll_frame,
            17,
            0,
            "Max Decode Pixels",
            tooltip="Edge cap of the text region decoded WITH gradient through the VAE (the memory-critical part). "
            "Words outside the capped window are read but not rewarded that step. 512 keeps the decode under ~1 GB "
            "with decoder checkpointing.",
        )
        components.entry(self.scroll_frame, 17, 1, self.ui_state, "text_reward_max_decode_pixels")
        components.label(
            self.scroll_frame,
            17,
            3,
            "Word List",
            tooltip="Optional text file, one word per line, replacing the built-in dictionary for the word, price, "
            "date, time, URL, abbreviation and number prompt families. Characters outside the OCR's vocabulary are "
            "dropped from targets.",
        )
        components.entry(self.scroll_frame, 17, 4, self.ui_state, "text_reward_word_list")

        components.label(
            self.scroll_frame,
            18,
            0,
            "Words Min / Max",
            tooltip="Number of words per prompt, drawn uniformly between these bounds.",
        )
        components.entry(self.scroll_frame, 18, 1, self.ui_state, "text_reward_min_words")
        components.entry(self.scroll_frame, 18, 2, self.ui_state, "text_reward_max_words")
        components.label(
            self.scroll_frame,
            18,
            3,
            "Save Sample Every",
            tooltip="Write every N-th reward rollout with what the OCR read to workspace/text_reward so reward "
            "hacking (text the OCR accepts but a person cannot read) is visible. 0 disables.",
        )
        components.entry(self.scroll_frame, 18, 4, self.ui_state, "text_reward_save_interval")

        components.label(
            self.scroll_frame,
            19,
            0,
            "Text Recognizer",
            tooltip="Recognizer behind the reward gradient. trocr-base-str (Microsoft TrOCR, scene-text fine-tune, "
            "~330M) reads stylised, cursive and neon text best; parseq (~23M) is the cheap docTR option; "
            "crnn_vgg16_bn is the weakest. Also used by the dataset tool to count boxed words.",
        )
        components.options(
            self.scroll_frame,
            19,
            1,
            ["trocr-base-str", "trocr-large-str", "trocr-base-printed", "parseq", "crnn_vgg16_bn"],
            self.ui_state,
            "text_reward_recognizer",
        )
        components.label(
            self.scroll_frame,
            19,
            3,
            "Text Reward Source",
            tooltip="Online Rollout: sample from noise on synthetic prompts every reward step (pays a multi-step "
            "prefix). Offline Images: start from pre-generated images (Turbo + adapter via SwarmUI, curated by the "
            "Text Alignment Dataset Tool): VAE-encode, re-noise to a late sigma and take the one grad step. About "
            "six times cheaper per sample and every sample carries a misspelling to correct.",
        )
        components.options_kv(
            self.scroll_frame,
            19,
            4,
            [("Online Rollout", TextRewardSource.ONLINE_ROLLOUT), ("Offline Images", TextRewardSource.OFFLINE_IMAGES)],
            self.ui_state,
            "text_reward_source",
        )

        components.label(
            self.scroll_frame,
            20,
            0,
            "Text Dataset Folder",
            tooltip="Offline Images only. Folder written by the Text Alignment Dataset Tool: image + .txt prompt "
            "(+ .json) per sample. Samples per Epoch is ignored; every image is visited once per epoch.",
        )
        components.path_entry(self.scroll_frame, 20, 1, self.ui_state, "text_reward_dataset_dir", mode="dir")
        components.label(
            self.scroll_frame,
            21,
            0,
            "Text Dataset Repeats",
            tooltip="Offline Images only. Visit every dataset image this many times per epoch; the reward interval "
            "is derived from images x repeats against the primary loader length, so raise it to fire more often "
            "from a small dataset.",
        )
        components.entry(self.scroll_frame, 21, 1, self.ui_state, "text_reward_dataset_repeats")
        components.label(
            self.scroll_frame,
            21,
            3,
            "Text Init Sigma",
            tooltip="Offline Images only. 0: re-noise the dataset image straight to the grad sigma and take one "
            "step (the plain offline reward; the crop is the source's own letters lightly re-denoised, so the "
            "gradient can barely change letter identity). Above 0 (try 0.8): re-noise to this high sigma where "
            "layout and surface survive but letters have not formed, roll the policy down to the grad step "
            "(CFG applies, costs those extra no-grad forwards), re-detect on the policy's own x0 and score that. "
            "Must sit above the grad sigma window.",
        )
        components.entry(self.scroll_frame, 21, 4, self.ui_state, "text_reward_init_sigma")
        components.label(
            self.scroll_frame,
            22,
            0,
            "Grad Sigma Min / Max",
            tooltip="Window of (shifted) sigma the grad-bearing step is drawn from; 0 / 1 keeps the original rule "
            "(the second half of the rollout grid). Letter identity is decided around 0.4-0.8; below ~0.3 the "
            "model only sharpens, and the gradient into the velocity is scaled by sigma anyway. Applies to all "
            "sources.",
        )
        components.entry(self.scroll_frame, 22, 1, self.ui_state, "text_reward_grad_sigma_min")
        components.entry(self.scroll_frame, 22, 2, self.ui_state, "text_reward_grad_sigma_max")
        components.label(
            self.scroll_frame,
            22,
            3,
            "Reuse Probe Graph",
            tooltip="Online Rollout and the hybrid offline init: the forward that finds the text boxes is also the "
            "grad-bearing forward (one transformer forward less per reward step, same maths). Its graph is held "
            "through the full-frame decode and the detector, which is only cheap with gradient checkpointing on "
            "(~1.1 GB at 1024px); without checkpointing this is ignored. Turn off if reward steps OOM.",
        )
        components.switch(self.scroll_frame, 22, 4, self.ui_state, "text_reward_reuse_probe_graph")
        components.label(
            self.scroll_frame,
            23,
            0,
            "Grad Steps (DRaFT-K)",
            tooltip="Consecutive grad-bearing Euler steps starting at the drawn grad step; the scored x0 is the "
            "one-step estimate after the last of them. 1 is the original single forward on the blurry mid-sigma "
            "mean. 2-3 lets the loss see letters as they resolve (CFG applies, unconditional branch detached). "
            "Each extra step holds one more transformer graph (~1.1 GB at 1024px with gradient checkpointing) and "
            "costs one more forward; clamped to the end of the rollout grid. Applies to all sources.",
        )
        components.entry(self.scroll_frame, 23, 1, self.ui_state, "text_reward_grad_steps")
        components.button(
            self.scroll_frame,
            20,
            3,
            "Write Wildcard",
            command=self._write_text_wildcard,
            tooltip="Write 2000 synthetic text prompts (same generator as the online reward) to "
            "SwarmUI/Data/Wildcards/text_alignment.txt so <wildcard:text_alignment> generates the offline set.",
        )

        components.button(
            self.scroll_frame,
            9,
            0,
            "Check Pairs",
            command=self._check_pairs,
            tooltip="Check that your chosen and rejected concept folders line up before training.",
        )
        components.button(
            self.scroll_frame,
            9,
            1,
            "Review Pairs",
            command=self._review_pairs,
            tooltip="Visually review your chosen/rejected image pairs. Remove bad pairs and their counterparts.",
        )
        components.button(
            self.scroll_frame,
            9,
            2,
            "Re-pair by Similarity",
            command=self._repair_rejected,
            tooltip="Within each caption group, rename rejected images so each is paired with the chosen image it most "
            "resembles (DINOv2 structural similarity). Sharpens the DPO signal by holding composition constant so the "
            "gradient keys on the quality difference. Renames rejected files in place — back up first.",
        )
        components.button(
            self.scroll_frame,
            9,
            3,
            "DPO Bucket Analysis",
            command=self._bucket_analysis,
            tooltip="Show per-aspect-bucket pair counts and what to add or remove for clean batches at a given batch size.",
        )

        components.label(
            self.scroll_frame,
            10,
            0,
            "Training Type:",
            tooltip="Shows whether DPO is starting from a fresh adapter or refining a loaded adapter. The output is "
            "always an adapter file. DPO requires LoRA training; full finetuning is not supported.",
        )
        components.label(
            self.scroll_frame,
            10,
            1,
            training_type,
            tooltip="DPO always writes an adapter file. New Adapter means the adapter starts from scratch. Existing Adapter means DPO refines a loaded adapter.",
        )

    def _write_text_wildcard(self):
        from modules.util.text_reward_util import write_wildcard_file

        path = os.path.join("E:/AI/Data/Packages/SwarmUI/Data/Wildcards", "text_alignment.txt")
        try:
            n = write_wildcard_file(
                path,
                2000,
                seed=0x7E57,
                words=load_word_list(self.train_config.text_reward_word_list),
                min_words=self.train_config.text_reward_min_words,
                max_words=self.train_config.text_reward_max_words,
            )
        except Exception as ex:
            messagebox.showerror("Write Wildcard", str(ex))
            return
        messagebox.showinfo("Write Wildcard", f"Wrote {n} prompts to {path}")

    def _check_pairs(self):
        try:
            concept_pairs = self._load_concept_pairs()
        except Exception as ex:
            messagebox.showerror("Check Pairs Error", str(ex))
            return

        try:
            result = check_dpo_pairs(concept_pairs)
        except Exception as ex:
            messagebox.showerror("Check Pairs Error", f"Error checking pairs: {ex}")
            return

        total_matched = result.get("total_matched", 0)
        total_chosen_stray = result.get("total_chosen_stray", 0)
        total_rejected_stray = result.get("total_rejected_stray", 0)

        lines = [f"Matched pairs: {total_matched}"]
        if total_chosen_stray > 0:
            lines.append(f"Chosen strays (no rejected match): {total_chosen_stray}")
        if total_rejected_stray > 0:
            lines.append(f"Rejected strays (no chosen match): {total_rejected_stray}")

        if result["format_stats"]:
            fmt_parts = [f"{ext}: {count}" for ext, count in sorted(result["format_stats"].items())]
            lines.append(f"\nFormats: {', '.join(fmt_parts)}")

        for pair_info in result["pairs"]:
            lines.append("\n--- Pair ---")
            chosen_path = pair_info.get("chosen_path")
            rejected_path = pair_info.get("rejected_path")
            chosen_stray = pair_info.get("chosen_stray", 0)
            rejected_stray = pair_info.get("rejected_stray", 0)
            lines.append(f"  Chosen: {chosen_path}")
            lines.append(f"  Rejected: {rejected_path}")
            lines.append(
                f"  Matched: {pair_info['matched']}, Chosen stray: {chosen_stray}, Rejected stray: {rejected_stray}"
            )

        multiline = result.get("multiline_captions", 0)
        if multiline:
            lines.append(f"\nMultiline captions: {multiline}")

        has_strays = total_chosen_stray > 0 or total_rejected_stray > 0
        if has_strays:
            lines.append(f"\nTotal strays: {total_chosen_stray + total_rejected_stray}")

        messagebox.showinfo("Check Pairs Results", "\n".join(lines))

        if has_strays:
            remove = messagebox.askyesno(
                "Remove Strays?",
                f"Found {total_chosen_stray + total_rejected_stray} stray file(s) with no matching pair.\n\n"
                f"Remove them and their caption files?",
            )
            if remove:
                removed = self._remove_strays(concept_pairs, result)
                messagebox.showinfo("Strays Removed", f"Removed {removed} stray file(s) and their captions.")

        if multiline:
            fix = messagebox.askyesno(
                "Fix Multiline Captions?",
                f"Found {multiline} caption file(s) with newlines.\n\n"
                f"Newlines are treated as separate captions by OneTrainer, which can "
                f"sabotage training. Flatten them to single-line captions?",
            )
            if fix:
                fixed = fix_multiline_captions(concept_pairs)
                messagebox.showinfo("Captions Fixed", f"Flattened {fixed} caption file(s) to single lines.")

        # Caption-content mismatch resolution: chosen.txt vs rejected.txt for
        # matched image pairs. Runs last so any earlier stray/multiline fixes
        # are reflected in the matched-pair set this scans.
        try:
            mismatches = find_caption_mismatches(concept_pairs)
        except Exception as ex:
            messagebox.showerror("Caption Mismatch Error", f"Error checking captions: {ex}")
            return

        if not mismatches:
            return

        from modules.ui.DPOCaptionMismatchWindow import (
            DPOCaptionMismatchChoiceDialog,
            DPOCaptionMismatchWindow,
        )

        dialog = DPOCaptionMismatchChoiceDialog(self.master.winfo_toplevel(), len(mismatches))
        self.master.winfo_toplevel().wait_window(dialog)
        choice = dialog.result
        if choice == "correct_all":
            corrected = correct_all_captions_to_chosen(mismatches)
            messagebox.showinfo(
                "Captions Corrected",
                f"Overwrote {corrected} rejected caption file(s) with the chosen caption.",
            )
        elif choice == "manual":
            DPOCaptionMismatchWindow(self.master.winfo_toplevel(), concept_pairs)

    def _remove_strays(self, concept_pairs, result) -> int:
        from modules.util.dpo_curation_util import dpo_pair_key
        from modules.util.path_util import supported_image_extensions

        exts = supported_image_extensions()
        removed = 0

        for i, (chosen_path, rejected_path) in enumerate(concept_pairs):
            pair_info = result["pairs"][i]
            if pair_info.get("chosen_stray", 0) == 0 and pair_info.get("rejected_stray", 0) == 0:
                continue

            chosen_keys: dict[str, str] = {}
            rejected_keys: dict[str, str] = {}

            for root, _dirs, files in os.walk(chosen_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in exts:
                        full = os.path.join(root, fname)
                        chosen_keys[dpo_pair_key(full, chosen_path)] = full

            for root, _dirs, files in os.walk(rejected_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in exts:
                        full = os.path.join(root, fname)
                        rejected_keys[dpo_pair_key(full, rejected_path)] = full

            matched = set(chosen_keys) & set(rejected_keys)

            for key, path in chosen_keys.items():
                if key not in matched:
                    remove_finalized_pair(path, None)
                    removed += 1
            for key, path in rejected_keys.items():
                if key not in matched:
                    remove_finalized_pair(None, path)
                    removed += 1

        return removed

    def _review_pairs(self):
        try:
            concept_pairs = self._load_concept_pairs()
        except Exception as ex:
            messagebox.showerror("Review Pairs Error", str(ex))
            return

        from modules.ui.DPOReviewWindow import DPOReviewWindow

        DPOReviewWindow(self.master.winfo_toplevel(), concept_pairs)

    def _bucket_analysis(self):
        from modules.ui.DPOBucketAnalysisWindow import DPOBucketAnalysisWindow

        DPOBucketAnalysisWindow(self.master.winfo_toplevel(), self.train_config)

    def _repair_rejected(self):
        try:
            concept_pairs = self._load_concept_pairs()
        except Exception as ex:
            messagebox.showerror("Re-pair Error", str(ex))
            return

        proceed = messagebox.askyesno(
            "Re-pair Rejected Images?",
            "This renames rejected image files in place so each is paired with the most visually similar chosen image "
            "in its caption group (computed with DINOv2). A quality floor (0.85) keeps any pair from being pushed "
            "below 0.85 cosine or below where it started, so re-pairing only helps. Single-pair captions are left "
            "untouched and no pairs are lost.\n\nThe first run downloads ~330MB of model weights. Back up your "
            "dataset before continuing.\n\nProceed?",
        )
        if not proceed:
            return

        # DINOv2 inference can take a while, so run it on a worker thread with an
        # indeterminate progress dialog rather than freezing the Tk main loop.
        parent = self.master.winfo_toplevel()
        dialog = ctk.CTkToplevel(parent)
        dialog.title("Re-pairing")
        dialog.transient(parent)
        dialog.resizable(False, False)
        ctk.CTkLabel(dialog, text="Computing embeddings and re-pairing rejected images...").pack(padx=20, pady=(20, 10))
        progress = ctk.CTkProgressBar(dialog, width=320, mode="indeterminate")
        progress.pack(padx=20, pady=(0, 20))
        progress.start()

        result: dict = {}

        def _worker():
            try:
                result["summary"] = repair_rejected_pairs(concept_pairs)
            except Exception as ex:  # surfaced on the UI thread by _poll
                result["error"] = str(ex)

        import threading

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        def _poll():
            if thread.is_alive():
                dialog.after(150, _poll)
                return
            progress.stop()
            dialog.destroy()
            if "error" in result:
                messagebox.showerror("Re-pair Error", f"Error re-pairing: {result['error']}")
                return
            summary = result.get("summary", {})
            messagebox.showinfo(
                "Re-pair Complete",
                f"Re-paired {summary.get('pairs_repaired', 0)} rejected image(s) across "
                f"{summary.get('groups_processed', 0)} multi-pair caption group(s).\n"
                f"Single-pair groups skipped: {summary.get('groups_skipped_single', 0)}\n"
                f"Total matched pairs considered: {summary.get('pairs_total', 0)}",
            )

        dialog.after(150, _poll)

    def _load_concept_pairs(self):
        concept_pairs = dpo_concept_pattern_dirs(self._load_concepts())
        if not concept_pairs:
            raise RuntimeError(
                "No DPO concepts found. Set the chosen/rejected patterns on a concept in the Concepts tab."
            )
        return concept_pairs

    def _load_concepts(self):
        concepts = self.train_config.concepts
        if concepts is None:
            concept_file = self.train_config.concept_file_name
            if not concept_file or not os.path.isfile(concept_file):
                raise RuntimeError("No concepts configured. Set up concepts in the Concepts tab first.")
            from modules.util.config.ConceptConfig import ConceptConfig

            with open(concept_file, "r") as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]
        return concepts
