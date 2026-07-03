# Review: PR #1403 — RLHF DPO (Direct Preference Optimization) training

**Date:** 2026-07-02
**Target:** https://github.com/Nerogar/OneTrainer/pull/1403 (branch `RLHF`)
**Review worktree:** [`E:\AI\Data\Packages\OneTrainer-RLHF`](file:///E:/AI/Data/Packages/OneTrainer-RLHF) — all file:line references below are relative to this checkout
**Base:** Nerogar/OneTrainer `master` @ `07254ad0` (branch is merged up to upstream tip)
**Diff:** 43 files, +6,163 / −135, 47 non-merge commits

**Verdict: not ready to confirm as done.** The PR description documents a design that no longer exists in the code, a small amount of fork churn/leakage is present, and verification confirmed several real correctness bugs — one of which hard-crashes SDXL DPO on the first step. The curation tooling **stays in the PR** (decision 2026-07-02: DPO training is useless without a way to build datasets for it), but it gets slimmed: the similarity re-pairing feature and the caption-mismatch finder are removed, and the remaining tooling issues are fixed in place (§2).

---

## 1. Confirmed correctness findings (most severe first)

### 1.1 `modules/modelSetup/BaseModelSetup.py:167` — SDXL DPO crashes on the first training step

`_create_dpo_batched_batch` only duplicates `torch.Tensor` values into the 2B chosen+rejected batch. SDXL's `original_resolution` / `crop_resolution` / `crop_offset` collate into *lists* of tensors (default_collate transposes per-sample tuples), so they stay size B while latents become 2B. `add_time_ids` comes out `[B,6]` against `[2B]` embeddings, and the UNet's `add_embedding` raises a shape RuntimeError on the very first step. Only SDXL consumes these keys — Flux/SD3/other families are unaffected.

**Fix:** duplicate list/tuple-of-tensor entries elementwise in `_create_dpo_batched_batch`.

### 1.2 `modules/modelSetup/mixin/ModelSetupNoiseMixin.py:78` — text-encoder dropout is not paired between chosen and rejected

`_apply_dpo_paired_rng` pairs only noise and timesteps. Every model family's `encode_text` draws conditioning dropout per-sample across the full 2B batch (e.g. `StableDiffusionXLModel.py:277-288`), so with `dropout_probability > 0`, ~2·p·(1−p) of pairs compare chosen-with-prompt against rejected-with-zeroed-prompt — silently corrupted preference gradients, no error raised. Ref and policy forwards share the same `batch_seed`, so the corruption is purely the chosen/rejected asymmetry in the margin. Architecture-wide.

**Fix:** pair the dropout mask between batch halves (same mechanism as `_apply_dpo_paired_rng`), or disable conditioning dropout under DPO.

### 1.3 `modules/modelSetup/BaseModelSetup.py:363` — resume-from-backup silently moves the "frozen" DPO reference

In EXISTING_ADAPTER mode the reference is lazily cloned from live adapter weights at the first `calculate_dpo_loss` call. On resume, `GenericTrainer.py:104-116` sets `model_names.lora = last_backup_path`, so the post-restore lazy capture clones the *already-DPO-trained* adapter: rewards/margins discontinuously reset and the KL anchor drifts on every resume — contradicting the code's own "frozen for the entire training run" comment. `_dpo_ref_params` is never persisted.

**Fix:** load the reference from `config.lora_model_name` (which is untouched by the resume override) at capture time, instead of cloning live weights.

### 1.4 `modules/trainer/GenericTrainer.py:373` + `modules/dataLoader/BaseDataLoader.py:39` — `rlhf_dpo_validation=True` with `rlhf_enabled=False` kills a normal run mid-training

The trainer's DPO-validation branch checks only `rlhf_dpo_validation`; the loader gates the paired pipeline on `rlhf_dpo_validation and rlhf_enabled`. With the flags disagreeing (both are independently settable switches in the RLHF tab; nothing sanitizes the combination), validation batches lack `latent_image_rejected` and `calculate_dpo_loss` raises `RuntimeError` at the first validation interval — hours into a run. `config.validation` is not required: `rlhf_dpo_validation` alone creates the validation loader (`GenericTrainer.py:164`, `:1032`).

**Related startup trap:** in a DPO run, enabling plain `validation` without a pattern-bearing VALIDATION concept raises at startup (`DataLoaderMgdsMixin.py:51-54`). The Pair Tool's exported `concepts.json` does auto-create a valid VALIDATION concept (`dpo_curation_util.py:618-634`), but users with hand-built datasets hit an undocumented requirement — no tooltip mentions it.

**Fix:** gate the trainer branch on both flags (or derive one from the other), and document/validate the VALIDATION-concept requirement.

### 1.5 `modules/trainer/GenericTrainer.py:463-469` and `:384` — early stopping and best-checkpoint selection are defeated in exactly the reward-hacking regime they exist for

Two compounding confirmed bugs:

- **(a) Margin tiebreaker:** at equal (rounded) accuracy, a larger validation reward margin counts as a new best. Validation batch size is forced to 1, so small val sets saturate accuracy at exactly 1.0 early — after which the margin (precisely the hackable quantity the code's own comment at `GenericTrainer.py:392-394` excludes; loss→0 is margin→∞) becomes the sole criterion. Patience never increments, `commands.stop()` never fires, and `backup/dpo-best.pt` is continuously overwritten with progressively hacked weights that the end-of-run restore (`:1065-1076`) bakes into the final model.
- **(b) Nondeterministic validation:** DPO validation calls `calculate_dpo_loss` → `predict()` without `deterministic=True` (standard validation passes it at `:421-423`; `calculate_dpo_loss` has no such parameter). Each validation pass evaluates the held-out pairs at different `global_step`-seeded noise/timesteps, so `dpo/val_accuracy` and margin fluctuate with identical weights — and patience/save-best key off that RNG noise. Upward jitter above the recorded best re-crowns a checkpoint via (a).

**Fix:** deterministic seeding for the DPO validation path (note `deterministic=True` as-is pins all samples to t=0.5 — a fixed per-sample seed is preferable), plus a margin-insensitive or margin-capped best/patience criterion.

> **Decision 2026-07-02:** patience/save-best is being removed from the PR entirely (§2.3), which eliminates this finding — (b) downgrades to a metrics-smoothness nit once val metrics are only logged, never acted on.

### 1.6 `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py:397,401,417` — cache group-key change invalidates every existing user's disk cache

`concept.dpo_chosen_pattern` / `concept.dpo_rejected_pattern` were appended unconditionally to `variations_group_in_name` for both DiskCaches and VariationSorting. Verified at the pinned mgds@`9320a69`: the group key is `sha256(json.dumps([...values...]))` and the on-disk cache directory is named by it — growing the list 4→6 changes the hash for **every** concept, DPO or not (the fields exist with default `""` after ConceptConfig round-trip). Old-hash directories are never cleaned up. Users with `clear_cache_before_training=False` — i.e. everyone the cache matters to — get a silent full re-encode (hours on large datasets) plus orphaned old cache doubling disk usage.

**Fix:** include the pattern values in the key only when non-empty (keeps the key byte-identical to upstream's 4-element form for all non-DPO concepts) — the code comment's correctness rationale only applies when patterns are set.

### 1.7 `modules/trainer/GenericTrainer.py:986` — adaptive beta only updates on rank 0 in multi-GPU

`set_dpo_runtime_beta` sits inside `if multi.is_master():`. Each rank is a separate process with its own `model_setup`, so non-master ranks keep `_dpo_runtime_beta=None` and fall back to static `config.rlhf_dpo_beta` (`BaseModelSetup.py:214`) forever — averaged gradients silently mix losses at different beta scales. Nothing blocks multi-GPU + RLHF (`create.py:1376-1378` selects MultiTrainer purely on `config.multi_gpu`; the only RLHF guard is LoRA-only). Divergence is bounded by the controller's [β₀/4, 4β₀] clamp — moderate severity.

**Deeper issue:** `accumulated_dpo_metrics` is rank-local and never all-reduced, so hoisting the call out of `is_master()` is not enough — the margin needs a cross-rank reduction (or the beta a broadcast).

### 1.8 `modules/ui/RLHFTab.py:150` — `rlhf_dpo_validation_percentage` is dead and its tooltip is false

Nothing reads the field (only definition/migration/default in `TrainConfig.py:557/854/1279` and the UI binding). The Pair Tool uses its own local `StringVar("10")` (`DPOCurationWindow.py:1279`), yet the RLHFTab tooltip (`:148`) claims the config field controls the Pair Tool's holdout. Delete the field or wire it up.

### 1.9 Low severity (confirmed)

- `BaseModelSetup.py:286-289` — timestep-quartile diagnostic misbuckets: the per-batch `t.max() > 1.0` continuous-vs-discrete sniff fails when a discrete batch happens to sample only timesteps {0,1} (systematic with very low `max_noising_strength`), putting near-zero timesteps in the top quartile; and the hardcoded `/1000` misnormalizes any scheduler with `num_train_timesteps != 1000`. Diagnostics-only; pass the actual `num_train_timesteps` through instead of sniffing.
- `TrainConfig.py:598` — `config_version=15` while migrations 15 **and** 16 are registered: saved configs are stamped behind the applied chain, so migrations 15/16 re-run on every load (harmless today only because they're idempotent setdefaults; a future non-idempotent migration at those indices would corrupt configs). Subsumed by the migration-collapse recommendation in §3.

### Checked and cleared

- **Refuted:** "standard per-concept validation silently skipped when both validation modes are on" — the two modes use mutually exclusive loader pipelines by construction; the early `return` at `GenericTrainer.py:410` only skips code that could never process a batch, and mixing modes fails loudly (RuntimeError / printed warning), not silently.
- **IPO objective branch** (`BaseModelSetup.py:253-258`): formula matches Azar et al. (`(margin − 1/(2τ))²`), label smoothing correctly excluded, adaptive beta correctly gated to SIGMOID only. Default `τ=1000` is a deliberate diffusion-scale calibration. One UX nit: enabling adaptive beta with IPO is silently ignored rather than warned.
- **Ruff:** all 42 changed `.py` files pass `ruff check` clean. No test files committed on the branch. No attribution trailers in commits.

---

## 2. Slimming the tooling (decision: tooling stays in the PR)

Roughly **4,300 of 6,163 added lines are curation/triage tooling** with zero training-path consumers (the single coupling edge is `DeriveDPORejectedPath.py:3` importing `resolve_aspect_ratio` from `dpo_curation_util`):

| File | Lines |
|---|---|
| `modules/ui/DPOCurationWindow.py` | 1,441 |
| `modules/util/dpo_curation_util.py` | 1,318 |
| `modules/ui/DPOCaptionMismatchWindow.py` | 319 |
| `modules/util/dpo_bucket_analysis_util.py` | 303 |
| `modules/util/image_metadata_util.py` | 299 |
| `modules/ui/DPOBucketAnalysisWindow.py` | 291 |
| `modules/ui/DPOReviewWindow.py` | 193 |
| `modules/util/dpo_swiss_service.py` | 143 |
| + tool-button half of `RLHFTab.py`, `TrainUI.py` window hooks | |

Splitting it into a follow-up PR was considered and **rejected**: the DPO training feature is useless on its own if there is no way to build preference datasets for it. Instead, slim the tooling in place:

### 2.1 Remove: similarity re-pairing + DINOv2 download (~400 lines)

`dpo_curation_util.py:976-1300` — `assign_rejected_to_chosen`, `align_pool_by_similarity`, `load_dino_model`, `embed_images_dino`, `_rejected_target_path`, `_apply_rejected_renames`, `repair_rejected_pairs`, plus `DpoScanCache.get_embedding`, the `align_pool_by_similarity` call in `DPOCurationWindow.py:1111`, and the `repair_rejected_pairs` button wiring in `RLHFTab.py:432`. This is the piece a maintainer will push back on hardest: it lazily downloads `facebook/dinov2-base` via transformers at first use and pulls in scipy Hungarian assignment, for a convenience feature.

### 2.2 Remove: caption-mismatch finder (~500 lines) — obsolete by design

`modules/ui/DPOCaptionMismatchWindow.py` (319 lines), `find_caption_mismatches` / `correct_all_captions_to_chosen` / `apply_caption_to_pair` in `dpo_curation_util.py:830-912`, and the RLHFTab wiring (`:315-338`). The premise died with the concept-type design that `bd902bdf` retired:

- Training only reads the **chosen** caption: `sample_prompt_path` derives from `image_path` (`DataLoaderText2ImageMixin.py:88`), which post-`FilterDPOChosenPaths` contains only chosen images. The rejected side contributes exactly `image_path_rejected` → `latent_image_rejected`; no rejected `.txt` is ever loaded.
- The exporter already writes captions **only to the chosen side** ("Single-concept pattern pairing reads the caption from the chosen side only", `dpo_curation_util.py:~491`), so Pair-Tool exports can't even produce a mismatch.
- `correct_all_captions_to_chosen` overwrites rejected `.txt` files that nothing consumes — a no-op for training.

**Keep `fix_multiline_captions`** — it operates on chosen captions that training actually uses (mgds reads only the first line).

### 2.3 Remove: patience / save-best / end-of-run restore (~100 lines) — kills finding 1.5

Not an upstream feature, and the smallest-line / highest-risk removal in the PR:

| Site | Lines |
|---|---|
| `GenericTrainer.py` — state init (`:169-172`), call site (`:406`), `__check_dpo_patience` (`:463-493`), `__save_dpo_best` (`:495-509`), end-of-run best-restore block (`:1065-1077`) | ~64 |
| `TrainConfig.py` — 3 fields (`rlhf_dpo_patience_enabled`, `rlhf_dpo_patience_value`, `rlhf_dpo_save_best`), 3 migration setdefaults, 3 `data.append` entries | ~9 |
| `RLHFTab.py` — three label+widget blocks (Early Stopping, Patience, Save Best) with tooltips (`:152-179`) | ~25 |

Only ~1.6% of the diff, but the payoff is outsized:

- **Finding 1.5 disappears entirely** — the margin tiebreaker that defeats early stopping under reward hacking, and the RNG-jitter sensitivity of patience/save-best. The nondeterministic DPO validation (1.5b) downgrades from "corrupts the final model" to a metrics-smoothness nit, since val metrics are then only logged, never acted on.
- **The end-of-run weight swap goes away** — that block silently overrides EMA and replaces the user's final weights with `dpo-best.pt`, and `rlhf_dpo_save_best` defaults to **True**, so it's on for everyone. The most surprising behavior in the PR for an upstream maintainer to swallow.
- The `__save_dpo_best` VRAM-clone efficiency nit (§5) goes with it.
- Scope story improves: the PR stops touching training-flow control (stopping, checkpoint selection) and stays "a new loss + data pipeline".

Users lose nothing they can't do manually: `dpo/val_accuracy` and `dpo/val_reward_margin` still land in TensorBoard, and OneTrainer already has manual stop + regular backups. Patience/save-best can return as a follow-up PR with the tiebreaker and determinism fixed. (Dropping only the early-stopping counter while keeping save-best saves ~40-50 lines and is not worth it — they share the `is_new_best` logic, and save-best is where the danger lives.)

### 2.4 Fix in place (remaining tooling issues)

- **blake3 dependency** (`requirements-global.txt:6`) — single consumer (`compute_pixel_hash`); the same file already uses stdlib `hashlib.sha256` for file hashing, and pixel hashing is decode-bound, not hash-bound. Replace with `hashlib.blake2b`/`sha256`, drop the dependency.
- **Dead code**: `export_curated_pairs` + `has_existing_exports` (`dpo_curation_util.py:637-710`) are superseded by the per-pair export path (`export_single_pair`/`finalize_export`) and called only by local tests. Delete.
- **Duplicated image display**: near-identical copies of the image-fit/display block across the DPO windows (fewer once the mismatch window is gone), none using `image_util.load_image` — so DPO thumbnails skip EXIF transpose while the rest of the UI applies it. Extract one shared fit-and-label helper built on `load_image`.
- **Three dot-dir-skipping image walkers** (`walk_skipping_dotted`, `dpo_pattern_util._walk`, `dpo_bucket_analysis_util._iter_image_files`) with subtly different semantics — consolidate on one (home it on the training side, e.g. `path_util`/`dpo_pattern_util`, so tooling imports from training and not the reverse).
- **Six inline copies** of the chosen/rejected indexing loop in `dpo_curation_util` (four after §2.2's deletion); none prune dot-dirs, so `.thumbnails` count as pairs. Extract one `_index_images(root)` helper built on the shared walker.
- **`_TRAINER_BUCKET_RATIOS`** (`dpo_curation_util.py:36`) hand-copies `AspectBucketing.all_possible_input_aspects` — import the mgds class attribute it claims to mirror, so bucket drift can't happen silently.
- **UI-thread thumbnail decode**: `DPOCurationWindow` decodes full-resolution images synchronously on the Tk main thread and stores them in unbounded caches (`_thumb_cache`/`_triage_thumb_cache`) — use `Image.draft`/`thumbnail`, decode in the existing worker thread, bound the caches.
- **`image_metadata_util.py`'s hand-rolled PNG chunk parser**: PIL provides `img.info`/`img.text` without decoding pixels. Lower priority — the byte scanner was written deliberately for speed and covers Forge/Comfy/SwarmUI + WebP UTF-16LE quirks; replacing it risks metadata-parsing regressions for modest line savings.

**Estimated result:** ~1,000 lines off the tooling plus ~100 from patience/save-best (§2.3), plus ~350 of churn from §3 — a ~4,700-line PR where every remaining line is load-bearing.

---

## 3. Fork leakage / churn to remove regardless

- **`TrainConfig.py:525-528, 867-870, 1247-1250`** — `transfer_step1` / `transfer_step2` / `transfer_guidance` / `transfer_train_lora`: dead fields from the bitcrushed-blend distillation/transfer work; grep confirms zero readers anywhere. Pure fork leakage.
- **`TrainConfig.py:598-616, 836-880`** — the migration chain replays the fork's private development history (migrations 10→16), seeding orphan keys (`rlhf_dpo_shared_noise`, `rlhf_dpo_execution_mode`, `rlhf_dpo_patience_mode`) that no longer exist as config fields, plus the version-15-vs-migration-16 numbering bug (§1.9). Upstream is at version 10 and has never seen any intermediate state. **Collapse to a single `__migration_10` (10→11) with the final key set, minus dead keys.**
- ~~**`modules/cloud/BaseCloud.py:134`, `modules/cloud/LinuxCloud.py:252`** — formatting-only hunks~~ **(resolved: keep as-is)** — dxqb explicitly OK'd updating the cloud formatting in this PR, so the `self, local: Path` fix (commit `68c2559b`) stays. A de-churn pass had reverted it; that revert was dropped 2026-07-03.
- **`modules/ui/ConceptWindow.py`** — of the 230-line diff, only ~15 lines are real change (the two `dpo_chosen_pattern`/`dpo_rejected_pattern` rows + a tooltip sentence); the rest is cosmetic re-wrapping from the fork's ruff-format style. Same for reformat-only hunks in `GenericTrainer.py` (re-wraps, deleted upstream comments). Revert to upstream formatting and re-add the real changes — Nerogar does not run ruff format, so these hunks are pure churn and future merge-conflict surface.
- **Dead config surface** (delete): `rlhf_dpo_ref_mode` (declared/defaulted/migrated but `effective_dpo_ref_mode()` derives from `lora_model_name` and ignores it — users can set it with zero effect; the `DPORefMode` enum stays, it's used by the derivation), `rlhf_mode` + the one-member `RLHFMode` enum file, `rlhf_dpo_validation_percentage` (§1.8).
- **Dead plumbing**: the `is_validation` parameter on `_output_modules`, threaded through the abstract signature and **all 15 `*BaseDataLoader.py` files**, is never read by any implementation (`_output_modules_from_out_names` accepts and ignores it). Deleting it removes most of the per-model-family churn in the PR. (The `_create_mgds(…, is_validation)` usage is live — only the `_output_modules` threading is dead.)

---

## 4. The PR description needs a rewrite

The body documents the **old** design, heavily diverged from the code:

| PR body says | Code actually has |
|---|---|
| Four DPO concept types (`DPO_CHOSEN`, …) | Retired in `bd902bdf`; `ConceptType` has no DPO members — pairing is per-concept `dpo_chosen_pattern`/`dpo_rejected_pattern` path derivation |
| Execution modes (SEQUENTIAL / POLICY_CONCURRENT / FULL_CONCURRENT), shared-noise option | Removed in `4b5c0871` — single execution path, always-paired RNG |
| Four forward passes per step | Two batched passes (`669b64b9`) |
| `PairByFilename.py`, `DPOExecutionMode.py` | Files don't exist on the branch |
| ELO scoring mode | Swiss tournament + selection + triage modes |
| Beta default 5000 | 200, plus adaptive beta controller and IPO objective (never mentioned) |
| Patience EITHER/BOTH modes on accuracy + chosen reward | Accuracy-only with margin tiebreaker (`1801977f`) |

A maintainer reviewing code against that description will be confused within minutes — and dxqb is looking at the PR now (the linked comment is their 👀 reaction). The 47-commit history with visible design pivots is also worth squashing into a few logical commits.

---

## 5. Smaller cleanups worth doing in the core PR

- **Hot path:** six sequential `.item()` calls per micro-batch building `_last_dpo_metrics` (`BaseModelSetup.py:273`) — stack the scalars and sync once, or accumulate on-GPU per accumulation window; the quartile block adds eight more syncs when enabled (use `torch.bincount`, one sync).
- `DeriveDPORejectedPath.start()` re-opens every pair's two image headers **every epoch** for a static aspect warning (`:78`) — check only at variation 0 or memoize.
- `match_chosen` calls `parse.parse(pattern, target)` per item, reconstructing the parser each time (`dpo_pattern_util.py:46`) — `lru_cache` a `parse.compile(pattern)` per pattern.
- `__save_dpo_best` uses `p.data.clone().cpu()` (`GenericTrainer.py:499`) — the clone doubles adapter VRAM transiently; `p.data.detach().to('cpu', copy=True)` skips it.
- `BaseDataLoader.py:39` mutates `config.rlhf_enabled` on the copied config as a side channel for "build the DPO validation pipeline" — the explicit `is_validation` flag (currently dead, §3) is the right seam; the mutation makes `rlhf_enabled` mean different things depending on which config copy a reader got (this ambiguity is what produced finding 1.4).

Tooling-side issues are covered by the slim-down plan in §2.4.

---

## 6. Suggested order of operations

1. **Slim the PR (§2)**: remove similarity re-pairing + DINOv2 download, remove the caption-mismatch finder (keep `fix_multiline_captions`), remove patience/save-best/end-of-run restore, delete dead `export_curated_pairs`/`has_existing_exports`, replace blake3 with stdlib hashlib and drop the dependency, then apply the §2.4 dedup/fix items.
2. **De-churn (§3)**: revert formatting-only hunks (cloud files, ConceptWindow/GenericTrainer rewraps); delete fork-leaked `transfer_*` fields, dead config fields, dead `is_validation` threading; collapse migrations to a single 10→11.
3. **Fix the confirmed bugs (§1)**: the SDXL crash (1.1), dropout pairing (1.2), and cache-key compatibility (1.6) are the ones a maintainer or user hits immediately; the reference-resume (1.3) and validation-gate (1.4) fixes protect the headline feature's integrity. (1.5 is eliminated by the §2.3 patience removal.)
4. **Rewrite the PR body** against the actual design; squash the history into a few logical commits.

**Housekeeping note:** the main `bitcrushed-blend` working tree has uncommitted changes to `modules/util/dpo_curation_util.py` — check whether those were meant for this PR before reshaping the branch.
