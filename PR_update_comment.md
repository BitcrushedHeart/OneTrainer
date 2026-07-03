Final update.

## Bug Fixes
- SDXL DPO used to crash on the very first training step, because the batch got built at the wrong size for SDXL's resolution inputs. Fixed.
- Chosen and rejected could get different caption dropout, which quietly poisoned the training signal. Caption dropout is now off during the DPO pass, so both sides always see the same prompt.
- Resuming an "existing adapter" run from a backup used to silently move the frozen reference onto the half-trained weights. The reference is now saved with the backup and restored properly.
- Turning on DPO validation without turning on RLHF itself could crash a normal run partway through. Both flags are checked together now.
- Adding the DPO pattern fields changed the cache key for everyone, so non-DPO users would get a surprise full re-cache. The key only changes when RLHF is actually on now, so existing caches stay valid.
- Adaptive beta only updated on one GPU in multi-GPU runs (the others kept the old beta and the gradients got mixed). All GPUs now share the same beta.
- The timestep-margin diagnostic bucketed timesteps wrong for schedulers that aren't 1000 steps, and could misplace very low timesteps. Fixed.

## Code Improvements
- Removed the DINOv2-based "re-pair by similarity" feature and its model download. It was heavy for what it actually did.
- Removed the caption-mismatch finder. It's pointless now that only the chosen caption is ever used for training.
- Removed the patience / early-stopping / save-best logic. It isn't really an upstream thing, and the "best" pick could end up chasing a reward-hacked metric.
- Dropped the blake3 dependency and used Python's built-in hashlib instead, so there's one less thing to install.
- Deleted some dead export code and de-duplicated repeated bits in the tooling.

## Config Changes
- Removed leftover config fields that weren't doing anything. Some of them leaked in from another branch (transfer_step1/2/guidance/train_lora), plus a dead RLHFMode enum, rlhf_dpo_ref_mode, and rlhf_dpo_validation_percentage.
- Collapsed a messy chain of config-version migrations down to a single clean one.
- The cloud config-file formatting tidy-up stays (dxqb okayed that one).
