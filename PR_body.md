# DPO / RLHF training for OneTrainer

## What is this

This adds preference training (DPO) to OneTrainer. The idea is simple: for a given prompt you show the model two images, one you like and one you don't, and it trains a LoRA (an adapter, so no full finetune) to make more of the good kind and less of the bad kind. You don't have to score anything on a scale or hand-label rewards. You just say "this one is better than that one" and it learns from the comparison.

DPO stands for Direct Preference Optimization. It started in the language-model world, where you compare two text answers. Here it's been adapted to image models.

It's model-agnostic. It hooks into OneTrainer's existing `predict()` path, so Flux, SDXL, SD1.5, Z-Image and the rest all work the same way with no per-model special casing.

## How it works

Language-model DPO scores an answer by its token log-probabilities. Image models don't give you that, so this uses the denoising MSE as the score instead. In plain terms: at a random noise level, how well does the model reconstruct the image? Lower error means the model "likes" that image more, so negative MSE becomes the preference score. That's the whole trick that makes it model-agnostic.

DPO needs a reference model to compare against, otherwise the adapter just runs off and reward-hacks itself. Normally the reference is a second full copy of the model, which is rough on VRAM. Here it isn't:

- **New adapter** (you didn't load a base LoRA): the reference is just the raw base model weights. Nothing extra to load.
- **Existing adapter** (you loaded a LoRA and want to keep training it): at the start it freezes a snapshot of your adapter's weights and uses that snapshot as the reference. So the reference is a small set of frozen tensors, not a whole second model sitting in VRAM. That frozen snapshot is saved next to your backups and restored on resume, so picking a run back up doesn't accidentally move the reference onto your half-trained weights.

Because of this, VRAM stays close to a normal LoRA run. The chosen and rejected images get batched together into one forward pass, so a step is really just two forwards total (reference and policy), not four.

There's a `beta` setting that controls how hard the model is allowed to pull away from the reference. Because MSE values are small, the default is 200, which is higher than you'd see in text DPO. Higher beta means it stays closer to the reference and trains more gently.

One important detail: the chosen and rejected image use the **same noise and the same timestep** on every step. The only thing that differs between the two sides is the image itself, so the model can't get thrown off by noise luck. Caption (conditioning) dropout is also turned off during the DPO pass, so the two sides can never end up seeing different prompts.

## Making a dataset

You need pairs: for each prompt, a chosen image and a rejected one. There's a **Pair Tool** in the Tools tab to build these.

It reads the generation metadata baked into images (ComfyUI, SwarmUI and Forge PNG metadata, plus JPEG and WebP) to pull out the prompt and aspect ratio, then groups your images by prompt. From there you pick winners three ways:

- **Tournament** shows you two images at a time (A vs B) and schedules the matchups for you.
- **Selection** shows a whole group and you just pick the best and the worst.
- **Triage** lets you mark each image Good or Bad, then builds the pairs from those.

When you're done it exports a folder layout that's ready to train on (`chosen/train`, `chosen/val`, `rejected/train`, `rejected/val`) with a `concepts.json`. There are a couple of small helper tools in there too, like a bucket / batch-size analyzer and a pair review window.

Pairing at train time is by **file pattern**, not by any special concept type. You point a concept at your chosen and rejected patterns (something like `chosen/{}` and `rejected/{}`) and it matches them up by filename. Only the chosen image's caption is used for training.

## The knobs

- **beta** (default 200): how tightly to stay near the reference. Lower lets it move more.
- **supervised_mix** (default 0.25): mixes a normal training loss on the chosen image in on top of the preference loss. Helps keep the adapter from drifting off into weird territory.
- **label smoothing**: optional, for when your labels are noisy and you don't fully trust every pair.
- **IPO**: an alternative loss to the default sigmoid DPO one. Optional, some people find it more stable.
- **adaptive beta**: optional, nudges beta up or down based on the reward margin as training goes. Works on multi-GPU too.
- **DPO validation**: optional, runs on a held-out set of pairs and logs accuracy and reward margin to TensorBoard so you can actually see whether it's learning the preference.

## Limits

- LoRA only. Full finetune isn't supported and will raise a clear error at startup rather than half-working.
- No reward model, no full-parameter DPO, no external reference model. The reference is always either the base weights or a frozen snapshot of your adapter, as described above.
- It works with gradient accumulation and multi-GPU.
