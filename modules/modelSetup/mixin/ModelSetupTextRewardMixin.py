"""DRaFT-1 style on-image text reward as an add-on term.

One reward step, for one synthetic prompt:

1. Roll the current policy (adapter on) from fresh noise to step ``k`` under
   ``no_grad`` on a shifted linear sigma grid, with classifier-free guidance
   if configured. Only the prefix pays for CFG.
2. One grad-bearing conditional forward at ``x_k`` gives ``v_k`` and the
   one-step clean estimate ``x0 = x_k - sigma_k * v_k``.
3. Decode ``x0`` once without grad, run the frozen detector, align its word
   boxes to the prompt's words.
4. Decode only the latent window around the matched boxes *with* grad
   (decoder gradient-checkpointed), crop the words, and push the frozen
   recognizer's CTC loss back through pixels, VAE and the one transformer
   forward into the adapter.

With ``OFFLINE_IMAGES`` there is no prefix and no step 3 on the policy: the
pre-generated image is re-noised to ``sigma_k``, and its VAE latent plus the
detector's boxes and readings on the *clean* source are computed once per
image and cached under ``<cache_dir>/text_reward``. A sample is then exactly
one transformer forward (the grad one) and one grad-bearing crop decode; the
metric readings come from a no-grad read of that crop, and the preview's full
decode runs only on save steps, after the backward.

Conditioning is pre-encoded per epoch (:meth:`build_text_reward_bank`)
because the text encoder is offloaded while training; embeddings are cached
by prompt across epochs, so an offline dataset is encoded once per process.
"""

from __future__ import annotations

import os
import random
from abc import ABCMeta
from collections.abc import Callable

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.distill_dmd2_util import euler_flow_step
from modules.util.enum.TextRewardSource import TextRewardSource
from modules.util.text_reward_util import (
    TextRewardBank,
    TextRewardSample,
    generate_text_specs,
    load_text_reward_dataset,
    load_word_list,
    policy_read_metrics,
    text_reward_metrics,
)
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

TEXT_REWARD_ENCODE_CHUNK = 16


def text_reward_sigmas(steps: int, shift: float, device: torch.device) -> Tensor:
    """``(1, steps + 1)`` sigma grid from 1 to 0 with the flow-matching time
    shift ``shift * s / (1 + (shift - 1) * s)`` — the same warp the sampler's
    scheduler applies."""
    s = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    sigmas = shift * s / (1.0 + (shift - 1.0) * s)
    return sigmas.unsqueeze(0)


class ModelSetupTextRewardMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._text_reward_model = None
        self._text_reward_vae_ckpt_enabled = False
        self._last_text_reward_metrics: dict[str, float] = {}
        self._text_reward_sample_count = 0
        self._text_reward_recognizer_arch: str | None = None
        self._text_reward_cache = None
        self._text_reward_prompt_embeddings: dict[str, Tensor] | None = None
        self._text_reward_prompt_embeddings_identity: str | None = None

    # -- reward model ---------------------------------------------------------

    def text_reward_model(self):
        if self._text_reward_model is None:
            from modules.util.text_reward_model import RECOGNIZER_ARCH, TextRewardModel

            self._text_reward_model = TextRewardModel(
                self.train_device, recognizer_arch=self._text_reward_recognizer_arch or RECOGNIZER_ARCH
            )
        return self._text_reward_model

    def get_last_text_reward_metrics(self) -> dict[str, float]:
        return dict(self._last_text_reward_metrics)

    # -- per-epoch prompt bank -------------------------------------------------

    @staticmethod
    def _text_reward_text_encoder_identity(model: BaseModel, config: TrainConfig) -> str:
        """What a cached prompt embedding depends on from the text-encoder
        side: the model class, which encoder (its own model name, else the base
        model's) and the loaded object, its configured and runtime weight
        dtype, the autocast dtype, and the prompt token limit."""
        from modules.model.ZImageModel import PROMPT_MAX_LENGTH

        te_part = getattr(config, "text_encoder", None)
        name = getattr(te_part, "model_name", "") or getattr(config, "base_model_name", "") or ""
        weight_dtype = getattr(te_part, "weight_dtype", None)
        text_encoder = getattr(model, "text_encoder", None)
        try:
            runtime_dtype = next(text_encoder.parameters()).dtype
        except (AttributeError, StopIteration):
            runtime_dtype = None
        train_dtype = getattr(model, "text_encoder_train_dtype", None)
        return (
            f"{type(model).__name__}|{name}|{id(text_encoder)}|{weight_dtype}|{runtime_dtype}"
            f"|autocast={train_dtype}|max_len={PROMPT_MAX_LENGTH}"
        )

    def _text_reward_prompt_cache(self, model: BaseModel, config: TrainConfig) -> dict[str, Tensor]:
        """The per-process ``raw prompt -> CPU embedding`` cache, emptied
        whenever the text-encoder identity changes."""
        identity = self._text_reward_text_encoder_identity(model, config)
        cache = getattr(self, "_text_reward_prompt_embeddings", None)
        if cache is None or getattr(self, "_text_reward_prompt_embeddings_identity", None) != identity:
            cache = {}
            self._text_reward_prompt_embeddings = cache
            self._text_reward_prompt_embeddings_identity = identity
        return cache

    def _text_reward_encode_prompts(
        self, model: BaseModel, config: TrainConfig, prompts: list[str], cache: dict[str, Tensor]
    ) -> None:
        """Encode ``prompts`` (unique, all cache misses) in chunks of
        :data:`TEXT_REWARD_ENCODE_CHUNK` with the text encoder paged onto the
        train device once, storing each CPU embedding under its raw prompt."""
        model.text_encoder_to(self.train_device)
        try:
            with torch.no_grad():
                for start in range(0, len(prompts), TEXT_REWARD_ENCODE_CHUNK):
                    chunk_prompts = prompts[start : start + TEXT_REWARD_ENCODE_CHUNK]
                    # encode_text templates the list it is given in place: hand
                    # it a copy so the cache keys stay the raw prompts.
                    encoded = model.encode_text(
                        train_device=self.train_device,
                        batch_size=len(chunk_prompts),
                        text=list(chunk_prompts),
                        pad_to_longest=True,
                    )
                    for prompt, embedding in zip(chunk_prompts, encoded, strict=True):
                        cache[prompt] = embedding.detach().to("cpu")
        finally:
            if config.latent_caching:
                model.text_encoder_to(self.temp_device)
            torch_gc()

    def build_text_reward_bank(self, model: BaseModel, config: TrainConfig, epoch: int) -> TextRewardBank:
        """Draw this epoch's prompts and look their embeddings up in the
        per-process prompt cache; only misses are encoded, with the text
        encoder paged onto the train device once (and not at all when every
        prompt hits). Entries this epoch does not use are evicted. Embeddings
        live on the CPU."""
        self._text_reward_recognizer_arch = config.text_reward_recognizer
        vocab = self.text_reward_model().vocab
        image_paths: list[str] | None = None
        image_sha256s: list[str | None] | None = None
        offline = config.text_reward_source == TextRewardSource.OFFLINE_IMAGES
        if offline:
            rows = load_text_reward_dataset(config.text_reward_dataset_dir, vocab)
            if not rows:
                raise RuntimeError(f"text reward dataset is empty: {config.text_reward_dataset_dir}")
            rng = random.Random(0x7E57 + epoch)
            rng.shuffle(rows)
            specs = [r[0] for r in rows]
            prompts = [r[1] for r in rows]
            image_paths = [r[2] for r in rows]
            image_sha256s = [getattr(r, "sha256", None) for r in rows]
        else:
            specs = generate_text_specs(
                max(1, config.text_reward_samples_per_epoch),
                seed=0x7E57 + epoch,
                vocab=vocab,
                words=load_word_list(config.text_reward_word_list),
                min_words=config.text_reward_min_words,
                max_words=config.text_reward_max_words,
            )
            prompts = [s.prompt for s in specs]

        # The unconditional branch only runs in the online rollout prefix; the
        # offline step re-noises the source image and never reads it.
        negative_prompt = None
        if not offline and config.text_reward_cfg_scale > 1.0:
            negative_prompt = config.text_reward_negative_prompt or ""

        cache = self._text_reward_prompt_cache(model, config)
        wanted = list(prompts) if negative_prompt is None else [*prompts, negative_prompt]
        wanted_set = set(wanted)
        for stale in [p for p in cache if p not in wanted_set]:
            del cache[stale]
        misses = [p for p in dict.fromkeys(wanted) if p not in cache]
        if misses:
            self._text_reward_encode_prompts(model, config, misses, cache)

        embeddings = [cache[p] for p in prompts]
        negative = cache[negative_prompt] if negative_prompt is not None else None
        return TextRewardBank(specs, embeddings, negative, image_paths, image_sha256s)

    # -- one reward step ---------------------------------------------------------

    def _text_reward_cond_batch(self, embedding: Tensor, latent_shape: tuple[int, ...], model: BaseModel) -> dict:
        embedding = embedding.to(self.train_device)
        return {
            "latent_image": torch.zeros(latent_shape, device=self.train_device, dtype=model.train_dtype.torch_dtype()),
            "tokens_mask": torch.ones((1, embedding.shape[0]), device=self.train_device, dtype=torch.long),
            "text_encoder_hidden_state": embedding.unsqueeze(0),
        }

    @staticmethod
    def _text_reward_target_resolution(config: TrainConfig) -> int:
        """Effective longer-edge target of the offline encode: a multiple of 16, at least 64."""
        return max(64, int(config.text_reward_resolution) // 16 * 16)

    def _text_reward_load_pixels(self, image_path: str, config: TrainConfig) -> Tensor:
        """Load an offline sample as ``(1, 3, H, W)`` float pixels in ``[0, 1]``
        on the CPU, resized so its longer edge matches the effective
        ``text_reward_resolution`` and both edges are multiples of 16."""
        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        target = self._text_reward_target_resolution(config)
        w, h = image.size
        scale = target / max(w, h)
        nw = max(16, int(round(w * scale / 16)) * 16)
        nh = max(16, int(round(h * scale / 16)) * 16)
        if (nw, nh) != (w, h):
            image = image.resize((nw, nh), Image.LANCZOS)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def _text_reward_encode_pixels(self, model: BaseModel, pixels01: Tensor) -> Tensor:
        """VAE-encode ``[0, 1]`` pixels to a *scaled* latent ``(1, C, H/8, W/8)``."""
        pixels = (pixels01 * 2.0 - 1.0).to(device=self.train_device)
        vae_dtype = next(model.vae.parameters()).dtype
        posterior = model.vae.encode(pixels.to(dtype=vae_dtype), return_dict=False)[0]
        latent = posterior.mode() if hasattr(posterior, "mode") else posterior
        return model.scale_latents(latent.float())

    def _text_reward_encode_image(self, model: BaseModel, image_path: str, config: TrainConfig) -> Tensor:
        """Load an offline sample and VAE-encode it to a *scaled* latent
        ``(1, C, H/8, W/8)``. The image is resized so its longer edge matches
        ``text_reward_resolution`` and both edges are multiples of 16."""
        return self._text_reward_encode_pixels(model, self._text_reward_load_pixels(image_path, config))

    def _text_reward_vae_to_train_device(self, model: BaseModel) -> None:
        if next(model.vae.parameters()).device.type != self.train_device.type:
            model.vae_to(self.train_device)

    @staticmethod
    def _text_reward_vae_identity(model: BaseModel, config: TrainConfig) -> str:
        """What the cached latent depends on from the VAE side: which VAE (its
        own model name, else the base model's), its configured and actual
        weight dtype, and the latent scaling."""
        vae_part = getattr(config, "vae", None)
        name = getattr(vae_part, "model_name", "") or getattr(config, "base_model_name", "") or ""
        weight_dtype = getattr(vae_part, "weight_dtype", None)
        runtime_dtype = next(model.vae.parameters()).dtype
        vae_config = getattr(model.vae, "config", None)
        scaling = getattr(vae_config, "scaling_factor", None)
        shift = getattr(vae_config, "shift_factor", None)
        return f"{name}|{weight_dtype}|{runtime_dtype}|scale={scaling}|shift={shift}"

    def _text_reward_source_cache(self, config: TrainConfig):
        from modules.util.text_reward_cache import TEXT_REWARD_CACHE_SUBDIR, TextRewardSourceCache

        if not config.cache_dir:
            return None
        folder = os.path.join(config.cache_dir, TEXT_REWARD_CACHE_SUBDIR)
        cache = getattr(self, "_text_reward_cache", None)
        if cache is None or cache.folder != folder:
            cache = TextRewardSourceCache(folder)
            self._text_reward_cache = cache
        return cache

    def _text_reward_offline_source(self, model: BaseModel, sample: TextRewardSample, config: TrainConfig, reward):
        """``(scaled source latent on CPU, DetectedWords)`` for an offline
        sample: from ``<cache_dir>/text_reward`` when a matching entry exists,
        otherwise encoded and detected on the clean resized source (no decode)
        and written back. Only a miss touches the VAE; it is paged onto the
        train device for the encode and returned to the temp device afterwards
        if it was not there already."""
        from modules.util.text_reward_cache import text_reward_cache_key, text_reward_image_identity
        from modules.util.text_reward_model import DetectedWords

        cache = self._text_reward_source_cache(config)
        key = text_reward_cache_key(
            image_identity=text_reward_image_identity(sample.image_path, sample.image_sha256),
            resolution=self._text_reward_target_resolution(config),
            vae_identity=self._text_reward_vae_identity(model, config),
            detector_arch=getattr(reward, "detector_arch", ""),
            recognizer_arch=getattr(reward, "recognizer_arch", ""),
            min_box_confidence=reward.min_box_confidence,
            min_box_height=reward.min_box_height,
        )
        if cache is not None:
            hit = cache.load(key)
            if hit is not None:
                return hit.latent, DetectedWords(boxes=hit.boxes, readings=hit.readings)

        pixels01 = self._text_reward_load_pixels(sample.image_path, config)
        was_on_train_device = next(model.vae.parameters()).device.type == self.train_device.type
        if not was_on_train_device:
            model.vae_to(self.train_device)
        try:
            with torch.no_grad():
                latent = self._text_reward_encode_pixels(model, pixels01).detach().to(device="cpu", dtype=torch.float32)
        finally:
            if not was_on_train_device:
                model.vae_to(self.temp_device)
        detected = reward.detect(pixels01.to(device=reward.device))[0]
        del pixels01
        if cache is not None:
            cache.save(key, latent, detected.boxes, detected.readings)
        return latent, detected

    def _text_reward_decode(self, model: BaseModel, scaled_latent: Tensor, *, tiled: bool = False) -> Tensor:
        """VAE decode of a scaled latent to ``[0, 1]`` float pixels. ``tiled``
        uses the VAE's tiled decoder (no-grad full frames only; a few hundred
        MB instead of several GB at 1024²)."""
        vae_dtype = next(model.vae.parameters()).dtype
        use_tiling = tiled and hasattr(model.vae, "enable_tiling")
        saved = None
        if use_tiling:
            # Small tiles: the reward step runs with ~4 GB free next to the
            # optimizer states and DPO reference; a 512 px tile alone is ~1 GB.
            saved = (model.vae.tile_sample_min_size, model.vae.tile_latent_min_size)
            model.vae.tile_sample_min_size = 256
            model.vae.tile_latent_min_size = 32
            model.vae.enable_tiling()
        try:
            image = model.vae.decode(model.unscale_latents(scaled_latent.to(dtype=vae_dtype)), return_dict=False)[0]
        finally:
            if use_tiling:
                model.vae.disable_tiling()
                model.vae.tile_sample_min_size, model.vae.tile_latent_min_size = saved
        return (image.float() + 1.0) * 0.5

    def calculate_text_reward_loss(
        self,
        model: BaseModel,
        sample: TextRewardSample,
        config: TrainConfig,
        train_progress: TrainProgress,
        backward_fn: Callable[[Tensor], None],
    ) -> Tensor:
        """Run one reward sample; gradients land through ``backward_fn``.
        Returns the detached CTC value (zero when no word was found)."""
        reward = self.text_reward_model()
        offline = sample.image_path is not None
        if not offline:
            self._text_reward_vae_to_train_device(model)
        if not self._text_reward_vae_ckpt_enabled and hasattr(model.vae, "enable_gradient_checkpointing"):
            model.vae.enable_gradient_checkpointing()
            self._text_reward_vae_ckpt_enabled = True

        channels = model.vae.config.latent_channels
        source_latent = None
        detected = None
        if offline:
            # OFFLINE_IMAGES: the clean latent comes from the pre-generated
            # image; its own size (padded to 16 px) sets the latent shape. The
            # latent and the detector's boxes/readings on the clean source are
            # cached on disk per image (see modules/util/text_reward_cache.py).
            source_latent, detected = self._text_reward_offline_source(model, sample, config, reward)
            source_latent = source_latent.to(device=self.train_device)
            lh, lw = int(source_latent.shape[-2]), int(source_latent.shape[-1])
        else:
            res = self._text_reward_target_resolution(config)
            lh = lw = res // 8
        latent_shape = (1, channels, lh, lw)
        cond_batch = self._text_reward_cond_batch(sample.embedding, latent_shape, model)
        uncond_batch = None
        cfg = float(config.text_reward_cfg_scale)
        if cfg > 1.0 and sample.negative_embedding is not None:
            uncond_batch = self._text_reward_cond_batch(sample.negative_embedding, latent_shape, model)

        shift = float(model.calculate_timestep_shift(lw, lh)) if hasattr(model, "calculate_timestep_shift") else 1.0
        steps = max(1, int(config.text_reward_rollout_steps))
        sigmas = text_reward_sigmas(steps, shift, self.train_device)

        rng = random.Random(train_progress.global_step * 31 + 7)
        grad_step = rng.randint(steps // 2, steps - 1) if config.text_reward_random_grad_step else steps - 1

        generator = torch.Generator(device=self.train_device)
        generator.manual_seed(train_progress.global_step * 3 + 1)
        sigma_k = sigmas[:, grad_step]

        if source_latent is not None:
            # Re-noise the pre-generated image to sigma_k along the flow path
            # (x = (1 - s) x0 + s eps): the ReFL trick. No prefix to run.
            eps = torch.randn(latent_shape, generator=generator, device=self.train_device, dtype=torch.float32)
            s = sigma_k.view(1, 1, 1, 1)
            x_k = ((1.0 - s) * source_latent.float() + s * eps).detach()
            del source_latent, eps
        else:
            latent = torch.randn(latent_shape, generator=generator, device=self.train_device, dtype=torch.float32)
            with torch.no_grad():
                for i in range(grad_step):
                    sigma, next_sigma = sigmas[:, i], sigmas[:, i + 1]
                    v = self.predict_distill_velocity(model, cond_batch, config, train_progress, latent, sigma)
                    if uncond_batch is not None:
                        v_u = self.predict_distill_velocity(model, uncond_batch, config, train_progress, latent, sigma)
                        v = v_u + cfg * (v - v_u)
                    latent = euler_flow_step(latent, v, sigma, next_sigma)
            x_k = latent.detach()
        sigma_b = sigma_k.view(1, 1, 1, 1).to(dtype=x_k.dtype)

        # Where the boxes come from, and why nothing full-frame coexists with
        # the transformer's autograd graph (that pairing OOMs a 24 GB card):
        # * OFFLINE_IMAGES: boxes and readings come from the clean source image
        #   (the cached detection above), not from the policy's x0, so there is
        #   no no-grad probe forward and no full-frame decode. Boxes are
        #   relative and the latent grid is derived from the same resized
        #   source, so they land on the grad forward's x0 unchanged. The only
        #   full decode left is the preview, which runs after backward_fn.
        # * ONLINE_ROLLOUT: detect BEFORE the grad-bearing forward. The grad
        #   forward below sees the same x_k and sigma, so its x0 matches the
        #   probe's bit-for-bit and the boxes stay valid.
        full01 = None
        if not offline:
            with torch.no_grad():
                v_probe = self.predict_distill_velocity(model, cond_batch, config, train_progress, x_k, sigma_k)
                x0_probe = x_k - sigma_b * v_probe.to(dtype=x_k.dtype)
                del v_probe
                full01 = self._text_reward_decode(model, x0_probe, tiled=True)
                del x0_probe
                detected = reward.detect(full01)[0]
        alignment = reward.align(detected, sample.spec.words)

        v_k = self.predict_distill_velocity(model, cond_batch, config, train_progress, x_k, sigma_k)
        x0 = x_k - sigma_b * v_k.to(dtype=x_k.dtype)

        from modules.util.text_reward_model import boxes_inside_region, latent_crop_region

        save_due = (
            config.text_reward_save_interval > 0
            and self._text_reward_sample_count % config.text_reward_save_interval == 0
        )
        matched = [(detected.boxes[bi], sample.spec.words[ti]) for bi, ti in alignment.pairs]
        loss_value = torch.zeros((), device=self.train_device)
        ctc_words = 0
        policy_readings: dict[int, str] = {}  # offline: box index -> no-grad reading of the policy crop
        if matched:
            max_edge = max(64, int(config.text_reward_max_decode_pixels)) // 8
            region = latent_crop_region([b for b, _ in matched], lh, lw, max_area=max_edge * max_edge)
            keep = boxes_inside_region([b for b, _ in matched], region, lh, lw)
            if keep:
                y0, y1, x0_, x1 = region
                if offline:
                    self._text_reward_vae_to_train_device(model)
                crop01 = self._text_reward_decode(model, x0[:, :, y0:y1, x0_:x1])
                ch, cw = (y1 - y0) * 8, (x1 - x0_) * 8
                rel_boxes = []
                targets = []
                for i in keep:
                    (bx0, by0, bx1, by1), word = matched[i]
                    rel_boxes.append(
                        (
                            (bx0 * lw * 8 - x0_ * 8) / cw,
                            (by0 * lh * 8 - y0 * 8) / ch,
                            (bx1 * lw * 8 - x0_ * 8) / cw,
                            (by1 * lh * 8 - y0 * 8) / ch,
                        )
                    )
                    targets.append(word)
                with torch.autocast(device_type=self.train_device.type, enabled=False):
                    loss = reward.ctc_loss(crop01, rel_boxes, targets)
                backward_fn(loss)
                loss_value = loss.detach()
                ctc_words = len(targets)
                del loss
                if offline:
                    # Policy reading for the metrics: the recognizer on the
                    # (already decoded) crop, after the backward freed the graph.
                    with torch.no_grad(), torch.autocast(device_type=self.train_device.type, enabled=False):
                        crop_readings = reward.read(crop01.detach(), rel_boxes)
                    for i, reading in zip(keep, crop_readings, strict=True):
                        policy_readings[alignment.pairs[i][0]] = reading
                del crop01
        x0_preview = x0.detach() if offline and save_due else None
        del x0, v_k

        if offline:
            metrics = policy_read_metrics(policy_readings, list(sample.spec.words), alignment)
            readings = [policy_readings[bi] for bi in sorted(policy_readings)]  # boxes are in reading order
        else:
            metrics = text_reward_metrics([detected.readings], [list(sample.spec.words)], [alignment])
            readings = detected.readings
        self._last_text_reward_metrics = {
            "ctc": float(loss_value),
            "word_ratio": metrics.word_ratio,
            "exact": metrics.exact,
            "missing": metrics.missing,
            "scored_words": float(ctc_words),
            f"family_ratio/{sample.spec.family}": metrics.word_ratio,
            f"family_count/{sample.spec.family}": 1.0,
        }
        self._text_reward_sample_count += 1
        if save_due:
            if x0_preview is not None:
                # Preview only: a detached tiled decode, after backward_fn and
                # after the grad forward's graph was dropped.
                self._text_reward_vae_to_train_device(model)
                with torch.no_grad():
                    full01 = self._text_reward_decode(model, x0_preview, tiled=True)
                del x0_preview
            self._save_text_reward_sample(config, train_progress, full01, sample, readings)
        del full01
        return loss_value

    @staticmethod
    def _save_text_reward_sample(config: TrainConfig, train_progress: TrainProgress, image01: Tensor, sample, readings):
        try:
            from PIL import Image

            folder = os.path.join(config.workspace_dir, "text_reward")
            os.makedirs(folder, exist_ok=True)
            array = (image01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            stem = os.path.join(folder, f"{train_progress.global_step:08d}")
            Image.fromarray(array).save(stem + ".png")
            with open(stem + ".txt", "w", encoding="utf-8") as f:
                f.write(sample.spec.prompt + "\n")
                f.write("read: " + " ".join(readings) + "\n")
        except Exception as exception:  # noqa: BLE001 - never let a debug image abort a run
            print(f"text reward sample not saved: {exception}")
