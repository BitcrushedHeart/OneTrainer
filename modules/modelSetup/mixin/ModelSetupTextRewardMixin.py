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

Conditioning is pre-encoded per epoch (:meth:`build_text_reward_bank`)
because the text encoder is offloaded while training.
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
    text_reward_metrics,
)
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


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

    def build_text_reward_bank(self, model: BaseModel, config: TrainConfig, epoch: int) -> TextRewardBank:
        """Draw this epoch's prompts and encode them with the text encoder
        paged onto the train device once. Embeddings live on the CPU."""
        self._text_reward_recognizer_arch = config.text_reward_recognizer
        vocab = self.text_reward_model().vocab
        image_paths: list[str] | None = None
        if config.text_reward_source == TextRewardSource.OFFLINE_IMAGES:
            rows = load_text_reward_dataset(config.text_reward_dataset_dir, vocab)
            if not rows:
                raise RuntimeError(f"text reward dataset is empty: {config.text_reward_dataset_dir}")
            rng = random.Random(0x7E57 + epoch)
            rng.shuffle(rows)
            specs = [r[0] for r in rows]
            prompts = [r[1] for r in rows]
            image_paths = [r[2] for r in rows]
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
        model.text_encoder_to(self.train_device)
        try:
            embeddings: list[Tensor] = []
            chunk = 16
            with torch.no_grad():
                for start in range(0, len(specs), chunk):
                    chunk_prompts = list(prompts[start : start + chunk])
                    encoded = model.encode_text(
                        train_device=self.train_device, batch_size=len(chunk_prompts), text=chunk_prompts
                    )
                    embeddings.extend(e.detach().to("cpu") for e in encoded)
                negative = None
                if config.text_reward_cfg_scale > 1.0:
                    negative = (
                        model.encode_text(
                            train_device=self.train_device,
                            batch_size=1,
                            text=[config.text_reward_negative_prompt or ""],
                        )[0]
                        .detach()
                        .to("cpu")
                    )
        finally:
            if config.latent_caching:
                model.text_encoder_to(self.temp_device)
            torch_gc()
        return TextRewardBank(specs, embeddings, negative, image_paths)

    # -- one reward step ---------------------------------------------------------

    def _text_reward_cond_batch(self, embedding: Tensor, latent_shape: tuple[int, ...], model: BaseModel) -> dict:
        embedding = embedding.to(self.train_device)
        return {
            "latent_image": torch.zeros(latent_shape, device=self.train_device, dtype=model.train_dtype.torch_dtype()),
            "tokens_mask": torch.ones((1, embedding.shape[0]), device=self.train_device, dtype=torch.long),
            "text_encoder_hidden_state": embedding.unsqueeze(0),
        }

    def _text_reward_encode_image(self, model: BaseModel, image_path: str, config: TrainConfig) -> Tensor:
        """Load an offline sample and VAE-encode it to a *scaled* latent
        ``(1, C, H/8, W/8)``. The image is resized so its longer edge matches
        ``text_reward_resolution`` and both edges are multiples of 16."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        target = max(64, int(config.text_reward_resolution) // 16 * 16)
        w, h = image.size
        scale = target / max(w, h)
        nw = max(16, int(round(w * scale / 16)) * 16)
        nh = max(16, int(round(h * scale / 16)) * 16)
        if (nw, nh) != (w, h):
            image = image.resize((nw, nh), Image.LANCZOS)
        import numpy as np

        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        pixels = (pixels * 2.0 - 1.0).to(device=self.train_device)
        vae_dtype = next(model.vae.parameters()).dtype
        posterior = model.vae.encode(pixels.to(dtype=vae_dtype), return_dict=False)[0]
        latent = posterior.mode() if hasattr(posterior, "mode") else posterior
        return model.scale_latents(latent.float())

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
            torch.cuda.empty_cache()
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
        vae_device = next(model.vae.parameters()).device
        if vae_device.type != self.train_device.type:
            model.vae_to(self.train_device)
        if not self._text_reward_vae_ckpt_enabled and hasattr(model.vae, "enable_gradient_checkpointing"):
            model.vae.enable_gradient_checkpointing()
            self._text_reward_vae_ckpt_enabled = True

        channels = model.vae.config.latent_channels
        source_latent = None
        if sample.image_path is not None:
            # OFFLINE_IMAGES: the clean latent comes from the pre-generated
            # image; its own size (padded to 16 px) sets the latent shape.
            with torch.no_grad():
                source_latent = self._text_reward_encode_image(model, sample.image_path, config)
            lh, lw = int(source_latent.shape[-2]), int(source_latent.shape[-1])
        else:
            res = max(64, int(config.text_reward_resolution) // 16 * 16)
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

        # Detect BEFORE the grad-bearing forward so the full-image decode never
        # coexists with the transformer's autograd graph (that pairing OOMs a
        # 24 GB card). The grad forward below sees the same x_k and sigma, so
        # its x0 matches this one bit-for-bit and the boxes stay valid.
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

        matched = [(detected.boxes[bi], sample.spec.words[ti]) for bi, ti in alignment.pairs]
        loss_value = torch.zeros((), device=self.train_device)
        ctc_words = 0
        if matched:
            max_edge = max(64, int(config.text_reward_max_decode_pixels)) // 8
            region = latent_crop_region([b for b, _ in matched], lh, lw, max_area=max_edge * max_edge)
            keep = boxes_inside_region([b for b, _ in matched], region, lh, lw)
            if keep:
                y0, y1, x0_, x1 = region
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
        del x0, v_k

        metrics = text_reward_metrics([detected.readings], [list(sample.spec.words)], [alignment])
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
        if (
            config.text_reward_save_interval > 0
            and (self._text_reward_sample_count - 1) % config.text_reward_save_interval == 0
        ):
            self._save_text_reward_sample(config, train_progress, full01, sample, detected.readings)
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
