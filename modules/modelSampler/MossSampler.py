import copy
from collections.abc import Callable

from modules.model.MossModel import LATENT_DOWNSCALE, MossModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from tqdm import tqdm

# MOSS's default guidance: the negative branch runs with this block skipped
# (zero-based), see https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
UNCOND_SKIP_LAYERS = (22,)


class MossSampler(BaseModelSampler):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        model: MossModel,
        model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __sample_base(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        seed: int,
        random_seed: bool,
        diffusion_steps: int,
        cfg_scale: float,
        noise_scheduler: NoiseScheduler,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            # MOSS samples with its own scheduler (shift baked into its config, snapped to the
            # training grid), so the sample config's noise_scheduler does not apply
            noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            image_processor = self.pipeline.image_processor
            transformer = self.pipeline.transformer
            vae = self.pipeline.vae
            train_dtype = self.model.train_dtype.torch_dtype()

            do_cfg = cfg_scale != 1.0
            skip_layers = [i for i in UNCOND_SKIP_LAYERS if i < len(transformer.layers)]

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            prompt_embedding, prompt_mask = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
            )
            if do_cfg:
                negative_embedding, negative_mask = self.model.encode_text(
                    text=negative_prompt,
                    train_device=self.train_device,
                )

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latent image: 128-channel packed latents at 1/16 scale, sampled in fp32
            latent_image = torch.randn(
                size=(
                    1,
                    transformer.config.latent_channels,
                    height // LATENT_DOWNSCALE,
                    width // LATENT_DOWNSCALE,
                ),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = noise_scheduler.timesteps

            # denoising loop
            self.model.transformer_to(self.train_device)
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                # CFG combines fp32 predictions even when the transformer runs in bf16
                noise_pred = transformer(
                    hidden_states=latent_image.to(dtype=train_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embedding.to(dtype=train_dtype),
                    encoder_attention_mask=prompt_mask,
                    return_dict=False,
                )[0].float()

                if do_cfg:
                    noise_pred_negative = transformer(
                        hidden_states=latent_image.to(dtype=train_dtype),
                        timestep=timestep,
                        encoder_hidden_states=negative_embedding.to(dtype=train_dtype),
                        encoder_attention_mask=negative_mask,
                        return_dict=False,
                        skip_layers=skip_layers,
                    )[0].float()
                    noise_pred = noise_pred_negative + cfg_scale * (noise_pred - noise_pred_negative)

                latent_image = noise_scheduler.step(noise_pred, timestep, latent_image, return_dict=False)[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()
            self.model.vae_to(self.train_device)

            latents = self.model.unscale_latents(latent_image).to(dtype=vae.dtype)
            image = vae.decode(latents, return_dict=False)[0].clamp(-1, 1)

            image = image_processor.postprocess(image, output_type="pil")

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image[0],
            )

    def sample(
        self,
        sample_config: SampleConfig,
        destination: str,
        image_format: ImageFormat | None = None,
        video_format: VideoFormat | None = None,
        audio_format: AudioFormat | None = None,
        on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, LATENT_DOWNSCALE),
            width=self.quantize_resolution(sample_config.width, LATENT_DOWNSCALE),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output,
            destination,
            image_format,
            video_format,
            audio_format,
        )

        on_sample(sampler_output)


factory.register(BaseModelSampler, MossSampler, ModelType.MOSS)
