"""Pre-download a training run's Hugging Face base model parts into the HF cache.

Cloud runs that upload without a GPU call this while the pod is still on the CPU-only rate, so the
model loader later finds every file cached instead of downloading the text encoder / VAE / transformer
with the GPU attached. Mirrors what the loaders' _prepare_sub_modules fetch: the subfolders a run
overrides with its own file (transformer, VAE, prior) are left out.
"""

import os

from modules.util.config.TrainConfig import TrainConfig


def hf_prefetch_repo(config: TrainConfig) -> str | None:
    """The base model's Hugging Face repo id, or None when it is a local path or a single file."""
    name = (config.base_model_name or "").strip()
    if not name or os.path.exists(name) or name.startswith(("/", "\\", ".")) or ":" in name:
        return None
    if name.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin", ".gguf")):
        return None
    return name if name.count("/") == 1 else None


def hf_prefetch_patterns(config: TrainConfig) -> list[str]:
    """snapshot_download allow_patterns covering the parts the model loader will read from the base repo."""
    patterns = ["model_index.json", "tokenizer*/*", "scheduler/*", "text_encoder*/*", "unet/*"]
    if not config.transformer.model_name:
        patterns.append("transformer/*")
    if not config.vae.model_name:
        patterns.append("vae/*")
    if not config.prior.model_name:
        patterns.append("prior/*")
    if not config.decoder.model_name:
        patterns.append("decoder/*")
    return patterns
