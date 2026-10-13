"""Keep caption text out of text caches (``TrainConfig.cache_scrub_captions``).

A text cache stores two caption-bearing things besides the encoder output the
model actually trains on: the prompt strings stamped into ``cache.json`` for
sourceless training, and the token-id tensors in every ``.pt`` (decodable with
the public tokenizer). With scrubbing on, SmartDiskCache stamps ``""`` for the
prompts and writes the token tensors as zeros of the same shape and dtype.

Nothing reads cached token VALUES once the encoder output is cached, with one
exception: output embeddings, which locate their placeholder tokens in
``tokens`` after the cached hidden state is loaded (BaseModel
._apply_output_embeddings). Token scrubbing is therefore skipped whenever an
output embedding is configured; prompt blanking still applies.
"""

import inspect
import re

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod

from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

_TOKEN_TENSOR_NAME = re.compile(r"tokens(_\d+)?")
_KNOB_NAMES = ("store_sourceless_prompts", "scrub_split_names")


def uses_output_embeddings(config: TrainConfig) -> bool:
    if config.training_method == TrainingMethod.EMBEDDING and config.embedding.is_output_embedding:
        return True
    return any(embedding.is_output_embedding for embedding in config.additional_embeddings)


def token_tensor_names(split_names: list[str]) -> list[str]:
    """``tokens`` / ``tokens_<n>`` split names (not ``tokens_mask*``)."""
    return [name for name in split_names if _TOKEN_TENSOR_NAME.fullmatch(name)]


def smart_disk_cache_supports_scrub() -> bool:
    parameters = inspect.signature(SmartDiskCache.__init__).parameters
    return all(name in parameters for name in _KNOB_NAMES)


def text_cache_scrub_kwargs(config: TrainConfig, text_split_names: list[str]) -> dict:
    """Extra SmartDiskCache kwargs for a text cache; ``{}`` when scrubbing is off."""
    if not getattr(config, "cache_scrub_captions", False):
        return {}
    if not smart_disk_cache_supports_scrub():
        print(
            "WARNING: cache_scrub_captions is on, but the installed mgds SmartDiskCache has no "
            "caption-scrub support; text caches will store caption text. Update mgds."
        )
        return {}
    if uses_output_embeddings(config):
        print(
            "cache_scrub_captions: output embeddings read cached token ids, so token tensors are kept; "
            "prompt strings are still blanked."
        )
        scrub = []
    else:
        scrub = token_tensor_names(text_split_names)
    return {"store_sourceless_prompts": False, "scrub_split_names": scrub}
