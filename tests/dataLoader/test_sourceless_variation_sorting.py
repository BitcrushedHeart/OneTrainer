"""Sourceless runs must serve each row's own concept values.

A sourceless SmartDiskCache serves "concept" from the snapshot stamped in
cache.json, with its rows already grouped and balanced (repeats applied).
The VariationSorting after the caches used to group and balance those rows
again, so for repeats > 1 its first concept's block outgrew the epoch and every
output index resolved "concept" (beta, loss weight, mask overrides) and
"prompt" to that block: V54 trained every DPO pair at the On Policy beta.
Sourceless pipelines now pass those names through row for row.
"""

import os

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.VariationSorting import VariationSorting
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

# Three concepts sharing one cache root, like the DPO concepts of a run; the
# first one has repeats > 1 so its block is the one that used to swallow the rest.
CONCEPTS = [
    {"name": "on-policy", "balancing": 3.0, "dpo_beta_override": 40.0, "supervised_mix_override": 1.5},
    # Opted out of adaptive beta: the flag must reach its rows, and only its rows.
    {
        "name": "off-policy",
        "balancing": 1.0,
        "dpo_beta_override": 20.0,
        "supervised_mix_override": 3.0,
        "adaptive": False,
    },
    {"name": "minor-edits", "balancing": 1.0, "dpo_beta_override": 1.0, "supervised_mix_override": 1.0},
]
ROWS_PER_CONCEPT = 3
FIELDS = (
    "dpo_beta_override",
    "dpo_adaptive_beta",
    "supervised_mix_override",
    "unmasked_weight_override",
    "loss_weight",
    # Per-concept timestep sampling, served from the same snapshot.
    "min_noising_strength_override",
    "max_noising_strength_override",
    "noising_weight_override",
    "noising_bias_override",
    "timestep_shift_override",
    "noising_range_after_shift",
)


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def __init__(self):
        super().__init__()
        self.stop_check_fun = lambda: False

    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


class ModelSetupStub:
    def prepare_text_caching(self, model, config):
        pass


class Source(PipelineModule, RandomAccessPipelineModule):
    """Rows 0-2 belong to concept 0, 3-5 to concept 1, 6-8 to concept 2."""

    def __init__(self, images, texts, shared_root):
        super().__init__()
        self.images, self.texts, self.shared_root = images, texts, shared_root

    def length(self):
        return len(self.images)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return [
            "image_path",
            "sample_prompt_path",
            "latent_image",
            "crop_resolution",
            "text_embedding",
            "prompt",
            "concept",
        ]

    def get_item(self, variation, index, requested_name=None):
        return {
            "image_path": self.images[index],
            "sample_prompt_path": self.texts[index],
            "latent_image": torch.tensor([index, variation], dtype=torch.float32),
            "crop_resolution": (512, 512),
            "text_embedding": torch.tensor([index + 100, variation], dtype=torch.float32),
            "prompt": f"prompt-{index}",
            "concept": _concept(index // ROWS_PER_CONCEPT, self.shared_root).to_dict(),
        }


def _concept(index: int, cache_dir: str) -> ConceptConfig:
    spec = CONCEPTS[index]
    concept = ConceptConfig.default_values()
    concept.name = spec["name"]
    concept.path = f"/data/{spec['name']}"
    concept.seed = 100 + index
    concept.cache_dir = cache_dir
    concept.balancing = spec["balancing"]
    concept.dpo_beta_override = spec["dpo_beta_override"]
    concept.dpo_adaptive_beta = spec.get("adaptive", True)
    concept.supervised_mix_override = spec["supervised_mix_override"]
    # Values exact in float32: the output module casts floats to tensors.
    concept.unmasked_weight_override = 0.25 * (index + 1)
    concept.loss_weight = 1.0 + index
    concept.min_noising_strength_override = 0.125 * index
    concept.max_noising_strength_override = 0.5 + 0.125 * index
    concept.noising_weight_override = 0.5 * index
    concept.noising_bias_override = -0.25 * index
    concept.timestep_shift_override = 1.0 + index
    concept.noising_range_after_shift = index == 1
    return concept


def _cache_modules(run_cache: str, shared: str, *, sourceless: bool, before_cache_fun=None):
    loader = DataLoaderHarness()
    loader._concepts_for_cache = [_concept(i, shared) for i in range(len(CONCEPTS))]
    config = TrainConfig.default_values()
    config.cache_dir = run_cache
    config.latent_caching = True
    config.sourceless_training = sourceless
    # Keep prompt text in the test caches so the prompt alignment is checkable.
    config.cache_scrub_captions = False
    return loader._cache_modules_from_names(
        model=None,
        model_setup=ModelSetupStub(),
        image_split_names=["latent_image"],
        image_aggregate_names=["crop_resolution", "image_path"],
        text_split_names=["text_embedding"],
        sort_names=["latent_image", "crop_resolution", "image_path", "text_embedding", "prompt", "concept"],
        config=config,
        text_caching=True,
        before_cache_image_fun=before_cache_fun,
    )


def _variation_sorting(modules) -> VariationSorting:
    return next(m for m in modules if isinstance(m, VariationSorting))


def test_sourced_pipeline_groups_and_balances_the_uncached_names(tmp_path):
    sorting = _variation_sorting(_cache_modules(str(tmp_path / "run"), str(tmp_path / "shared"), sourceless=False))
    assert sorting.names == ["prompt", "concept"]
    assert sorting.balancing_in_name == "concept.balancing"
    assert sorting.group_enabled_in_name == "concept.enabled"


def test_sourceless_pipeline_passes_the_uncached_names_through(tmp_path):
    sorting = _variation_sorting(_cache_modules(str(tmp_path / "run"), str(tmp_path / "shared"), sourceless=True))
    assert sorting.names == ["prompt", "concept"]
    assert sorting.balancing_in_name is None
    assert sorting.balancing_strategy_in_name is None
    assert sorting.variations_group_in_names is None
    assert sorting.group_enabled_in_name is None


def _dataset(modules, output_names):
    return MGDS(
        device=torch.device("cpu"),
        concepts=[],
        settings={},
        definition=[modules, OutputPipelineModule(names=output_names)],
        batch_size=1,
        state=PipelineState(),
        seed=7,
    )


def _epoch(ds):
    ds.start_next_epoch()
    rows = []
    for batch in ds:
        path = batch["image_path"][0] if isinstance(batch["image_path"], list) else batch["image_path"]
        row = int(os.path.basename(path).split(".")[0])
        prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        rows.append((row, prompt, {field: float(batch[field]) for field in FIELDS}))
    return rows


def _expected(row: int) -> dict:
    concept = _concept(row // ROWS_PER_CONCEPT, "")
    return {field: float(getattr(concept, field)) for field in FIELDS}


def test_sourceless_epoch_serves_each_rows_own_concept(tmp_path):
    run_cache, shared = str(tmp_path / "run"), str(tmp_path / "shared")
    images, texts = [], []
    for i in range(ROWS_PER_CONCEPT * len(CONCEPTS)):
        for ext, paths in ((".png", images), (".txt", texts)):
            path = tmp_path / "data" / f"{i:02d}{ext}"
            os.makedirs(path.parent, exist_ok=True)
            path.write_bytes(b"x")
            paths.append(str(path))
    output_names = ["image_path", "prompt"] + [(f"concept.{field}", field) for field in FIELDS]

    # The sourced pass builds the shared caches and stamps the concept snapshots.
    sourced = _cache_modules(run_cache, shared, sourceless=False, before_cache_fun=lambda: None)
    sourced_rows = _epoch(_dataset([Source(images, texts, shared)] + sourced, output_names))
    expected_rows = ROWS_PER_CONCEPT * int(sum(spec["balancing"] for spec in CONCEPTS))
    assert len(sourced_rows) == expected_rows
    for row, prompt, values in sourced_rows:
        assert prompt == f"prompt-{row}"
        assert values == _expected(row)

    # Sourceless: the same rows, each with its own concept values and prompt.
    sourceless = _cache_modules(run_cache, shared, sourceless=True, before_cache_fun=lambda: None)
    sourceless_rows = _epoch(_dataset([PlaceholderModule()] + sourceless, output_names))
    assert sorted(r[0] for r in sourceless_rows) == sorted(r[0] for r in sourced_rows)
    for row, prompt, values in sourceless_rows:
        assert prompt == f"prompt-{row}"
        assert values == _expected(row)

    # The previous wiring (a balancing VariationSorting after the sourceless
    # caches) hands the first concept's block to every row; kept as the
    # reason the passthrough exists.
    rebalanced = [m for m in sourceless if not isinstance(m, VariationSorting)] + [
        VariationSorting(
            names=["prompt", "concept"],
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name=["concept.path", "concept.seed", "concept.include_subdirectories", "concept.text"],
            group_enabled_in_name="concept.enabled",
        )
    ]
    wrong = _epoch(_dataset([PlaceholderModule()] + rebalanced, output_names))
    assert all(values == _expected(0) for _, _, values in wrong)
    assert any(values != _expected(row) for row, _, values in wrong)
