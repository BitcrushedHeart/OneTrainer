"""Benchmark requesting `concept` from a sourceless cache (the VariationSorting
epoch-start pattern) on the real cache.

VariationSorting groups every row by concept.* at epoch start, which routes to
SmartDiskCache.get_item("concept"). Before the runtime-value fast path that meant
a torch.load of each tensor .pt just to return a concept dict. This measures how
many torch.load calls N concept requests trigger now.

    BENCH_N=4000 venv/Scripts/python.exe scripts/util/_bench_concept_request.py
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineState
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

import torch

REAL_CACHE = os.environ.get("BENCH_CACHE", r"F:/workspace/SoReal!/cache")
N = int(os.environ.get("BENCH_N", "4000"))

_load_count = [0]
_orig_load = torch.load


def _counting_load(*a, **k):
    _load_count[0] += 1
    return _orig_load(*a, **k)


torch.load = _counting_load


def _trimmed_index():
    with open(os.path.join(REAL_CACHE, "image", "cache.json"), encoding="utf-8") as f:
        full = json.load(f)
    full["entries"] = dict(list(full["entries"].items())[:N])
    full["hash_index"] = {}
    return full


def main() -> int:
    image_cache = SmartDiskCache(
        cache_dir=os.path.join(REAL_CACHE, "image"),
        split_names=["latent_image"],
        aggregate_names=["crop_resolution", "image_path"],
        variations_in_name="concept.image_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path", "concept.seed", "concept.image"],
        group_enabled_in_name="concept.enabled",
        modeltype="Z_IMAGE",
        source_path_in_name="image_path",
        sourceless=True,
    )
    trimmed = _trimmed_index()
    image_cache._load_cache_index = lambda: json.loads(json.dumps(trimmed))
    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "c", "path": "c"}],
        settings={},
        definition=[[PlaceholderModule(), image_cache], OutputPipelineModule(names=["image_path", "latent_image"])],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    ds.start_next_epoch()  # init + aggregate (already fast)

    rows = len(image_cache._sourceless_filepaths)
    _load_count[0] = 0  # reset: measure only the concept-request phase
    t0 = time.time()
    served = 0
    for in_index in range(rows):
        item = image_cache.get_item(in_index, "concept")
        if isinstance(item, dict) and isinstance(item.get("concept"), dict):
            served += 1
    elapsed = time.time() - t0

    print(f"rows={rows}  concept served={served}  torch.load calls during concept requests={_load_count[0]}")
    print(f"elapsed={elapsed:.2f}s")
    verdict = "FAST (served from index, no .pt reads)" if _load_count[0] < rows * 0.1 else "SLOW (.pt read per concept)"
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
