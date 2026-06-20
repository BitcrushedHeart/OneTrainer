"""End-to-end smoke test for the sourceless cache metadata bake.

Builds a tiny synthetic cache (fake tensors — no real dataset needed), bakes it
with a normal sourced pass, then:

1. asserts the cache.json index is metadata-complete (shared coverage helper),
2. asserts a SECOND sourced pass re-resolves NOTHING and re-saves NOTHING
   (convergence — the bug this fix targets),
3. loads the cache in sourceless mode and drains an epoch (the bake is usable),
4. strips one entry's metadata and asserts sourceless init now hard-raises
   (the guard).

    venv/Scripts/python.exe scripts/util/_smoke_sourceless_cache.py

Exits 0 on success, non-zero on the first failed expectation.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.util.sourceless_cache_util import sourceless_cache_problems

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

N = 4


class PairedSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, image_paths, text_paths):
        super().__init__()
        self.image_paths = image_paths
        self.text_paths = text_paths

    def length(self):
        return len(self.image_paths)

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ["image_path", "sample_prompt_path", "latent_image", "text_embedding", "prompt", "concept"]

    def get_item(self, variation, index, requested_name=None):
        return {
            "image_path": self.image_paths[index],
            "sample_prompt_path": self.text_paths[index],
            "latent_image": torch.tensor([index, variation], dtype=torch.float32),
            "text_embedding": torch.tensor([index + 100, variation], dtype=torch.float32),
            "prompt": f"prompt-{index}",
            "concept": {
                "name": f"concept-{index}",
                "path": f"concept-{index}",
                "seed": 7 + index,
                "enabled": True,
                "image_variations": 1,
                "text_variations": 1,
                "balancing": 1.0,
                "balancing_strategy": "REPEATS",
                "loss_weight": 1.0,
                "include_subdirectories": False,
                "image": {"resolution": "test"},
                "text": {"source": "sample"},
            },
        }


def _cache_modules(cache_root, *, sourceless):
    image_cache = SmartDiskCache(
        cache_dir=str(cache_root / "image"),
        split_names=["latent_image"],
        aggregate_names=["image_path"],
        variations_in_name="concept.image_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path", "concept.seed", "concept.image"],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="image_path",
        sourceless=sourceless,
    )
    text_cache = SmartDiskCache(
        cache_dir=str(cache_root / "text"),
        split_names=["text_embedding"],
        aggregate_names=[],
        variations_in_name="concept.text_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path", "concept.seed", "concept.text"],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="sample_prompt_path",
        sourceless=sourceless,
        content_key_in_name="prompt",
    )
    return image_cache, text_cache


def _build_dataset(cache_root, image_paths=None, text_paths=None, *, sourceless=False):
    modules = []
    if sourceless:
        modules.append(PlaceholderModule())
    else:
        modules.append(PairedSource(image_paths, text_paths))
    modules.extend(_cache_modules(cache_root, sourceless=sourceless))
    output = OutputPipelineModule(names=["image_path", "latent_image", "text_embedding"])
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "concept-a", "path": "concept-a"}],
        settings={},
        definition=[modules, output],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _drain(ds):
    ds.start_next_epoch()
    return list(ds)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="ot_sourceless_smoke_"))
    cache_root = root / "cache"
    image_paths = [str(root / "img" / f"{i}.jpg") for i in range(N)]
    text_paths = [str(root / "img" / f"{i}.txt") for i in range(N)]
    for p in image_paths + text_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Path(p).write_bytes(b"x")

    # 1. bake with a normal sourced pass
    _drain(_build_dataset(cache_root, image_paths, text_paths, sourceless=False))
    problems = sourceless_cache_problems(str(cache_root))
    if problems:
        print("FAIL: cache not metadata-complete after bake:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("ok: cache metadata-complete after one sourced pass")

    # 2. convergence — a second sourced pass must resolve/re-save nothing
    image_cache, _text_cache = _cache_modules(cache_root, sourceless=False)
    resolved = {"n": 0}
    saved = {"n": 0}
    original_get_previous = SmartDiskCache._get_previous_item
    original_save = SmartDiskCache._save_cache_index

    def counting_get_previous(self, variation, name, index):
        if name in ("concept", "prompt", "prompt_1", "prompt_2"):
            resolved["n"] += 1
        return original_get_previous(self, variation, name, index)

    def counting_save(self, *a, **k):
        saved["n"] += 1
        return original_save(self, *a, **k)

    SmartDiskCache._get_previous_item = counting_get_previous
    SmartDiskCache._save_cache_index = counting_save
    try:
        _drain(_build_dataset(cache_root, image_paths, text_paths, sourceless=False))
    finally:
        SmartDiskCache._get_previous_item = original_get_previous
        SmartDiskCache._save_cache_index = original_save
    if resolved["n"] != 0:
        print(f"FAIL: second sourced pass re-resolved concept/prompt {resolved['n']} times (should be 0)")
        return 1
    if saved["n"] != 0:
        print(f"FAIL: second sourced pass re-saved the index {saved['n']} times (should be 0)")
        return 1
    print("ok: second sourced pass converged (0 re-resolutions, 0 index re-saves)")

    # 3. sourceless load must succeed and yield rows
    rows = _drain(_build_dataset(cache_root, sourceless=True))
    if not rows:
        print("FAIL: sourceless pass produced no rows")
        return 1
    print(f"ok: sourceless pass produced {len(rows)} rows")

    # 4. guard — strip one image entry's metadata, expect a hard raise
    image_index_path = cache_root / "image" / "cache.json"
    index = json.loads(image_index_path.read_text(encoding="utf-8"))
    victim = next(iter(index["entries"]))
    index["entries"][victim].pop("sourceless", None)
    index["entries"][victim].pop("sourceless_rows", None)
    image_index_path.write_text(json.dumps(index), encoding="utf-8")

    try:
        _drain(_build_dataset(cache_root, sourceless=True))
    except RuntimeError as e:
        if "sourceless metadata" in str(e):
            print("ok: guard raised on stripped entry")
            print("\nSMOKE PASSED")
            return 0
        print(f"FAIL: raised, but not the metadata guard: {e}")
        return 1
    print("FAIL: sourceless init did NOT raise on a metadata-stripped entry")
    return 1


if __name__ == "__main__":
    sys.exit(main())
