"""Benchmark the sourceless image-cache aggregate load on the real cache.

Reproduces the production sourceless path (AspectBucketing is placeholdered out
of the pipeline, so variant_key_from_aspect returns None for every row) and
measures how many torch.load calls the aggregate load makes and how fast it
runs. A StubAB mimics that disconnected bucketing.

BENCH_MODE=stub  -> aspect_bucketing = disconnected stub (production sourceless)
BENCH_MODE=none  -> aspect_bucketing = None (the path my earlier fix handles)

    BENCH_N=4000 BENCH_MODE=stub venv/Scripts/python.exe scripts/util/_bench_sourceless_aggregate.py
"""

import json
import os
import sys
import threading
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
MODE = os.environ.get("BENCH_MODE", "stub")
TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "30"))

_load_count = [0]
_orig_load = torch.load


def _counting_load(*a, **k):
    _load_count[0] += 1
    return _orig_load(*a, **k)


torch.load = _counting_load


class StubAB:
    """Mimics the AspectBucketing instance SmartDiskCache holds in sourceless:
    it's a real object (not None) but disconnected from the pipeline, so
    variant_key_from_aspect always returns None."""

    frame_dim_enabled = False

    def variant_key_from_aspect(self, variation, index, aspect):
        return None

    def compute_bucket_method_hash(self):
        return None


def _trimmed_index():
    with open(os.path.join(REAL_CACHE, "image", "cache.json"), encoding="utf-8") as f:
        full = json.load(f)
    entries = dict(list(full["entries"].items())[:N])
    full["entries"] = entries
    full["hash_index"] = {}
    return full


def _build():
    ab = StubAB() if MODE == "stub" else None
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
        aspect_bucketing=ab,
    )
    trimmed = _trimmed_index()
    image_cache._load_cache_index = lambda: json.loads(json.dumps(trimmed))
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "c", "path": "c"}],
        settings={},
        definition=[[PlaceholderModule(), image_cache], OutputPipelineModule(names=["image_path", "latent_image"])],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def main() -> int:
    print(f"mode={MODE}  N={N}  cache={REAL_CACHE}")
    ds = _build()
    done = [False]
    err = [None]

    def run():
        try:
            ds.start_next_epoch()
            done[0] = True
        except Exception as e:  # noqa: BLE001
            err[0] = e

    t0 = time.time()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(TIMEOUT)
    elapsed = time.time() - t0

    loads = _load_count[0]
    print(f"elapsed={elapsed:.1f}s  completed={done[0]}  torch.load calls={loads}")
    if err[0] is not None:
        print(f"ERROR: {err[0]!r}")
    if elapsed > 0:
        print(f"torch.load rate = {loads / elapsed:.0f}/s")
    verdict = "FAST (synth, no .pt reads)" if loads < N * 0.1 else "SLOW (.pt read per row)"
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
