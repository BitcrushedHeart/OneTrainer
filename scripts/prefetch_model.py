"""Download the Hugging Face base model parts a training config will load, into $HF_HOME.

Run on a cloud pod before the GPU is attached (see RunpodCloud.attach_gpu), so the trainer starts
from a warm cache. Failures only warn: the trainer downloads whatever is missing itself.

Usage: run-cmd.sh prefetch_model --config-path=<remote config json>
"""

from util.import_util import script_imports

script_imports()

import argparse
import json
import traceback

from modules.util.config.TrainConfig import TrainConfig
from modules.util.hf_prefetch_util import hf_prefetch_patterns, hf_prefetch_repo

import huggingface_hub


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download a training config's Hugging Face base model.")
    parser.add_argument("--config-path", type=str, required=True, dest="config_path")
    args = parser.parse_args()

    config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        config.from_dict(json.load(f))

    repo = hf_prefetch_repo(config)
    if repo is None:
        print(f"prefetch: base model {config.base_model_name!r} is not a Hugging Face repo id; nothing to download")
        return 0

    patterns = hf_prefetch_patterns(config)
    print(f"prefetch: downloading {repo} ({', '.join(patterns)})", flush=True)
    try:
        path = huggingface_hub.snapshot_download(repo, allow_patterns=patterns)
    except Exception:
        traceback.print_exc()
        print("prefetch: failed; the trainer will download what it needs itself")
        return 0
    print(f"prefetch: done -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
