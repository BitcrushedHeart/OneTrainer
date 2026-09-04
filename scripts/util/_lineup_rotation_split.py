"""Bake group-selective Base/Turbo hybrids to localize the distillation.

The uniform alpha sweep (_lineup_rotation_interp.py) showed the few-step
distillation behavior survives down to alpha=0.5, so a scalar can't separate
"looks like Turbo" from "is distilled". This script splits by TENSOR GROUP
instead: selected groups of the 48 differing late tensors are taken from
Turbo verbatim, the rest stay Base. Full-strength copies are bit-exact, need
no SVD, and are on-spectrum by construction.

Because attention and mlp partition the diff, "Base + attn from Turbo" is
identical to "Turbo with mlp reverted to Base" -- each bake answers both
directions. Test each hybrid for (a) does it read as distilled, (b) does it
look like Turbo; the 2x2 localizes where each property lives. Refine with
finer groups (qk / v / w13 / w2, or layer ranges) as needed.

Groups match against the differing-tensor key stems:
  attn = attention.to_q/to_k/to_v      mlp = feed_forward.w1/w2/w3
  qk   = to_q/to_k                     v   = to_v
  w13  = w1/w3                         w2  = w2
Combine with '+' (e.g. --groups qk+v). --layers restricts further (e.g. 26-29).
"""

import argparse
import re
import time
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\08Jul\SoReal!_Base_V1.0.safetensors"
DEFAULT_TURBO = r"E:\AI\Data\Models\StableDiffusion\SoReal!\08Jul\SoReal!_Turbo_V1.0.safetensors"

GROUP_STEMS = {
    "attn": ("attention.to_q", "attention.to_k", "attention.to_v"),
    "mlp": ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
    "qk": ("attention.to_q", "attention.to_k"),
    "v": ("attention.to_v",),
    "w13": ("feed_forward.w1", "feed_forward.w3"),
    "w2": ("feed_forward.w2",),
}
_LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.+)\.weight$")


def parse_layers(spec: str) -> range:
    lo, _, hi = spec.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def bake(base: Path, turbo: Path, groups: str, layers: range, out_path: Path) -> None:
    stems = tuple(s for g in groups.split("+") for s in GROUP_STEMS[g])
    t0 = time.time()
    sd: dict[str, torch.Tensor] = {}
    n_diff, n_taken = 0, 0
    with (
        safe_open(str(base), framework="pt", device="cpu") as fb,
        safe_open(str(turbo), framework="pt", device="cpu") as ft,
    ):
        src_meta = fb.metadata() or {}
        keys = sorted(fb.keys())
        assert set(keys) == set(ft.keys()), "key sets differ"
        for k in keys:
            wb = fb.get_tensor(k)
            wt = ft.get_tensor(k)
            if torch.equal(wb, wt):
                sd[k] = wb
                continue
            n_diff += 1
            m = _LAYER_KEY.match(k)
            take = m is not None and int(m.group(1)) in layers and m.group(2).endswith(stems)
            sd[k] = wt if take else wb
            n_taken += take
            print(f"  {'TURBO' if take else 'base '}  {k}", flush=True)

    meta = dict(src_meta)
    meta["modelspec.title"] = out_path.stem
    meta["comment"] = (
        f"group-selective hybrid: {n_taken}/{n_diff} differing late tensors taken from Turbo "
        f"(groups={groups}, layers={layers.start}-{layers.stop - 1}), rest bit-exact Base"
    )
    save_file(sd, str(out_path), metadata=meta)
    print(
        f"wrote {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)  "
        f"[{n_taken}/{n_diff} from Turbo, {time.time() - t0:.0f}s]\n",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--turbo", default=DEFAULT_TURBO)
    ap.add_argument(
        "--groups",
        nargs="+",
        default=["attn", "mlp"],
        help="one output per entry; combine groups within an entry with '+' (e.g. qk+v)",
    )
    ap.add_argument("--layers", default="22-29", help="layer range taken from Turbo, e.g. 26-29")
    args = ap.parse_args()

    base, turbo = Path(args.base), Path(args.turbo)
    layers = parse_layers(args.layers)
    for groups in args.groups:
        tag = groups.replace("+", "")
        suffix = f"_TurboRot_{tag}" + (f"_L{args.layers}" if args.layers != "22-29" else "")
        bake(base, turbo, groups, layers, base.with_name(f"{base.stem}{suffix}.safetensors"))


if __name__ == "__main__":
    main()
