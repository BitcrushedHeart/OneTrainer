"""Retarget a Base-trained DoRA-OFT adapter into per-variant warm-start adapters.

Emits, from one Base-trained adapter (e.g. V40), two *adapter* files -- one per
distilled variant -- meant to be loaded via ``lora_model_name`` as the starting
weights of a short per-variant training run on Turbo / Lightning. This moves the
sentinel alignment from bake space into adapter space: instead of blending merged
checkpoints at per-family strengths, the late-family ``oft_R`` skew vectors are
scaled by the same least-squares ``s*`` (and DoRA multipliers mapped
``m -> 1 + s*(m - 1)``), so ``variant + adapter`` starts where the aligned bake
would have landed, and training only has to learn the residual a scalar cannot
express.

Turbo is the sentinel (strength 1.0 everywhere), so its init is the source adapter
verbatim; Lightning's late families get the derived strengths. Outside the late
blocks all variants are bit-identical to Base, so the source parameters are already
exact there.

Scaling the skew vector by ``s`` rotates by ~``s * theta``; against the bake-time
linear weight blend ``W + s*C`` this differs at O(theta^2) -- run ``--verify`` to
print the actual worst-case relative deviation (expected well under the bf16 file
floor at trained angles).

Reads are strictly one safetensors mapping at a time with copies forced out of the
mmap (three simultaneous 11.5GB mappings die with in-page access violations while a
training run holds this machine's memory; a bare ``.to(torch.float32)`` on an
fp32-stored adapter is a no-op that keeps the mapping alive past close).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)  # multithreaded Windows MKL segfaults on batched linalg

from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.oft_align import (  # noqa: E402
    _read_oft_config,
    merged_full,
    reconstruct_oft_rotation,
)
from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS  # noqa: E402

LINEUP = r"E:\AI\Data\Models\StableDiffusion\SoReal!\27Jul"
DEFAULT_ADAPTER = r"E:\AI\Data\Models\Lora\SoReal!_V40_OFT_9.safetensors"
DEFAULT_SENTINEL = rf"{LINEUP}\SoReal!_Turbo_V1.1.safetensors"
DEFAULT_TARGETS = {
    "Turbo": rf"{LINEUP}\SoReal!_Turbo_V1.1.safetensors",
    "Lightning": rf"{LINEUP}\SoReal!_Lightning_V1.1.safetensors",
}
LATE_START = 22

_ADAPTER_KEY = re.compile(r"^(?:transformer\.)?layers\.(\d+)\.(.+)\.oft_R\.weight$")
_STEM_TO_FAM = {s: f for f, ss in DEFAULT_LATE_FAMILY_STEMS.items() for s in ss}


def _late_family(key: str) -> str | None:
    """Family name if ``key`` is a late-block oft_R / dora_multiplier key, else None."""
    stem_key = key
    for suffix in (".oft_R.weight", ".dora_multiplier"):
        if key.endswith(suffix):
            stem_key = key[: -len(suffix)]
            break
    else:
        return None
    m = re.match(r"^(?:transformer\.)?layers\.(\d+)\.(.+)$", stem_key)
    if not m or int(m.group(1)) < LATE_START:
        return None
    return _STEM_TO_FAM.get(m.group(2))


def _read_adapter(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read a whole adapter via plain file I/O (no mmap; see _read_late_weights)."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        md = header.pop("__metadata__", {})
        tensors = {}
        for k, info in header.items():
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            buf = bytearray(f.read(end - begin))
            tensors[k] = torch.frombuffer(buf, dtype=_ST_DTYPES[info["dtype"]]).reshape(info["shape"])
    return tensors, md


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _read_late_weights(ckpt_path: str, stems: set[str]) -> dict[str, torch.Tensor]:
    """Read the late-block weights for ``stems`` via plain file I/O (no mmap).

    While a training run holds this machine's memory, mmap page-ins from a second
    11.5GB mapping fail intermittently with in-page access violations that kill the
    process. Buffered reads degrade gracefully instead, so the checkpoint header is
    parsed by hand and only the needed tensors are read.
    """
    with open(ckpt_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        prefix = ""
        for k in header:
            if k.endswith(".weight") and ".layers." in k:
                prefix = "transformer." if k.startswith("transformer.") else ""
                break
        out = {}
        for s in stems:
            info = header[f"{prefix}{s}.weight"]
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            buf = bytearray(f.read(end - begin))
            t = torch.frombuffer(buf, dtype=_ST_DTYPES[info["dtype"]]).reshape(info["shape"])
            out[s] = t.to(torch.float32)
    return out


def family_strengths_sequential(
    adapter: dict[str, torch.Tensor],
    adapter_path: str,
    sentinel_path: str,
    target_path: str,
) -> dict[str, float]:
    """family_match_strengths math, but one checkpoint mapping at a time."""
    cfg = _read_oft_config(adapter_path)
    late = {}  # stem -> (fam, vec)
    for k, v in adapter.items():
        m = _ADAPTER_KEY.match(k)
        if not m:
            continue
        layer, stem = int(m.group(1)), m.group(2)
        fam = _STEM_TO_FAM.get(stem)
        if fam is None or layer < LATE_START:
            continue
        late[f"layers.{layer}.{stem}"] = (fam, v.to(torch.float32))

    stems = set(late)
    w_target = _read_late_weights(target_path, stems)
    w_sent = _read_late_weights(sentinel_path, stems)

    acc: dict[str, list[float]] = {f: [0.0, 0.0] for f in DEFAULT_LATE_FAMILY_STEMS}
    for stem, (fam, vec) in late.items():
        R = reconstruct_oft_rotation(vec, cfg)
        Wx, Ws = w_target[stem], w_sent[stem]
        target = merged_full(Ws, R)  # sentinel at strength 1.0
        C = merged_full(Wx, R) - Wx
        T0 = target - Wx
        acc[fam][0] += torch.sum(C * T0, dtype=torch.float64).item()
        acc[fam][1] += torch.sum(C * C, dtype=torch.float64).item()

    return {f: (n / d if d > 0.0 else 1.0) for f, (n, d) in acc.items()}


def verify_linearity(
    adapter: dict[str, torch.Tensor],
    adapter_path: str,
    target_path: str,
    strengths: dict[str, float],
) -> float:
    """Worst relative deviation of skew-scaling vs the bake-time linear blend."""
    cfg = _read_oft_config(adapter_path)
    late = {}
    for k, v in adapter.items():
        m = _ADAPTER_KEY.match(k)
        if not m:
            continue
        layer, stem = int(m.group(1)), m.group(2)
        fam = _STEM_TO_FAM.get(stem)
        if fam is not None and layer >= LATE_START:
            late[f"layers.{layer}.{stem}"] = (fam, v.to(torch.float32))
    weights = _read_late_weights(target_path, set(late))
    worst = 0.0
    for stem, (fam, vec) in late.items():
        s = strengths[fam]
        W = weights[stem]
        blend = W + s * (merged_full(W, reconstruct_oft_rotation(vec, cfg)) - W)
        scaled = merged_full(W, reconstruct_oft_rotation(vec * s, cfg))
        worst = max(worst, float((scaled - blend).norm() / W.norm()))
    return worst


def retarget(
    adapter: dict[str, torch.Tensor],
    strengths: dict[str, float],
) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in adapter.items():
        fam = _late_family(k)
        if fam is None or strengths.get(fam, 1.0) == 1.0:
            out[k] = v
            continue
        s = strengths[fam]
        if k.endswith(".oft_R.weight"):
            out[k] = (v.to(torch.float32) * s).to(v.dtype)
        else:  # .dora_multiplier: m -> 1 + s*(m - 1)
            out[k] = (1.0 + s * (v.to(torch.float32) - 1.0)).to(v.dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--sentinel", default=DEFAULT_SENTINEL)
    ap.add_argument("--out-dir", default=r"E:\AI\Data\Models\Lora")
    ap.add_argument("--verify", action="store_true", help="print skew-scaling vs linear-blend deviation")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    stem = Path(args.adapter).stem
    adapter, md = _read_adapter(args.adapter)

    for variant, ckpt in DEFAULT_TARGETS.items():
        out_path = Path(args.out_dir) / f"{stem}_{variant}_init.safetensors"
        if out_path.exists() and not args.force:
            print(f"SKIP {out_path.name}: exists (use --force)")
            continue

        if ckpt == args.sentinel:
            strengths = dict.fromkeys(DEFAULT_LATE_FAMILY_STEMS, 1.0)
            print(f"{variant}: sentinel, late strengths 1.0 (verbatim copy)")
        else:
            strengths = family_strengths_sequential(adapter, args.adapter, args.sentinel, ckpt)
            print(f"{variant}: late strengths " + "  ".join(f"{f}={s:.4f}" for f, s in strengths.items()))
            for f, s in strengths.items():
                if s <= 0.5 or s >= 1.5:
                    print(
                        f"  WARNING: {f}_late strength {s:.3f} outside [0.5, 1.5] -- "
                        f"a scalar can't align this family; inspect before training."
                    )
            if args.verify:
                dev = verify_linearity(adapter, args.adapter, ckpt, strengths)
                print(f"  verify: worst skew-scaling vs linear-blend rel deviation {dev:.2e}")

        out_md = dict(md)
        out_md["soreal_retarget"] = json.dumps(
            {
                "source_adapter": Path(args.adapter).name,
                "variant": variant,
                "variant_checkpoint": Path(ckpt).name,
                "sentinel": Path(args.sentinel).name,
                "late_strengths": strengths,
            }
        )
        save_file(retarget(adapter, strengths), str(out_path), metadata=out_md)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
