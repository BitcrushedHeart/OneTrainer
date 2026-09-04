"""Pre-flight check on an OFT / DoRA-OFT adapter before it is baked into the lineup.

Why this exists
---------------
The SoReal! lineup transplants one Base-trained adapter onto Base, Turbo and
Lightning. Turbo and Lightning run at 4-10 steps, so they cannot average out a
perturbation the way Base's 25-30 steps can: an adapter that looks fine on Base can
read as plastic / 3D-rendered on Turbo. Measurement of the two adapters that bracket
this failure (see the envelope constants below) shows the difference is *not* in how
the adapter is transported to Turbo -- transplanting it and re-deriving it additively
from the lineup delta agree to ~1% of that delta, well under the bf16 file's own
quantisation noise. The difference is simply how hard the adapter perturbs the model,
and how much of that perturbation lands in the non-norm-preserving magnitude channel.

So the useful gate is on the adapter itself, before baking anything:

  theta_merge      pooled relative weight change the bake causes on the late blocks
  worst row norm   largest per-output-row norm change -- the V22 collapse signature,
                   which pooled Frobenius does not see (row-concentrated damage is
                   visible while the pooled norm barely moves)
  mean |m-1|       average DoRA magnitude deviation; the magnitude channel is the one
                   that is not norm-preserving, and weight decay pushes adaptation
                   into it (adjust_wds gives oft_R standard decay but shields
                   dora_multiplier, so decay suppresses rotation and the optimiser
                   compensates through magnitude)

Block sizes are derived per tensor from the stored parameter width, never from
ot_config -- OFTModule.initialize_weights adjusts a requested size to divide
in_features, so one adapter can hold several block sizes at once.

Usage
-----
  python scripts/util/_oft_preflight.py --adapter path/to/adapter.safetensors
  python scripts/util/_oft_preflight.py --adapter a.safetensors --turbo path/to/Turbo.safetensors

Passing --turbo makes the late-block set exact (the tensors that actually differ
between Base and Turbo) instead of inferred from layer index and module family, and
adds theta_distill for scale. Exit status is 1 if any metric leaves the envelope, so
this can gate a bake script.
"""

import argparse
import re
import sys
from pathlib import Path

import torch

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.oft_align import (  # noqa: E402
    AdapterOFTConfig,
    _read_oft_config,
    block_size_from_elements,
    merged_full,
    reconstruct_oft_rotation,
)
from modules.util.oft_merge import DEFAULT_LATE_FAMILY_STEMS  # noqa: E402

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\24Jul\SoReal!_Base_V1.1.safetensors"

# Envelope thresholds, calibrated on exactly two adapters. This is a tripwire, not a
# certification -- it says "unlike the one that broke Turbo, like the one that did
# not", and should be recalibrated as more adapters are characterised.
#
#   V38_OFT_5 (wd 0.015, block 128) -- produced the plastic / 3D-render look on Turbo
#       theta_merge 0.0101   worst row 0.0508   mean|m-1| 0.00478
#   V39_OFT   (wd 0.001, block 400) -- did not
#       theta_merge 0.0024   worst row 0.0153   mean|m-1| 0.00132
#
# Thresholds sit between the two, nearer the known-good side.
THETA_MERGE_MAX = 0.004
ROW_NORM_MAX = 0.025
DORA_DEV_MAX = 0.0025

_ADAPTER_KEY = re.compile(r"^(?:transformer\.)?(.+)\.oft_R\.weight$")
_LAYER_FAMILY = re.compile(r"^layers\.(\d+)\.(.+)$")
LATE_START = 22


def _norm(key: str) -> str:
    return key.removeprefix("transformer.").removesuffix(".weight")


def _is_late(stem: str) -> bool:
    """Fallback late-block test when no Turbo checkpoint is supplied."""
    m = _LAYER_FAMILY.match(stem)
    if not m:
        return False
    layer, tail = int(m.group(1)), m.group(2)
    stems = {s for ss in DEFAULT_LATE_FAMILY_STEMS.values() for s in ss}
    return layer >= LATE_START and tail in stems


def _late_stems(base_path: str, turbo_path: str | None) -> tuple[set[str], str]:
    if turbo_path is None:
        return set(), f"inferred (layer >= {LATE_START}, qkv/mlp families)"
    with (
        safe_open(base_path, framework="pt", device="cpu") as fb,
        safe_open(turbo_path, framework="pt", device="cpu") as ft,
    ):
        if set(fb.keys()) != set(ft.keys()):
            raise ValueError("base and turbo key sets differ; wrong pair?")
        found = {_norm(k) for k in fb.keys() if not torch.equal(fb.get_tensor(k), ft.get_tensor(k))}  # noqa: SIM118
    return found, f"exact (differ between base and turbo: {len(found)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--turbo", default=None, help="optional; makes the late-block set exact")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--theta-merge-max", type=float, default=THETA_MERGE_MAX)
    ap.add_argument("--row-norm-max", type=float, default=ROW_NORM_MAX)
    ap.add_argument("--dora-dev-max", type=float, default=DORA_DEV_MAX)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = args.device

    late, how = _late_stems(args.base, args.turbo)
    cfg_meta = _read_oft_config(args.adapter)

    # ---- phase 1: read everything, then close the files ----
    # Interleaving safetensors mmap with LAPACK segfaults under Windows MKL
    # (see modules/util/oft_align.py).
    with safe_open(args.adapter, framework="pt", device="cpu") as fa:
        rot, mult = {}, {}
        for k in fa.keys():  # noqa: SIM118
            m = _ADAPTER_KEY.match(k)
            if m:
                rot[m.group(1)] = fa.get_tensor(k).to(torch.float32)
            elif k.endswith("dora_multiplier"):
                mult[_norm(k[: -len(".dora_multiplier")])] = fa.get_tensor(k).to(torch.float32)

    if not late:
        late = {s for s in rot if _is_late(s)}
    targets = sorted(set(rot) & late)
    if not targets:
        print("ERROR: adapter covers none of the late blocks; nothing to check.", file=sys.stderr)
        raise SystemExit(2)

    with safe_open(args.base, framework="pt", device="cpu") as fb:
        keys = {_norm(k): k for k in fb.keys()}  # noqa: SIM118
        missing = [s for s in targets if s not in keys]
        if missing:
            raise ValueError(f"{len(missing)} adapter modules absent from base, e.g. {missing[:3]}")
        weights = {s: fb.get_tensor(keys[s]).to(torch.float32) for s in targets}
        turbo_w = {}
        if args.turbo:
            with safe_open(args.turbo, framework="pt", device="cpu") as ft:
                turbo_w = {s: ft.get_tensor(keys[s]).to(torch.float32) for s in targets}

    # ---- phase 2: compute ----
    blocks: dict[int, int] = {}
    tm_n = tm_d = td_n = td_d = 0.0
    worst_tensor = (0.0, "")
    worst_row = (0.0, "")

    for stem in targets:
        W = weights[stem].to(dev)
        vec = rot[stem].to(dev)
        bs = block_size_from_elements(vec.shape[-1])
        blocks[W.shape[1]] = bs
        R = reconstruct_oft_rotation(vec, AdapterOFTConfig(bs, cfg_meta.clip, cfg_meta.scaled))
        merged = merged_full(W, R)
        if stem in mult:
            merged = merged * mult[stem].to(dev).view(-1, 1)

        tm_n += float(((merged - W) ** 2).sum())
        tm_d += float((W**2).sum())
        t = float((merged - W).norm() / W.norm())
        if t > worst_tensor[0]:
            worst_tensor = (t, stem)
        rn = float((merged.norm(dim=1) / W.norm(dim=1).clamp_min(1e-12) - 1.0).abs().max())
        if rn > worst_row[0]:
            worst_row = (rn, stem)
        if stem in turbo_w:
            Wt = turbo_w[stem].to(dev)
            td_n += float(((Wt - W) ** 2).sum())
            td_d += float((W**2).sum())
            del Wt

        del W, vec, R, merged
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    theta_merge = (tm_n / tm_d) ** 0.5
    dora = torch.cat([(mult[s] - 1.0).abs().flatten() for s in targets if s in mult]) if mult else None

    # ---- report ----
    def flag(v: float, lim: float) -> str:
        return f"[ok,   envelope <= {lim}]" if v <= lim else f"[HIGH, envelope <= {lim}]"

    print(f"\nadapter        {Path(args.adapter).name}")
    print(f"base           {Path(args.base).name}")
    print(f"late blocks    {len(targets)} modules, {how}")
    print(
        f"block sizes    {'  '.join(f'{k}->{v}' for k, v in sorted(blocks.items()))}"
        f"   (ot_config requested {cfg_meta.block_size})"
    )
    print()
    print(f"  theta_merge      {theta_merge:.4f}   {flag(theta_merge, args.theta_merge_max)}")
    print(f"  worst row norm   {worst_row[0]:.4f}   {flag(worst_row[0], args.row_norm_max)}")
    if dora is not None:
        print(f"  mean |m-1|       {float(dora.mean()):.5f}  {flag(float(dora.mean()), args.dora_dev_max)}")
        print(f"  max  |m-1|       {float(dora.max()):.5f}")
    else:
        print("  mean |m-1|       n/a      (pure OFT, no DoRA multipliers)")
    print(f"  worst tensor     {worst_tensor[1]}  ({worst_tensor[0]:.4f})")
    print(f"  worst row in     {worst_row[1]}")
    if td_d > 0:
        td = (td_n / td_d) ** 0.5
        print(f"\n  theta_distill    {td:.4f}   (base -> turbo, for scale)")
        print(f"  merge/distill    {theta_merge / td:.3f}")

    breaches = [
        n
        for n, v, lim in (
            ("theta_merge", theta_merge, args.theta_merge_max),
            ("worst row norm", worst_row[0], args.row_norm_max),
            *([("mean |m-1|", float(dora.mean()), args.dora_dev_max)] if dora is not None else []),
        )
        if v > lim
    ]
    if breaches:
        print(f"\n  VERDICT: OUTSIDE envelope on {', '.join(breaches)} -- expect few-step")
        print("           variants (Turbo/Lightning) to degrade before Base does.")
        print("           Lower weight decay and/or raise oft_block_size, or bake the")
        print("           late families at reduced strength.")
        raise SystemExit(1)
    print("\n  VERDICT: within the V39 envelope -- no adapter-strength reason to expect")
    print("           Turbo to degrade. Envelope is calibrated on two adapters only;")
    print("           confirm with a generation A/B at Turbo's step count.")


if __name__ == "__main__":
    main()
