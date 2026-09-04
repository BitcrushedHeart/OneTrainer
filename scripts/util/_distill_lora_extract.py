"""Extract the SoReal! V0.94 distillation delta as per-group LoRA files.

Base and Distilled V0.94 differ only in attention to_q/to_k/to_v and
feed_forward w1/w2/w3 (plus refiners); everything else is bit-identical.
This splits that delta (Distilled - Base) into per-group LoRA files so the
causal reversion sweep can be run in SwarmUI/ComfyUI by leave-one-out
loading on the Base checkpoint instead of re-baking 12GB checkpoints.

Groups: {qkv,mlp} x {early 0-6, mid 7-21, late 22-29} + refiners (noise/context
refiner, cap_embedder, and anything else outside the main layer stack).

Output format (ComfyUI Lumina2/Z-Image LoRA path, verified against both the
standalone and SwarmUI-bundled comfy/lora.py):
  - 2D deltas: diffusion_model.<key>.lora_B.weight (out x r),
    .lora_A.weight (r x in), .alpha = r  -> scale alpha/rank = 1 at strength 1
  - 1D bias deltas: diffusion_model.<weight_key>.diff_b
  - 1D weight deltas (norms): diffusion_model.<key>.diff
  - --mode diff stores exact full-rank 2D deltas as .diff (fallback if the
    low-rank gate fails)

Checkpoint keys are already diffusers-style (unfused to_q/to_k/to_v), which is
exactly what z_image_to_diffusers' key_map anchors on, including the sliced
patch onto ComfyUI's fused native qkv.
"""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\28Jun\SoReal!_Base_V0.94.safetensors"
DEFAULT_DIST = r"E:\AI\Data\Models\StableDiffusion\SoReal!\28Jun\SoReal!_Distilled_V0.94.safetensors"
DEFAULT_OUT = r"E:\AI\Data\Models\Lora\Distillers"

# Matches both diffusers-style unfused attention (to_q/to_k/to_v) and native
# fused attention (single qkv), so the same grouping works on either checkpoint layout.
MAIN_LAYER_RE = re.compile(r"^layers\.(\d+)\.(attention\.(?:to_[qkv]|qkv)|feed_forward\.w[123])\.(weight|bias)$")

# diffusers-style names that z_image_to_diffusers puts in the LoRA key_map.
# Anything changed but outside this set is invisible to ComfyUI's LoRA loader.
BLOCK_PREFIX_RE = re.compile(r"^(layers|context_refiner|noise_refiner)\.\d+\.")
BLOCK_PATCHABLE = {
    # diffusers-style unfused attention
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.norm_q",
    "attention.norm_k",
    "attention.to_out.0",
    # native fused attention (ComfyUI/Z-Image checkpoint layout)
    "attention.qkv",
    "attention.out",
    "attention.q_norm",
    "attention.k_norm",
    "attention_norm1",
    "attention_norm2",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "ffn_norm1",
    "ffn_norm2",
    "adaLN_modulation.0",
}
BASIC_PATCHABLE = {
    "final_layer.linear",
    "final_layer.adaLN_modulation.1",
    "x_embedder",
    "cap_embedder.0",
    "cap_embedder.1",
    "t_embedder.mlp.0",
    "t_embedder.mlp.2",
}


def weight_anchor(key: str) -> str | None:
    """The LoRA map anchor (weight key minus '.weight') for a tensor key, or None if unpatchable."""
    if key.endswith(".weight"):
        stem = key[: -len(".weight")]
    elif key.endswith(".bias"):
        stem = key[: -len(".bias")]
    else:
        return None  # x_pad_token / cap_pad_token etc. — no .weight anchor in the map
    m = BLOCK_PREFIX_RE.match(stem)
    if m:
        return stem if stem[m.end() :] in BLOCK_PATCHABLE else None
    return stem if stem in BASIC_PATCHABLE else None


def group_of(key: str, early_end: int, mid_end: int) -> str:
    m = MAIN_LAYER_RE.match(key)
    if not m:
        return "refiners"
    i = int(m.group(1))
    fam = "qkv" if m.group(2).startswith("attention") else "mlp"
    band = "early" if i <= early_end else ("mid" if i <= mid_end else "late")
    return f"{fam}_{band}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--distilled", default=DEFAULT_DIST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument(
        "--mode",
        choices=("lowrank", "diff"),
        default="lowrank",
        help="lowrank: SVD LoRA extraction; diff: exact full deltas as .diff keys",
    )
    ap.add_argument("--oversample", type=int, default=128, help="svd_lowrank oversampling columns")
    ap.add_argument("--niter", type=int, default=7, help="svd_lowrank power iterations")
    ap.add_argument("--early-end", type=int, default=6)
    ap.add_argument("--mid-end", type=int, default=21)
    ap.add_argument(
        "--groups", default=None, help="comma-separated group filter, e.g. 'mlp_late,qkv_late' (default: all)"
    )
    ap.add_argument("--suffix", default="", help="appended to output filenames, e.g. '_exact'")
    ap.add_argument("--prefix", default="SoReal!_V0.94_distill", help="output filename prefix and modelspec title stem")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    args = ap.parse_args()

    save_dtype = getattr(torch, args.save_dtype)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    stats: dict[str, dict] = defaultdict(
        lambda: {"tensors": 0, "params": 0, "captured": 0.0, "total": 0.0, "worst": (1.0, "")}
    )
    unpatchable: list[str] = []
    group_filter = {g.strip() for g in args.groups.split(",")} if args.groups else None
    t0 = time.time()

    with (
        safe_open(args.base, framework="pt", device="cpu") as fa,
        safe_open(args.distilled, framework="pt", device="cpu") as fb,
    ):
        keys_a, keys_b = set(fa.keys()), set(fb.keys())
        if keys_a != keys_b:
            raise SystemExit(
                f"key sets differ: only-base={sorted(keys_a - keys_b)} only-dist={sorted(keys_b - keys_a)}"
            )

        changed = []
        for k in sorted(keys_a):
            a, b = fa.get_tensor(k), fb.get_tensor(k)
            if a.shape != b.shape:
                raise SystemExit(f"shape mismatch on {k}: {a.shape} vs {b.shape}")
            if not torch.equal(a, b):
                changed.append(k)
        print(f"{len(changed)}/{len(keys_a)} tensors differ")

        for n, k in enumerate(changed):
            anchor = weight_anchor(k)
            if anchor is None:
                unpatchable.append(k)
                continue
            group = group_of(k, args.early_end, args.mid_end)
            if group_filter is not None and group not in group_filter:
                continue
            delta = (fb.get_tensor(k).to(torch.float32) - fa.get_tensor(k).to(torch.float32)).to(args.device)
            total = (delta**2).sum().item()
            st = stats[group]
            st["tensors"] += 1
            st["params"] += delta.numel()
            st["total"] += total
            lora_key = f"diffusion_model.{anchor}"

            if delta.ndim == 1:
                suffix = ".diff_b" if k.endswith(".bias") else ".diff"
                groups[group][lora_key + suffix] = delta.to("cpu", save_dtype)
                st["captured"] += total  # exact
            elif args.mode == "diff":
                groups[group][lora_key + ".diff"] = delta.to("cpu", save_dtype)
                st["captured"] += total  # exact
            else:
                r = min(args.rank, *delta.shape)
                q = min(r + args.oversample, *delta.shape)
                u, s, v = torch.svd_lowrank(delta, q=q, niter=args.niter)
                u, s, v = u[:, :r], s[:r], v[:, :r]
                sq = s.sqrt()
                groups[group][lora_key + ".lora_B.weight"] = (u * sq).to("cpu", save_dtype).contiguous()
                groups[group][lora_key + ".lora_A.weight"] = (sq.unsqueeze(1) * v.T).to("cpu", save_dtype).contiguous()
                groups[group][lora_key + ".alpha"] = torch.tensor(float(r))
                captured = (s**2).sum().item()
                st["captured"] += captured
                frac = captured / total if total > 0 else 1.0
                if frac < st["worst"][0]:
                    st["worst"] = (frac, k)
            del delta
            if (n + 1) % 10 == 0 or n + 1 == len(changed):
                print(f"  [{n + 1}/{len(changed)}] {k}  ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== per-group summary ===")
    print(f"{'group':>12} {'tensors':>8} {'params':>14} {'energy_captured':>16} {'worst_tensor_frac':>18}")
    for group in sorted(groups):
        st = stats[group]
        frac = st["captured"] / st["total"] if st["total"] > 0 else 1.0
        worst = f"{st['worst'][0]:.4f} {st['worst'][1]}" if st["worst"][1] else "exact"
        print(f"{group:>12} {st['tensors']:>8} {st['params']:>14,} {frac:>16.6f}   {worst}")

    if unpatchable:
        print("\nWARNING: changed tensors NOT expressible as ComfyUI LoRA patches (left out):")
        for k in unpatchable:
            print(f"  {k}")

    for group, sd in groups.items():
        st = stats[group]
        path = out_dir / f"{prefix}_{group}{args.suffix}.safetensors"
        meta = {
            # SwarmUI's per-group key sniffer can't classify these (no adaLN keys, groups
            # never span refiners+layers), so declare the class explicitly via modelspec.
            "modelspec.architecture": "z-image/lora",
            "modelspec.title": f"{prefix} {group}",
            "modelspec.resolution": "1024x1024",
            "source_base": args.base,
            "source_distilled": args.distilled,
            "group": group,
            "mode": args.mode,
            "rank": str(args.rank),
            "energy_captured": f"{st['captured'] / st['total']:.6f}" if st["total"] > 0 else "1.0",
            "comment": "distillation delta (Distilled - Base); load on Base at strength 1.0; leave-one-out = reversion sweep",
        }
        save_file(sd, str(path), metadata=meta)
        size_mb = path.stat().st_size / 1e6
        print(f"wrote {path}  ({size_mb:.0f} MB)")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
