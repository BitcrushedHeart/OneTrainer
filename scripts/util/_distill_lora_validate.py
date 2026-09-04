"""Numeric validation gate for the extracted distillation LoRAs.

Independently reconstructs Base + sum(group LoRAs) from the saved files and
compares against the true Distilled checkpoint, per tensor. Catches key-naming,
transpose, alpha and grouping bugs, and quantifies rank-truncation loss.

This is the numeric half of the gate. The visual half (Base + all groups @1.0
vs Distilled, same seeds, 4-step grid in SwarmUI) is still required before any
leave-one-out conclusion counts.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from _distill_lora_extract import DEFAULT_BASE, DEFAULT_DIST, DEFAULT_OUT, weight_anchor
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--distilled", default=DEFAULT_DIST)
    ap.add_argument("--lora-dir", default=DEFAULT_OUT)
    ap.add_argument("--pattern", default="SoReal!_V0.94_distill_*.safetensors")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    lora_files = sorted(Path(args.lora_dir).glob(args.pattern))
    if not lora_files:
        raise SystemExit(f"no LoRA files matching {args.pattern} in {args.lora_dir}")

    # anchor -> (group, {suffix: tensor}); anchor is e.g. "diffusion_model.layers.3.attention.to_q"
    patches: dict[str, dict] = {}
    file_of: dict[str, str] = {}
    for f in lora_files:
        with safe_open(str(f), framework="pt", device="cpu") as fl:
            for k in fl.keys():  # noqa: SIM118 — safe_open handle, not a dict
                for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha", ".diff", ".diff_b"):
                    if k.endswith(suffix):
                        anchor = k[: -len(suffix)]
                        patches.setdefault(anchor, {})[suffix] = fl.get_tensor(k)
                        file_of[anchor] = f.stem
                        break
                else:
                    print(f"UNRECOGNIZED key in {f.name}: {k}")

    def delta_hat(anchor: str, is_bias: bool) -> torch.Tensor | None:
        p = patches.get(anchor)
        if p is None:
            return None
        if is_bias:
            return p.get(".diff_b")
        if ".diff" in p:
            return p[".diff"]
        if ".lora_A.weight" in p and ".lora_B.weight" in p:
            a = p[".lora_A.weight"].to(args.device, torch.float32)
            b = p[".lora_B.weight"].to(args.device, torch.float32)
            rank = a.shape[0]
            alpha = p[".alpha"].item() if ".alpha" in p else rank
            return (b @ a) * (alpha / rank)
        return None

    per_group = defaultdict(lambda: {"tensors": 0, "resid_sq": 0.0, "delta_sq": 0.0, "worst": (0.0, "")})
    missing, covered = [], set()

    with (
        safe_open(args.base, framework="pt", device="cpu") as fa,
        safe_open(args.distilled, framework="pt", device="cpu") as fb,
    ):
        for k in sorted(fa.keys()):
            a, b = fa.get_tensor(k), fb.get_tensor(k)
            if torch.equal(a, b):
                continue
            delta = (b.to(torch.float32) - a.to(torch.float32)).to(args.device)
            anchor_stem = weight_anchor(k)
            anchor = f"diffusion_model.{anchor_stem}" if anchor_stem else None
            dh = delta_hat(anchor, k.endswith(".bias")) if anchor else None
            if dh is None:
                missing.append(k)
                continue
            covered.add((anchor, "bias" if k.endswith(".bias") else "weight"))
            dh = dh.to(args.device, torch.float32)
            resid = (delta - dh).pow(2).sum().item()
            dsq = delta.pow(2).sum().item()
            g = per_group[file_of[anchor]]
            g["tensors"] += 1
            g["resid_sq"] += resid
            g["delta_sq"] += dsq
            rel = (resid / dsq) ** 0.5 if dsq > 0 else 0.0
            if rel > g["worst"][0]:
                g["worst"] = (rel, k)
            del delta, dh

    print("=== reconstruction residuals (Base + LoRA vs Distilled) ===")
    print(f"{'file':>42} {'tensors':>8} {'energy_captured':>16} {'rel_resid':>10}  worst_tensor")
    for name in sorted(per_group):
        g = per_group[name]
        cap = 1.0 - g["resid_sq"] / g["delta_sq"] if g["delta_sq"] > 0 else 1.0
        rel = (g["resid_sq"] / g["delta_sq"]) ** 0.5 if g["delta_sq"] > 0 else 0.0
        print(f"{name:>42} {g['tensors']:>8} {cap:>16.6f} {rel:>10.4f}  {g['worst'][0]:.4f} {g['worst'][1]}")

    if missing:
        print(f"\nWARNING: {len(missing)} changed tensors have NO patch in any LoRA file:")
        for k in missing:
            print(f"  {k}")

    extra = []
    for a, p in patches.items():
        has_weight = any(s in p for s in (".diff", ".lora_A.weight", ".lora_B.weight"))
        if has_weight and (a, "weight") not in covered:
            extra.append(a)
        if ".diff_b" in p and (a, "bias") not in covered:
            extra.append(a + " (bias)")
    if extra:
        print(f"\nNOTE: {len(extra)} anchors in LoRA files matched no changed checkpoint tensor:")
        for a in extra[:20]:
            print(f"  {a}")

    print("\nnumeric gate done — visual 4-step gate still required (Base + all groups @1.0 vs Distilled).")


if __name__ == "__main__":
    main()
