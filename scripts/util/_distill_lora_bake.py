"""Bake distill-delta patch files onto the Base checkpoint as a new full model.

Companion to _distill_lora_extract.py: applies its output files (.diff exact
patches or .lora_A/.lora_B low-rank pairs) onto Base weights and writes a new
checkpoint. Streams tensor-by-tensor; untouched tensors are copied bit-exact.

Default: Base + mlp_late_exact + qkv_late_exact -> SoReal!_Turbo_V0.94
(the confirmed minimal distillation core, blocks 22-29).
"""

import argparse
import time
from pathlib import Path

import torch

from _distill_lora_extract import DEFAULT_BASE, DEFAULT_OUT
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_PATCHES = [
    rf"{DEFAULT_OUT}\SoReal!_V0.94_distill_mlp_late_exact.safetensors",
    rf"{DEFAULT_OUT}\SoReal!_V0.94_distill_qkv_late_exact.safetensors",
]
DEFAULT_TURBO_OUT = r"E:\AI\Data\Models\StableDiffusion\SoReal!\28Jun\SoReal!_Turbo_V0.94.safetensors"


def load_patches(files: list[str]) -> dict[str, torch.Tensor]:
    """anchor stem (checkpoint key minus '.weight'/'.bias' context) -> fp32 delta"""
    deltas: dict[str, torch.Tensor] = {}

    def put(stem: str, delta: torch.Tensor) -> None:
        if stem in deltas:
            raise SystemExit(f"duplicate patch for {stem} — overlapping patch files?")
        deltas[stem] = delta

    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fl:
            keys = set(fl.keys())
            for k in sorted(keys):
                if not k.startswith("diffusion_model."):
                    raise SystemExit(f"unexpected key prefix in {f}: {k}")
                stem = k[len("diffusion_model.") :]
                if k.endswith(".diff"):
                    put(stem[: -len(".diff")] + ".weight", fl.get_tensor(k).to(torch.float32))
                elif k.endswith(".diff_b"):
                    put(stem[: -len(".diff_b")] + ".bias", fl.get_tensor(k).to(torch.float32))
                elif k.endswith(".lora_B.weight"):
                    anchor = k[: -len(".lora_B.weight")]
                    b = fl.get_tensor(k).to(torch.float32)
                    a = fl.get_tensor(anchor + ".lora_A.weight").to(torch.float32)
                    alpha_key = anchor + ".alpha"
                    scale = fl.get_tensor(alpha_key).item() / a.shape[0] if alpha_key in keys else 1.0
                    put(stem[: -len(".lora_B.weight")] + ".weight", (b @ a) * scale)
                elif k.endswith((".lora_A.weight", ".alpha")):
                    continue  # consumed alongside .lora_B.weight
                else:
                    raise SystemExit(f"unrecognized key in {f}: {k}")
        print(f"loaded {len(deltas)} cumulative patches after {Path(f).name}")
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=DEFAULT_TURBO_OUT)
    ap.add_argument("--title", default="SoReal! Turbo V0.94")
    ap.add_argument(
        "--reference",
        default=None,
        help="optional checkpoint (e.g. true Distilled) to report patched-tensor residuals against",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("patches", nargs="*", default=None, help=f"patch files (default: {DEFAULT_PATCHES})")
    args = ap.parse_args()

    if Path(args.out).exists():
        raise SystemExit(f"refusing to overwrite existing {args.out}")
    patch_files = args.patches or DEFAULT_PATCHES
    deltas = load_patches(patch_files)

    t0 = time.time()
    out_sd: dict[str, torch.Tensor] = {}
    applied = set()
    with safe_open(args.base, framework="pt", device="cpu") as fa:
        for k in fa.keys():  # noqa: SIM118 — safe_open handle, not a dict
            w = fa.get_tensor(k)
            if k in deltas:
                if deltas[k].shape != w.shape:
                    raise SystemExit(f"shape mismatch on {k}: patch {deltas[k].shape} vs base {w.shape}")
                w = (w.to(args.device, torch.float32) + deltas[k].to(args.device)).to("cpu", w.dtype)
                applied.add(k)
            out_sd[k] = w
    unused = set(deltas) - applied
    if unused:
        raise SystemExit(f"patches with no matching base tensor: {sorted(unused)}")
    print(f"applied {len(applied)} patches, {len(out_sd) - len(applied)} tensors copied bit-exact")

    meta = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": "z-image",
        "modelspec.title": args.title,
        "modelspec.date": "2026-07-03",
        "modelspec.implementation": "https://github.com/huggingface/diffusers",
        "ot_branch": "bitcrushed-blend",
        "comment": f"Base + distillation core ({', '.join(Path(p).stem for p in patch_files)}); "
        "early/mid blocks reverted to Base for diversity, 10-step distillation retained",
    }
    save_file(out_sd, args.out, metadata=meta)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size / 1e9:.2f} GB)  in {time.time() - t0:.0f}s")

    if args.reference:
        print("\n=== patched tensors vs reference ===")
        worst = (0.0, "")
        with safe_open(args.reference, framework="pt", device="cpu") as fr:
            for k in sorted(applied):
                r = fr.get_tensor(k).to(args.device, torch.float32)
                d = (out_sd[k].to(args.device, torch.float32) - r).norm().item() / (r.norm().item() + 1e-12)
                if d > worst[0]:
                    worst = (d, k)
        print(f"worst relative deviation from reference on patched tensors: {worst[0]:.6f}  ({worst[1]})")


if __name__ == "__main__":
    main()
