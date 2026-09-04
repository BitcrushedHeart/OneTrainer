"""Bake SoReal!_Lightning_V0.95_SpectrumVariant: late-blocks-only + Base magnitudes.

Lightning, unlike Turbo, still carries changes outside mlp_late/qkv_late.
This makes it structurally match the (now spectrum-projected) Turbo:
  1. REVERT to Base V0.95 everything except layers 22-29 attention.to_{q,k,v}
     and feed_forward.w{1,2,3} weights (i.e. qkv/mlp early+mid, refiners,
     norms, to_out, embedders -- all Base).
  2. For the kept 48 late-block tensors: spectrum-project Lightning onto
     Base's magnitudes (SVD swap: Lightning singular vectors + Base singular
     values), same as _distill_spectrum_project.py did for Turbo.

Reports how much was reverted and how much delta energy the projection
discards -- Lightning is the separately-trained 4-step model, so its delta
may be less spectrum-neutral than Turbo's 0.037%.
"""

import argparse
import re
import time
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Base_V0.95.safetensors"
DEFAULT_LIGHT = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Lightning_V0.95.safetensors"

KEEP_RE = re.compile(r"^layers\.(\d+)\.(attention\.to_[qkv]|feed_forward\.w[123])\.weight$")
LATE_START = 22  # keep bands: layers >= 22 (matches extractor mid_end=21)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--lightning", default=DEFAULT_LIGHT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()

    light = Path(args.lightning)
    out_path = light.with_name(light.stem + "_SpectrumVariant.safetensors")

    sd: dict[str, torch.Tensor] = {}
    n_proj, n_reverted, n_same = 0, 0, 0
    disc_num, disc_den = 0.0, 0.0
    reverted: list[str] = []

    with (
        safe_open(args.base, framework="pt", device="cpu") as fb,
        safe_open(args.lightning, framework="pt", device="cpu") as fl,
    ):
        keys = sorted(fb.keys())
        assert set(keys) == set(fl.keys()), "key sets differ"
        for k in keys:
            wb_cpu = fb.get_tensor(k)
            m = KEEP_RE.match(k)
            if not (m and int(m.group(1)) >= LATE_START):
                # outside kept groups -> Base wins (revert if Lightning differed)
                if not torch.equal(wb_cpu, fl.get_tensor(k)):
                    n_reverted += 1
                    reverted.append(k)
                else:
                    n_same += 1
                sd[k] = wb_cpu
                continue
            wl_cpu = fl.get_tensor(k)
            if torch.equal(wb_cpu, wl_cpu):
                n_same += 1
                sd[k] = wb_cpu
                continue
            wl = wl_cpu.to(args.device, torch.float32)
            wb = wb_cpu.to(args.device, torch.float32)
            u, s_l, vh = torch.linalg.svd(wl, full_matrices=False)
            s_b = torch.linalg.svdvals(wb)
            sd[k] = (u @ torch.diag(s_b) @ vh).to("cpu", wl_cpu.dtype).contiguous()
            disc = float(((s_l - s_b) ** 2).sum())
            delta2 = float(((wl - wb) ** 2).sum())
            disc_num += disc
            disc_den += delta2
            n_proj += 1
            print(f"  proj {k}  spec-mismatch/delta = {disc / delta2 if delta2 > 0 else 0.0:.5f}", flush=True)
            del wl, wb, u, vh
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\nprojected {n_proj}, reverted-to-Base {n_reverted}, identical {n_same}", flush=True)
    if disc_den > 0:
        print(f"total delta energy discarded by projection (kept tensors): {disc_num / disc_den:.6f}")
    if reverted:
        print("reverted (changed outside late qkv/mlp):")
        for k in reverted:
            print(f"  {k}")

    meta = {
        "modelspec.architecture": "z-image",
        "modelspec.title": out_path.stem,
        "source_geometry": args.lightning,
        "source_spectrum": args.base,
        "comment": "Lightning late-block (22-29 qkv/mlp) rotations on Base magnitudes; all other tensors reverted to Base",
        "delta_energy_discarded": f"{disc_num / disc_den:.6f}" if disc_den > 0 else "0",
        "reverted_tensors": str(n_reverted),
    }
    save_file(sd, str(out_path), metadata=meta)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
