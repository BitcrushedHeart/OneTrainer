"""Bake Base-toward-Turbo interpolations of the SoReal V1.0 spectrum-aligned pair.

Base and Turbo V1.0 are bit-identical except the late (22-29) qkv/mlp weights,
whose difference is a pure two-sided rotation with equal singular values (see
_distill_spectrum_project.py). That makes "align Base toward Turbo" a single
scalar dial: lerp the 48 differing tensors and leave everything else bit-exact.

At these rotation angles (rel delta ~1-3%) the chord (lerp) and the geodesic
(fractional rotation) agree to second order; the only artifact of lerping two
equal-spectrum matrices is a ~theta^2/8 spectral shrink. --respectrum (default
on) removes even that by snapping each interpolated tensor's singular values
back to the shared Base/Turbo spectrum, so every output stays on the lineup's
equal-magnitude manifold and differs from both parents by a pure rotation.

alpha = 0 reproduces Base, alpha = 1 reproduces Turbo. Sweep and A/B at Base's
sampling settings (CFG, full step count): the aesthetic part of the distill
rotation tends to transfer at partial strength before the few-step bias bites.

fp32 on GPU, TF32 off; saved as bf16 to match the sources.
"""

import argparse
import time
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\08Jul\SoReal!_Base_V1.0.safetensors"
DEFAULT_TURBO = r"E:\AI\Data\Models\StableDiffusion\SoReal!\08Jul\SoReal!_Turbo_V1.0.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--turbo", default=DEFAULT_TURBO)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--no-respectrum",
        action="store_true",
        help="skip snapping interpolated singular values back to the shared spectrum",
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    base, turbo = Path(args.base), Path(args.turbo)

    outs: dict[float, dict[str, torch.Tensor]] = {a: {} for a in args.alphas}
    n_interp = 0
    t0 = time.time()
    with (
        safe_open(str(base), framework="pt", device="cpu") as fb,
        safe_open(str(turbo), framework="pt", device="cpu") as ft,
    ):
        keys = sorted(fb.keys())
        assert set(keys) == set(ft.keys()), "key sets differ"
        for k in keys:
            wb_cpu = fb.get_tensor(k)
            wt_cpu = ft.get_tensor(k)
            if torch.equal(wb_cpu, wt_cpu):
                for a in args.alphas:
                    outs[a][k] = wb_cpu  # bit-exact copy, shared across alphas
                continue
            wb = wb_cpu.to(args.device, torch.float32)
            wt = wt_cpu.to(args.device, torch.float32)
            s_shared = None
            if not args.no_respectrum and wb.ndim == 2:
                s_shared = torch.linalg.svdvals(wb)
            for a in args.alphas:
                w = torch.lerp(wb, wt, a)
                if s_shared is not None:
                    u, s, vh = torch.linalg.svd(w, full_matrices=False)
                    shrink = float(((s - s_shared) ** 2).sum().sqrt() / s_shared.norm())
                    w = u @ torch.diag(s_shared) @ vh
                else:
                    shrink = float("nan")
                outs[a][k] = w.to("cpu", wb_cpu.dtype).contiguous()
                print(f"  a={a:.2f} {k} {tuple(wb.shape)}  spectral shrink removed = {shrink:.2e}", flush=True)
            n_interp += 1
            del wb, wt
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\ninterpolated {n_interp} tensors x {len(args.alphas)} alphas  [{time.time() - t0:.0f}s]", flush=True)
    with safe_open(str(base), framework="pt", device="cpu") as fb:
        src_meta = fb.metadata() or {}
    for a in args.alphas:
        out_path = base.with_name(f"{base.stem}_TowardTurbo_a{a:.2f}.safetensors")
        meta = dict(src_meta)
        meta["modelspec.title"] = out_path.stem
        meta["comment"] = (
            f"late qkv/mlp rotation interpolation Base->Turbo at alpha={a} "
            f"(respectrum={'off' if args.no_respectrum else 'on'}); all other tensors bit-exact Base"
        )
        save_file(outs[a], str(out_path), metadata=meta)
        print(f"wrote {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
