"""Bake spectrum-projected variants of the SoReal V0.95 Base/Turbo pair.

Rationale (see _distill_rotation_diag.py / _distill_rotation_blockfit.py):
the distillation delta is ~99.95% a dense two-sided rotation -- singular
values barely move. So we can snap each model onto the OTHER model's
spectrum while keeping its own singular vectors (= the rotation), giving a
pair whose relative difference is a pure rotation with EQUAL magnitudes --
near-aligned for further training.

Closed form: for changed 2D weight W_src with counterpart W_other,
  SVD  W_src = U S_src V^T   ->   out = U * S_other * V^T
This is the nearest matrix to W_src expressible as R_L W_other R_R
(von Neumann); discarded energy per tensor = ||S_src - S_other||^2.

Outputs (suffix _SpectrumVariant, saved next to the sources):
  Turbo_SpectrumVariant: Turbo geometry + Base magnitudes
  Base_SpectrumVariant : Base geometry + Turbo magnitudes
Unchanged tensors are copied bit-exact; changed non-2D (norm gains, pad
tokens, biases) keep the source model's values and are reported.

fp32 on GPU, TF32 off; saved as bf16 to match the sources.
"""

import argparse
import time
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Base_V0.95.safetensors"
DEFAULT_TURBO = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Turbo_V0.95.safetensors"
MIN_DIM = 64  # don't SVD-project degenerate 2D shapes (e.g. [1, d] pad tokens)


def project_pass(src_path: str, other_path: str, out_path: Path, device: str) -> None:
    """Build one variant: src geometry + other's singular values."""
    t0 = time.time()
    sd: dict[str, torch.Tensor] = {}
    n_proj, n_kept_changed, disc_num, disc_den = 0, 0, 0.0, 0.0
    kept_changed: list[str] = []

    with (
        safe_open(src_path, framework="pt", device="cpu") as fs,
        safe_open(other_path, framework="pt", device="cpu") as fo,
    ):
        keys = sorted(fs.keys())
        assert set(keys) == set(fo.keys()), "key sets differ"
        for n, k in enumerate(keys):
            ws_cpu = fs.get_tensor(k)
            wo_cpu = fo.get_tensor(k)
            if torch.equal(ws_cpu, wo_cpu):
                sd[k] = ws_cpu  # bit-exact copy
                continue
            if ws_cpu.ndim != 2 or min(ws_cpu.shape) < MIN_DIM:
                sd[k] = ws_cpu  # changed but not projectable -> keep source values
                kept_changed.append(f"{k} {tuple(ws_cpu.shape)}")
                n_kept_changed += 1
                continue
            ws = ws_cpu.to(device, torch.float32)
            wo = wo_cpu.to(device, torch.float32)
            u, s_src, vh = torch.linalg.svd(ws, full_matrices=False)
            s_other = torch.linalg.svdvals(wo)
            out = u @ torch.diag(s_other) @ vh
            sd[k] = out.to("cpu", ws_cpu.dtype).contiguous()
            # discarded delta energy = ||out - ws||^2 = ||s_src - s_other||^2 (exact)
            disc = float(((s_src - s_other) ** 2).sum())
            delta2 = float(((ws - wo) ** 2).sum())
            disc_num += disc
            disc_den += delta2
            n_proj += 1
            frac = disc / delta2 if delta2 > 0 else 0.0
            print(f"  [{n + 1}/{len(keys)}] proj {k} {tuple(ws.shape)}  spec-mismatch/delta = {frac:.5f}", flush=True)
            del ws, wo, u, vh, out
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\n{out_path.name}: projected {n_proj} tensors; changed-but-kept {n_kept_changed}", flush=True)
    if disc_den > 0:
        print(f"  total delta energy discarded by projection: {disc_num / disc_den:.6f}", flush=True)
    if kept_changed:
        print("  changed non-projectable tensors (kept source values):")
        for s in kept_changed:
            print(f"    {s}")

    meta = {
        "modelspec.architecture": "z-image",
        "modelspec.title": out_path.stem,
        "source_geometry": src_path,
        "source_spectrum": other_path,
        "comment": "spectrum-projected: source singular vectors (rotation) + counterpart singular values (magnitudes); "
        "relative diff between the pair is a pure two-sided rotation",
        "delta_energy_discarded": f"{disc_num / disc_den:.6f}" if disc_den > 0 else "0",
    }
    save_file(sd, str(out_path), metadata=meta)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)  [{time.time() - t0:.0f}s]\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--turbo", default=DEFAULT_TURBO)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False

    base, turbo = Path(args.base), Path(args.turbo)
    # Turbo geometry + Base magnitudes
    project_pass(str(turbo), str(base), turbo.with_name(turbo.stem + "_SpectrumVariant.safetensors"), args.device)
    # Base geometry + Turbo magnitudes
    project_pass(str(base), str(turbo), base.with_name(base.stem + "_SpectrumVariant.safetensors"), args.device)


if __name__ == "__main__":
    main()
