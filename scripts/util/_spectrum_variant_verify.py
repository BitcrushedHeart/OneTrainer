"""Verify the SpectrumVariant lineup claim:

Base, Turbo (=SpectrumVariant), Lightning_SpectrumVariant differ ONLY in the
rotations of late-block (layers 22-29) qkv + mlp weights. Three checks per
variant vs Base:

  1. bit-identity: every key outside the 48 late qkv/mlp tensors is
     torch.equal; any unexpected diff is a FAIL.
  2. equal magnitudes: for the 48, rel singular-value mismatch
     ||sig_v - sig_b|| / ||sig_b||  should be ~bf16 noise (<~1e-3).
  3. pure rotation: with R_L = U_v U_b^T, R_R = V_b V_v^T (orthogonal by
     construction), R_L W_b R_R == U_v Sig_b V_v^T; so
     ||U_v Sig_b V_v^T - W_v|| / ||W_v||  measures how far the stored tensor
     is from an exact rotation of Base -- should be at the bf16 storage
     floor (~1e-3), since that's literally what the bake computed pre-quantization.
"""

import re
import time

import torch

from safetensors import safe_open

DIR = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul"
BASE = rf"{DIR}\SoReal!_Base_V0.95.safetensors"
VARIANTS = {
    "Turbo": rf"{DIR}\SoReal!_Turbo_V0.95.safetensors",
    "LightningSV": rf"{DIR}\SoReal!_Lightning_V0.95_SpectrumVariant.safetensors",
}
KEEP_RE = re.compile(r"^layers\.(2[2-9])\.(attention\.to_[qkv]|feed_forward\.w[123])\.weight$")
DEVICE = "cuda"

torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()

for name, path in VARIANTS.items():
    n_ident, unexpected = 0, []
    late = []
    with (
        safe_open(BASE, framework="pt", device="cpu") as fb,
        safe_open(path, framework="pt", device="cpu") as fv,
    ):
        keys = sorted(fb.keys())
        assert set(keys) == set(fv.keys()), f"{name}: key sets differ"
        for k in keys:
            wb, wv = fb.get_tensor(k), fv.get_tensor(k)
            is_late = bool(KEEP_RE.match(k))
            if torch.equal(wb, wv):
                n_ident += 1
                if is_late:
                    unexpected.append(f"{k} (late tensor UNCHANGED?)")
                continue
            if not is_late:
                unexpected.append(k)
                continue
            late.append(k)

        print(f"\n=== {name} vs Base ===")
        print(f"check 1 (bit-identity outside the 48): {n_ident} identical, "
              f"{len(late)}/48 late differ, {len(unexpected)} UNEXPECTED -> "
              f"{'PASS' if not unexpected and len(late) == 48 else 'FAIL'}")
        for k in unexpected:
            print(f"  UNEXPECTED: {k}")

        worst_sig, worst_rot = (0.0, ""), (0.0, "")
        for k in late:
            wb = fb.get_tensor(k).to(DEVICE, torch.float32)
            wv = fv.get_tensor(k).to(DEVICE, torch.float32)
            sig_b = torch.linalg.svdvals(wb)
            u_v, sig_v, vh_v = torch.linalg.svd(wv, full_matrices=False)
            sig_rel = float((sig_v - sig_b).norm() / sig_b.norm())
            rot_rel = float((u_v @ torch.diag(sig_b) @ vh_v - wv).norm() / wv.norm())
            if sig_rel > worst_sig[0]:
                worst_sig = (sig_rel, k)
            if rot_rel > worst_rot[0]:
                worst_rot = (rot_rel, k)
            del wb, wv, u_v, vh_v
            torch.cuda.empty_cache()
        print(f"check 2 (magnitudes equal): worst sigma rel-diff = {worst_sig[0]:.2e}  ({worst_sig[1]})")
        print(f"check 3 (pure rotation):    worst rotation resid = {worst_rot[0]:.2e}  ({worst_rot[1]})")
        print("   (bf16 storage floor ~1e-3; both PASS iff at/below that)")

print(f"\ndone in {time.time() - t0:.0f}s")
