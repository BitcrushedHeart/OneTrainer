"""End-to-end training-pipeline visualization for Flux.1 + roundtrip for Flux.2/SDXL.

For each clean image in testimages/ (non-mask, non-t* files), at 512px and 768px buckets:

  - 0_inputimage.png                       the bucketed image (what the VAE eats)
  - 1_sdxl.png                             SDXL  encode->decode roundtrip
  - 2_flux1.png                            Flux.1 encode->decode roundtrip
  - 3_flux2.png                            Flux.2 encode->decode roundtrip
  - flux1_clean_latent.png                 Flux.1 training-space latent (4x4 grid of 16 channels,
                                           globally normalized [-3,+3] -> [0,255])
  - flux1_noisy_sigma{X.XXX}_latent.png    Flux.1 training tensor at each timestep
                                           (clean_latent * (1-sigma) + N(0,1) * sigma in latent space
                                           -- this is what BaseFluxSetup.py:269 produces)
  - flux1_noisy_sigma{X.XXX}_decoded.png   That tensor unscaled and decoded back to pixel space
                                           -- visualizes what the noisy latent "looks like"

Outputs to: testimages/training/{512,768}/{stem}/
"""

from pathlib import Path

import torch

from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

FLUX1_VAE_PATH = "E:/AI/Data/Models/VAE/ae.safetensors"
FLUX2_VAE_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B"
    / "snapshots/17c3b160520b7dd44665dbf0b9ed9dd30c15cd06/vae"
)
SDXL_BAKED_CKPT = "E:/AI/Data/Models/StableDiffusion/MyModels/SoReal_PendingRLHF.safetensors"

SOURCE_DIR = Path(__file__).parent / "testimages"

# Flux.1 (from ae config.json)
FLUX1_SHIFT = 0.1159
FLUX1_SCALE = 0.3611
# SDXL standard (no shift)
SDXL_SCALE = 0.13025

PATCH_MULTIPLE = 16
RESOLUTIONS = [512, 768]
TIMESTEPS_SIGMA = [0.001, 0.25, 0.5, 0.75, 1.0]
SEED = 42


def bucket_dims(orig_w: int, orig_h: int, target_area: int, multiple: int = PATCH_MULTIPLE) -> tuple[int, int]:
    """Aspect-preserving resize to area ~= target_area, snapped to `multiple`."""
    aspect = orig_w / orig_h
    h = max(multiple, int(round(np.sqrt(target_area / aspect) / multiple)) * multiple)
    w = max(multiple, int(round(np.sqrt(target_area * aspect) / multiple)) * multiple)
    return w, h


def to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    """1,3,H,W or 3,H,W in [-1,1] -> H,W,3 uint8."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.float().cpu().clamp(-1, 1)
    return ((tensor + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()


def channel_grid(latent_chw: torch.Tensor, cols: int, rows: int, clip: float = 3.0) -> np.ndarray:
    """16,H,W (or any C,H,W) -> (rows*H, cols*W) uint8 grid, globally normalized to [-clip,+clip] -> [0,255]."""
    C, H, W = latent_chw.shape
    assert cols * rows >= C, f"{C} channels won't fit in {cols}x{rows} grid"
    arr = latent_chw.float().cpu().numpy()
    grid = np.zeros((rows * H, cols * W), dtype=np.float32)
    for c in range(C):
        r, col = c // cols, c % cols
        norm = np.clip((arr[c] + clip) / (2 * clip), 0.0, 1.0)
        grid[r * H : (r + 1) * H, col * W : (col + 1) * W] = norm
    return (grid * 255).astype(np.uint8)


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device} ({dtype})\n")

# ============================ Flux.1 VAE ============================
print("Loading Flux.1 VAE (manual config + LDM key conversion)...")
flux1_kwargs = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": FLUX1_SCALE,
    "shift_factor": FLUX1_SHIFT,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}
flux1_vae = AutoencoderKL(**flux1_kwargs)
flux1_state = load_file(FLUX1_VAE_PATH)
flux1_converted = convert_ldm_vae_checkpoint(flux1_state, flux1_kwargs)
flux1_result = flux1_vae.load_state_dict(flux1_converted, strict=False)
assert not flux1_result.missing_keys and not flux1_result.unexpected_keys, "Flux.1 VAE key mismatch"
flux1_vae = flux1_vae.to(device=device, dtype=dtype).eval()
print(f"  Flux.1: {len(flux1_converted)} keys matched.")

# ============================ Flux.2 VAE ============================
print("Loading Flux.2 VAE (AutoencoderKLFlux2 from HF cache)...")
flux2_vae = AutoencoderKLFlux2.from_pretrained(str(FLUX2_VAE_DIR), torch_dtype=dtype).to(device).eval()
print(f"  Flux.2: loaded from {FLUX2_VAE_DIR.name}")

# ============================ SDXL VAE ============================
print("Loading SDXL VAE (extracted from baked checkpoint)...")
sdxl_state: dict[str, torch.Tensor] = {}
with safe_open(SDXL_BAKED_CKPT, framework="pt") as f:
    for k in f.keys():  # noqa: SIM118 — safetensors safe_open exposes only .keys()
        if k.startswith("first_stage_model."):
            sdxl_state[k] = f.get_tensor(k)
sdxl_kwargs = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": SDXL_SCALE,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
}
sdxl_vae = AutoencoderKL(**sdxl_kwargs)
sdxl_converted = convert_ldm_vae_checkpoint(sdxl_state, sdxl_kwargs)
sdxl_result = sdxl_vae.load_state_dict(sdxl_converted, strict=False)
assert not sdxl_result.missing_keys and not sdxl_result.unexpected_keys, (
    f"SDXL VAE key mismatch: missing={sdxl_result.missing_keys[:3]} unexpected={sdxl_result.unexpected_keys[:3]}"
)
sdxl_vae = sdxl_vae.to(device=device, dtype=dtype).eval()
print(f"  SDXL: {len(sdxl_converted)} keys matched.\n")

# ============================ Inputs ============================
images = sorted(
    p for p in SOURCE_DIR.glob("*.webp") if p.is_file() and not p.stem.startswith("t0") and "masklabel" not in p.stem
)
# Also pick up any new pngs/jpgs that aren't masks
for p in sorted(SOURCE_DIR.glob("*.png")):
    if "masklabel" not in p.stem and p.is_file():
        images.append(p)

print(f"Found {len(images)} clean source images:")
for p in images:
    print(f"  {p.name}")
print()

# ============================ Pipeline ============================
gen = torch.Generator(device=device).manual_seed(SEED)

for res in RESOLUTIONS:
    area = res * res
    print(f"\n=== {res}px (target area = {area:,}) ===")

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        w, h = bucket_dims(*img.size, area)
        img_resized = img.resize((w, h), Image.LANCZOS)

        out_dir = SOURCE_DIR / "training" / str(res) / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        img_resized.save(out_dir / "0_inputimage.png", optimize=True)

        arr = np.asarray(img_resized).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

        # ---- Flux.1: full training pipeline ----
        with torch.no_grad():
            f1_raw = flux1_vae.encode(tensor).latent_dist.mean  # (1,16,h/8,w/8)
            f1_clean_decoded = flux1_vae.decode(f1_raw).sample
        f1_scaled = (f1_raw - FLUX1_SHIFT) * FLUX1_SCALE  # training-space, what model sees

        Image.fromarray(channel_grid(f1_scaled[0], cols=4, rows=4), mode="L").save(
            out_dir / "flux1_clean_latent.png", optimize=True
        )
        Image.fromarray(to_uint8_rgb(f1_clean_decoded), mode="RGB").save(out_dir / "2_flux1.png", optimize=True)

        # ---- Flux.1: flow-matching noise at each timestep IN LATENT SPACE ----
        # Matches modules/modelSetup/BaseFluxSetup.py:269 (noisy = clean*(1-sigma) + noise*sigma)
        for sigma in TIMESTEPS_SIGMA:
            noise = torch.randn(f1_scaled.shape, generator=gen, device=device, dtype=dtype)
            noisy_scaled = f1_scaled * (1.0 - sigma) + noise * sigma
            Image.fromarray(channel_grid(noisy_scaled[0], cols=4, rows=4), mode="L").save(
                out_dir / f"flux1_noisy_sigma{sigma:.3f}_latent.png", optimize=True
            )
            # decode for pixel-space preview: invert scale/shift, decode
            noisy_raw = noisy_scaled / FLUX1_SCALE + FLUX1_SHIFT
            with torch.no_grad():
                noisy_decoded = flux1_vae.decode(noisy_raw).sample
            Image.fromarray(to_uint8_rgb(noisy_decoded), mode="RGB").save(
                out_dir / f"flux1_noisy_sigma{sigma:.3f}_decoded.png", optimize=True
            )

        # ---- Flux.2: encode/decode roundtrip ----
        with torch.no_grad():
            f2_raw = flux2_vae.encode(tensor).latent_dist.mean  # (1,32,h/8,w/8)
            f2_decoded = flux2_vae.decode(f2_raw).sample
        Image.fromarray(to_uint8_rgb(f2_decoded), mode="RGB").save(out_dir / "3_flux2.png", optimize=True)

        # ---- SDXL: encode/decode roundtrip ----
        with torch.no_grad():
            sdxl_raw = sdxl_vae.encode(tensor).latent_dist.mean  # (1,4,h/8,w/8)
            sdxl_decoded = sdxl_vae.decode(sdxl_raw).sample
        Image.fromarray(to_uint8_rgb(sdxl_decoded), mode="RGB").save(out_dir / "1_sdxl.png", optimize=True)

        f1_stats = f1_scaled[0].float().cpu()
        print(
            f"  {img_path.stem[:32]:<32s} {w}x{h} -> "
            f"F1 lat {tuple(f1_scaled.shape[1:])} mean={f1_stats.mean():+.2f} std={f1_stats.std():.2f}  "
            f"F2 lat {tuple(f2_raw.shape[1:])}  SDXL lat {tuple(sdxl_raw.shape[1:])}"
        )

total_files = len(images) * len(RESOLUTIONS) * (5 + 2 * len(TIMESTEPS_SIGMA))
print(
    f"\nDone. {total_files} files written across {SOURCE_DIR / 'training'}/{{{','.join(str(r) for r in RESOLUTIONS)}}}/"
)
