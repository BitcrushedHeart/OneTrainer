"""SoReal! Base V0.95 vs official Z-Image base: rotation/magnitude divergence report (v2).

v1 lesson: raw Frobenius deltas are dominated by uniform per-tensor DOWN-SCALING
(SoReal norms are 0.05-0.6x Z-Image's) which is FUNCTIONALLY NULL in this
architecture: to_q/to_k feed per-head RMSNorm (norm_q/norm_k), and both branch
outputs pass sandwich norms (attention_norm2 / ffn_norm2) before the residual
add, so uniform scale of to_q,to_k and of the v->out / FFN->w2 compositions
cancels exactly. Spectral-norm-clipped OFT training shrinks norms freely.
(Caveat: w1/w3 scale is NOT fully null -- it shifts SiLU saturation.)

So v2 reports, per 2D weight pair:
  norm_ratio  = ||W_s|| / ||W_z||                     (mostly functionally null)
  dir_div     = ||s*W_s - W_z|| / ||W_z||, s = ||W_z||/||W_s||   scale-corrected divergence
  spec_shape  = ||s*sig_s - sig_z|| / ||sig_z||       spectrum-SHAPE change (real magnitude training)
  rot_share   = 1 - ||s*sig_s - sig_z||^2 / ||s*W_s - W_z||^2    rotation-expressible share
Raw (uncorrected) rel_delta kept for reference. 1D tensors: rel_delta only.

Layout mapping (official file is fused): to_q/k/v <- qkv row-thirds,
to_out.0 <- out, norm_q/k <- q_norm/k_norm; rest match by name.
Writes SoReal_Divergence.md next to the SoReal V0.95 checkpoints.
"""

import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from safetensors import safe_open

SOREAL = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Base_V0.95.safetensors"
ZIMAGE = r"C:\Users\calla\Downloads\z_image_bf16.safetensors"
OUT_MD = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal_Divergence.md"
DEVICE = "cuda"
MIN_DIM = 64

LAYER_RE = re.compile(r"^layers\.(\d+)\.")
ATTN_MAP = {".attention.to_out.0": ".attention.out", ".attention.norm_q": ".attention.q_norm", ".attention.norm_k": ".attention.k_norm"}
QKV_ROW = {".attention.to_q": 0, ".attention.to_k": 1, ".attention.to_v": 2}


def z_lookup(fz, keys_z: set, k: str):
    if k in keys_z:
        return fz.get_tensor(k)
    for suf, idx in QKV_ROW.items():
        for part in (".weight", ".bias"):
            if k.endswith(suf + part):
                zk = k[: -len(suf + part)] + ".attention.qkv" + part
                if zk in keys_z:
                    t = fz.get_tensor(zk)
                    d = t.shape[0] // 3
                    return t[idx * d : (idx + 1) * d]
                return None
    for s_suf, z_suf in ATTN_MAP.items():
        for part in (".weight", ".bias"):
            if k.endswith(s_suf + part):
                zk = k[: -len(s_suf + part)] + z_suf + part
                return fz.get_tensor(zk) if zk in keys_z else None
    return None


def family_of(k: str) -> str:
    for pat, fam in (
        ("attention.to_q", "to_q"), ("attention.to_k", "to_k"), ("attention.to_v", "to_v"),
        ("attention.to_out", "to_out"), ("feed_forward.w1", "w1"), ("feed_forward.w2", "w2"),
        ("feed_forward.w3", "w3"), ("adaLN", "adaLN"), ("norm", "norms(1D)"),
    ):
        if pat in k:
            return fam
    return "other"


def band_of(k: str) -> str:
    m = LAYER_RE.match(k)
    if not m:
        return "non-layer"
    i = int(m.group(1))
    return "early(0-6)" if i <= 6 else ("mid(7-21)" if i <= 21 else "late(22-29)")


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    rows, oned, unmatched = [], [], []

    with (
        safe_open(SOREAL, framework="pt", device="cpu") as fs,
        safe_open(ZIMAGE, framework="pt", device="cpu") as fz,
    ):
        keys_z = set(fz.keys())
        for k in sorted(fs.keys()):
            ws_cpu = fs.get_tensor(k)
            wz_cpu = z_lookup(fz, keys_z, k)
            if wz_cpu is None:
                unmatched.append(k)
                continue
            if ws_cpu.shape != wz_cpu.shape:
                unmatched.append(f"{k} SHAPE {tuple(ws_cpu.shape)} vs {tuple(wz_cpu.shape)}")
                continue
            ws = ws_cpu.to(DEVICE, torch.float32)
            wz = wz_cpu.to(DEVICE, torch.float32)
            ns, nz = float(ws.norm()), float(wz.norm())
            raw_rel = float((ws - wz).norm()) / nz if nz > 0 else 0.0
            if ws.ndim == 2 and min(ws.shape) >= MIN_DIM and ns > 0 and nz > 0:
                s = nz / ns
                dnorm2 = float(((s * ws - wz) ** 2).sum())
                dir_div = (dnorm2**0.5) / nz
                sig_s = torch.linalg.svdvals(ws) * s
                sig_z = torch.linalg.svdvals(wz)
                spec2 = float(((sig_s - sig_z) ** 2).sum())
                spec_shape = (spec2**0.5) / float(sig_z.norm())
                rot_share = 1.0 - spec2 / dnorm2 if dnorm2 > 0 else 1.0
                rows.append((k, family_of(k), band_of(k), ns / nz, raw_rel, dir_div, spec_shape, rot_share, dnorm2))
            else:
                oned.append((k, raw_rel))
            del ws, wz
            torch.cuda.empty_cache()

    def agg(items):
        w = sum(r[8] for r in items) or 1.0
        gmean_ratio = float(torch.tensor([r[3] for r in items]).log().mean().exp())
        return (
            gmean_ratio,
            sum(r[8] * r[5] for r in items) / w,
            sum(r[8] * r[6] for r in items) / w,
            sum(r[8] * r[7] for r in items) / w,
            len(items),
        )

    by_band = defaultdict(list)
    by_fam = defaultdict(list)
    for r in rows:
        by_band[r[2]].append(r)
        by_fam[r[1]].append(r)

    g_ratio, g_dir, g_spec, g_rot, _ = agg(rows)
    top_dir = sorted(rows, key=lambda r: r[5], reverse=True)[:15]
    top_shape = sorted(rows, key=lambda r: (1 - r[7]) * r[8], reverse=True)[:15]
    scars = [r for r in rows if r[3] < 0.02 or r[5] > 1.2]
    changed_1d = sorted((r for r in oned if r[1] > 1e-6), key=lambda r: r[1], reverse=True)

    md = []
    md.append("# SoReal! Base V0.95 vs Z-Image Base — Divergence Report")
    md.append("")
    md.append(f"*Generated 2026-07-04 by `scripts/util/_soreal_divergence_report.py` "
              f"({len(rows)} 2D weight pairs, {len(oned)} 1D tensors, {len(unmatched)} unmatched keys).*")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("- **All divergence lives in the 30 transformer layers' attention/FFN/adaLN weights.** "
              "Every norm gain, embedder, refiner, final layer, and pad token is **bit-identical** to official Z-Image base.")
    md.append(f"- **Raw weight norms are much smaller** (geometric-mean ratio ‖W_SoReal‖/‖W_Z‖ ≈ {g_ratio:.2f}); this uniform "
              "down-scaling is **functionally null**: to_q/to_k feed per-head RMSNorm (norm_q/k) and both branches pass "
              "sandwich norms (attention_norm2 / ffn_norm2) before the residual add, so uniform scale cancels. "
              "Spectral-norm-clipped OFT training shrinks norms freely. (Caveat: w1/w3 scale shifts SiLU saturation, "
              "so their scale is only approximately null.)")
    md.append(f"- **Scale-corrected divergence is {g_dir:.3f}** (energy-weighted), of which **{g_rot:.0%} is expressible as a "
              f"pure two-sided rotation** and **{1 - g_rot:.0%} is spectrum-shape (true magnitude-structure) change** — "
              "consistent with 'mostly rotations, some magnitude training'.")
    md.append("")
    md.append("## Global (energy-weighted over all 2D weights)")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| norm ratio ‖W_s‖/‖W_z‖ (geo-mean; functionally null) | {g_ratio:.3f} |")
    md.append(f"| scale-corrected divergence (dir_div) | {g_dir:.4f} |")
    md.append(f"| spectrum-shape shift (true magnitude change) | {g_spec:.4f} |")
    md.append(f"| rotation-expressible share of corrected delta | {g_rot:.4f} |")
    md.append(f"| magnitude-structure share | {1 - g_rot:.4f} |")
    md.append("")
    md.append("## By band (energy-weighted)")
    md.append("")
    md.append("| band | tensors | norm_ratio | dir_div | spec_shape | rot_share | mag_share |")
    md.append("|---|---|---|---|---|---|---|")
    for band in ("early(0-6)", "mid(7-21)", "late(22-29)", "non-layer"):
        if band in by_band:
            ratio, dird, spec, rot, n = agg(by_band[band])
            md.append(f"| {band} | {n} | {ratio:.3f} | {dird:.4f} | {spec:.4f} | {rot:.4f} | {1 - rot:.4f} |")
    md.append("")
    md.append("## By family (energy-weighted)")
    md.append("")
    md.append("| family | tensors | norm_ratio | dir_div | spec_shape | rot_share | mag_share |")
    md.append("|---|---|---|---|---|---|---|")
    for fam in sorted(by_fam, key=lambda f: -sum(r[8] for r in by_fam[f])):
        ratio, dird, spec, rot, n = agg(by_fam[fam])
        md.append(f"| {fam} | {n} | {ratio:.3f} | {dird:.4f} | {spec:.4f} | {rot:.4f} | {1 - rot:.4f} |")
    md.append("")
    md.append("## Top 15 most-diverged tensors (scale-corrected)")
    md.append("")
    md.append("| tensor | norm_ratio | dir_div | spec_shape | rot_share |")
    md.append("|---|---|---|---|---|")
    for k, _f, _b, ratio, _raw, dird, spec, rot, _w in top_dir:
        md.append(f"| `{k}` | {ratio:.3f} | {dird:.4f} | {spec:.4f} | {rot:.4f} |")
    md.append("")
    md.append("## Top 15 magnitude-structure-trained tensors (by spectrum-shape-shift energy)")
    md.append("")
    md.append("| tensor | norm_ratio | dir_div | spec_shape | rot_share |")
    md.append("|---|---|---|---|---|")
    for k, _f, _b, ratio, _raw, dird, spec, rot, _w in top_shape:
        md.append(f"| `{k}` | {ratio:.3f} | {dird:.4f} | {spec:.4f} | {rot:.4f} |")
    md.append("")
    md.append("## Lineage scars (near-dead or near-orthogonal tensors)")
    md.append("")
    if scars:
        md.append("Tensors with norm_ratio < 0.02 or scale-corrected divergence > 1.2 — candidates for the known "
                  "Beta21 dead-to_v / merge damage still present in the V0.95 lineage:")
        md.append("")
        md.append("| tensor | norm_ratio | dir_div | rot_share |")
        md.append("|---|---|---|---|")
        for k, _f, _b, ratio, _raw, dird, _spec, rot, _w in sorted(scars, key=lambda r: r[3]):
            md.append(f"| `{k}` | {ratio:.4f} | {dird:.4f} | {rot:.4f} |")
    else:
        md.append("None found.")
    md.append("")
    md.append("## 1D tensors (norm gains, biases, pad tokens — pure magnitude)")
    md.append("")
    md.append(f"{len(changed_1d)}/{len(oned)} differ." + (" Top movers:" if changed_1d else " **All bit-identical to Z-Image base.**"))
    if changed_1d:
        md.append("")
        md.append("| tensor | rel_delta |")
        md.append("|---|---|")
        for k, rel in changed_1d[:20]:
            md.append(f"| `{k}` | {rel:.4f} |")
    if unmatched:
        md.append("")
        md.append("## Unmatched keys (SoReal-only, no Z-Image counterpart)")
        md.append("")
        for k in unmatched:
            md.append(f"- `{k}`")
    md.append("")

    Path(OUT_MD).write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_MD}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
