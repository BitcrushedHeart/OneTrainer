"""CPU toy: does a sigma-gated perceptual term on x0_hat beat plain v-MSE?

Rectified flow in pixel space, so the gate / weight mechanics are exactly the
trainer's (``x_t = (1 - s) x0 + s eps``, ``v = eps - x0``,
``x0_hat = x_t - s v_hat``), minus the VAE.

Data (32x32 grayscale, seeded): each image is a *structure* term that is
predictable from context plus *texture* terms whose phase is not:

    S      = normalized sum of 3 Gaussian blobs (random centre, width, amplitude)
    stripe = cos(2 pi f (x cos th + y sin th) + phi), f in [0.25, 0.40] cyc/px,
             random orientation th and phase phi
    grain  = high-passed white noise
    x      = 0.7 (2 S - 1) + S (0.35 stripe + 0.15 grain), clamped to [-1, 1]

The texture rides on the structure (its *presence* is predictable) but its
phase is a fresh draw per image, so the posterior mean E[x0 | x_t] at low s
averages the stripes away -- exactly the MSE blur the perceptual term targets.

Perceptual space: a tiny conv autoencoder trained on the train split; the loss
is the mean squared distance between the AE *decoder's* intermediate block
features of x0_hat and x0, each block cross-normalized with the prediction's
per-sample, per-channel spatial statistics (LPL style, arXiv 2411.04873;
outlier masking omitted). ``--features lpips`` swaps in VGG-LPIPS on 3-channel,
64 px upsampled images.

Weight: ``w(s) = lam * min(((1 - s) / s)^2, gamma) * g(s)``, g = 1 below
sigma_full, cosine taper to 0 at sigma_max. ``lam`` is calibrated once (after a
short MSE-only warmup) so the perceptual gradient on v_hat over gated samples
is ~0.15x the MSE gradient, then shared by every perceptual arm.

Arms (identical init, batches, noise and sigmas):
    A         MSE only
    B         MSE + perceptual, gamma 5, gate 0.5 -> 0.75
    B-narrow  as B, gate 0.25 -> 0.4
    B-flat    as B without the min-SNR factor (w = lam g)
    C         as B without the gate (g = 1)
    D         perceptual replaces MSE inside the gate (MSE weight 1 - g)

Metrics on ODE samples (Euler on the shift-warped grid, shared noise) at
8 / 32 / 128 steps: high-frequency energy ratio vs held-out, Frechet distance
in the AE bottleneck, Laplacian-variance ratio, pairwise-L2 diversity ratio,
held-out v-MSE per sigma band on a fixed grid, and the 8-vs-128-step
divergence.

Run: venv/Scripts/python.exe scripts/util/_perceptual_toy.py
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SCRATCH = Path(
    r"C:\Users\calla\AppData\Local\Temp\claude\E--AI-Data-Packages-OneTrainer"
    r"\1a2105e3-4ef9-4db8-8e32-0a0f4ac8b92f\scratchpad"
)
SIZE = 32

ARMS = {
    "A": {"perc": False},
    "B": {"perc": True, "gamma": 5.0, "full": 0.5, "mx": 0.75, "replace": False},
    "B-narrow": {"perc": True, "gamma": 5.0, "full": 0.25, "mx": 0.4, "replace": False},
    "B-flat": {"perc": True, "gamma": None, "full": 0.5, "mx": 0.75, "replace": False},
    "C": {"perc": True, "gamma": 5.0, "full": None, "mx": None, "replace": False},
    "D": {"perc": True, "gamma": 5.0, "full": 0.5, "mx": 0.75, "replace": True},
}
BANDS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


# -- data ---------------------------------------------------------------------


def make_dataset(n: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    xx = xx.float()[None, None]
    yy = yy.float()[None, None]

    k = 3
    cx = torch.rand(n, k, 1, 1, generator=g) * SIZE
    cy = torch.rand(n, k, 1, 1, generator=g) * SIZE
    width = 3.0 + 5.0 * torch.rand(n, k, 1, 1, generator=g)
    amp = 0.5 + torch.rand(n, k, 1, 1, generator=g)
    blobs = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width**2))
    structure = blobs.sum(1, keepdim=True)
    structure = structure / structure.amax(dim=(2, 3), keepdim=True)

    theta = torch.rand(n, 1, 1, 1, generator=g) * math.pi
    freq = 0.25 + 0.15 * torch.rand(n, 1, 1, 1, generator=g)
    phase = torch.rand(n, 1, 1, 1, generator=g) * 2 * math.pi
    stripes = torch.cos(2 * math.pi * freq * (xx * torch.cos(theta) + yy * torch.sin(theta)) + phase)

    white = torch.randn(n, 1, SIZE, SIZE, generator=g)
    grain = white - F.avg_pool2d(white, 3, stride=1, padding=1)
    grain = grain / grain.std(dim=(2, 3), keepdim=True)

    x = 0.7 * (2 * structure - 1) + structure * (0.35 * stripes + 0.15 * grain)
    return x.clamp(-1, 1)


# -- models -------------------------------------------------------------------


def sigma_embedding(sigma: Tensor, dim: int = 32) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = sigma.reshape(-1, 1) * 1000.0 * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.film = nn.Linear(emb, 2 * cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.norm1(self.conv1(x))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(h * (1 + scale) + shift)
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class TinyUNet(nn.Module):
    def __init__(self, ch=(16, 32, 64), emb: int = 64):
        super().__init__()
        c0, c1, c2 = ch
        self.emb = nn.Sequential(nn.Linear(32, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(1, c0, 3, padding=1)
        self.d0 = ResBlock(c0, c0, emb)
        self.down0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.d1 = ResBlock(c1, c1, emb)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.mid = ResBlock(c2, c2, emb)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.u1 = ResBlock(2 * c1, c1, emb)
        self.up0 = nn.Conv2d(c1, c0, 3, padding=1)
        self.u0 = ResBlock(2 * c0, c0, emb)
        self.out = nn.Conv2d(c0, 1, 3, padding=1)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        e = self.emb(sigma_embedding(sigma))
        h0 = self.d0(self.inp(x), e)
        h1 = self.d1(self.down0(h0), e)
        h2 = self.mid(self.down1(h1), e)
        u = self.up1(F.interpolate(h2, scale_factor=2, mode="nearest"))
        u = self.u1(torch.cat([u, h1], 1), e)
        u = self.up0(F.interpolate(u, scale_factor=2, mode="nearest"))
        u = self.u0(torch.cat([u, h0], 1), e)
        return self.out(u)


class TinyAE(nn.Module):
    """Encoder to a 16x8x8 bottleneck, decoder with three feature blocks."""

    def __init__(self):
        super().__init__()
        self.e0 = nn.Conv2d(1, 16, 3, padding=1)
        self.e1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.e2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.e3 = nn.Conv2d(64, 16, 1)
        self.d0 = nn.Conv2d(16, 64, 3, padding=1)
        self.d1 = nn.Conv2d(64, 32, 3, padding=1)
        self.d2 = nn.Conv2d(32, 16, 3, padding=1)
        self.d3 = nn.Conv2d(16, 1, 3, padding=1)

    def encode(self, x: Tensor) -> Tensor:
        h = F.silu(self.e0(x))
        h = F.silu(self.e1(h))
        h = F.silu(self.e2(h))
        return self.e3(h)

    def decoder_features(self, z: Tensor) -> list[Tensor]:
        f0 = F.silu(self.d0(z))
        f1 = F.silu(self.d1(F.interpolate(f0, scale_factor=2, mode="nearest")))
        f2 = F.silu(self.d2(F.interpolate(f1, scale_factor=2, mode="nearest")))
        return [f0, f1, f2]

    def decode(self, z: Tensor) -> Tensor:
        return torch.tanh(self.d3(self.decoder_features(z)[-1]))

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def train_ae(train: Tensor, seed: int, steps: int = 2000) -> TinyAE:
    # 2000 steps at 3e-3 keeps ~0.8 of the held-out high-frequency energy;
    # 800 at 2e-3 kept 0.25, i.e. the feature space would not see the stripes.
    torch.manual_seed(seed)
    ae = TinyAE()
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, train.shape[0], (128,), generator=g)
        x = train[idx]
        loss = F.mse_loss(ae(x), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# -- perceptual term ------------------------------------------------------------


class Perceptual:
    def __init__(self, kind: str, ae: TinyAE):
        self.kind = kind
        self.ae = ae
        self.lpips = None
        if kind == "lpips":
            import lpips

            self.lpips = lpips.LPIPS(net="vgg", verbose=False).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

    def __call__(self, x0_hat: Tensor, x0: Tensor) -> Tensor:
        """Per-sample perceptual distance, shape (B,)."""
        if self.kind == "lpips":
            a = F.interpolate(x0_hat, size=64, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b = F.interpolate(x0, size=64, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            return self.lpips(a, b).reshape(-1)
        pred_feats = self.ae.decoder_features(self.ae.encode(x0_hat))
        with torch.no_grad():
            tgt_feats = self.ae.decoder_features(self.ae.encode(x0))
        total = 0.0
        for a, b in zip(pred_feats, tgt_feats, strict=True):
            # cross-normalization: both sides with the prediction's statistics
            mu = a.mean(dim=(2, 3), keepdim=True).detach()
            sd = a.std(dim=(2, 3), keepdim=True).detach() + 1e-4
            total = total + (((a - mu) / sd - (b - mu) / sd) ** 2).mean(dim=(1, 2, 3))
        return total / len(pred_feats)


def gate(sigma: Tensor, full: float | None, mx: float | None) -> Tensor:
    if full is None:
        return torch.ones_like(sigma)
    t = ((sigma - full) / (mx - full)).clamp(0, 1)
    return 0.5 * (1 + torch.cos(math.pi * t))


def snr_weight(sigma: Tensor, gamma: float | None) -> Tensor:
    if gamma is None:
        return torch.ones_like(sigma)
    return (((1 - sigma) / sigma) ** 2).clamp(max=gamma)


# -- training -------------------------------------------------------------------


def shift_sigma(u: Tensor, shift: float) -> Tensor:
    return shift * u / (1 + (shift - 1) * u)


def sample_sigma(n: int, shift: float, g: torch.Generator) -> Tensor:
    return shift_sigma(torch.randn(n, generator=g).sigmoid(), shift)


def make_batch(train: Tensor, batch: int, shift: float, g: torch.Generator):
    idx = torch.randint(0, train.shape[0], (batch,), generator=g)
    x0 = train[idx]
    eps = torch.randn(x0.shape, generator=g)
    sigma = sample_sigma(batch, shift, g)
    s = sigma[:, None, None, None]
    return x0, eps, sigma, (1 - s) * x0 + s * eps, eps - x0


def arm_losses(arm: dict, lam: float, perc: Perceptual, x0, sigma, xt, v, v_hat):
    """Returns (loss, mse_per_sample, perceptual_per_sample or None)."""
    mse = ((v_hat - v) ** 2).mean(dim=(1, 2, 3))
    if not arm["perc"]:
        return mse.mean(), mse, None
    g = gate(sigma, arm["full"], arm["mx"])
    w = lam * snr_weight(sigma, arm["gamma"]) * g
    x0_hat = xt - sigma[:, None, None, None] * v_hat
    p = perc(x0_hat, x0)
    mse_w = (1 - g) if arm["replace"] else torch.ones_like(g)
    return (mse_w * mse + w * p).mean(), mse, p


def train_arm(
    name: str,
    arm: dict,
    train: Tensor,
    steps: int,
    batch: int,
    shift: float,
    seed: int,
    lam: float,
    perc: Perceptual,
) -> TinyUNet:
    torch.manual_seed(seed)
    model = TinyUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for step in range(steps):
        x0, _eps, sigma, xt, v = make_batch(train, batch, shift, g)
        v_hat = model(xt, sigma)
        loss, mse, p = arm_losses(arm, lam, perc, x0, sigma, xt, v, v_hat)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0 or step == 0:
            extra = f" perc {p.mean().item():.4f}" if p is not None else ""
            print(f"  [{name}] step {step + 1}/{steps} mse {mse.mean().item():.4f}{extra} ({time.time() - t0:.0f}s)")
    model.eval()
    return model


def calibrate_lambda(
    train: Tensor, steps: int, batch: int, shift: float, seed: int, perc: Perceptual, target_ratio: float
) -> tuple[float, dict]:
    """MSE-only warmup, then lam so ||d perc / d v_hat|| ~= target_ratio * ||d mse / d v_hat||
    over gated samples, with arm B's unscaled weight (lam = 1)."""
    model = train_arm("calib", ARMS["A"], train, steps, batch, shift, seed, 0.0, perc)
    arm = ARMS["B"]
    g = torch.Generator().manual_seed(seed + 7)
    x0s, sigmas, xts, vs = [], [], [], []
    while sum(x.shape[0] for x in x0s) < 256:
        x0, _eps, sigma, xt, v = make_batch(train, 512, shift, g)
        keep = sigma < arm["mx"]
        x0s.append(x0[keep])
        sigmas.append(sigma[keep])
        xts.append(xt[keep])
        vs.append(v[keep])
    x0 = torch.cat(x0s)[:256]
    sigma = torch.cat(sigmas)[:256]
    xt = torch.cat(xts)[:256]
    v = torch.cat(vs)[:256]
    with torch.no_grad():
        v_hat = model(xt, sigma)
    v_hat = v_hat.detach().requires_grad_(True)
    mse = ((v_hat - v) ** 2).mean(dim=(1, 2, 3)).mean()
    (g_mse,) = torch.autograd.grad(mse, v_hat)
    w = snr_weight(sigma, arm["gamma"]) * gate(sigma, arm["full"], arm["mx"])
    p = (w * perc(xt - sigma[:, None, None, None] * v_hat, x0)).mean()
    (g_perc,) = torch.autograd.grad(p, v_hat)
    n_mse = g_mse.norm().item()
    n_perc = g_perc.norm().item()
    lam = target_ratio * n_mse / n_perc
    info = {"grad_norm_mse": n_mse, "grad_norm_perc_unit": n_perc, "lam": lam, "n_gated": int(x0.shape[0])}
    return lam, info


# -- evaluation -----------------------------------------------------------------


@torch.no_grad()
def ode_sample(model: TinyUNet, noise: Tensor, steps: int, shift: float, chunk: int = 256) -> Tensor:
    grid = shift_sigma(torch.linspace(1.0, 0.0, steps + 1), shift)
    outs = []
    for start in range(0, noise.shape[0], chunk):
        x = noise[start : start + chunk].clone()
        for i in range(steps):
            s = grid[i].expand(x.shape[0])
            x = x + (grid[i + 1] - grid[i]) * model(x, s)
        outs.append(x)
    return torch.cat(outs)


def high_freq_energy(x: Tensor, cutoff: float = 0.2) -> Tensor:
    spec = torch.fft.fft2(x.float()).abs() ** 2
    fy = torch.fft.fftfreq(SIZE)[:, None]
    fx = torch.fft.fftfreq(SIZE)[None, :]
    mask = (fy**2 + fx**2).sqrt() > cutoff
    return spec[..., mask].sum(dim=-1).mean(dim=1)


def laplacian_var(x: Tensor) -> Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
    return F.conv2d(x, k).var(dim=(1, 2, 3))


def pairwise_diversity(x: Tensor, n: int = 256) -> float:
    flat = x[:n].flatten(1)
    return torch.cdist(flat, flat).sum().item() / (n * (n - 1))


def frechet(a: Tensor, b: Tensor) -> float:
    a = a.double()
    b = b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = torch.cov(a.T)
    cb = torch.cov(b.T)
    ev, evec = torch.linalg.eigh(ca)
    sqrt_ca = (evec * ev.clamp(min=0).sqrt()) @ evec.T
    m = sqrt_ca @ cb @ sqrt_ca
    tr_sqrt = torch.linalg.eigvalsh((m + m.T) / 2).clamp(min=0).sqrt().sum()
    return ((mu_a - mu_b).pow(2).sum() + ca.trace() + cb.trace() - 2 * tr_sqrt).item()


@torch.no_grad()
def v_mse_bands(model: TinyUNet, holdout: Tensor, seed: int) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed + 99)
    eps = torch.randn(holdout.shape, generator=g)
    grid = torch.arange(0.025, 1.0, 0.05)
    per_sigma = []
    for s in grid:
        sigma = s.expand(holdout.shape[0])
        xt = (1 - s) * holdout + s * eps
        per_sigma.append(((model(xt, sigma) - (eps - holdout)) ** 2).mean().item())
    per_sigma = torch.tensor(per_sigma)
    out = {}
    for lo, hi in BANDS:
        sel = (grid >= lo) & (grid < hi)
        out[f"vmse[{lo:.2f},{hi:.2f})"] = per_sigma[sel].mean().item()
    return out


@torch.no_grad()
def evaluate(model: TinyUNet, holdout: Tensor, ae: TinyAE, noise: Tensor, shift: float, seed: int) -> dict:
    ref_hf = high_freq_energy(holdout).mean().item()
    ref_lap = laplacian_var(holdout).mean().item()
    ref_div = pairwise_diversity(holdout)
    ref_feat = ae.encode(holdout).flatten(1)
    out = {}
    samples = {}
    for steps in (8, 32, 128):
        x = ode_sample(model, noise, steps, shift).clamp(-1, 1)
        samples[steps] = x
        out[f"hf{steps}"] = high_freq_energy(x).mean().item() / ref_hf
        out[f"fd{steps}"] = frechet(ae.encode(x).flatten(1), ref_feat)
        out[f"lap{steps}"] = laplacian_var(x).mean().item() / ref_lap
        out[f"div{steps}"] = pairwise_diversity(x) / ref_div
    out["div8v128"] = (samples[8] - samples[128]).flatten(1).norm(dim=1).mean().item()
    out.update(v_mse_bands(model, holdout, seed))
    return out


# -- main -----------------------------------------------------------------------


def print_table(results: dict[str, dict]) -> None:
    cols = [
        "hf8",
        "hf32",
        "hf128",
        "fd8",
        "fd32",
        "fd128",
        "lap32",
        "div8",
        "div32",
        "div8v128",
        *[f"vmse[{lo:.2f},{hi:.2f})" for lo, hi in BANDS],
        "time_s",
    ]
    short = {c: c.replace("vmse[", "vm[") for c in cols}
    print()
    print(f"{'arm':10s}" + "".join(f"{short[c]:>12s}" for c in cols))
    for name, r in results.items():
        print(f"{name:10s}" + "".join(f"{r[c]:12.4f}" for c in cols))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--shift", type=float, default=3.16)
    ap.add_argument("--features", choices=("ae", "lpips"), default="ae")
    ap.add_argument("--target-ratio", type=float, default=0.15)
    ap.add_argument("--calib-steps", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=SCRATCH / "perceptual_toy.json")
    args = ap.parse_args()

    train = make_dataset(8000, args.seed)
    holdout = make_dataset(1000, args.seed + 1)
    print(f"data: train {tuple(train.shape)} holdout {tuple(holdout.shape)}")

    t0 = time.time()
    ae = train_ae(train, args.seed)
    with torch.no_grad():
        recon = ae(holdout)
        ae_mse = F.mse_loss(recon, holdout).item()
        ae_hf = (high_freq_energy(recon).mean() / high_freq_energy(holdout).mean()).item()
    print(f"ae: recon mse {ae_mse:.4f}, high-freq energy kept {ae_hf:.3f} ({time.time() - t0:.0f}s)")
    perc = Perceptual(args.features, ae)
    print(f"unet params: {sum(p.numel() for p in TinyUNet().parameters()):,}")

    lam, calib = calibrate_lambda(train, args.calib_steps, args.batch, args.shift, args.seed, perc, args.target_ratio)
    print(f"lambda calibration: {calib}")

    noise = torch.randn(args.n_samples, 1, SIZE, SIZE, generator=torch.Generator().manual_seed(args.seed + 5))
    results = {}
    for name in args.arms.split(","):
        arm = ARMS[name]
        t_arm = time.time()
        model = train_arm(name, arm, train, args.steps, args.batch, args.shift, args.seed, lam, perc)
        t_train = time.time() - t_arm
        results[name] = evaluate(model, holdout, ae, noise, args.shift, args.seed)
        results[name]["time_s"] = t_train
        results[name]["eval_s"] = time.time() - t_arm - t_train
        print(f"[{name}] train {t_train:.0f}s eval {results[name]['eval_s']:.0f}s")

    print_table(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "lam": lam, "calib": calib, "results": results},
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
