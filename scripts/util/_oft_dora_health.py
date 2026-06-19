"""Per-row DoRA health for an OFT/DoRA-OFT adapter (handles both formats).

NEW relative format (dora_multiplier): the applied per-row scale IS the
multiplier (init 1.0). Reports its distribution + a healthy/check verdict --
a good run keeps it gentle (~[0.1, 3], centered near 1.0), not 10s/100s.

OLD absolute format (dora_scale + initial_norm): applied scale =
dora_scale / max(initial_norm, 1e-8). Surfaces exploding/dead rows and whether
dead rows coincide with tiny initial_norm (base-poisoning) vs healthy
initial_norm (training instability). Buckets by layer family.

Usage: python _oft_dora_health.py <adapter.safetensors> [<adapter2> ...]
"""

import sys

import torch

from safetensors import safe_open


def family(key: str) -> str:
    k = key.lower()
    for tok in (
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "add_q",
        "add_k",
        "add_v",
        "ff_context",
        "feed_forward",
        "mlp",
        "ff.",
        "proj_mlp",
        "proj_out",
        "proj_in",
        "norm",
        "attn",
    ):
        if tok in k:
            return tok
    return "other"


def analyze_multiplier(f, keys, mult_keys):
    """NEW relative format: applied per-row scale == dora_multiplier (init 1.0)."""
    all_s = []
    fam_hot = {}
    for mk in mult_keys:
        s = f.get_tensor(mk).float().flatten()
        all_s.append(s)
        hot = int((s.abs() > 2).sum())
        if hot:
            fam = family(mk)
            fam_hot[fam] = fam_hot.get(fam, 0) + hot
    scale = torch.cat(all_s)
    total = scale.numel()

    # whole-file NaN/Inf + rotation magnitude (oft_R.weight std) scan
    nan = inf = 0
    oft_std = []
    for k in keys:
        t = f.get_tensor(k).float()
        nan += int(torch.isnan(t).sum())
        inf += int(torch.isinf(t).sum())
        if k.endswith("oft_R.weight"):
            oft_std.append(float(t.std()))

    print(f"  format: dora_multiplier (relative, init 1.0)  layers={len(mult_keys)}  rows={total}")
    print(
        f"  multiplier: min={float(scale.min()):.4g} max={float(scale.max()):.4g} "
        f"mean={float(scale.mean()):.4g} median={float(scale.median()):.4g} std={float(scale.std()):.4g}"
    )
    for thr in (2, 3, 5, 10, 50):
        c = int((scale.abs() > thr).sum())
        if c:
            print(f"    |mult|>{thr}: {c} rows ({100 * c / total:.3f}%)")
    dead = int((scale.abs() < 1e-6).sum())
    neg = int((scale < 0).sum())
    print(f"  near-zero (<1e-6): {dead}   negative: {neg}")
    if oft_std:
        print(
            f"  oft_R weight std (mean/min/max over {len(oft_std)} layers): "
            f"{sum(oft_std) / len(oft_std):.5g} / {min(oft_std):.5g} / {max(oft_std):.5g}"
        )
    print(f"  NaN: {nan}   Inf: {inf}")
    if fam_hot:
        print("  |mult|>2 rows by layer family:", dict(sorted(fam_hot.items(), key=lambda x: -x[1])))
    if float(scale.abs().max()) < 5.0 and dead == 0 and nan == 0 and inf == 0:
        print("  VERDICT: HEALTHY -- gentle multiplier near 1.0, no runaway / dead / NaN")
    else:
        print("  VERDICT: CHECK -- multiplier out of band, or dead rows / NaN present")


def analyze(path: str):
    print(f"\n################ {path}")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        mult_keys = sorted(k for k in keys if k.endswith("dora_multiplier"))
        if mult_keys:
            analyze_multiplier(f, keys, mult_keys)
            return

        dora_keys = sorted(k for k in keys if k.endswith("dora_scale"))

        all_scale = []
        all_dora = []
        all_norm = []
        fam_dead = {}
        fam_explode = {}
        for dk in dora_keys:
            base = dk[: -len("dora_scale")]
            nk = base + "initial_norm"
            d = f.get_tensor(dk).float().flatten()
            n = f.get_tensor(nk).float().flatten() if nk in keys else torch.ones_like(d)
            scale = d / n.clamp(min=1e-8)
            all_scale.append(scale)
            all_dora.append(d)
            all_norm.append(n)
            fam = family(dk)
            dead = d.abs() < 1e-6
            explode = scale.abs() > 10
            if int(dead.sum()):
                fam_dead[fam] = fam_dead.get(fam, 0) + int(dead.sum())
            if int(explode.sum()):
                fam_explode[fam] = fam_explode.get(fam, 0) + int(explode.sum())

        scale = torch.cat(all_scale)
        dora = torch.cat(all_dora)
        norm = torch.cat(all_norm)
        total = scale.numel()

        dead_mask = dora.abs() < 1e-6
        n_dead = int(dead_mask.sum())
        print(f"  rows total={total}")
        print(
            f"  applied scale = dora/clamp(norm): "
            f"min={float(scale.min()):.4g} max={float(scale.max()):.4g} "
            f"absmax={float(scale.abs().max()):.4g} median={float(scale.median()):.4g}"
        )
        for thr in (5, 10, 50, 100, 1000):
            c = int((scale.abs() > thr).sum())
            if c:
                print(f"    |scale|>{thr}: {c} rows ({100 * c / total:.3f}%)")
        print(f"  dead dora rows (|dora|<1e-6): {n_dead} ({100 * n_dead / total:.3f}%)")
        if n_dead:
            dn = norm[dead_mask]
            print(
                f"    their initial_norm: min={float(dn.min()):.4g} "
                f"median={float(dn.median()):.4g} max={float(dn.max()):.4g}"
            )
            # how many of the dead rows had a TINY base norm (base-poisoning signature)
            tiny = int((dn < 1e-3).sum())
            print(
                f"    of dead rows, {tiny}/{n_dead} have initial_norm<1e-3 "
                f"({100 * tiny / n_dead:.1f}%)  -> high% = base-poisoning; low% = instability"
            )
        print(
            f"  global initial_norm: min={float(norm.min()):.4g} "
            f"<1e-3: {int((norm < 1e-3).sum())} rows, <1e-4: {int((norm < 1e-4).sum())} rows"
        )
        if fam_dead:
            print("  dead rows by layer family:", dict(sorted(fam_dead.items(), key=lambda x: -x[1])))
        if fam_explode:
            print("  |scale|>10 rows by layer family:", dict(sorted(fam_explode.items(), key=lambda x: -x[1])))
        # WHY do the exploding rows explode: tiny denominator (base) or big numerator (training)?
        exp_mask = scale.abs() > 10
        if int(exp_mask.sum()):
            en, ed = norm[exp_mask], dora[exp_mask].abs()
            print(f"  EXPLODING (|scale|>10) rows diagnosis: n={int(exp_mask.sum())}")
            print(
                f"    initial_norm (denominator): min={float(en.min()):.4g} "
                f"median={float(en.median()):.4g} max={float(en.max()):.4g}  "
                f"<1e-3: {int((en < 1e-3).sum())}  <1e-2: {int((en < 1e-2).sum())}"
            )
            print(
                f"    |dora_scale| (numerator):   min={float(ed.min()):.4g} "
                f"median={float(ed.median()):.4g} max={float(ed.max()):.4g}"
            )
            print("    -> small denominator => base/merge; large numerator => training instability")


def main():
    for p in sys.argv[1:]:
        analyze(p)


if __name__ == "__main__":
    main()
