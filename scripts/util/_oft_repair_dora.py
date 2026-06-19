"""Repair the DoRA magnitude of an OFT/DoRA-OFT adapter, nine ways, in one pass.

Edits ONLY dora_scale (the per-output-row magnitude). oft_R is per-INPUT-block
and shared across output rows, so it cannot be edited per pathological row and is
left untouched. All other tensors/metadata are preserved verbatim.

Applied per-row magnitude:  scale[o] = dora_scale[o] / clamp(initial_norm[o], 1e-8).
Because the OFT rotation R is orthogonal (norm-preserving), the merged row norm
equals dora_scale[o]; so scale==1  <=>  the row keeps its base magnitude (pure
rotation).  Setting dora_scale := initial_norm forces scale==1.

Outputs (in <out_dir>, originals untouched):
  _fix1 (pureOFT)   : dora_scale := initial_norm for ALL rows (scale==1). Keeps the
                      DoRA keys. Rotation-only; discards ALL learned magnitude.
  _fix2 (surgical)  : reset dora_scale := initial_norm ONLY for pathological rows
                      ( |dora|<dead_thr  OR  |scale|>extreme ). ~99.9% of healthy
                      magnitudes preserved.
  _fix3 (clamp)     : clamp |scale| into [lo,hi] (sign preserved) for ALL rows, then
                      dora_scale := scale*initial_norm. Smoother than fix2, but also
                      pulls healthy rows below lo up to lo (reported).
  _fix4 (no-DoRA)   : DELETE dora_scale + initial_norm keys entirely -> loader runs
                      its non-DoRA OFT path. Mathematically == fix1; exercises a
                      different loader branch (cross-check).
  _fix5 (surgical-2): like fix2 but FIXED threshold 2.0 -> rows with |scale|>2 reset
                      to scale=1 (kills the overcooked 2-5x band V21 never had).
  _fix6 (clamp-0-2) : FIXED clamp [0, 2], NO floor -> overcooked rows capped at 2x,
                      all learned magnitudes <=2x kept as-is. NOTE: leaves dead rows
                      dead (a zero scale cannot be lifted by a zero floor).
  _fix7 (revive-cap): revive dead rows to scale=1 (base magnitude) AND cap |scale| at
                      2x; every other learned magnitude (incl. small ones) untouched.
                      = fix6 but with the dead to_out/to_k channels restored.
  _fix8 (atten-only): FIXED clamp [0, 1], NO floor -> dead rows stay dead, learned
                      attenuation (scale<1) kept, and ALL amplification (scale>1)
                      capped at 1x. Matches V21's regime (max scale ~1.03, no amplify).
  _fix9 (atten+revive): like fix8 (cap amplification at 1x, keep attenuation) BUT revive
                      dead rows to a LOW scale (default 0.1, arg 7) instead of leaving
                      them at 0 -> weakly-alive channels rather than hundreds killed.

fix5 vs fix6 differ only in how they treat the overcooked (>2x) band:
fix5 resets those rows to 1 (base magnitude); fix6 caps them at 2.
fix7 = fix6 + reviving the dead channels (the cleanest magnitude-retaining salvage).

Usage:
  python _oft_repair_dora.py <in.safetensors> <out_dir>
        [dead_thr=1e-6] [extreme=5.0] [lo=0.5] [hi=2.0]
"""

import os
import sys

import torch

from safetensors import safe_open
from safetensors.torch import save_file


def load(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():  # noqa: SIM118
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def pairs(tensors):
    out = []
    for k in tensors:
        if k.endswith("dora_scale"):
            nk = k[: -len("dora_scale")] + "initial_norm"
            if nk in tensors:
                out.append((k, nk))
    return out


def summarize(tensors, label):
    pr = pairs(tensors)
    if not pr:
        print(f"  [{label:>14}] no DoRA keys present -> pure OFT rotation only")
        return
    mins, maxs, dead, extreme5, total = [], [], 0, 0, 0
    for dk, nk in pr:
        d = tensors[dk].float().flatten()
        n = tensors[nk].float().flatten()
        s = d / n.clamp(min=1e-8)
        mins.append(float(s.min()))
        maxs.append(float(s.max()))
        dead += int((d.abs() < 1e-6).sum())
        extreme5 += int((s.abs() > 5).sum())
        total += d.numel()
    print(
        f"  [{label:>14}] scale=[{min(mins):.4g}, {max(maxs):.4g}]  dead={dead}  |scale|>5={extreme5}  / {total} rows"
    )


def make_surgical(dead_thr, extreme_thr):
    """Reset dora_scale := initial_norm (scale=1) for dead OR |scale|>extreme_thr rows."""

    def mutate(d, n, c):
        s = d.float() / n.float().clamp(min=1e-8)
        mask = (d.float().abs() < dead_thr) | (s.abs() > extreme_thr)
        out = d.clone()
        out[mask] = n[mask]
        c["changed"] += int(mask.sum())
        return out

    return mutate


def make_clamp(dead_thr, report_extreme, lo_b, hi_b):
    """Clamp |scale| into [lo_b, hi_b] (sign preserved); dora_scale := scale*initial_norm."""

    def mutate(d, n, c):
        df, nf = d.float(), n.float()
        s = df / nf.clamp(min=1e-8)
        mag = s.abs()
        sgn = torch.sign(s)
        sgn[sgn == 0] = 1.0
        new_mag = mag.clamp(min=lo_b, max=hi_b)
        changed = new_mag != mag
        patho = (df.abs() < dead_thr) | (mag > report_extreme)
        c["changed"] += int(changed.sum())
        c["healthy"] += int((changed & ~patho).sum())
        c["floor"] += int((mag < lo_b).sum())
        c["ceil"] += int((mag > hi_b).sum())
        return sgn * new_mag * nf

    return mutate


def make_revive_cap(dead_thr, revive_value, hi_b):
    """Revive dead rows to scale=revive_value; cap |scale| at hi_b; keep other magnitudes."""

    def mutate(d, n, c):
        df, nf = d.float(), n.float()
        s = df / nf.clamp(min=1e-8)
        mag = s.abs()
        sgn = torch.sign(s)
        sgn[sgn == 0] = 1.0
        dead = df.abs() < dead_thr
        capped = sgn * mag.clamp(max=hi_b)
        revived = torch.full_like(capped, revive_value)
        new_scale = torch.where(dead, revived, capped)
        c["floor"] += int(dead.sum())  # 'floor' counter reused as revived-dead count
        c["ceil"] += int((mag > hi_b).sum())
        c["changed"] += int((dead | (mag > hi_b)).sum())
        return new_scale * nf

    return mutate


def write_safetensors(tensors, outp, meta):
    try:
        save_file(tensors, outp, metadata=meta)
        print(f"\nwrote {outp}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\nSKIPPED {outp}: {e}\n  (file likely open in SwarmUI/ComfyUI — unload it and re-run)")
        return False


def main():
    inp, outdir = sys.argv[1], sys.argv[2]
    dead_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    extreme = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    lo = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    hi = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    revive = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1

    os.makedirs(outdir, exist_ok=True)
    base, meta = load(inp)
    stem = os.path.splitext(os.path.basename(inp))[0]
    print(f"input: {inp}")
    summarize(base, "input")

    def emit(suffix, mutate, note=None):
        t = {k: v.clone() for k, v in base.items()}
        counters = {"changed": 0, "healthy": 0, "floor": 0, "ceil": 0}
        for dk, nk in pairs(t):
            t[dk] = mutate(t[dk], t[nk], counters).to(t[dk].dtype)
        outp = os.path.join(outdir, f"{stem}_{suffix}.safetensors")
        if write_safetensors(t, outp, meta):
            summarize(t, suffix)
            if note:
                print("   ", note(counters))

    # fix1) pure OFT: scale == 1 everywhere (keys kept)
    def m1(d, n, c):
        c["changed"] += d.numel()
        return n.clone()

    emit("fix1", m1)

    # fix2) surgical with CLI extreme threshold
    emit("fix2", make_surgical(dead_thr, extreme), note=lambda c: f"rows reset to scale=1: {c['changed']}")

    # fix3) clamp into CLI [lo, hi]
    emit(
        "fix3",
        make_clamp(dead_thr, extreme, lo, hi),
        note=lambda c: (
            f"band [{lo}, {hi}] rows changed: {c['changed']} "
            f"(healthy floored-up: {c['healthy']}; at floor: {c['floor']}, at ceil: {c['ceil']})"
        ),
    )

    # fix4) delete DoRA keys entirely -> loader's non-DoRA OFT path
    t4 = {k: v.clone() for k, v in base.items() if not k.endswith(("dora_scale", "initial_norm"))}
    removed = len(base) - len(t4)
    outp = os.path.join(outdir, f"{stem}_fix4.safetensors")
    if write_safetensors(t4, outp, meta):
        summarize(t4, "fix4")
        print(f"    removed {removed} DoRA keys (dora_scale + initial_norm); kept {len(t4)} keys")

    # fix5) surgical, FIXED extreme=2 -> overcooked >2x rows reset to scale=1
    emit("fix5", make_surgical(dead_thr, 2.0), note=lambda c: f"|scale|>2 reset to scale=1: {c['changed']}")

    # fix6) FIXED clamp [0, 2], no floor -> overcooked >2x rows capped at 2
    emit(
        "fix6",
        make_clamp(dead_thr, 2.0, 0.0, 2.0),
        note=lambda c: f"clamp[0,2] rows capped at 2: {c['ceil']} (total changed: {c['changed']})",
    )

    # fix7) revive dead rows to scale=1 AND cap |scale| at 2; keep all other magnitudes
    emit(
        "fix7",
        make_revive_cap(dead_thr, 1.0, 2.0),
        note=lambda c: f"revived {c['floor']} dead -> scale 1; capped {c['ceil']} at 2 (total changed: {c['changed']})",
    )

    # fix8) attenuation-only: clamp [0, 1] (dead stays dead) -> DoRA may only reduce norms
    emit(
        "fix8",
        make_clamp(dead_thr, 1.0, 0.0, 1.0),
        note=lambda c: f"clamp[0,1] capped {c['ceil']} amplified rows at 1 (dead left dead)",
    )

    # fix9) attenuation-only + revive dead to a LOW scale (not 0, not full 1)
    emit(
        "fix9",
        make_revive_cap(dead_thr, revive, 1.0),
        note=lambda c: f"revived {c['floor']} dead -> scale {revive}; capped {c['ceil']} amplified at 1",
    )

    print(
        f"\nLEGEND: fix1=pure-OFT(scale=1)  fix2=surgical(>{extreme})  fix3=clamp[{lo},{hi}]  "
        f"fix4=DoRA-deleted  fix5=surgical(>2)  fix6=clamp[0,2]  fix7=revive-dead+cap2  "
        f"fix8=atten-only[0,1]  fix9=atten+revive-dead-to-{revive}"
    )


if __name__ == "__main__":
    main()
