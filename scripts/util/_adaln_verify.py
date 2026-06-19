"""Verify an AdaLN-transplanted safetensors file.

Compares OUT against SRC (donor of AdaLN) and DST (donor of everything else).
"""

from __future__ import annotations

import argparse
import sys

import torch

from safetensors import safe_open

ADALN_TOKENS = ("adaLN", "adaln", "modulation", "img_mod", "txt_mod")


def is_adaln_key(key: str) -> bool:
    kl = key.lower()
    return any(tok.lower() in kl for tok in ADALN_TOKENS)


def get(fp: str, key: str) -> torch.Tensor:
    with safe_open(fp, framework="pt") as f:
        return f.get_tensor(key)


def list_keys(fp: str) -> list[str]:
    with safe_open(fp, framework="pt") as f:
        return list(f.keys())


def get_metadata(fp: str) -> dict:
    with safe_open(fp, framework="pt") as f:
        md = f.metadata() or {}
    return dict(md)


def verify(src: str, dst: str, out: str, sample_non: int = 8) -> int:
    keys_d = list_keys(dst)
    keys_o = list_keys(out)
    if set(keys_d) != set(keys_o):
        only_d = set(keys_d) - set(keys_o)
        only_o = set(keys_o) - set(keys_d)
        print(f"  FAIL: key set mismatch  only_in_dst={list(only_d)[:3]}  only_in_out={list(only_o)[:3]}")
        return 1
    print(f"  keys total: {len(keys_o)}  (matches dst)")

    adaln = [k for k in keys_o if is_adaln_key(k)]
    non = [k for k in keys_o if not is_adaln_key(k)]
    print(f"  adaLN keys: {len(adaln)}  non-adaLN: {len(non)}")

    failures = 0
    adaln_ok = 0
    adaln_changed = 0
    for k in adaln:
        ts = get(src, k)
        to = get(out, k)
        if torch.equal(ts, to):
            adaln_ok += 1
        td = get(dst, k)
        if not torch.equal(td, to):
            adaln_changed += 1
    print(f"  adaLN equal SRC: {adaln_ok}/{len(adaln)}")
    print(f"  adaLN changed vs DST: {adaln_changed}/{len(adaln)} (>0 expected; equal-by-coincidence is fine)")
    if adaln_ok != len(adaln):
        failures += 1

    if non:
        step = max(1, len(non) // sample_non)
        sample = non[::step][:sample_non]
        ok = 0
        for k in sample:
            td = get(dst, k)
            to = get(out, k)
            if torch.equal(td, to):
                ok += 1
        print(f"  non-adaLN sample equal DST: {ok}/{len(sample)}  (sampled out of {len(non)})")
        if ok != len(sample):
            failures += 1

    md_d = get_metadata(dst)
    md_o = get_metadata(out)
    print(f"  metadata preserved: {md_d == md_o}  (dst keys: {len(md_d)}, out keys: {len(md_o)})")
    if md_d != md_o:
        failures += 1

    return 0 if failures == 0 else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rc = verify(args.src, args.dst, args.out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
