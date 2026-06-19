"""Transplant AdaLN + modulation layers from a source safetensors into a destination one.

Result = destination tensors, with every key matching the AdaLN/modulation patterns
replaced by the source's tensor for the same key. Metadata is preserved from the
destination file (since the rest of the model comes from it).

Usage:
    python _adaln_transplant.py --src SRC.safetensors --dst DST.safetensors --out OUT.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

ADALN_TOKENS = ("adaLN", "adaln", "modulation", "img_mod", "txt_mod")


def is_adaln_key(key: str) -> bool:
    kl = key.lower()
    return any(tok.lower() in kl for tok in ADALN_TOKENS)


def read_metadata(fp: str) -> dict[str, str]:
    with safe_open(fp, framework="pt") as f:
        md = f.metadata() or {}
    return dict(md)


def transplant(src: str, dst: str, out: str) -> None:
    src_p, dst_p, out_p = Path(src), Path(dst), Path(out)
    if not src_p.is_file():
        sys.exit(f"src not found: {src}")
    if not dst_p.is_file():
        sys.exit(f"dst not found: {dst}")
    if out_p.exists():
        sys.exit(f"output already exists, refusing to overwrite: {out}")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(src, framework="pt") as fs, safe_open(dst, framework="pt") as fd:
        src_keys = set(fs.keys())
        dst_keys = list(fd.keys())
        adaln_keys = [k for k in dst_keys if is_adaln_key(k)]

        missing_in_src = [k for k in adaln_keys if k not in src_keys]
        if missing_in_src:
            sys.exit(f"AdaLN keys missing in src ({len(missing_in_src)}): {missing_in_src[:5]}...")

        shape_mismatch = []
        for k in adaln_keys:
            ss = tuple(fs.get_slice(k).get_shape())
            ds = tuple(fd.get_slice(k).get_shape())
            if ss != ds:
                shape_mismatch.append((k, ss, ds))
        if shape_mismatch:
            for m in shape_mismatch[:5]:
                print("  shape mismatch:", m)
            sys.exit(f"refusing to transplant: {len(shape_mismatch)} shape mismatches")

        print(f"keys total: {len(dst_keys)}  adaLN keys to replace: {len(adaln_keys)}")

        tensors: dict = {}
        for k in dst_keys:
            tensors[k] = (fs if k in adaln_keys else fd).get_tensor(k)

    metadata = read_metadata(dst)
    save_file(tensors, out, metadata=metadata if metadata else None)
    print(f"wrote {out}  ({sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source file (donor of AdaLN/modulation tensors)")
    ap.add_argument("--dst", required=True, help="destination file (everything else comes from here)")
    ap.add_argument("--out", required=True, help="output path (must not exist)")
    args = ap.parse_args()
    transplant(args.src, args.dst, args.out)


if __name__ == "__main__":
    main()
