"""Buffered (no-mmap) safetensors readers for adapter/checkpoint tooling.

Promoted from scripts/util/_oft_adapter_retarget.py so scripts and tests can
share them. Rationale (see that script's docstring): while a training run holds
this machine's memory, page-ins from a second multi-GB mmap fail intermittently
with in-page access violations that kill the process, and a same-dtype
``.to()`` on an mmapped tensor is a no-op alias that keeps the mapping alive
past close. These readers parse the safetensors header by hand and read tensor
data into fresh ``bytearray`` buffers -- no mmap ever exists, so neither trap
applies.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import torch

ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    # Marker buffers and misc metadata tensors: scaled_oft/cans_oft are BOOL,
    # and integer tensors appear in some checkpoint formats. Without these,
    # reading a scaled-OFT adapter died on a raw KeyError('BOOL').
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
}


def _read_header(f) -> tuple[dict, dict[str, str], int]:
    header_len = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(header_len))
    metadata = header.pop("__metadata__", {})
    return header, metadata, 8 + header_len


def _read_tensor(f, info: dict, data_start: int) -> torch.Tensor:
    dtype = ST_DTYPES.get(info["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype {info['dtype']!r} (known: {sorted(ST_DTYPES)})")
    begin, end = info["data_offsets"]
    f.seek(data_start + begin)
    buf = bytearray(f.read(end - begin))  # writable buffer: no frombuffer warning, no sharing
    return torch.frombuffer(buf, dtype=dtype).reshape(info["shape"])


def detect_layer_prefix(header_keys: Iterable[str]) -> str:
    """'transformer.' or '' -- whether layer-stack keys carry the transformer prefix."""
    for k in header_keys:
        if k.endswith(".weight") and ".layers." in k:
            return "transformer." if k.startswith("transformer.") else ""
    return ""


def read_safetensors_buffered(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Whole file: (tensors in their stored dtype, string metadata)."""
    with open(path, "rb") as f:
        header, metadata, data_start = _read_header(f)
        tensors = {k: _read_tensor(f, info, data_start) for k, info in header.items()}
    return tensors, metadata


def read_tensors_buffered(path: str, keys: set[str]) -> dict[str, torch.Tensor]:
    """Named tensors only, in their stored dtype. Raises KeyError on a missing key."""
    with open(path, "rb") as f:
        header, _, data_start = _read_header(f)
        out = {}
        for k in keys:
            if k not in header:
                raise KeyError(f"{path}: tensor {k!r} not present")
            out[k] = _read_tensor(f, header[k], data_start)
    return out


def read_layer_weights_buffered(path: str, stems: set[str]) -> dict[str, torch.Tensor]:
    """Read ``{stem}.weight`` for each stem (``layers.N.attention.to_q`` style),
    auto-detecting the checkpoint's ``transformer.`` prefix. Keys in the result
    are the bare stems; tensors are returned in their stored dtype."""
    with open(path, "rb") as f:
        header, _, data_start = _read_header(f)
        prefix = detect_layer_prefix(header.keys())
        out = {}
        for stem in stems:
            key = f"{prefix}{stem}.weight"
            if key not in header:
                raise KeyError(f"{path}: tensor {key!r} not present")
            out[stem] = _read_tensor(f, header[key], data_start)
    return out
