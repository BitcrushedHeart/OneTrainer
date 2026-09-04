"""Shared fixtures for the OFT conjugation / transfer-mode tests.

Auto-loaded by pytest for every test under tests/. There are no __init__.py
files in this tree (rootdir/prepend import mode), so a root conftest is the
only way to share helpers across test directories. The repo has no pytest ini
for this file to shadow, so rootdir detection is unaffected in practice.
"""

import json

from modules.module.LoRAModule import DoRAOFTModule, OFTModule

import torch
from torch import nn

import pytest
from safetensors.torch import save_file

# Tiny Z-Image-shaped geometry shared by the conjugation/transfer tests.
# hidden=16, mlp=24 with a requested block of 6 reproduces the real lineup's
# "requested vs adjusted" trap in miniature: in=16 adjusts 6 -> 4 while in=24
# keeps 6, so two block sizes coexist in one adapter (like 480/512 in V40).
TINY_HIDDEN = 16
TINY_MLP = 24
TINY_BLOCK_REQUEST = 6
TINY_STEM_SHAPES = {
    "attention.to_q": (TINY_HIDDEN, TINY_HIDDEN),
    "attention.to_k": (TINY_HIDDEN, TINY_HIDDEN),
    "attention.to_v": (TINY_HIDDEN, TINY_HIDDEN),
    "feed_forward.w1": (TINY_MLP, TINY_HIDDEN),
    "feed_forward.w2": (TINY_HIDDEN, TINY_MLP),
    "feed_forward.w3": (TINY_MLP, TINY_HIDDEN),
}


@pytest.fixture
def tiny_geometry():
    """The tiny lineup geometry as data, for tests that need the shapes/blocks."""
    return {
        "hidden": TINY_HIDDEN,
        "mlp": TINY_MLP,
        "block_request": TINY_BLOCK_REQUEST,
        "stem_shapes": dict(TINY_STEM_SHAPES),
        # Actual adapter block per in_features after OFTModule's adjustment:
        # 16 -> 4 (request 6 rejected), 24 -> 6 (request 6 divides).
        "block_for_in": {TINY_HIDDEN: 4, TINY_MLP: TINY_BLOCK_REQUEST},
    }


@pytest.fixture
def make_tiny_adapter():
    """Factory writing a tiny OFT / DoRA-OFT adapter safetensors file.

    Builds real OFTModule/DoRAOFTModule instances over toy Linears so the file
    carries exactly the keys and marker buffers OneTrainer saves (oft_R.weight,
    oft_R.clipped_oft, dora_multiplier, ...), plus ot_config metadata holding
    the REQUESTED block size (the adjusted sizes are only discoverable from the
    tensors, as with real adapters).
    """

    def _make(
        path,
        *,
        layers: int = 4,
        dora: bool = False,
        prefix: str = "transformer.",
        file_dtype: torch.dtype = torch.float32,
        block_request: int = TINY_BLOCK_REQUEST,
        theta: float = 1e-2,
        clip: float | None = 0.99,
        scaled: bool = False,
        seed: int = 0,
    ) -> str:
        gen = torch.Generator().manual_seed(seed)
        state = {}
        for layer in range(layers):
            for stem, (out_f, in_f) in TINY_STEM_SHAPES.items():
                cls = DoRAOFTModule if dora else OFTModule
                module = cls(
                    prefix="m",
                    orig_module=nn.Linear(in_f, out_f, bias=False),
                    oft_block_size=block_request,
                    block_share=False,
                    oft_scaled=scaled,
                    oft_clipped_norm=clip,
                )
                with torch.no_grad():
                    module.oft_R.weight.copy_(torch.randn(module.oft_R.weight.shape, generator=gen) * theta)
                    if dora:
                        module.dora_multiplier.copy_(
                            1.0 + 0.1 * torch.randn(module.dora_multiplier.shape, generator=gen)
                        )
                for k, v in module.state_dict().items():
                    key = f"{prefix}layers.{layer}.{stem}.{k}"
                    state[key] = v.detach().to(file_dtype).contiguous()
        ot_config = {
            "peft_type": "OFT_2",
            "dora_oft": dora,
            "oft_block_size": block_request,
            "oft_clipped_norm": clip,
            "oft_scaled": scaled,
            "oft_block_share": False,
            "oft_cans": False,
        }
        save_file(state, str(path), metadata={"ot_config": json.dumps(ot_config)})
        return str(path)

    return _make


@pytest.fixture
def make_ckpt_tensors():
    """Factory for a tiny checkpoint's layer-stack weights, keyed by bare stem
    (``layers.N.attention.to_q``), fp32."""

    def _make(*, layers: int = 4, seed: int = 100) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        return {
            f"layers.{layer}.{stem}": torch.randn(out_f, in_f, generator=gen)
            for layer in range(layers)
            for stem, (out_f, in_f) in TINY_STEM_SHAPES.items()
        }

    return _make


@pytest.fixture
def write_tiny_ckpt():
    """Factory writing bare-stem-keyed tensors as a checkpoint safetensors file."""

    def _write(path, tensors: dict[str, torch.Tensor], *, prefix: str = "", file_dtype: torch.dtype = torch.float32):
        state = {f"{prefix}{stem}.weight": t.to(file_dtype).contiguous() for stem, t in tensors.items()}
        save_file(state, str(path))
        return str(path)

    return _write


@pytest.fixture
def make_orthogonal_blocks():
    """Factory for an (r, b, b) stack of orthogonal blocks.

    angle=None -> fully random orthogonal blocks (QR of randn).
    angle=a    -> matrix_exp of a skew matrix with spectral norm a: a genuine
                  rotation at a controlled distance from identity. The
                  alternating blockwise Procrustes solver is only used in the
                  near-identity regime (distillation deltas are ~1-2%), so
                  ground-truth transports in tests use small angles.
    """

    def _make(
        r: int,
        b: int,
        *,
        seed: int,
        angle: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        if angle is None:
            m = torch.randn(r, b, b, generator=gen, dtype=torch.float64)
            q, rr = torch.linalg.qr(m)
            # Sign-fix columns so the factorization is deterministic.
            s = torch.diagonal(rr, dim1=-2, dim2=-1).sign()
            q = q * s.unsqueeze(-2)
            return q.to(dtype)
        skew = torch.randn(r, b, b, generator=gen, dtype=torch.float64)
        skew = skew - skew.transpose(-2, -1)
        norms = torch.linalg.matrix_norm(skew, ord=2)
        skew = skew * (angle / norms).reshape(r, 1, 1)
        return torch.matrix_exp(skew).to(dtype)

    return _make
