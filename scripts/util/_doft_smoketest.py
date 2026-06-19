"""Smoke test for the ported relative-multiplier DoRA-OFT module.

Confirms DoRAOFTModule constructs against our OFTModule signature, inits the
dora_multiplier to 1.0, forwards without NaN, saves the new `dora_multiplier`
key (and not the legacy dora_scale/initial_norm), and that the multiplier
actually scales the rotated output per row.
"""

import math

from modules.module.LoRAModule import DoRAOFTModule

import torch
from torch import nn

from safetensors.torch import save_file

torch.manual_seed(0)

orig = nn.Linear(64, 32)  # in_features=64, out_features=32
m = DoRAOFTModule("lora_test", orig, oft_block_size=16, coft=False, coft_eps=1e-3, block_share=False, scaled_oft=True)
m.initialize_weights()
m.hook_to_module()  # sets orig_forward; required before check_initialized()/forward()
m.check_initialized()
m.eval()

print("dora_multiplier shape :", tuple(m.dora_multiplier.shape), "(expect (32,))")
print("init == all ones      :", bool(torch.all(m.dora_multiplier == 1.0)))

keys = sorted(m.state_dict().keys())
print("state_dict keys       :", keys)
print("has dora_multiplier   :", any(k.endswith("dora_multiplier") for k in keys))
print("has legacy dora_scale :", any(k.endswith("dora_scale") for k in keys))
print("has legacy init_norm  :", any(k.endswith("initial_norm") for k in keys))

x = torch.randn(4, 64)
with torch.no_grad():
    out1 = m(x)  # multiplier == 1.0
    bias = orig.bias.detach()
    m.dora_multiplier.data.fill_(2.0)
    out2 = m(x)  # multiplier == 2.0

ok_shape = tuple(out1.shape) == (4, 32)
ok_finite = bool(torch.isfinite(out1).all() and torch.isfinite(out2).all())
# multiplier scales (result - bias): out2 == 2*(out1 - bias) + bias == 2*out1 - bias
ok_wiring = bool(torch.allclose(out2, 2 * out1 - bias, atol=1e-4))

print("forward shape ok      :", ok_shape, tuple(out1.shape))
print("forward finite        :", ok_finite)
print("multiplier wiring ok  :", ok_wiring, "(out2 == 2*out1 - bias)")

# --- apply_to_module (bake) invariant: ||merged_row|| == |mult| * ||base_row|| ---
with torch.no_grad():
    base_norm = orig.weight.data.reshape(32, -1).float().norm(dim=1).clone()
    m.dora_multiplier.data.uniform_(0.5, 1.5)
    mult = m.dora_multiplier.data.abs().clone()
    m.apply_to_module()
    merged_norm = orig.weight.data.reshape(32, -1).float().norm(dim=1)
ok_bake = bool(torch.allclose(merged_norm, mult * base_norm, rtol=5e-3, atol=1e-3))
print("apply_to_module bake  :", ok_bake, "(||merged_row|| == |mult| * ||base_row||)")

assert ok_shape and ok_finite and ok_wiring and ok_bake and tuple(m.dora_multiplier.shape) == (32,)

# --- oft_clipped_norm (PR #1492): clipping must keep the Neumann series convergent at large ||Q|| ---
big = torch.randn(4, 120) * 3.0  # in=64, block=16 -> 4 blocks, n_elements=120; large => ||Q|| >> 1
eye16 = torch.eye(16).unsqueeze(0)
mc = DoRAOFTModule("clip", nn.Linear(64, 32), 16, False, 1e-3, False, scaled_oft=False, oft_clipped_norm=True)
mc.initialize_weights()
mc.train()
with torch.no_grad():
    mc.oft_R.weight.copy_(big)
    for _ in range(6):  # let power iteration converge
        mc.oft_R._cayley_batch(mc.oft_R.weight, 16, True, 5)
    Rc = mc.oft_R._cayley_batch(mc.oft_R.weight, 16, True, 5)
    err_clipped = float((Rc.transpose(-1, -2) @ Rc - eye16).norm(dim=(1, 2)).max() / (16**0.5))
mu = DoRAOFTModule("noclip", nn.Linear(64, 32), 16, False, 1e-3, False, scaled_oft=False, oft_clipped_norm=False)
mu.initialize_weights()
with torch.no_grad():
    Ru = mu.oft_R._cayley_batch(big, 16, True, 5)
    err_unclipped = float((Ru.transpose(-1, -2) @ Ru - eye16).norm(dim=(1, 2)).max() / (16**0.5))
print(f"clip @ large ||Q||    : clipped_err={err_clipped:.3f}  unclipped_err={err_unclipped:.1f}")
assert math.isfinite(err_clipped) and err_clipped < err_unclipped

# Emit a tiny new-format adapter so the health script's dora_multiplier branch is exercised.
out_path = "scripts/util/_doft_smoke_adapter.safetensors"
save_file({k: v.detach().contiguous() for k, v in m.state_dict().items()}, out_path)
print("wrote", out_path)
print("\nSMOKE TEST PASSED")
