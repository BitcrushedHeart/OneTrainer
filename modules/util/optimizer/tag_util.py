import math

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig

import torch


def _verify_dora_oft_mult_patch():
    """Check that adv_optm honors `_is_dora_oft_mult` by dropping standard wd.

    The branch lives in the vendored adv_optm (util/scaled_optm.py), so a plain
    `pip install -U adv_optm` silently reverts it. Losing it is not loud: training
    still runs, but the DoRA-OFT magnitude quietly decays toward 0 (row collapse)
    instead of toward its 1.0 anchor. Probe the function object that param_update
    actually calls, since it binds `adjust_wds` by value at import time.
    """
    try:
        from adv_optm.util.param_update import adjust_wds
    except ImportError:
        return

    probe = torch.zeros(1)
    probe._is_dora_oft_mult = True
    wd, _ = adjust_wds(1.0, 1.0, probe)
    if wd != 0.0:
        print(
            "WARN: adv_optm's adjust_wds() does not honor _is_dora_oft_mult. The DoRA-OFT "
            "magnitude multiplier will receive standard weight decay and drift toward 0 "
            "(row collapse) instead of its centered-wd anchor at 1.0. Re-apply the "
            "scaled_optm.py patch -- it was most likely lost in an adv_optm reinstall."
        )


def tag_peft_parameters(model: torch.nn.Module | None, config: TrainConfig):
    """
    Tags PEFT parameters with attributes like `_is_lora_A`, `_is_lora_B`,
    `_is_oft`, `_is_dora_scale`, and `_is_dora_oft_mult` based on their names.
    This is required to apply correct scaling and optimizations for certain features in adv_optm library.
    """
    tagged_dora_oft_mult = False

    def apply_tags(name: str, p: torch.nn.Parameter):
        if name.endswith(("lora_down.weight", "lokr_w1_b", "lokr_w2_b")):
            # Down projection
            p._is_lora_A = True
        elif name.endswith(("lora_up.weight", "lokr_w1_a", "lokr_w2_a")):
            # Up projection
            p._is_lora_B = True
        elif name.endswith("dora_multiplier"):
            # DoRAOFTModule's relative per-output-row magnitude, initialized to 1.0.
            # Tagged apart from _is_dora_scale because its identity value is 1.0
            # rather than the base-weight norm: standard toward-zero decay shrinks
            # the row past the base weight toward collapse, so adv_optm's
            # adjust_wds() drops standard wd for this tag and leaves only centered
            # wd (which anchors it at 1.0).
            nonlocal tagged_dora_oft_mult
            p._is_dora_oft_mult = True
            tagged_dora_oft_mult = True
        elif name.endswith(("dora_scale", "dora_log_multiplier")):
            # Vector in shape of >= 2D tensor
            p._is_dora_scale = True
        elif name.endswith("oft_R.weight"):
            # Set of independent flattened matrices (rank, n_elements)
            p._is_oft = True
            if config.oft_scaled:
                n_el = p.shape[-1]
                b = (1 + math.sqrt(1 + 8 * n_el)) / 2
                scaling_factor = 2 * math.sqrt(b - 1)
                p._oft_scale_factor = scaling_factor

    for module in vars(model).values():
        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                for param_name, p in lora_module.named_parameters():
                    apply_tags(param_name, p)
        elif isinstance(module, torch.nn.Module):
            for param_name, p in module.named_parameters():
                apply_tags(param_name, p)

    if tagged_dora_oft_mult:
        _verify_dora_oft_mult_patch()
