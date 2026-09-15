import torch
from torch import Tensor

generator = None


def set_seed(seed: int, device: torch.device):
    global generator
    if generator is None or generator.device != device:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)


def copy_stochastic_(target: Tensor, source: Tensor, generator: torch.Generator | None = None):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
        generator: the random generator; None uses the module-level generator set by set_seed
    """

    if generator is None:
        generator = globals()["generator"]

    # create a random 16 bit integer
    result = torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=generator,
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(
    input: Tensor,
    other: Tensor,
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
    chunk_numel: int | None = None,
):
    """
    adds alpha * other to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
        generator: the random generator; None uses the module-level generator set by set_seed
        chunk_numel: if set, process input in slices along dim 0 of at most this many elements
            (at least one row), bounding the float32 + int32 transients of large tensors
    """
    if chunk_numel is not None and input.dim() > 0 and input.shape[0] > 1 and input.numel() > chunk_numel:
        rows = max(1, chunk_numel // (input.numel() // input.shape[0]))
        for start in range(0, input.shape[0], rows):
            add_stochastic_(input[start : start + rows], other[start : start + rows], alpha, generator)
        return

    # input + alpha * other. (Before this was fixed it computed other + alpha * input,
    # which only coincided for alpha == 1 or other is input -- the two ways CAME calls it.)
    result = input.to(dtype=torch.float32, copy=True)

    result.add_(other, alpha=alpha)
    copy_stochastic_(input, result, generator)


def addcdiv_stochastic_(input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0):
    """
    adds (tensor1 / tensor2 * value) to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        tensor1: the numerator tensor
        tensor2: the denominator tensor
        value: a multiplier for tensor1/tensor2
    """
    result = input.clone() if input.dtype == torch.float32 else input.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(input, result)
