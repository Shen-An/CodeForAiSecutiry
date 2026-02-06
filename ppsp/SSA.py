"""SSA / TI+DI building blocks.

This module provides two utilities commonly used in adversarial attacks:

- TI (Translation-Invariant) smoothing via a Gaussian kernel (as weight tensor)
- DI (Input Diversity) stochastic resize + pad

References:
- TI: https://arxiv.org/abs/1904.02884
- DI: https://arxiv.org/abs/1803.06978
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gkern(kernlen: int = 15, nsig: float = 3.0, channels: int = 3, device: torch.device | None = None) -> torch.Tensor:
    """Create a depthwise Gaussian kernel for TI.

    Returns tensor of shape (C, 1, K, K) suitable for depthwise conv2d.

    Note: unlike some reference implementations, this function does NOT force
    cuda(); pass device or move the returned tensor outside.
    """
    if kernlen <= 0 or kernlen % 2 == 0:
        raise ValueError("kernlen must be a positive odd integer")

    # Create 1D Gaussian
    x = np.linspace(-nsig, nsig, kernlen)
    try:
        # Prefer SciPy if available
        import scipy.stats as st  # type: ignore

        kern1d = st.norm.pdf(x)
    except Exception:
        # Fallback: manual Gaussian (unnormalized)
        kern1d = np.exp(-(x ** 2) / (2 * (nsig ** 2)))

    kernel_raw = np.outer(kern1d, kern1d)
    kernel = (kernel_raw / kernel_raw.sum()).astype(np.float32)

    kernel_c = np.stack([kernel] * int(channels), axis=0)  # (C, K, K)
    kernel_c = np.expand_dims(kernel_c, 1)  # (C, 1, K, K)
    t = torch.from_numpy(kernel_c)
    if device is not None:
        t = t.to(device)
    return t


def DI(x: torch.Tensor, resize_rate: float = 1.15, diversity_prob: float = 0.5) -> torch.Tensor:
    """Input diversity (random resize + pad).

    Args:
        x: (B, C, H, W)
        resize_rate: >= 1.0
        diversity_prob: in [0, 1]
    """
    if resize_rate < 1.0:
        raise ValueError("resize_rate must be >= 1.0")
    if not (0.0 <= diversity_prob <= 1.0):
        raise ValueError("diversity_prob must be in [0, 1]")

    img_size = int(x.shape[-1])
    img_resize = int(img_size * float(resize_rate))

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), device=x.device, dtype=torch.int32)
    rnd_int = int(rnd.item())

    rescaled = F.interpolate(x, size=[rnd_int, rnd_int], mode="bilinear", align_corners=False)
    h_rem = img_resize - rnd_int
    w_rem = img_resize - rnd_int

    pad_top = int(torch.randint(low=0, high=h_rem + 1, size=(1,), device=x.device, dtype=torch.int32).item())
    pad_bottom = h_rem - pad_top
    pad_left = int(torch.randint(low=0, high=w_rem + 1, size=(1,), device=x.device, dtype=torch.int32).item())
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)

    if torch.rand(1, device=x.device) < diversity_prob:
        return padded
    return x
