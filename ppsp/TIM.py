import numpy as np
import torch
import torch.nn.functional as F


def gkern(kernlen: int = 15, nsig: float = 3.0) -> np.ndarray:
    """返回 2D Gaussian kernel（对齐 stats.norm.pdf(x) 的默认行为：sigma=1）。

    你的要求是使用等价于 stats.norm.pdf(x)（即 loc=0, scale=1）。
    这里仍用 nsig 控制 x 的取值范围 [-nsig, nsig]，但 pdf 的标准差固定为 1。
    最终对 2D kernel 做 sum 归一化。
    """
    x = np.linspace(-nsig, nsig, kernlen).astype(np.float32)

    # 等价于 stats.norm.pdf(x) 默认参数：loc=0, scale=1
    kern1d = (np.exp(-0.5 * (x ** 2)) / np.sqrt(2.0 * np.pi)).astype(np.float32)

    kernel_raw = np.outer(kern1d, kern1d).astype(np.float32)
    kernel = kernel_raw / np.sum(kernel_raw)
    return kernel.astype(np.float32)


def build_tim_kernel(
    *,
    kernlen: int = 15,
    nsig: float = 3.0,
    channels: int = 3,
) -> torch.Tensor:
    """构建 TIM 的 depthwise 卷积核 (C,1,k,k)。"""
    kernel = gkern(kernlen=kernlen, nsig=nsig)
    stack_kernel = np.stack([kernel] * int(channels))
    stack_kernel = np.expand_dims(stack_kernel, axis=1)  # (C,1,k,k)
    return torch.from_numpy(stack_kernel).float()


def tim_grad(
    grad: torch.Tensor,
    *,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """对梯度应用 TIM：depthwise 高斯卷积。

    grad: (B,C,H,W)
    kernel: (C,1,k,k)
    """
    k = int(kernel.shape[-1])
    pad = k // 2
    return F.conv2d(grad, kernel.to(grad.device, grad.dtype), padding=pad, groups=grad.size(1))
