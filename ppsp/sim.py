import torch
import torch.nn.functional as F
from typing import Iterable, Sequence


def si_gradient(
    model,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    *,
    scales: Iterable[float] = (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16),
) -> torch.Tensor:
    """SI-MI-FGSM 的 SI（Scale Invariance）梯度估计。

    注意：该函数是“对 x_adv * scale 前向，但对 x_adv 求梯度并累加”的传统 SI 写法。
    如果你希望“先生成多尺度副本再做后续增强/BSR”，请使用 make_si_copies()。
    """
    if not x_adv.requires_grad:
        x_adv = x_adv.requires_grad_(True)

    grad_sum = torch.zeros_like(x_adv)

    for s in scales:
        x_scaled = x_adv * float(s)
        output = model(x_scaled)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            output = output.view(output.size(0), -1)

        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        grad_sum = grad_sum + grad

    return grad_sum


def make_si_copies(x: torch.Tensor, *, scales: Sequence[float]) -> torch.Tensor:
    """在 batch 维度上拼接 SI 的多尺度副本（用于“SI 放在 BSR 创建副本前”）。

    输入:
      x: (B,C,H,W)
      scales: 例如 (1, 1/2, 1/4, 1/8, 1/16)

    输出:
      x_si: (B*len(scales), C, H, W)

    说明：
    - 这里按你的 SI 代码语义做“直接乘 scale”。
    - 不做 resize，不改空间尺寸，只改像素幅值。
    """
    if len(scales) == 0:
        return x
    copies = [x * float(s) for s in scales]
    return torch.cat(copies, dim=0)
