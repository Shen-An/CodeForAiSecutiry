import torch
def admix(x, portion=0.2, size=3):
    """混合输入变换（Admix）。

    与原 bsr.py 兼容：随机打乱 batch 并做线性混合，然后在 batch 维度拼接 size 份。
    """
    indices = torch.randperm(x.size(0), device=x.device)
    admixed = []
    for _ in range(int(size)):
        admixed_x = x + float(portion) * x[indices]
        admixed.append(admixed_x)
    return torch.cat(admixed, dim=0)