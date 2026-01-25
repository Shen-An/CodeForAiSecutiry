
import torch
import torch.nn.functional as F


def input_diversity(x, image_width=299, image_resize=331, prob=0.5):
    """输入多样性（DI-FGSM 风格）。

    备注：使用 nearest 与原实现保持一致。
    """
    if torch.rand(1, device=x.device).item() < prob:
        rnd = torch.randint(image_width, image_resize + 1, (1,), device=x.device).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='nearest')

        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,), device=x.device).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem + 1, (1,), device=x.device).item()
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        padded = F.interpolate(padded, size=(image_width, image_width), mode='nearest')
        return padded
    return x