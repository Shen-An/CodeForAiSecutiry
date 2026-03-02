import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from torchvision import models, transforms
from models import ModelRepository


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    """Configuration for the SID Attack"""
    output_dir = './outputs'
    trans_dir = './trans_results'  # 保存 i=5 时的 20 张变换图片
    max_epsilon = 16.0
    momentum = 1.0
    N = 20  # 谱变换数量 (生成 20 张)
    K = 2  # Number of blocks
    beta = 0.1  # Downsampling factor
    p = 0.5  # Probability of image block fusion
    omega = 0.5  # Weight of linear fusion


opt = Config()


# ==========================================
# 2. 原版 DCT 函数 (SID核心)
# ==========================================
def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    return 2 * V.view(*x_shape)


def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


# ==========================================
# 3. 原版频域融合与多尺度逻辑
# ==========================================
def get_length(length, num_block):
    length = int(length)
    rand = np.random.uniform(size=num_block, low=0.1, high=0.9)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def random_flip_batch(x):
    mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < 0.5).float()
    return mask * torch.flip(x, dims=(3,)) + (1 - mask) * x


def frequency_fusion_batch(patch, x):
    org_x = x.clone()
    _, _, patch_w, patch_h = patch.shape
    rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)
    rescale_flip_x = random_flip_batch(rescale_x)

    dctx = dct_2d(rescale_flip_x)
    dctp = dct_2d(patch)

    _, _, w, h = dctx.shape
    low_w = int(w * 0.4)
    low_h = int(h * 0.4)

    dctx[:, :, 0:low_w, 0:low_h] = dctp[:, :, 0:low_w, 0:low_h]
    return idct_2d(dctx)


def linear_fusion_batch(patch, x, omega=0.5):
    rescale_x = F.interpolate(x.clone(), size=[patch.shape[2], patch.shape[3]], mode='bilinear', align_corners=False)
    return random_flip_batch(rescale_x) * omega + patch * (1 - omega)


def local_fusion_batch(x, num_block=2, probabilities=0.5, omega=0.5):
    N, C, w, h = x.shape
    width_length = get_length(w, num_block)
    height_length = get_length(h, num_block)

    x_split_w = torch.split(x, width_length, dim=2)
    x_split_h_l = [torch.split(sw, height_length, dim=3) for sw in x_split_w]

    ret_list = []
    for strip in x_split_h_l:
        temp_list = []
        for patch in strip:
            mask_fuse = (torch.rand(N, 1, 1, 1, device=x.device) >= probabilities)
            mask_freq = (torch.rand(N, 1, 1, 1, device=x.device) < 0.5)

            freq_out = frequency_fusion_batch(patch, x)
            linear_out = linear_fusion_batch(patch, x, omega)

            fused = torch.where(mask_freq, freq_out, linear_out)
            x_enh = torch.where(mask_fuse, fused, patch)
            temp_list.append(random_flip_batch(x_enh))

        ret_list.append(torch.cat(temp_list, dim=3))

    return torch.cat(ret_list, dim=2)


def multi_scale_batch(x_batch):
    N, C, H, W = x_batch.shape
    outputs = []
    for n in range(N):
        resize_ratio = 1 - (n * opt.beta / opt.N)
        img_slice = x_batch[n:n + 1]
        if resize_ratio == 1:
            ret = img_slice
        else:
            new_size = int(H * resize_ratio)
            rescaled = F.interpolate(img_slice, size=[new_size, new_size], mode='bilinear', align_corners=False)
            h_rem = H - new_size
            w_rem = W - new_size
            pad_top = random.randint(0, h_rem)
            pad_left = random.randint(0, w_rem)
            ret = F.pad(rescaled, [pad_left, w_rem - pad_left, pad_top, h_rem - pad_top], value=0)

        if random.random() < 0.5:
            ret = torch.flip(ret, dims=(3,))
        outputs.append(ret)

    return torch.cat(outputs, dim=0)


# ==========================================
# 4. Attack Logic (保存 i=5 时的全部 20 张)
# ==========================================
def SID(images, gt, model, min_val, max_val, image_name):
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    grad = torch.zeros_like(x)

    pure_name = os.path.splitext(image_name)[0]

    for i in range(num_iter):
        x.requires_grad = True

        # SID 核心变换
        x_batch = x.repeat(opt.N, 1, 1, 1)
        x_emb = local_fusion_batch(x_batch, opt.K, opt.p, opt.omega)
        x_input = multi_scale_batch(x_emb)

        # 👉 核心需求：在 i=5 时，保存这 20 张带有 SID 噪声/伪影的中间图
        if i == 0:
            print(f"--> [Iter 5] 正在提取 {opt.N} 张 SID 变换中间图...")
            for j in range(opt.N):
                t_img = x_input[j].detach().cpu().permute(1, 2, 0).numpy()
                t_img = (np.clip(t_img, 0, 1) * 255).astype(np.uint8)
                save_path = os.path.join(opt.trans_dir, f"{pure_name}_iter5_trans_{j:02d}.png")
                Image.fromarray(t_img).save(save_path)
            print(f"--> 保存完成！路径: {os.path.abspath(opt.trans_dir)}")

        # 前向反向传播求梯度
        output = model(x_input)
        if isinstance(output, (list, tuple)):
            output = output[0]

        loss = F.cross_entropy(output, gt.repeat(opt.N))
        model.zero_grad()
        loss.backward()

        noise = x.grad.data / opt.N
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x.detach() + alpha * torch.sign(noise)
        x = torch.clamp(x, min_val, max_val)

    return x.detach()


# ==========================================
# 5. Main Execution (单图执行)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.trans_dir, exist_ok=True)

    # Setup Model
    model_repo = ModelRepository(device)
    model_info = model_repo.get_source_model('tf2torch_inception_v3')
    model = model_info['model']
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # 👉 指定你的图片路径
    img_path = r'E:\TransferAttack\RuiGuoCode\data\images\ILSVRC2012_val_00013393.png'

    if not os.path.exists(img_path):
        print(f"Error: 找不到指定的图片文件 -> {img_path}")
        return

    filename = os.path.basename(img_path)
    label_val = 1  # 默认标签

    print(f"Starting SID attack on single image: {filename} ...")

    # Load Image
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    y = torch.tensor([label_val], device=device)

    # constraints
    images_min = torch.clamp(x - (opt.max_epsilon / 255.0), 0.0, 1.0)
    images_max = torch.clamp(x + (opt.max_epsilon / 255.0), 0.0, 1.0)

    # Run Attack (内部处理了 i=5 保存逻辑)
    x_adv = SID(x, y, model, images_min, images_max, filename)

    # 保存最终生成的对抗样本
    adv_img = x_adv.squeeze(0).cpu().permute(1, 2, 0).numpy()
    adv_img = (np.clip(adv_img, 0, 1) * 255).astype(np.uint8)
    out_path = os.path.join(opt.output_dir, filename)
    Image.fromarray(adv_img).save(out_path)

    print(f"Attack finished. 最终对抗样本已保存至 {out_path}")


if __name__ == "__main__":
    main()