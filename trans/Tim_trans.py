import os
import sys

# --- 关键步骤 1: 动态修复路径，不改动其他文件 ---
# 获取当前脚本所在目录 (E:\TransferAttack\RuiGuoCode\trans)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (E:\TransferAttack\RuiGuoCode)
root_dir = os.path.dirname(current_dir)

# 将工作目录切换到根目录，这样模型加载权重时用的相对路径就能匹配了
os.chdir(root_dir)
# 将根目录加入系统路径，确保能 import models 和其他模块
sys.path.append(root_dir)
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from scipy import stats

from TIM_cli import attack_batch

from preprocess import AdvPNGDataset


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = stats.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


# 创建高斯卷积核 - 调整维度顺序
kernel = gkern(15, 3).astype(np.float32)
# TensorFlow的维度顺序是 [height, width, in_channels, out_channels]
# PyTorch需要 [out_channels, in_channels, height, width]
# 对于深度可分离卷积，我们需要 [out_channels * in_channels/groups, 1, height, width]
stack_kernel = np.stack([kernel, kernel, kernel])  # 形状变为 (3, 15, 15)
stack_kernel = np.expand_dims(stack_kernel, axis=1)  # 形状变为 (3, 1, 15, 15)
stack_kernel = torch.from_numpy(stack_kernel).float()


def get_model_output(model, x):
    """
    统一处理模型输出，确保返回张量
    """
    output = model(x)

    # 处理不同类型的输出
    if isinstance(output, tuple):
        # 如果是元组，取第一个元素（通常是主输出）
        output = output[0]
    elif isinstance(output, list):
        # 如果是列表，取第一个元素
        output = output[0]

    # 确保输出是张量
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")

    return output


def get_model_prediction(model, x):
    """
    从模型输出中获取预测结果
    处理不同形状的输出
    """
    output = get_model_output(model, x)

    # 处理不同维度的输出
    if output.dim() == 1:
        # 一维张量，直接取argmax
        return torch.argmax(output).item()
    elif output.dim() == 2:
        # 二维张量 (batch, num_classes)
        return torch.argmax(output, dim=1).item()
    else:
        # 更高维度，展平为二维
        if output.dim() > 2:
            # 展平除了batch维度外的所有维度
            output = output.view(output.size(0), -1)
            return torch.argmax(output, dim=1).item()
        else:
            raise ValueError(f"Unexpected output dimension: {output.dim()}, shape: {output.shape}")


def attack(x, y, model, eps=16 / 255, iterations=10, mu=1.0):
    """
    PyTorch版本的动量迭代攻击
    使用高斯卷积替代随机缩放填充

    Args:
        x: 原始图像 [1, 3, H, W]
        y: 真实标签
        model: 目标模型
        eps: 最大扰动
        iterations: 迭代次数
        mu: 动量系数
    Returns:
        x_adv: 对抗样本
    """
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True)

    # 计算迭代步长
    alpha = eps / iterations

    # 初始化动量 - 使用与输入相同尺寸
    momentum = torch.zeros_like(x).to(device)

    # 将高斯卷积核移动到设备
    kernel_device = stack_kernel.to(device)

    for i in range(iterations):
        x_adv.requires_grad = True

        # 获取模型输出
        output = get_model_output(model, x_adv)

        # 处理输出形状以确保可以计算交叉熵损失
        if output.dim() == 1:
            # 如果是一维输出，添加batch维度
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            # 如果是更高维度的输出，展平为二维
            output = output.view(output.size(0), -1)

        # 计算损失
        loss = F.cross_entropy(output, y)

        # 计算梯度
        grad = torch.autograd.grad(loss, [x_adv])[0]

        # 应用高斯卷积（替代随机缩放填充）
        # 深度可分离卷积，每个通道独立卷积
        # 使用padding=7确保输出尺寸与输入相同 (299 -> 299)
        grad_conv = F.conv2d(grad, kernel_device, padding=7, groups=3)

        # 动量更新
        grad_conv = grad_conv / torch.mean(torch.abs(grad_conv), dim=(1, 2, 3), keepdim=True)
        momentum = mu * momentum + grad_conv

        # 更新对抗样本
        x_adv = x_adv.detach() + alpha * torch.sign(momentum)

        # 裁剪到允许范围内
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()


## --- 辅助函数：扰动可视化 ---
def save_diff_image(orig, adv, folder, filename):
    """
    计算并保存扰动图。
    orig, adv: [C, H, W] Tensor, 范围 [0, 1]
    """
    # 1. 计算原始扰动 (带有正负号)
    diff = adv - orig

    # 2. 增强可视化方案 A: 绝对值并归一化 (直观查看哪里改动大)
    # 将扰动缩放到 0-1 范围，方便观察
    diff_abs = torch.abs(diff)
    if diff_abs.max() > 0:
        diff_visual = diff_abs / diff_abs.max()
    else:
        diff_visual = diff_abs

    # 3. 增强可视化方案 B: 偏移映射 (0.5表示无变化，<0.5变暗，>0.5变亮)
    diff_shift = (diff + 0.5).clamp(0, 1)

    # 保存路径
    base_name = os.path.splitext(filename)[0]

    # 保存攻击后的图
    save_image(adv, os.path.join(folder, f"{base_name}_adv.png"))
    # 保存绝对值扰动 (哪里被改了)
    save_image(diff_visual, os.path.join(folder, f"{base_name}_diff_abs.png"))
    # 保存偏移扰动 (改了什么颜色)
    save_image(diff_shift, os.path.join(folder, f"{base_name}_diff_shift.png"))


def main():
    # 参数配置（可以根据需要修改或使用 argparse）
    input_dir = './data'
    output_diff_dir = './results/tim_trans'
    batch_size = 4
    eps = 16 / 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_diff_dir, exist_ok=True)

    from models import ModelRepository
    model_repo = ModelRepository(device)
    source_model = model_repo.get_source_model('tf2torch_resnet_v2_101')['model']
    source_model.eval()

    # 数据加载
    label_df = pd.read_csv(os.path.join(input_dir, 'labels.csv'))
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    dataset = AdvPNGDataset(os.path.join(input_dir, 'images'), label_df, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Starting TIM change analysis...")

    for x_batch, y_batch0, filename_batch in tqdm(loader):
        x_batch = x_batch.to(device)
        y_batch0 = y_batch0.to(device)

        # 执行 TIM 攻击
        x_adv_batch = attack_batch(
            x_batch, y_batch0, source_model,
            eps=eps, iterations=10, mu=1.0
        )

        # 逐张对比并保存
        for i in range(x_batch.size(0)):
            save_diff_image(
                x_batch[i].cpu(),
                x_adv_batch[i].cpu(),
                output_diff_dir,
                filename_batch[i]
            )

    print(f"Done! Images and changes saved to {output_diff_dir}")


if __name__ == "__main__":
    main()