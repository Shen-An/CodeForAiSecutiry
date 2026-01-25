import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import argparse


def save_scaled_images(input_image_path, output_dir):
    """
    专门用于可视化 SI-MI-FGSM 攻击中产生的缩放变换图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 加载并预处理图片
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_name = os.path.basename(input_image_path).split('.')[0]

    # 标准的预处理：Resize 到 299x299 (Inception标准) 并转为 Tensor
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    img_pil = Image.open(input_image_path).convert('RGB')
    x = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 299, 299]

    # 2. 定义 SI (Scale Invariance) 的缩放比例
    # 对应你代码中的 scales = [1.0, 1/2, 1/4, 1/8, 1/16]
    scales = [1.0, 0.5, 0.25, 0.125, 0.0625]

    print(f"正在处理图片: {input_image_path}")
    print(f"输出目录: {output_dir}")

    for scale in scales:
        # 在 SI-MI-FGSM 中，缩放是通过直接乘法实现的
        # x_scaled = x * scale
        x_scaled = x * scale

        # 为了保存成看得见的图片，我们需要注意：
        # 直接乘 scale 会让图片变暗（像素值变小）
        # 如果你想看“模型实际看到的输入”，直接保存 x_scaled
        # 如果你想看“缩放后的内容”，通常需要归一化，但这里我们保持与代码逻辑一致

        file_name = f"{img_name}_scale_{scale:.4f}.png"
        save_path = os.path.join(output_dir, file_name)

        # save_image 会自动处理 Tensor 到图片的转换
        save_image(x_scaled, save_path)
        print(f"  [已保存] 比例 {scale}: {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='../data/images/ILSVRC2012_val_00000356.png', type=str, help="输入图片路径")
    parser.add_argument("--out_dir", default='../results/sim_trans', type=str, help="保存目录")
    args = parser.parse_args()

    save_scaled_images(args.input, args.out_dir)