import os
import torch
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 加上这一句，强行允许 OpenMP 重复加载

def compute_magnitude_spectrum(clean_img, adv_img):
    # 1. 计算对抗扰动
    perturbation = adv_img - clean_img
    # 2. 转为灰度图 (在通道维度求平均)
    perturbation_gray = torch.mean(perturbation, dim=0)
    # 3. 2D FFT
    fft_result = torch.fft.fft2(perturbation_gray)
    # 4. 频移：将低频分量移动到中心
    fft_shifted = torch.fft.fftshift(fft_result)
    # 5. 获取幅度谱
    magnitude_spectrum = torch.abs(fft_shifted)
    return magnitude_spectrum


# 【修改点1】增加了 adv_prefix 参数，默认是空字符串
def process_folder(clean_dir, adv_dir, adv_prefix=""):
    transform = transforms.ToTensor()
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    filenames = [f for f in os.listdir(clean_dir) if f.lower().endswith(valid_exts)]

    accumulated_spectrum = None
    count = 0

    print(f"Processing images from {adv_dir}...")
    for fname in tqdm(filenames):
        clean_path = os.path.join(clean_dir, fname)

        # 【修改点2】在这里把前缀拼接到文件名上
        adv_filename = adv_prefix + fname
        adv_path = os.path.join(adv_dir, adv_filename)

        # 如果找不到文件，打印第一个缺失的路径方便排查，然后跳过
        if not os.path.exists(adv_path):
            if count == 0:
                print(f"  [警告] 找不到文件，请检查路径或前缀: {adv_path}")
            continue

        clean_img = transform(Image.open(clean_path).convert('RGB'))
        adv_img = transform(Image.open(adv_path).convert('RGB'))
        mag = compute_magnitude_spectrum(clean_img, adv_img)

        if accumulated_spectrum is None:
            accumulated_spectrum = mag
        else:
            accumulated_spectrum += mag
        count += 1

    print(f"Successfully processed {count} images.\n")
    if count == 0:
        raise ValueError("没有处理任何图片，请终止程序并检查路径/前缀！")

    avg_spectrum = accumulated_spectrum / count
    log_avg_spectrum = torch.log(avg_spectrum + 1e-8)
    return log_avg_spectrum.numpy()


def main():
    # 请换成你的绝对路径
    CLEAN_DIR = './data/images'
    ADV_BASELINE_DIR = '../TransferAttack/BSR-TI-DIM'  # 比如用BSR作为对比基线
    ADV_BGPF_DIR = '../TransferAttack/4block_flip_pf_SITIDIM/images'  # 你们的方法
    OUTPUT_EPS = './eps/frequency_comparison.eps'  # 导出EPS文件的路径

    os.makedirs(os.path.dirname(OUTPUT_EPS), exist_ok=True)  # 自动创建 eps 文件夹（如果不存在的话）
    # 【修改点3】在这里传入 adv_prefix="adv_"
    # 注意：如果你的 BGPF 生成的图也有 adv_ 前缀，那就两个都传 "adv_"
    # 如果 BGPF 只有干净的文件名，那就保持 adv_prefix=""
    spectrum_baseline = process_folder(CLEAN_DIR, ADV_BASELINE_DIR, adv_prefix="adv_")
    spectrum_bgpf = process_folder(CLEAN_DIR, ADV_BGPF_DIR, adv_prefix="")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(spectrum_baseline, cmap='jet')
    axes[0].set_title('Average Perturbation Spectrum\n(Baseline: BSR)', fontsize=14)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(spectrum_bgpf, cmap='jet')
    axes[1].set_title('Average Perturbation Spectrum\n(Ours: BGPF)', fontsize=14)
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_EPS, format='eps', bbox_inches='tight')
    print(f"EPS figure successfully saved to: {OUTPUT_EPS}")
    plt.show()


if __name__ == '__main__':
    main()