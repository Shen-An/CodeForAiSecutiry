import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --- 核心逻辑：输入多样性增强 (DIM) ---
def input_diversity(x, prob=1.0):
    """
    对 Batch 中的每一张图像独立应用随机 Resize 和 Padding (DIM)。
    为了演示保存，默认概率设为 1.0。
    """
    if prob <= 0:
        return x

    device = x.device
    B, C, H, W = x.shape
    rnd_scales = np.random.randint(299, 330, size=B)
    apply_flags = np.random.random(B) > (1 - prob)

    outputs = []
    for i in range(B):
        img = x[i:i + 1]
        if not apply_flags[i]:
            outputs.append(img)
            continue

        scale = rnd_scales[i]
        # 1. Resize 到随机尺寸
        resized = F.interpolate(img, size=(scale, scale), mode='nearest')
        # 2. 随机填充回 330x330
        h_rem = 330 - scale
        w_rem = 330 - scale
        pad_top = np.random.randint(0, h_rem + 1)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem + 1)
        pad_right = w_rem - pad_left
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        # 3. Resize 回模型标准输入 299x299
        final = F.interpolate(padded, size=(299, 299), mode='nearest')
        outputs.append(final)

    return torch.cat(outputs, dim=0)


# --- 基础数据集 ---
class SimpleDataset(Dataset):
    def __init__(self, img_dir, label_df, transform):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        fn = self.label_df.iloc[idx]['filename']
        img = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, fn


def main():
    # 配置路径
    input_dir = '../data/images'
    csv_path = '../data/labels.csv'
    output_dir = '../results/dim_trans'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    label_df = pd.read_csv(csv_path)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = SimpleDataset(input_dir, label_df, transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"Applying Input Diversity and saving to {output_dir}...")

    for x_batch, filenames in tqdm(loader):
        # 执行变换
        # 这里 prob=1.0 确保每张图都被变换，方便观察效果
        x_dim = input_diversity(x_batch, prob=1.0)

        # 保存图片
        for i in range(x_dim.size(0)):
            save_path = os.path.join(output_dir, f"dim_{filenames[i]}")
            save_image(x_dim[i], save_path)


if __name__ == '__main__':
    main()