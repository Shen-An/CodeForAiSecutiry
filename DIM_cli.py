import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms

from eval import eval_transferability
from models import ModelRepository, Normalize
from torch.utils.data import DataLoader, Dataset, TensorDataset



# --- 自定义数据集：从本地加载生成的 PNG 对抗样本 ---
class AdvPNGDataset(Dataset):
    def __init__(self, img_dir, label_df, transform):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        # 兼容文件名获取
        fn = row['filename']
        img_path = os.path.join(self.img_dir, fn)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 返回 image, label, filename
        return img, row['label'], fn



def mi_fgsm_attack(x, y, model, eps=16 / 255.0, iterations=20, mu=1.0, **kwargs):
    """
    MI-FGSM 攻击 (Momentum Iterative FGSM)
    目标：仅通过带动量的分类损失梯度进行攻击
    """
    model.eval()
    device = x.device

    # 1. 初始化
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)
    alpha_step = eps / iterations

    for i in range(iterations):
        x_adv.requires_grad = True

        # 2. 前向传播计算交叉熵损失
        output = model(x_adv)
        if isinstance(output, (tuple, list)):
            output = output[0]

        loss = F.cross_entropy(output, y)

        # 3. 反向传播获取梯度
        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.data

        # 4. 梯度的 L1 归一化 (MI-FGSM 标准操作)
        # 将当前步梯度除以其平均绝对值，使梯度在同一量级
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

        # 5. 动量累积：g_{t+1} = mu * g_t + grad_t
        momentum = mu * momentum + grad

        # 6. 符号更新：x_{t+1} = x_t + alpha * sign(momentum)
        x_adv = x_adv.detach() + alpha_step * torch.sign(momentum)

        # 7. 投影约束 (Clip)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()



# 输入多样性增强 (DIM) - Batched Version (Independent per image)
def input_diversity(x, prob=0.5):
    """
    对 Batch 中的每一张图像独立应用随机 Resize 和 Padding (DIM)。
    保证 Batchsize 仅仅是加速，每一张图的变换参数是独立的。
    """
    if prob <= 0:
        return x

    device = x.device
    B, C, H, W = x.shape

    # 我们需要在 [299, 329] 之间为每一张图随机选择一个尺寸
    # 生成 B 个随机目标尺寸
    rnd_scales = np.random.randint(299, 330, size=B)

    # 决定每一张图是否应用变换 (Probability check)
    apply_flags = np.random.random(B) > (1 - prob) # True means apply

    outputs = []

    for i in range(B):
        img = x[i:i+1] # keep 4dims: [1, C, H, W]

        if not apply_flags[i]:
             outputs.append(img)
             continue

        scale = rnd_scales[i]

        # 1. Resize
        resized = F.interpolate(img, size=(scale, scale), mode='nearest')

        # 2. Pad
        h_rem = 330 - scale
        w_rem = 330 - scale

        pad_top = np.random.randint(0, h_rem + 1)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem + 1)
        pad_right = w_rem - pad_left

        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # 3. Resize back to 299
        final = F.interpolate(padded, size=(299, 299), mode='nearest')
        outputs.append(final)

    return torch.cat(outputs, dim=0)


# MI-FGSM +dim
def dim_attack(x, y, model, eps=16 / 255, iterations=10, mu=1.0, prob=0.5):
    device = x.device
    x_adv = x.clone().detach()  # 停止梯度追踪detach
    alpha = eps / max(1, iterations);
    momentum = torch.zeros_like(x_adv, device=device);

    for _ in range(iterations):
        x_adv.requires_grad_(True)  # 启用对抗样本张量x_adv的梯度追踪功能

        x_div = input_diversity(x_adv, prob)

        output = get_model_output(model, x_div)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            output = output.view(output.size(0), -1)
        loss = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss, [x_adv])[0]  # grad [B,C,H,w],B 张样本、C通道、H×W 分辨率

        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = mu * momentum + grad;

        x_adv = x_adv.detach() + alpha * torch.sign(momentum)

        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()



def get_model_output(model, x):
    output = model(x);
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")
    return output;


def get_model_prediction(model, x):
    output = get_model_output(model, x)

    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)

    preds = torch.argmax(output, dim=1).cpu().numpy()

    return np.atleast_1d(preds)


def parse_args():
    parser = argparse.ArgumentParser(description="DIM attack")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/dim/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/dim/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--prob', default=0.5, type=float, help='input diversity probability')
    return parser.parse_args()




def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model_repo = ModelRepository(device)
    source_model_info = model_repo.get_source_model(args.model)
    source_model = source_model_info['model']

    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    label_df = pd.read_csv(label_csv_path)

    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    orig_dataset = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(orig_dataset, batch_size=args.batchsize, shuffle=False)

    source_results = []
    adv_images_storage = []

    print(f"\n[Step 1/3] Attacking in Memory...")

    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)

        x_adv_batch = dim_attack(x_batch, y_batch, source_model,
                                 eps=args.eps, iterations=args.iterations,
                                 mu=args.mu, prob=args.prob)

        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item())
            s_adv_idx = int(source_adv_preds[i])
            s_orig_idx = int(source_orig_preds[i])
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

    del source_model
    torch.cuda.empty_cache()
    eval_transferability(source_results, adv_images_storage,args,model_repo)
if __name__ == '__main__':
    torch.cuda.empty_cache();
    main()