import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from models import *
import argparse
import gc
import os

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
    """返回 0-based 预测类标（numpy int array），与 DIM_cli.py 一致。"""
    output = get_model_output(model, x)

    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)

    preds = torch.argmax(output, dim=1).detach().cpu().numpy()
    return np.atleast_1d(preds)

def attack_batch(x, y0_based, model, eps=16/255, iterations=10, mu=1.0):
    """对 batch 进行 TIM 攻击。

    注意：labels.csv 的 label 通常是 0-based；本仓库评估打印/保存时会 +1 适配。
    攻击内部仍用 0-based label 做 cross_entropy。
    """
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True)

    alpha = eps / max(1, iterations)
    momentum = torch.zeros_like(x).to(device)
    kernel_device = stack_kernel.to(device)

    for _ in range(iterations):
        x_adv.requires_grad_(True)

        output = get_model_output(model, x_adv)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            output = output.view(output.size(0), -1)

        loss = F.cross_entropy(output, y0_based)
        grad = torch.autograd.grad(loss, [x_adv])[0]

        grad_conv = F.conv2d(grad, kernel_device, padding=7, groups=3)
        grad_conv = grad_conv / (torch.mean(torch.abs(grad_conv), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = mu * momentum + grad_conv

        x_adv = x_adv.detach() + alpha * torch.sign(momentum)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()

def parse_args():
    parser = argparse.ArgumentParser(description="TIM attack (Transferability Evaluation)")
    parser.add_argument("--source_model", default='tf2torch_resnet_v2_101', type=str, help="source model name in ModelRepository")
    parser.add_argument('--output_adv_dir', default='./results/tim/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/tim/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    return parser.parse_args()

class AdvPNGDataset(torch.utils.data.Dataset):
    """从本地 images 目录按 labels.csv 加载图片，返回 (tensor, label, filename)。"""

    def __init__(self, img_dir, label_df, transform):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        fn = row['filename']
        img_path = os.path.join(self.img_dir, fn)
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['label']), fn

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型仓库（保持原有逻辑：不改模型加载方式/权重来源）
    model_repo = ModelRepository(device)

    source_model_info = model_repo.get_source_model(args.source_model)
    source_model = source_model_info['model']
    source_model.eval()

    # 数据读取（本地评估逻辑）
    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')

    import pandas as pd
    label_df = pd.read_csv(label_csv_path)

    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    from torch.utils.data import DataLoader, TensorDataset
    orig_dataset = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(orig_dataset, batch_size=args.batchsize, shuffle=False)

    source_results = []
    adv_images_storage = []

    print(f"\n[Step 1/3] Attacking in Memory...")

    for x_batch, y_batch0, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch = x_batch.to(device)
        y_batch0 = y_batch0.to(device)

        # 预测（0-based）
        source_orig_preds0 = get_model_prediction(source_model, x_batch)

        x_adv_batch = attack_batch(
            x_batch, y_batch0, source_model,
            eps=args.eps, iterations=args.iterations, mu=args.mu
        )

        source_adv_preds0 = get_model_prediction(source_model, x_adv_batch)

        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch0[i].item()) + 1
            s_adv_idx = int(source_adv_preds0[i]) + 1
            s_orig_idx = int(source_orig_preds0[i]) + 1
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

    # 释放源模型显存
    del source_model
    torch.cuda.empty_cache()

    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    # 目标模型：除去源模型
    target_names = [name for name in all_model_names if name != args.source_model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")
        current_model_info = model_repo.load_single_model(model_name)
        model = current_model_info['model']
        model.eval()

        model_preds0 = []
        with torch.no_grad():
            for [x_adv_batch] in tqdm(adv_mem_loader, desc=f"Scanning {model_name}"):
                x_adv_batch = x_adv_batch.to(device)
                preds0 = get_model_prediction(model, x_adv_batch)
                model_preds0.extend(preds0)

        target_predictions[model_name] = model_preds0

        del model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[Step 3/3] Saving results and images to disk...")

    from torchvision.utils import save_image
    os.makedirs(args.output_adv_dir, exist_ok=True)
    for idx, res in enumerate(source_results):
        fn = res["filename"]
        save_image(all_adv_tensors[idx], os.path.join(args.output_adv_dir, fn))

    final_rows = []
    model_success_counts = {name: 0 for name in target_names}

    for idx, res in enumerate(source_results):
        row = res.copy()
        for model_name in target_names:
            pred0 = int(target_predictions[model_name][idx])
            pred = pred0 + 1  # 适配本地数据集（与 DIM_cli.py 一致）
            fooled = (pred != res["true_label"])
            row[f"{model_name}_pred"] = pred
            row[f"{model_name}_fooled"] = fooled
            if fooled:
                model_success_counts[model_name] += 1
        final_rows.append(row)

    total_samples = len(source_results)
    source_rate = (sum(1 for r in source_results if r['source_attack_success']) / total_samples * 100) if total_samples else 0.0
    print(f"\nSource Model ({args.source_model}) Success Rate: {source_rate:.1f}%")
    for name, count in model_success_counts.items():
        print(f"  {name}: {count}/{total_samples} ({(count / total_samples * 100) if total_samples else 0.0:.1f}%)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(final_rows).to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()