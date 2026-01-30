import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from typing import List

from models import ModelRepository
from torch.utils.data import DataLoader, TensorDataset

from preprocess import AdvPNGDataset

# 新增：PPSP 透视/翻转/拉伸变换（分块）
from ppsp.perspective_rotate import _perspective_permutation_transform
# 新增：TIM
from ppsp.TIM import build_tim_kernel, tim_grad


class EnsembleModel(nn.Module):
    """集成模型，将多个模型的输出进行平均（按用户给定实现）。"""

    def __init__(self, models: List[nn.Module], device: torch.device):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models).to(device)
        self.device = device

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)

            # 处理不同类型的输出
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, list):
                output = output[0]

            # 确保所有输出形状一致
            if output.dim() == 1:
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                output = output.view(output.size(0), -1)

            outputs.append(output)

        # 对输出进行平均
        avg_output = torch.stack(outputs).mean(dim=0)
        return avg_output

    def eval(self):
        for model in self.models:
            model.eval()
        return self


def input_diversity(x, prob=0.5):
    """输入多样性增强（按用户给定实现）。"""
    if np.random.random() > prob:
        return x

    rnd = np.random.randint(299, 330)
    resized = F.interpolate(x, size=(rnd, rnd), mode='nearest')
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem + 1)
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem + 1)
    pad_right = w_rem - pad_left

    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return F.interpolate(padded, size=(299, 299), mode='nearest')


def get_model_output(model, x):
    """统一处理模型输出，确保返回二维张量 (batch_size, num_classes)（按用户给定实现）。"""
    output = model(x)

    if isinstance(output, tuple):
        output = output[0]
    elif isinstance(output, list):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")

    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)

    return output


def get_model_prediction(model, x):
    """从模型输出中获取预测结果（按用户给定实现）。"""
    output = get_model_output(model, x)
    return torch.argmax(output, dim=1).item()


def _apply_pf_copies(
    x: torch.Tensor,
    *,
    num_blocks_h: int,
    num_blocks_w: int,
    num_copies: int,
    distortion_scale: float,
    max_angle: float,
    flip_prob: float,
    stretch_factor: float,
) -> torch.Tensor:
    """对输入 x (B,C,H,W) 采样 num_copies 个『分块透视+旋转+（可选）块内水平翻转/拉伸』副本。

    返回：(B*num_copies, C, H, W)
    """
    return _perspective_permutation_transform(
        x,
        num_blocks_h=num_blocks_h,
        num_blocks_w=num_blocks_w,
        num_copies=num_copies,
        distortion_scale=float(distortion_scale),
        max_angle=float(max_angle),
        flip_prob=float(flip_prob),
        stretch_factor=float(stretch_factor),
    )


def attack(
    x,
    y,
    model,
    eps=16 / 255,
    iterations=10,
    mu=1.0,
    prob=0.5,
    *,
    use_pf: bool = False,
    num_blocks_h: int = 2,
    num_blocks_w: int = 2,
    num_copies: int = 20,
    distortion_scale: float = 0.26,
    max_angle_default: float = 0.0,
    flip_prob: float = 0.0,
    stretch_factor: float = 0.0,
    # --- TIM ---
    use_tim: bool = False,
    tim_kernel: int = 15,
    tim_sigma: float = 3.0,
):
    """MI-FGSM + DI +（可选）PPSP: 分块透视/翻转 + TIM: 高斯平滑梯度。

    - use_pf=False: 保持原脚本 DI-MI-FGSM 行为
    - use_pf=True : 在每次迭代里，对 x_div 采样 num_copies 个 PF 变换副本，
                    对副本取平均 loss 来估计梯度（提升可迁移性）
    - use_tim=True: 对每次迭代的梯度应用 TIM 高斯平滑
    """
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True)
    alpha = eps / iterations
    momentum = torch.zeros_like(x).to(device)

    # TIM kernel（每次攻击构建一次即可）
    tim_k = None
    if use_tim:
        # depthwise kernel: (C,1,k,k)
        tim_k = build_tim_kernel(kernlen=int(tim_kernel), nsig=float(tim_sigma), channels=int(x.shape[1]))

    for _ in range(iterations):
        x_adv.requires_grad = True

        # 1) DI
        x_div = input_diversity(x_adv, prob) if prob > 0 else x_adv

        # 2) PPSP PF copies（可选）
        if use_pf:
            x_aug = _apply_pf_copies(
                x_div,
                num_blocks_h=num_blocks_h,
                num_blocks_w=num_blocks_w,
                num_copies=num_copies,
                distortion_scale=distortion_scale,
                max_angle=max_angle_default,
                flip_prob=flip_prob,
                stretch_factor=stretch_factor,
            )
            y_expanded = y.repeat_interleave(num_copies)

            output = get_model_output(model, x_aug)
            loss = F.cross_entropy(output, y_expanded, reduction='mean')
        else:
            output = get_model_output(model, x_div)
            loss = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss, [x_adv])[0]

        # TIM: 平滑梯度（translation-invariant）
        if use_tim and tim_k is not None:
            grad = tim_grad(grad, kernel=tim_k)

        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = mu * momentum + grad

        x_adv = x_adv.detach() + alpha * torch.sign(momentum)

        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="4-model ensemble attack (must match provided implementation)")

    parser.add_argument('--output_adv_dir', default='./results/4block_flip_pf/images', type=str)
    parser.add_argument('--output_csv', default='./results/4block_flip_pf/results.csv', type=str)
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=1, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float)
    parser.add_argument('--prob', default=0.5, type=float)

    # --- PPSP: perspective + (optional) per-block horizontal flip / stretch ---
    parser.add_argument('--use_pf', action='store_true', help='enable PPSP perspective+flip transform (block-wise)')
    parser.set_defaults(use_pf=False)

    parser.add_argument('--num_blocks_h', default=2, type=int, help='number of blocks along height')
    parser.add_argument('--num_blocks_w', default=2, type=int, help='number of blocks along width')
    parser.add_argument('--num_copies', default=20, type=int, help='number of transformed copies per iter when use_pf')

    parser.add_argument('--distortion_scale', default=0.26, type=float, help='perspective distortion scale')
    parser.add_argument('--max_angle_default', default=0.0, type=float, help='max rotation angle inside PF transform')

    # 透视水平翻转（块内 flip）
    parser.add_argument('--flip_prob', default=0.0, type=float, help='probability of per-block horizontal flip (after perspective)')

    # （可选）块内拉伸
    parser.add_argument('--stretch_factor', default=0.0, type=float, help='max aspect stretch factor (0 disables)')

    # --- TIM (Translation-Invariant) ---
    parser.add_argument('--use_tim', dest='use_tim', action='store_true', help='enable TIM (gaussian smoothing on gradient)')
    parser.add_argument('--no_tim', dest='use_tim', action='store_false', help='disable TIM')
    parser.set_defaults(use_tim=False)
    parser.add_argument('--tim_kernel', default=7, type=int, help='TIM gaussian kernel size (odd), e.g. 15')
    parser.add_argument('--tim_sigma', default=3.0, type=float, help='TIM gaussian sigma, e.g. 3.0')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_repo = ModelRepository(device)

    # 四模型集成（按用户给定固定列表）
    model_names = ['tf2torch_inception_v3', 'tf2torch_inception_v4', 'tf2torch_resnet_v2_101', 'tf2torch_inc_res_v2']

    all_available_models = model_repo.get_all_model_names()
    print(f"Available models: {all_available_models}")

    ensemble_models = []
    for model_name in model_names:
        if model_name in all_available_models:
            model_info = model_repo.get_source_model(model_name)
            ensemble_models.append(model_info['model'])
            print(f"Added {model_name} to ensemble")
        else:
            print(f"Warning: {model_name} not found in available models")

    if not ensemble_models:
        raise ValueError("No models found for ensemble!")

    ensemble_model = EnsembleModel(ensemble_models, device).eval()
    print(f"Created ensemble model with {len(ensemble_models)} models")

    # 目标模型（迁移评测）：保持与 block_flip_pf.py 一致的 repo 覆盖范围
    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)
    print(f"\nSelected {len(target_models)} target models for testing")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    attack_params = {
        "eps": args.eps,
        "iterations": args.iterations,
        "mu": args.mu,
        "prob": args.prob,
        "use_pf": args.use_pf,
        "num_blocks_h": args.num_blocks_h,
        "num_blocks_w": args.num_blocks_w,
        "num_copies": args.num_copies,
        "distortion_scale": args.distortion_scale,
        "max_angle_default": args.max_angle_default,
        "flip_prob": args.flip_prob,
        "stretch_factor": args.stretch_factor,
        "use_tim": args.use_tim,
        "tim_kernel": args.tim_kernel,
        "tim_sigma": args.tim_sigma,
    }

    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    label_df = pd.read_csv(label_csv_path)

    # 过滤存在的文件
    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    # --- Step 1: 生成对抗样本（按用户给定 attack 实现）---
    orig_dataset = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(orig_dataset, batch_size=args.batchsize, shuffle=False)

    source_results = []
    adv_images_storage = []

    print(f"\n[Step 1/3] Attacking in Memory (Ensemble DI-MI-FGSM)...")
    for (x_batch, y_batch, filename_batch) in tqdm(loader, desc="Attacking"):
        x_batch = x_batch.to(device)
        # 注意：工程数据集标签在很多脚本里是 y+1，这里严格按你贴的 main：true_label = int(row['label'])
        # 但 AdvPNGDataset 具体返回的 y 是否需要 +1，以原工程习惯保持：y_batch+1
        y_batch = (y_batch + 1).to(device)

        # 原始预测（逐 batch 取一个 item 的 .item()，与用户实现一致）
        source_orig_pred = get_model_prediction(ensemble_model, x_batch)

        x_adv_batch = attack(x_batch, y_batch, ensemble_model, **attack_params)
        source_adv_pred = get_model_prediction(ensemble_model, x_adv_batch)

        adv_images_storage.append(x_adv_batch.cpu())

        # 逐样本记录（batchsize 可能 >1 时，pred 这里会取 batch 的第一个，保持与用户实现一致的 .item() 行为）
        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item())
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": int(source_orig_pred),
                "source_adv_pred": int(source_adv_pred),
                "source_attack_success": int(source_adv_pred) != true_label,
            })

    # 释放部分显存
    torch.cuda.empty_cache()

    # --- Step 2: 迁移评测（沿用 block_flip_pf 的 in-memory 扫描思路）---
    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    target_predictions = {name: [] for name in target_models.keys()}

    for model_name, model_info in target_models.items():
        print(f"  --> Testing target model: {model_name}")
        target_model = model_info['model']
        target_model.eval()

        model_preds = []
        with torch.no_grad():
            for [x_adv_b] in tqdm(adv_mem_loader, desc=f"Scanning {model_name}"):
                x_adv_b = x_adv_b.to(device)
                # 这里同样按用户实现获取预测：每个 batch 取 .item()，所以 batchsize>1 会丢信息
                # 为了不改变你要求的实现，保持这一行为：逐张跑。
                if x_adv_b.size(0) == 1:
                    pred = get_model_prediction(target_model, x_adv_b)
                    model_preds.append(pred)
                else:
                    for j in range(x_adv_b.size(0)):
                        pred = get_model_prediction(target_model, x_adv_b[j:j + 1])
                        model_preds.append(pred)

        target_predictions[model_name] = model_preds

        # 释放显存
        del target_model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    # --- Step 3: 保存图片与 CSV ---
    print(f"\n[Step 3/3] Saving results and images to disk...")

    os.makedirs(args.output_adv_dir, exist_ok=True)
    for idx, res in enumerate(source_results):
        fn = res["filename"]
        save_image(all_adv_tensors[idx], os.path.join(args.output_adv_dir, fn))

    final_rows = []
    model_success_counts = {name: 0 for name in target_models.keys()}

    for idx, res in enumerate(source_results):
        row = res.copy()
        for model_name in target_models.keys():
            pred = int(target_predictions[model_name][idx])
            fooled = (pred != res["true_label"])
            row[f"{model_name}_pred"] = pred
            row[f"{model_name}_fooled"] = fooled
            if fooled:
                model_success_counts[model_name] += 1
        final_rows.append(row)

    total_samples = len(source_results)
    source_success = sum(1 for r in source_results if r['source_attack_success'])
    print("\n" + "=" * 80)
    print("Summary of Attack Results")
    print("=" * 80)
    print(f"Ensemble model attack success rate: {source_success}/{total_samples} ({source_success / max(total_samples, 1) * 100:.1f}%)")

    for name, count in model_success_counts.items():
        print(f"  {name}: {count}/{total_samples} ({count / max(total_samples, 1) * 100:.1f}%)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(final_rows).to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
