import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Tuple, Optional
from models import ModelRepository
from torch.utils.data import DataLoader, TensorDataset
import math

# 假设 ppsp 和 preprocess 包在你的路径中可用
# 如果没有这些包的源码，你需要确保相关文件在项目目录下
from ppsp.BSR import get_block_lengths_bsr, shuffle_rotate_bsr, BSR_transform, _save_transformed_copies
from ppsp.DIM import input_diversity
from ppsp.admix import admix
from ppsp.perspective_rotate import _rand_perspective_params, _homography_dlt, _warp_perspective_grid_sample, \
    _perspective_permutation_transform
from preprocess import AdvPNGDataset, get_model_output, get_model_prediction
from ppsp.TIM import build_tim_kernel, tim_grad


def parse_args():
    parser = argparse.ArgumentParser(description="PPSP attack - Debug Specific Batch")

    # --- 基础路径配置 ---
    parser.add_argument("--model", default='tf2torch_inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/PPSP/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/PPSP/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)

    # --- 核心攻击参数 ---
    parser.add_argument('--batchsize', default=2, type=int, help="Processing exactly 2 images as requested")
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')

    # --- 增强参数 (BSR / Diversity / Admix / TIM / SI) ---
    parser.add_argument("--num_blocks_h", default=2, type=int, help="number of blocks h")
    parser.add_argument("--num_blocks_w", default=2, type=int, help="number of blocks w")
    parser.add_argument("--num_copies", default=20, type=int, help="number of copies")

    parser.add_argument('--use_diversity', dest='use_diversity', action='store_true')
    parser.add_argument('--no_diversity', dest='use_diversity', action='store_false')
    parser.set_defaults(use_diversity=False)
    parser.add_argument('--diversity_prob', default=0.5, type=float)

    parser.add_argument('--use_admix', dest='use_admix', action='store_true')
    parser.set_defaults(use_admix=False)
    parser.add_argument("--portion", default=0.2, type=float)
    parser.add_argument("--admix_size", default=3, type=int)

    parser.add_argument('--bsr', action='store_true', help='use BSR')
    parser.set_defaults(bsr=False)

    parser.add_argument("--max_angle_bsr", default=0.2, type=float)
    parser.add_argument("--max_angle_default", default=0, type=float)
    parser.add_argument("--distortion_scale", default=0.2, type=float)
    parser.add_argument('--flip_prob', default=0.5, type=float)
    parser.add_argument('--stretch_factor', default=0.0, type=float)

    # --- SI & TIM ---
    parser.add_argument('--use_si', dest='use_si', action='store_true')
    parser.set_defaults(use_si=False)
    parser.add_argument('--si_scales', default='1,0.5,0.25,0.125,0.0625', type=str)

    parser.add_argument('--use_tim', dest='use_tim', action='store_true')
    parser.set_defaults(use_tim=False)
    parser.add_argument('--tim_kernel', default=7, type=int)
    parser.add_argument('--tim_sigma', default=3, type=float)

    # --- 调试与保存配置 (关键部分) ---
    # 默认开启保存第1次迭代的 Trans 图，以便观察效果
    parser.add_argument('--save_iter', default=1, type=int,
                        help='Save trans images at this iteration (set to 1 to see initial transform)')
    parser.add_argument('--save_trans_dir', default='./results/PPSP/trans_debug', type=str,
                        help='Where to save intermediate transforms')
    parser.add_argument('--save_iter5', action='store_true', help='Legacy compatibility')

    # 默认开启 One Batch 模式
    parser.add_argument('--one_batch', action='store_true', help='Stop after first batch')
    parser.set_defaults(one_batch=True)

    return parser.parse_args()


def _save_trans_images(
        x_trans: torch.Tensor,
        filenames: list[str],
        *,
        out_dir: str,
        iter_idx_1based: int,
):
    """保存变换前的图片（SI/Admix/Div之后，BSR/Perspective之前）。"""
    os.makedirs(out_dir, exist_ok=True)
    # Clamp to ensure valid image range
    x_trans = x_trans.detach().clamp(0, 1).cpu()

    # 注意：x_trans 的 batch size 可能是原始 batch 的倍数 (si_mult * admix_mult)
    # filenames 列表在传入前已经被扩充了，直接对应即可
    for i in range(min(len(filenames), x_trans.size(0))):
        fn = filenames[i]
        base, _ = os.path.splitext(fn)
        # 为了防止文件名冲突（因为是扩充的），加上索引
        save_path = os.path.join(out_dir, f"{base}_iter{iter_idx_1based:02d}_idx{i}_trans.png")
        save_image(x_trans[i], save_path)


def mifgsm_attack_ppsp(x, y, model, eps=16 / 255, iterations=10, mu=1.0,
                       num_blocks_h=2, num_blocks_w=2, *,
                       max_angle_bsr=0.2,
                       max_angle_default=0.2,
                       num_copies=20, use_diversity=True, use_admix=False,
                       portion=0.2, admix_size=3, diversity_prob=0.5,
                       flip_prob: float = 0.4,
                       stretch_factor: float = 0.0,
                       distortion_scale=0.06, bsr: bool = False,
                       save_iter: int | None = None,
                       save_trans_dir: str | None = None,
                       filenames: Optional[list[str]] = None,
                       use_si: bool = False,
                       si_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625),
                       use_tim: bool = False,
                       tim_kernel: int = 7,
                       tim_sigma: float = 3):
    alpha = eps / iterations
    x_adv = x.clone().requires_grad_(True)
    momentum = torch.zeros_like(x).to(x.device)

    tim_k = None
    if use_tim:
        tim_k = build_tim_kernel(kernlen=int(tim_kernel), nsig=float(tim_sigma), channels=int(x.size(1))).to(x.device)

    # --- 新增：处理第 0 次迭代的保存 ---
    if save_iter == 0 and save_trans_dir is not None and filenames is not None:
        os.makedirs(os.path.join(save_trans_dir, 'initial_state'), exist_ok=True)
        for idx, fn in enumerate(filenames):
            save_image(x[idx].cpu(), os.path.join(save_trans_dir, 'initial_state', f"{fn}_orig.png"))
    # ------------------------------
    for i in range(iterations):
        # 1. Input Diversity
        if use_diversity:
            x_transformed = input_diversity(x_adv, prob=diversity_prob)
        else:
            x_transformed = x_adv

        # 2. Admix
        admix_mult = 1
        if use_admix:
            x_transformed = admix(x_transformed, portion=portion, size=admix_size)
            admix_mult = int(admix_size)

        # 3. Scale Invariance (SI)
        si_mult = 1
        current_filenames = filenames  # Keep track of filenames for saving

        if use_si:
            si_mult = len(si_scales)
            x_transformed = torch.cat([x_transformed * float(s) for s in si_scales], dim=0)
            if current_filenames is not None:
                # 扩展文件名列表以匹配 SI 扩展后的 batch
                # 注意：Admix 是内部处理的，通常 Admix 后的 tensor batch 已经是 B*admix，
                # SI 在此基础上再乘，所以 filenames 需要先扩充 Admix 倍数 (虽然 admix 函数没返回文件名，但通常我们只关心原始对应关系)
                # 简单起见，这里假设 Admix 不改变 filenames 对应顺序（因为是对同一个图混入不同图），
                # 但为了 debug saving，我们只需简单的 repeat list
                expanded_filenames = []
                for s_idx in range(si_scales):
                    # 这一步逻辑较复杂，为简化，简单复制列表即可
                    # 真正的对应关系取决于 tensor cat 的顺序
                    pass
                current_filenames = list(current_filenames) * si_mult * admix_mult

        y_expanded = y.repeat(admix_mult * si_mult * num_copies)

        iter_1based = i + 1

        # --- DEBUG SAVING: Trans (Before BSR/Perspective) ---
        if (
                save_iter is not None
                and save_trans_dir is not None
                and current_filenames is not None
                and iter_1based == save_iter
        ):
            # 这里的 x_transformed 是进过 DIV, ADMIX, SI 的图
            # 这里的 batch size = B * admix_mult * si_mult
            # 我们临时造一个足够长的 filenames 列表来防止索引越界
            temp_filenames = list(filenames) * (x_transformed.size(0) // len(filenames) + 1)

            _save_trans_images(
                x_transformed,
                temp_filenames[:x_transformed.size(0)],
                out_dir=os.path.join(save_trans_dir, 'trans_pre_bsr'),
                iter_idx_1based=iter_1based,
            )

        # 4. BSR / Perspective Transform
        if bsr:
            x_aug = BSR_transform(
                x_transformed,
                num_blocks_h,
                num_blocks_w,
                max_angle_bsr,
                num_copies,
                distortion_scale=distortion_scale,
                flip_prob=flip_prob,
                stretch_factor=stretch_factor,
            )
        else:
            x_aug = _perspective_permutation_transform(
                x_transformed,
                num_blocks_h=num_blocks_h,
                num_blocks_w=num_blocks_w,
                num_copies=num_copies,
                distortion_scale=distortion_scale,
                max_angle=max_angle_default,
                flip_prob=flip_prob,
                stretch_factor=stretch_factor,
            )

        # --- DEBUG SAVING: Copies (After BSR/Perspective) ---
        if (
                save_iter is not None
                and save_trans_dir is not None
                and filenames is not None
                and iter_1based == save_iter
        ):
            # 保存所有的 augmented copies
            _save_transformed_copies(
                x_aug,
                filenames,  # 内部函数应该知道如何处理 copy 数量的扩充
                out_dir=os.path.join(save_trans_dir, 'copies_final'),
                iter_idx_1based=iter_1based,
                num_copies=num_copies * admix_mult * si_mult,  # 修正 num_copies 传递
            )

        # 5. Forward & Loss
        output = model(x_aug)
        if isinstance(output, (tuple, list)):
            output = output[0]

        loss = F.cross_entropy(output, y_expanded)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()
        grad = x_adv.grad.data

        # 6. TIM
        if use_tim and tim_k is not None:
            grad = tim_grad(grad, kernel=tim_k)

        # Normalize Gradient
        l1 = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8
        grad = grad / l1

        momentum = mu * momentum + grad

        with torch.no_grad():
            x_adv = x_adv + alpha * torch.sign(momentum)
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv


def main():
    args = parse_args()
    # 找到这一行
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 替换为以下强制检查代码：
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 错误：未检测到 GPU (CUDA)！请检查 PyTorch 版本或显卡驱动。")
    device = torch.device('cuda')

    print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0)})")

    # ---------------------------------------------------------
    # [用户修改区] 请在此处指定你要跑的那两张图片的完整文件名
    # ---------------------------------------------------------
    SPECIFIC_FILENAMES = [
        "ILSVRC2012_val_00013528.png",  # 示例：替换为你的文件名
        "ILSVRC2012_val_00000470.png"  # 示例：替换为你的文件名
    ]
    print(f"\n[Config] Targeted Image Batch: {SPECIFIC_FILENAMES}")
    print(f"[Config] Save Trans Images: {'Yes' if args.save_iter > 0 else 'No'} (Iter: {args.save_iter})")
    print(f"[Config] Output Dir: {args.output_adv_dir}")
    print(f"[Config] Trans Debug Dir: {args.save_trans_dir}")
    # ---------------------------------------------------------

    if (args.save_iter and args.save_iter > 0) or args.save_iter5:
        os.makedirs(args.save_trans_dir, exist_ok=True)

    # 模型加载
    model_repo = ModelRepository(device)
    source_model_info = model_repo.get_source_model(args.model)
    source_model = source_model_info['model']

    # 数据集准备
    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')

    # 读取 CSV 并过滤
    full_label_df = pd.read_csv(label_csv_path)

    # === 关键修改：只保留指定的文件名 ===
    label_df = full_label_df[full_label_df['filename'].isin(SPECIFIC_FILENAMES)].copy()

    # 检查文件是否存在
    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    missing_files = label_df[~label_df['exists']]['filename'].tolist()
    if missing_files:
        print(f"Error: The following specific files were not found in {img_root}: {missing_files}")
        return
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    if len(label_df) == 0:
        print("Error: No matching images found. Please check SPECIFIC_FILENAMES and labels.csv.")
        return

    print(f"[Data] Loaded {len(label_df)} images for processing.")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # 这里 batch_size 强制使用 args.batchsize (默认2)
    dataset = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    source_results = []

    try:
        si_scales = tuple(float(s.strip()) for s in str(args.si_scales).split(','))
    except Exception as e:
        raise ValueError(f"Invalid --si_scales: {args.si_scales}") from e

    # --- 攻击循环 ---
    print(f"\n[Step 1] Starting Attack on Specific Batch...")

    # 应该只会循环一次（如果只有2张图且 batchsize=2）
    for batch_idx, (x_batch, y_batch, filename_batch) in enumerate(tqdm(loader, desc="Attacking")):
        x_batch = x_batch.to(device)
        y_batch = (y_batch + 1).to(device)

        print(f"\nProcessing batch files: {filename_batch}")

        source_orig_preds = get_model_prediction(source_model, x_batch)

        x_adv_batch = mifgsm_attack_ppsp(
            x_batch,
            y_batch,
            source_model,
            eps=args.eps,
            iterations=args.iterations,
            mu=args.mu,
            num_blocks_h=args.num_blocks_h,
            num_blocks_w=args.num_blocks_w,
            max_angle_bsr=args.max_angle_bsr,
            max_angle_default=args.max_angle_default,
            num_copies=args.num_copies,
            use_diversity=args.use_diversity,
            use_admix=args.use_admix,
            portion=args.portion,
            admix_size=args.admix_size,
            diversity_prob=args.diversity_prob,
            flip_prob=args.flip_prob,
            stretch_factor=args.stretch_factor,
            distortion_scale=args.distortion_scale,
            bsr=args.bsr,
            save_iter=args.save_iter,
            save_trans_dir=args.save_trans_dir,
            filenames=list(filename_batch),
            use_si=args.use_si,
            si_scales=si_scales,
            use_tim=args.use_tim,
            tim_kernel=args.tim_kernel,
            tim_sigma=args.tim_sigma,
        )

        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        # 保存对抗样本到硬盘
        os.makedirs(args.output_adv_dir, exist_ok=True)
        for i in range(x_adv_batch.size(0)):
            fn = filename_batch[i]
            save_path = os.path.join(args.output_adv_dir, fn)
            save_image(x_adv_batch[i].detach().cpu(), save_path)

            true_label = int(y_batch[i].item())
            s_adv_idx = int(source_adv_preds[i])
            s_orig_idx = int(source_orig_preds[i])

            source_results.append({
                "filename": fn,
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

        # 因为设定了 one_batch=True，这里跑完直接 break
        if args.one_batch:
            print(f"\n[Done] One batch processed. Stopping.")
            break

    # 简单打印结果
    print("\n--- Summary ---")
    for res in source_results:
        status = "SUCCESS" if res["source_attack_success"] else "FAILED"
        print(f"File: {res['filename']} | True: {res['true_label']} | Adv Pred: {res['source_adv_pred']} -> {status}")

    print(f"\nAdversarial images saved to: {args.output_adv_dir}")
    print(f"Transformed images saved to: {args.save_trans_dir}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()