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

from ppsp.BSR import get_block_lengths_bsr, shuffle_rotate_bsr, BSR_transform, _save_transformed_copies
from ppsp.DIM import input_diversity
from ppsp.admix import admix
from ppsp.perspective_rotate import _rand_perspective_params, _homography_dlt, _warp_perspective_grid_sample, \
    _perspective_permutation_transform
from preprocess import AdvPNGDataset, get_model_output, get_model_prediction
from ppsp.TIM import build_tim_kernel, tim_grad


def parse_args():
    parser = argparse.ArgumentParser(description="PPSP attack")
    parser.add_argument("--model", default='tf2torch_inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/PPSP/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/PPSP/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--diversity_prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument("--num_blocks_h", default=2, type=int, help="number of blocks h")
    parser.add_argument("--num_blocks_w", default=2, type=int, help="number of blocks w")
    parser.add_argument("--num_copies", default=20, type=int, help="number of copies")

    # NOTE: argparse 的 type=bool 不可靠（传入字符串 'False' 也会变 True），改用 store_true/store_false
    parser.add_argument('--use_diversity', dest='use_diversity', action='store_true', help='enable diversity')
    parser.add_argument('--no_diversity', dest='use_diversity', action='store_false', help='disable diversity')
    parser.set_defaults(use_diversity=False)

    parser.add_argument('--use_admix', dest='use_admix', action='store_true', help='enable admix')
    parser.add_argument('--no_admix', dest='use_admix', action='store_false', help='disable admix')
    parser.set_defaults(use_admix=False)

    parser.add_argument("--portion", default=0.2, type=float, help="portion admix")
    parser.add_argument("--admix_size", default=3, type=int, help="admix size")

    #参数开关：
    #   --bsr      => True, 使用 BSR（当前实现：随机置换 + 透视 + 旋转）
    #   不加参数   => False, 默认分块透视 + 分块旋转
    parser.add_argument('--bsr', action='store_true', help='use BSR (permute+perspective+rotation)')
    parser.set_defaults(bsr=False)

    # 旋转角度拆分：BSR 与默认透视分块使用不同 max_angle，便于组件化开关
    parser.add_argument("--max_angle_bsr", default=0.2, type=float, help="maximum rotation angle for BSR")
    parser.add_argument("--max_angle_default", default=1.2, type=float, help="maximum rotation angle for default (non-BSR)")

    # Perspective params
    parser.add_argument("--distortion_scale", default=0.26, type=float, help="perspective distortion scale (BCTOT D)")

    # 保存第 N 次迭代的中间副本/变换图
    # 约定：save_iter<=0 视为关闭；>0 则在该迭代保存
    # 为了兼容旧用法，保留 --save_iter5，但默认不需要它
    parser.add_argument('--save_iter5', action='store_true', help='(deprecated) enable save_iter (kept for compatibility)')
    parser.add_argument('--save_iter', default=0, type=int, help='which iteration to dump (1-based); <=0 disables')
    parser.add_argument('--save_trans_dir', default='./results/PPSP/trans_debug', type=str, help='dir to save intermediate transformed images')

    # 调试：只跑一个 batch，跑完就退出
    parser.add_argument('--one_batch', action='store_true', help='debug: only attack the first batch and exit after saving')
    parser.set_defaults(one_batch=False)

    # 水平翻转
    parser.add_argument('--flip_prob', default=0.6, type=float, help='probability to apply random horizontal flip before block transforms')

    # --- SI (Scale-Invariance) ---
    # 说明：
    # - 开启后使用 SI-MI-FGSM 的多尺度梯度估计（在 PPSP 的变换采样之后做 loss/grad）。
    # - 默认关闭，保持原 MI-FGSM 行为一致。
    parser.add_argument('--use_si', dest='use_si', action='store_true', help='enable SI (scale-invariance) gradient')
    parser.add_argument('--no_si', dest='use_si', action='store_false', help='disable SI gradient')
    parser.set_defaults(use_si=False)
    parser.add_argument(
        '--si_scales',
        default='1,0.5,0.25,0.125,0.0625',
        type=str,
        help='comma-separated SI scales, e.g. "1,0.5,0.25"',
    )

    # --- TIM (Translation-Invariant) ---
    parser.add_argument('--use_tim', dest='use_tim', action='store_true', help='enable TIM (gaussian smoothing on gradient)')
    parser.add_argument('--no_tim', dest='use_tim', action='store_false', help='disable TIM')
    parser.set_defaults(use_tim=False)
    parser.add_argument('--tim_kernel', default=7, type=int, help='TIM gaussian kernel size (odd), e.g. 7')
    parser.add_argument('--tim_sigma', default=3, type=float, help='TIM gaussian sigma, e.g. 3')

    return parser.parse_args()



def _save_trans_images(
    x_trans: torch.Tensor,
    filenames: list[str],
    *,
    out_dir: str,
    iter_idx_1based: int,
):
    """保存 trans的图像，batch 维为 B。"""
    os.makedirs(out_dir, exist_ok=True)
    x_trans = x_trans.detach().clamp(0, 1).cpu()

    for i, fn in enumerate(filenames):
        base, _ = os.path.splitext(fn)
        save_path = os.path.join(out_dir, f"{base}_iter{iter_idx_1based:02d}_trans.png")
        save_image(x_trans[i], save_path)


def mifgsm_attack_ppsp(x, y, model, eps=16 / 255, iterations=10, mu=1.0,
                      num_blocks_h=2, num_blocks_w=2, *,
                      max_angle_bsr=0.2,
                      max_angle_default=0.2,
                      num_copies=20, use_diversity=True, use_admix=False,
                      portion=0.2, admix_size=3, diversity_prob=0.5,
                      flip_prob: float = 0.4,
                      distortion_scale=0.06, bsr: bool = False,
                      save_iter: int | None = None,
                      save_trans_dir: str | None = None,
                      filenames: Optional[list[str]] = None,
                      use_si: bool = False,
                      si_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625),
                      use_tim: bool = False,
                      tim_kernel: int = 7,
                      tim_sigma: float = 3):
    """MI-FGSM + (BSR 或 默认分块透视+旋转)。

    - bsr=True:  BSR 路径（置换 + 分块透视 + 分块旋转），旋转角度由 max_angle_bsr 控制
    - bsr=False: 默认分块透视 + 分块旋转，旋转角度由 max_angle_default 控制

    save_iter: 若给定(1-based)，则在该次迭代保存 x_aug 的所有 num_copies 变换副本。

    use_si:
      True  => 使用 SI-MI-FGSM 的多尺度梯度估计（对 x_adv 求梯度，内部做 x_adv*scale）
      False => 使用原先单尺度 loss.backward() 的梯度
    """
    alpha = eps / iterations
    x_adv = x.clone().requires_grad_(True)
    momentum = torch.zeros_like(x).to(x.device)

    # 仅构建一次 TIM kernel，避免每次迭代重复开销
    tim_k = None
    if use_tim:
        tim_k = build_tim_kernel(kernlen=int(tim_kernel), nsig=float(tim_sigma), channels=int(x.size(1))).to(x.device)

    for i in range(iterations):
               
        if use_diversity:
            # input_diversity 内部每次调用都会采样一次随机 resize/pad，所以直接对整个 x_aug 调用即可
            # （batch 内每张图共享同一次 rnd/pad 采样是 DI 的常见实现；如果你要 per-image 独立采样，需要改 DIM 实现）
            x_transformed = input_diversity(x_adv, prob=diversity_prob)
        else:
            x_transformed = x_adv

        # 水平翻转：在进入 BSR/默认分块变换前按概率执行
        if flip_prob > 0.0 and torch.rand(1, device=x_transformed.device).item() < flip_prob:
            # x: (B,C,W,H) 这里 H 维在 dim=3，对其做翻转实现水平翻转
            x_transformed = torch.flip(x_transformed, dims=[3])

        # admix 放在 SI 之前：先对原 batch 做混合，再做多尺度复制
        admix_mult = 1
        if use_admix:
            x_transformed = admix(x_transformed, portion=portion, size=admix_size)
            admix_mult = int(admix_size)

        # --- SI (Scale-Invariance) 放在 BSR 创建副本前 ---
        si_mult = 1
        if use_si:
            si_mult = len(si_scales)
            # 直接按 scale 乘像素值并在 batch 维拼接，得到 (B*si_mult*admix_mult, C, H, W)
            x_transformed = torch.cat([x_transformed * float(s) for s in si_scales], dim=0)
            if filenames is not None:
                filenames = list(filenames) * si_mult

        # 此时 x_transformed 的 batch = B * admix_mult * si_mult
        # 后续 BSR/透视置换会再复制 num_copies 次，所以标签也要匹配 (admix_mult*si_mult*num_copies)
        y_expanded = y.repeat(admix_mult * si_mult * num_copies)

        iter_1based = i + 1
        if (
            save_iter is not None
            and save_trans_dir is not None
            and filenames is not None
            and iter_1based == save_iter
        ):
            # 保存 trans（进入 BSR/透视置换之前）
            _save_trans_images(
                x_transformed,
                filenames,
                out_dir=os.path.join(save_trans_dir, 'trans'),
                iter_idx_1based=iter_1based,
            )

        if bsr:
            x_aug = BSR_transform(
                x_transformed,
                num_blocks_h,
                num_blocks_w,
                max_angle_bsr,
                num_copies,
                distortion_scale=distortion_scale,
            )
        else:
            x_aug = _perspective_permutation_transform(
                x_transformed,
                num_blocks_h=num_blocks_h,
                num_blocks_w=num_blocks_w,
                num_copies=num_copies,
                distortion_scale=distortion_scale,
                max_angle=max_angle_default,
            )

 
        # dump intermediate transformed copies at iteration save_iter
        if (
            save_iter is not None
            and save_trans_dir is not None
            and filenames is not None
            and iter_1based == save_iter
        ):
            _save_transformed_copies(
                x_aug,
                filenames,
                out_dir=os.path.join(save_trans_dir, 'copies'),
                iter_idx_1based=iter_1based,
                num_copies=num_copies,
            )

        # 前向传播
        output = model(x_aug)

        # 处理不同输出类型
        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected output type: {type(output)}")

        loss = F.cross_entropy(output, y_expanded)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        # SI 已经用于“生成多尺度副本给 BSR”，这里保持单次 backward
        loss.backward()
        grad = x_adv.grad.data

        # TIM：对梯度做高斯平滑（Translation-Invariant）
        if use_tim and tim_k is not None:
            grad = tim_grad(grad, kernel=tim_k)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 如果启用保存，提前创建目录，确保你能看到 ./results/BSR/trans_debug
    if (args.save_iter and args.save_iter > 0) or args.save_iter5:
        os.makedirs(args.save_trans_dir, exist_ok=True)

    # 初始化模型仓库
    model_repo = ModelRepository(device)

    # --- 1. 攻击阶段：在内存中生成 ---
    source_model_info = model_repo.get_source_model(args.model)
    source_model = source_model_info['model']
    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    label_df = pd.read_csv(label_csv_path)

    # 过滤文件逻辑保持不变
    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    orig_dataset = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(orig_dataset, batch_size=args.batchsize, shuffle=False)

    source_results = []
    adv_images_storage = []  # 用于暂存对抗样本 (CPU Tensor)

    print(f"\n[Step 1/3] Attacking in Memory...")

    # 解析 si_scales（逗号分隔）
    try:
        si_scales = tuple(float(s.strip()) for s in str(args.si_scales).split(','))
    except Exception as e:
        raise ValueError(f"Invalid --si_scales: {args.si_scales}") from e

    for batch_idx, (x_batch, y_batch, filename_batch) in enumerate(tqdm(loader, desc="Attacking")):
        x_batch = x_batch.to(device)
        y_batch = (y_batch+1).to(device)

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
            distortion_scale=args.distortion_scale,
            bsr=args.bsr,
            save_iter=(args.save_iter if args.save_iter > 0 else (5 if args.save_iter5 else None)),
            save_trans_dir=(args.save_trans_dir if (args.save_iter > 0 or args.save_iter5) else None),
            filenames=list(filename_batch),
            use_si=args.use_si,
            si_scales=si_scales,
            use_tim=args.use_tim,
            tim_kernel=args.tim_kernel,
            tim_sigma=args.tim_sigma,
        )

        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item())
            s_adv_idx = int(source_adv_preds[i])
            s_orig_idx = int(source_orig_preds[i])
            print(true_label, s_adv_idx)
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

        # 调试：只 attack 一个 batch 就退出（保存会在 mifgsm_attack_PPSP 内部触发）
        if args.one_batch:
            print("[DEBUG] --one_batch enabled: stopping after first batch.")
            # 只保存本 batch 的对抗样本到 output_adv_dir，方便对照
            os.makedirs(args.output_adv_dir, exist_ok=True)
            for i in range(x_adv_batch.size(0)):
                save_image(x_adv_batch[i].detach().cpu(), os.path.join(args.output_adv_dir, filename_batch[i]))
            return

    # 彻底释放源模型显存
    del source_model
    torch.cuda.empty_cache()

    # --- 2. 验证阶段：直接使用内存中的 Tensor ---
    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    # 将 List 转换为单个大 Tensor，并构建简单的 TensorDataset
    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")

        # 假设这里依然通过 repo 加载
        current_model_info = model_repo.get_source_model(model_name)
        model = current_model_info['model']
        model.eval()

        model_preds = []
        with torch.no_grad():
            for [x_adv_batch] in tqdm(adv_mem_loader, desc=f"Scanning {model_name}"):
                x_adv_batch = x_adv_batch.to(device)
                preds = get_model_prediction(model, x_adv_batch)
                model_preds.extend(preds)

        target_predictions[model_name] = model_preds

        # 释放显存
        del model
        torch.cuda.empty_cache()
        gc.collect()  # 强制清理 CPU 内存引用
        torch.cuda.empty_cache()  # 清理显存

    # --- 3. 汇总结果与最后保存对抗样本到磁盘 ---
    print(f"\n[Step 3/3] Saving results and images to disk...")

    # 保存图片
    os.makedirs(args.output_adv_dir, exist_ok=True)
    for idx, res in enumerate(source_results):
        fn = res["filename"]
        # 从大 Tensor 中提取对应的图片并保存
        save_image(all_adv_tensors[idx], os.path.join(args.output_adv_dir, fn))

    # 统计与保存 CSV (逻辑保持不变)
    final_rows = []
    model_success_counts = {name: 0 for name in target_names}
    for idx, res in enumerate(source_results):
        row = res.copy()
        for model_name in target_names:
            pred = int(target_predictions[model_name][idx])
            fooled = (pred != res["true_label"])
            row[f"{model_name}_pred"] = pred
            row[f"{model_name}_fooled"] = fooled
            if fooled:
                model_success_counts[model_name] += 1
        final_rows.append(row)

    # 打印总结
    total_samples = len(source_results)
    source_rate = sum(1 for r in source_results if r['source_attack_success']) / total_samples * 100
    print(f"\nSource Model ({args.model}) Success Rate: {source_rate:.1f}%")
    for name, count in model_success_counts.items():
        print(f"  {name}: {count}/{total_samples} ({count / total_samples * 100:.1f}%)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(final_rows).to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")


if __name__ == '__main__':
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    main()