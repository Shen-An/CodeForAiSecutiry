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
from torchvision.transforms import functional as TF
from models import ModelRepository
from torch.utils.data import DataLoader,  TensorDataset
import math

from preprocess import AdvPNGDataset, get_model_output, load_source_model, get_model_prediction


def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def parse_args():
    parser = argparse.ArgumentParser(description="BSR attack")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/BSR/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/BSR/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--diversity_prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument("--num_blocks_h",default=2,type=int,help="number of blocks h")
    parser.add_argument("--num_blocks_w",default=2,type=int,help="number of blocks w")
    parser.add_argument("--max_angle",default=2,type=int,help="maximum angle")
    parser.add_argument("--num_copies",default=20,type=int,help="number of copies")
    parser.add_argument("--use_diversity",default=True,type=bool,help="use diversity");
    parser.add_argument("--use_admix",default=False,type=bool,help="use admix");
    parser.add_argument("--portion",default=0.2,help="portion admix");
    parser.add_argument("--admix_size",default=3,help="admix size");
    return parser.parse_args()


def admix(x, portion=0.2, size=3):
    """混合输入变换"""
    indices = torch.randperm(x.size(0))
    admixed = []
    for _ in range(size):
        admixed_x = x + portion * x[indices]
        admixed.append(admixed_x)
    return torch.cat(admixed, dim=0)


def input_diversity(x, image_width=299, image_resize=331, prob=0.5):
    """输入多样性变换"""
    if torch.rand(1).item() < prob:
        # 随机调整大小
        rnd = torch.randint(image_width, image_resize + 1, (1,)).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='nearest')

        # 随机填充
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem + 1, (1,)).item()
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        padded = F.interpolate(padded, size=(image_width, image_width), mode='nearest')
        return padded
    else:
        return x


def _rand_perspective_params(block_h: int, block_w: int, distortion_scale: float, device):
    """为单个块生成随机透视变换的(归一化)四角坐标。

    采用 BCTOT 的做法：对四个角点加入噪声 Δ∈[-δ, δ]，其中
    δ = distortion_scale * block_dimension。
    返回：src_pts(4,2), dst_pts(4,2) in [-1, 1] normalized coords.
    """
    # 注意：这里的 (x, y) 使用像素坐标系
    src = torch.tensor(
        [[0.0, 0.0], [block_w - 1.0, 0.0], [block_w - 1.0, block_h - 1.0], [0.0, block_h - 1.0]],
        device=device,
        dtype=torch.float32,
    )

    # 分别按 w/h 维度设定噪声尺度，避免长宽比不同导致过度形变
    dx = distortion_scale * block_w
    dy = distortion_scale * block_h
    noise = torch.empty((4, 2), device=device, dtype=torch.float32)
    noise[:, 0].uniform_(-dx, dx)
    noise[:, 1].uniform_(-dy, dy)

    dst = src + noise

    # 夹紧到块内，保证不会把有效区域拉出识别边界（语义一致性）
    dst[:, 0].clamp_(0.0, block_w - 1.0)
    dst[:, 1].clamp_(0.0, block_h - 1.0)

    # 转为 grid_sample 需要的 [-1, 1] 归一化坐标
    def normalize_xy(pts):
        x = pts[:, 0] / max(1.0, (block_w - 1.0)) * 2.0 - 1.0
        y = pts[:, 1] / max(1.0, (block_h - 1.0)) * 2.0 - 1.0
        return torch.stack([x, y], dim=1)

    return normalize_xy(src), normalize_xy(dst)


def _homography_dlt(src_n: torch.Tensor, dst_n: torch.Tensor) -> torch.Tensor:
    """DLT 求解单应性 H（3x3），满足 dst ~ H * src。

    src_n/dst_n: (4,2) 归一化坐标（[-1,1]）。
    返回 H: (3,3)
    """
    # 构造 8x9 线性方程组 A h = 0
    A_rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        x, y, u, v = x.item(), y.item(), u.item(), v.item()
        A_rows.append([ -x, -y, -1,   0,  0,  0,  u*x,  u*y,  u])
        A_rows.append([  0,  0,  0,  -x, -y, -1,  v*x,  v*y,  v])
    A = torch.tensor(A_rows, device=src_n.device, dtype=torch.float32)

    # SVD 取最小奇异值对应的向量
    _, _, Vh = torch.linalg.svd(A)
    h = Vh[-1, :]
    H = h.view(3, 3)
    # 标准化
    return H / (H[2, 2] + 1e-8)


def _warp_perspective_grid_sample(block: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """用 grid_sample 对单张 block 做透视变换。

    block: (1,C,H,W)
    H: (3,3) 将 src -> dst 的单应。为了“输出为 dst，采样 src”，需要用 H^{-1}。
    """
    B, C, Hh, Ww = block.shape
    device = block.device
    dtype = block.dtype

    # 生成输出网格 (dst) 的归一化坐标
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, Hh, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, Ww, device=device, dtype=torch.float32),
        indexing='ij',
    )
    ones = torch.ones_like(xs)
    grid_h = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).t()  # (3, H*W)

    H_inv = torch.linalg.inv(H)
    src_h = H_inv @ grid_h
    src_h = src_h / (src_h[2:3, :] + 1e-8)
    x_src = src_h[0, :].view(Hh, Ww)
    y_src = src_h[1, :].view(Hh, Ww)

    grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0).to(dtype)  # (1,H,W,2)

    return F.grid_sample(block, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def shuffle_rotate_bsr(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2):
    """BSR-Perspective：保持块位置不变，对每块做随机透视 + 旋转。

    相比随机置换块位置（破坏语义导致 gradient disorientation），
    透视扰动在局部保持结构一致性，更利于稳定期望梯度。
    """
    batch_size, channels, w, h = x.shape

    # 获取分块长度
    width_length = get_block_lengths_bsr(w, num_blocks_h)
    height_length = get_block_lengths_bsr(h, num_blocks_w)

    # 不再做 permute，直接按原顺序分块
    x_split_w = torch.split(x, width_length, dim=2)

    # 逐块做: perspective -> rotate
    processed_strips = []
    distortion_scale = 0.10  #  推荐的最优 D

    for w_block in x_split_w:
        h_blocks = torch.split(w_block, height_length, dim=3)
        processed_blocks = []

        for block in h_blocks:
            # block: (B,C,h_blk,w_blk)
            bH, bW = block.shape[2], block.shape[3]
            if bH <= 1 or bW <= 1:
                processed_blocks.append(block)
                continue

            # 为每个样本生成独立透视 + 旋转
            out_block = block.clone()
            angles = torch.clamp(torch.randn(batch_size, device=x.device) * 0.05, -max_angle, max_angle)

            for i in range(batch_size):
                single = block[i:i + 1]

                # 1) perspective
                src_n, dst_n = _rand_perspective_params(bH, bW, distortion_scale, device=x.device)
                H_mat = _homography_dlt(src_n, dst_n)
                single = _warp_perspective_grid_sample(single, H_mat)

                # 2) rotation (保留原逻辑)
                angle = angles[i]
                angle_matrix = torch.tensor(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0]],
                    dtype=torch.float32,
                    device=x.device,
                ).unsqueeze(0)
                grid = F.affine_grid(angle_matrix, single.size(), align_corners=False)
                single = F.grid_sample(single, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

                out_block[i:i + 1] = single

            processed_blocks.append(out_block)

        processed_strips.append(torch.cat(processed_blocks, dim=3))

    return torch.cat(processed_strips, dim=2)


def BSR_transform(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2, num_copies=20):
    """BSR变换：创建多个分块旋转打乱的副本"""
    transformed_copies = []
    for _ in range(num_copies):
        transformed_copy = shuffle_rotate_bsr(x, num_blocks_h, num_blocks_w, max_angle)
        transformed_copies.append(transformed_copy)

    return torch.cat(transformed_copies, dim=0)


def mifgsm_attack_BSR(x, y, model, eps=16 / 255, iterations=10, mu=1.0,
                      num_blocks_h=2, num_blocks_w=2, max_angle=0.2,
                      num_copies=20, use_diversity=True, use_admix=False,
                      portion=0.2, admix_size=3, diversity_prob=0.5):
    """BSR版本的MI-FGSM攻击"""
    alpha = eps / iterations
    x_adv = x.clone().requires_grad_(True)
    momentum = torch.zeros_like(x).to(x.device)

    for i in range(iterations):
        # 应用输入变换
        if use_diversity:
            x_transformed = input_diversity(x_adv, prob=diversity_prob)
        else:
            x_transformed = x_adv

        if use_admix:
            x_transformed = admix(x_transformed, portion=portion, size=admix_size)
            y_expanded = y.repeat(admix_size * num_copies)
        else:
            y_expanded = y.repeat(num_copies)

        # 应用BSR变换
        x_bsr = BSR_transform(x_transformed, num_blocks_h, num_blocks_w, max_angle, num_copies)

        # 前向传播
        output = model(x_bsr)

        # 处理不同输出类型
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, list):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected output type: {type(output)}")

        # 计算损失
        loss = F.cross_entropy(output, y_expanded)

        # 反向传播
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        # 梯度归一化和动量更新
        grad = x_adv.grad.data
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8
        grad_normalized = grad / grad_norm

        momentum = mu * momentum + grad_normalized

        # 更新对抗样本
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

    # 初始化模型仓库
    model_repo = ModelRepository(device)

    # --- 1. 攻击阶段：在内存中生成 ---
    source_model = load_source_model(args.model, device)

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

    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 记录原始预测
        source_orig_preds = get_model_prediction(source_model, x_batch)

        # 生成对抗样本
        x_adv_batch = mifgsm_attack_BSR(x_batch, y_batch, source_model,
                                iterations=args.iterations,
                                 mu=args.mu,num_blocks_h=args.num_blocks_h,
                                        num_blocks_w=args.num_blocks_w,
                                        max_angle=args.max_angle,
                                        num_copies=args.num_copies,
                                        use_diversity=args.use_diversity,
                                        use_admix=args.use_admix,
                                        portion=args.portion,
                                        admix_size=args.admix_size,
                                        diversity_prob=args.diversity_prob)

        # 记录攻击后预测
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        # 将生成的对抗样本移至 CPU 并存储，释放显存
        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item()) + 1
            s_adv_idx = int(source_adv_preds[i]) + 1
            s_orig_idx = int(source_orig_preds[i]) + 1
            # print(f"{s_orig_idx}\t{s_adv_idx}\t{true_label}")
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

    # 彻底释放源模型显存
    del source_model
    torch.cuda.empty_cache()

    # --- 2. 验证阶段：直接使用内存中的 Tensor ---
    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    # 将 List 转换为单个大 Tensor，并构建简单的 TensorDataset
    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False,pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")
        # 这里建议你根据之前讨论的，实现一个 load_single_model 或者 load_source_model
        # 假设这里依然通过 repo 加载
        current_model_info = model_repo.load_single_model(model_name)
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
    torch.cuda.empty_cache();
    main()