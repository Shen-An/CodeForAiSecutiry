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

from preprocess import AdvPNGDataset, get_model_output, get_model_prediction


def admix(x, portion=0.2, size=3):
    """混合输入变换（Admix）。

    与原 bsr.py 兼容：随机打乱 batch 并做线性混合，然后在 batch 维度拼接 size 份。
    """
    indices = torch.randperm(x.size(0), device=x.device)
    admixed = []
    for _ in range(int(size)):
        admixed_x = x + float(portion) * x[indices]
        admixed.append(admixed_x)
    return torch.cat(admixed, dim=0)


def input_diversity(x, image_width=299, image_resize=331, prob=0.5):
    """输入多样性（DI-FGSM 风格）。

    备注：使用 nearest 与原实现保持一致。
    """
    if torch.rand(1, device=x.device).item() < prob:
        rnd = torch.randint(image_width, image_resize + 1, (1,), device=x.device).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='nearest')

        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,), device=x.device).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem + 1, (1,), device=x.device).item()
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        padded = F.interpolate(padded, size=(image_width, image_width), mode='nearest')
        return padded
    return x


def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


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

    # 你要的单参数开关：
    #   --bsr      => True, 使用 BSR（当前实现：随机置换 + 透视 + 旋转）
    #   不加参数   => False, 默认分块透视 + 分块旋转（无相邻置换）
    parser.add_argument('--bsr', action='store_true', help='use BSR (permute+perspective+rotation)')
    parser.set_defaults(bsr=False)

    # 旋转角度拆分：BSR 与默认透视分块使用不同 max_angle，便于组件化开关
    parser.add_argument("--max_angle_bsr", default=0.2, type=float, help="maximum rotation angle for BSR")
    parser.add_argument("--max_angle_default", default=0.2, type=float, help="maximum rotation angle for default (non-BSR)")

    # Adjacent-Swap + Perspective params
    # p_swap 已不再使用（不做相邻置换/概率置换），因此移除
    parser.add_argument("--distortion_scale", default=0.26, type=float, help="perspective distortion scale (BCTOT D)")

    # 保存第 N 次迭代的中间副本/变换图
    # 约定：save_iter<=0 视为关闭；>0 则在该迭代保存
    # 为了兼容旧用法，保留 --save_iter5，但默认不需要它
    parser.add_argument('--save_iter5', action='store_true', help='(deprecated) enable save_iter (kept for compatibility)')
    parser.add_argument('--save_iter', default=0, type=int, help='which iteration to dump (1-based); <=0 disables')
    parser.add_argument('--save_trans_dir', default='./results/PPSP/trans_debug', type=str, help='dir to save intermediate transformed images')

    # 调试：只跑一个 batch，跑完就退出（通常配合 --save_iter 1）
    parser.add_argument('--one_batch', action='store_true', help='debug: only attack the first batch and exit after saving')
    parser.set_defaults(one_batch=False)

    return parser.parse_args()


@dataclass(frozen=True)
class BSPPParams:
    """（保留占位）单次变换参数集合。

    旧版用于相邻置换的长度参数已移除；当前仅保留以兼容历史接口。
    """

    width_length: Tuple[int, ...]
    height_length: Tuple[int, ...]


# === 相邻置换相关实现已彻底移除（不再使用） ===
# _build_block_index_map
# _swap_adjacent_in_lengths
# generate_bspp_params
# _perm_from_adjacent_swap
# _adjacent_swap_reconstruct


def _split_blocks(x: torch.Tensor, width_length: Tuple[int, ...], height_length: Tuple[int, ...]):
    """把 x 切成二维块列表 blocks[i][j]，且保持原始次序。"""
    x_split_w = torch.split(x, width_length, dim=2)
    blocks = []
    for w_block in x_split_w:
        blocks.append(list(torch.split(w_block, height_length, dim=3)))
    return blocks


def _merge_blocks(blocks):
    """把二维块列表 blocks[i][j] 拼回 (B,C,W,H)。"""
    strips = [torch.cat(row, dim=3) for row in blocks]
    return torch.cat(strips, dim=2)


def _rand_perspective_params(block_h: int, block_w: int, distortion_scale: float, device):
    """为单个块生成随机透视变换的(归一化)四角坐标。

    对四个角点加入噪声 Δ∈[-δ, δ]，其中
    δ = distortion_scale * block_dimension。
    返回：src_pts(4,2), dst_pts(4,2) in [-1, 1] normalized coords.
    """
    # 注意：这里的 (x, y) 使用像素坐标系
    src = torch.tensor(
        [[0.0, 0.0], [block_w - 1.0, 0.0], [block_w - 1.0, block_h - 1.0], [0.0, block_h - 1.0]],
        device=device,
        dtype=torch.float32,
    )

    # BCTOT: δ = distortion_scale * block_dimension
    dx = distortion_scale * float(block_w)
    dy = distortion_scale * float(block_h)
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
        A_rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A_rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
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

# 1. 尝试求逆，如果矩阵是奇异的（不可逆），直接返回原块
    try:
        H_inv = torch.linalg.inv(H)
    except (torch._C._LinAlgError, RuntimeError):
        # 当 distortion_scale 很大时，极易触发此处的异常处理
        return block
    src_h = H_inv @ grid_h
    src_h = src_h / (src_h[2:3, :] + 1e-8)
    x_src = src_h[0, :].view(Hh, Ww)
    y_src = src_h[1, :].view(Hh, Ww)

    grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0).to(dtype)  # (1,H,W,2)

    return F.grid_sample(block, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def shuffle_rotate_bsr(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2, *,
                       distortion_scale: float = 0.06,
                       params: Optional[BSPPParams] = None):
    """PPSP.

    BSR 置换逻辑：宽/高方向分别做随机置换（完全打乱），然后对每个块做透视 + 旋转。

    注意：
    - 仅包含置换 + 透视 + 旋转；不包含任何相邻置换概率参数。
    - params 保留为兼容外部调用签名（当前未使用）。
    """
    batch_size, channels, w, h = x.shape

    # 获取分块长度（与原实现一致：随机长度分块）
    width_length = get_block_lengths_bsr(w, num_blocks_h)
    height_length = get_block_lengths_bsr(h, num_blocks_w)

    # === 置换：替换为 notebook 版本（宽/高分别随机置换） ===
    width_perm = np.random.permutation(np.arange(num_blocks_h))
    height_perm = np.random.permutation(np.arange(num_blocks_w))

    # 宽度方向分块并按 width_perm 重新排列
    x_split_w = torch.split(x, width_length, dim=2)
    x_w_perm = torch.cat([x_split_w[i] for i in width_perm], dim=2)

    # 在 x_w_perm 上按高度分块（保持与 notebook 行为一致：每个 strip 内再切 h 块）
    x_split_h_blocks = []
    for w_block in torch.split(x_w_perm, width_length, dim=2):
        h_blocks = torch.split(w_block, height_length, dim=3)
        x_split_h_blocks.append(h_blocks)

    # 透视 + 旋转（保持原实现），并按 height_perm 重排每个 strip 的 h 块
    rotated_blocks = []
    for strip in x_split_h_blocks:
        rotated_strip = []
        for block in strip:
            bH, bW = block.shape[2], block.shape[3]
            if bH <= 1 or bW <= 1:
                rotated_strip.append(block)
                continue

            out_block = block.clone()
            angles = torch.clamp(torch.randn(batch_size, device=x.device) * 0.05, -max_angle, max_angle)

            for bi in range(batch_size):
                single = block[bi:bi + 1]

                # perspective（原实现）
                src_n, dst_n = _rand_perspective_params(bH, bW, distortion_scale, device=x.device)
                H_mat = _homography_dlt(src_n, dst_n)
                single = _warp_perspective_grid_sample(single, H_mat)

                # rotation（原实现）
                angle = angles[bi]
                angle_matrix = torch.tensor(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0]],
                    dtype=torch.float32,
                    device=x.device,
                ).unsqueeze(0)
                grid = F.affine_grid(angle_matrix, single.size(), align_corners=False)
                single = F.grid_sample(single, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

                out_block[bi:bi + 1] = single

            rotated_strip.append(out_block)

        rotated_strip_perm = [rotated_strip[i] for i in height_perm]
        rotated_blocks.append(torch.cat(rotated_strip_perm, dim=3))

    x_h_perm = torch.cat(rotated_blocks, dim=2)
    return x_h_perm


def BSR_transform(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2, num_copies=20, *,
                  distortion_scale: float = 0.06):
    """BSR变换：创建多个分块旋转打乱的副本"""
    transformed_copies = []
    for _ in range(num_copies):
        transformed_copy = shuffle_rotate_bsr(
            x,
            num_blocks_h,
            num_blocks_w,
            max_angle,
            distortion_scale=distortion_scale,
        )
        transformed_copies.append(transformed_copy)

    return torch.cat(transformed_copies, dim=0)


def _perspective_permutation_transform(
    x: torch.Tensor,
    *,
    num_blocks_h: int,
    num_blocks_w: int,
    num_copies: int,
    distortion_scale: float,
    max_angle: float,
) -> torch.Tensor:
    """不使用 BSR 时：只做『分块透视 + 分块旋转』（不做相邻置换），复制 num_copies 次。

    说明：
    - 这里的默认路径不再进行分块相邻置换（T_AS）。
    - 旋转与 BSR 中保持一致：对每个块做小角度随机旋转。
    """
    copies = []
    B, C, w, h = x.shape

    # 分块长度每个副本可独立采样（与 BSR 一致形成随机算子族）
    for _ in range(num_copies):
        width_length = get_block_lengths_bsr(w, num_blocks_h)
        height_length = get_block_lengths_bsr(h, num_blocks_w)

        # 1) split blocks（不做相邻置换）
        blocks = _split_blocks(x, width_length, height_length)

        # 2) per-block perspective + rotation
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = blocks[i][j]
                bH, bW = block.shape[2], block.shape[3]
                if bH <= 1 or bW <= 1:
                    continue

                out_block = block.clone()
                angles = torch.clamp(torch.randn(B, device=x.device) * 0.05, -max_angle, max_angle)

                for bi in range(B):
                    single = block[bi:bi + 1]

                    # perspective
                    src_n, dst_n = _rand_perspective_params(bH, bW, distortion_scale, device=x.device)
                    H_mat = _homography_dlt(src_n, dst_n)
                    single = _warp_perspective_grid_sample(single, H_mat)

                    # rotation
                    angle = angles[bi]
                    angle_matrix = torch.tensor(
                        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0]],
                        dtype=torch.float32,
                        device=x.device,
                    ).unsqueeze(0)
                    grid = F.affine_grid(angle_matrix, single.size(), align_corners=False)
                    single = F.grid_sample(single, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

                    out_block[bi:bi + 1] = single

                blocks[i][j] = out_block

        copies.append(_merge_blocks(blocks))

    return torch.cat(copies, dim=0)


def _save_transformed_copies(
    x_aug: torch.Tensor,
    filenames: list[str],
    *,
    out_dir: str,
    iter_idx_1based: int,
    num_copies: int,
):
    """保存 x_aug 中的中间副本。

    约定：x_aug 的 batch 维 = B * num_copies，按 copy-major 拼接：
    [copy0_B, copy1_B, ..., copy{num_copies-1}_B]
    """
    os.makedirs(out_dir, exist_ok=True)

    B = len(filenames)
    if B == 0:
        return

    # 安全兜底：尺寸不符就不保存，避免误存
    if x_aug.size(0) != B * num_copies:
        return

    x_aug = x_aug.detach().clamp(0, 1).cpu()

    for c in range(num_copies):
        start = c * B
        end = (c + 1) * B
        chunk = x_aug[start:end]
        for i, fn in enumerate(filenames):
            base, _ = os.path.splitext(fn)
            save_path = os.path.join(out_dir, f"{base}_iter{iter_idx_1based:02d}_copy{c:03d}.png")
            save_image(chunk[i], save_path)


def _save_trans_images(
    x_trans: torch.Tensor,
    filenames: list[str],
    *,
    out_dir: str,
    iter_idx_1based: int,
):
    """保存 trans（进入 BSR/透视置换之前）的图像，batch 维为 B。"""
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
                      distortion_scale=0.06, bsr: bool = False,
                      save_iter: int | None = None,
                      save_trans_dir: str | None = None,
                      filenames: Optional[list[str]] = None):
    """MI-FGSM + (BSR 或 默认分块透视+旋转)。

    - bsr=True:  BSR 路径（置换 + 分块透视 + 分块旋转），旋转角度由 max_angle_bsr 控制
    - bsr=False: 默认分块透视 + 分块旋转（不做相邻置换），旋转角度由 max_angle_default 控制

    save_iter: 若给定(1-based)，则在该次迭代保存 x_aug 的所有 num_copies 变换副本。
    """
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
        loss.backward()

        grad = x_adv.grad.data
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
            distortion_scale=args.distortion_scale,
            bsr=args.bsr,
            save_iter=(args.save_iter if args.save_iter > 0 else (5 if args.save_iter5 else None)),
            save_trans_dir=(args.save_trans_dir if (args.save_iter > 0 or args.save_iter5) else None),
            filenames=list(filename_batch),
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