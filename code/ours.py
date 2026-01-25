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

from eval import eval_transferability
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
    parser = argparse.ArgumentParser(description="ours attack")
    parser.add_argument("--model", default='tf2torch_inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/ours/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/ours/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--diversity_prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument("--num_blocks_h", default=2, type=int, help="number of blocks h")
    parser.add_argument("--num_blocks_w", default=2, type=int, help="number of blocks w")
    parser.add_argument("--max_angle", default=2, type=int, help="maximum angle")
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
    #   --bsr      => True, 使用 BSR（当前实现：相邻置换 + 透视 + 旋转）
    #   不加参数   => False, 不使用 BSR，改为“直接 透视 + 置换”（无旋转）
    parser.add_argument('--bsr', action='store_true', help='use BSR (AS+perspective+rotation)')
    parser.set_defaults(bsr=False)

    # Adjacent-Swap + Perspective params
    parser.add_argument("--p_swap", default=0.5, type=float, help="adjacent swap probability")
    parser.add_argument("--distortion_scale", default=0.06, type=float, help="perspective distortion scale ")

    # 保存第 N 次迭代的中间副本/变换图
    # 约定：save_iter<=0 视为关闭；>0 则在该迭代保存
    # 为了兼容旧用法，保留 --save_iter5，但默认不需要它
    parser.add_argument('--save_iter5', action='store_true', help='(deprecated) enable save_iter (kept for compatibility)')
    parser.add_argument('--save_iter', default=0, type=int, help='which iteration to dump (1-based); <=0 disables')
    parser.add_argument('--save_trans_dir', default='./results/BSR/trans_debug', type=str, help='dir to save intermediate transformed images')

    # 调试：只跑一个 batch，跑完就退出（通常配合 --save_iter 1）
    parser.add_argument('--one_batch', action='store_true', help='debug: only attack the first batch and exit after saving')
    parser.set_defaults(one_batch=False)

    return parser.parse_args()


@dataclass(frozen=True)
class BSPPParams:
    """单次 BSPP 变换的参数集合。

    - width_length/height_length: 固定分块长度（保证同一次迭代内多处使用一致）
    - swapped_width_length/swapped_height_length: 相邻置换后的长度序列
    """

    width_length: Tuple[int, ...]
    height_length: Tuple[int, ...]
    swapped_width_length: Tuple[int, ...]
    swapped_height_length: Tuple[int, ...]


def _build_block_index_map(num_blocks_h: int, num_blocks_w: int):
    """把二维(i,j)映射到一维idx，以及反向映射。"""
    def to_idx(i: int, j: int) -> int:
        return i * num_blocks_w + j

    def from_idx(idx: int) -> Tuple[int, int]:
        return idx // num_blocks_w, idx % num_blocks_w

    return to_idx, from_idx


def _swap_adjacent_in_lengths(lengths: Tuple[int, ...], p_swap: float) -> Tuple[int, ...]:
    """在长度序列中，以概率 p_swap 选择一对相邻元素进行对调。

    对应文档的 T_AS：在 L_h 或 L_w 上做相邻置换，而不是直接交换块Tensor。
    这样重建时的拼接尺寸天然一致，不会出现 cat 维度不匹配。
    """
    if len(lengths) <= 1:
        return lengths

    out = list(lengths)
    # 单次应用：随机选一个相邻对；若未命中概率则不交换
    if np.random.rand() < p_swap:
        j = np.random.randint(0, len(out) - 1)
        out[j], out[j + 1] = out[j + 1], out[j]
    return tuple(int(x) for x in out)


def generate_bspp_params(width: int, height: int, num_blocks_h: int, num_blocks_w: int, p_swap: float = 0.3) -> BSPPParams:
    """生成一次 BSPP 变换所需的参数。"""
    width_length = get_block_lengths_bsr(width, num_blocks_h)
    height_length = get_block_lengths_bsr(height, num_blocks_w)
    # 选项1：每次只在一个方向做相邻交换（避免二维同时交换带来的尺寸重映射复杂性）
    if np.random.rand() < 0.5:
        swapped_width_length = _swap_adjacent_in_lengths(width_length, p_swap=p_swap)
        swapped_height_length = height_length
    else:
        swapped_width_length = width_length
        swapped_height_length = _swap_adjacent_in_lengths(height_length, p_swap=p_swap)
    return BSPPParams(
        width_length=width_length,
        height_length=height_length,
        swapped_width_length=swapped_width_length,
        swapped_height_length=swapped_height_length,
    )


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


def _perm_from_adjacent_swap(orig: Tuple[int, ...], swapped: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
    """若 swapped 是通过对 orig 做一次相邻交换得到的，返回被交换的一对索引(j, j+1)。

    若没有发生交换（或不符合一次相邻交换的形式），返回 None。
    """
    if orig == swapped:
        return None
    if len(orig) != len(swapped):
        return None
    diffs = [k for k in range(len(orig)) if orig[k] != swapped[k]]
    if len(diffs) != 2:
        return None
    a, b = diffs
    if b != a + 1:
        return None
    if orig[a] == swapped[b] and orig[b] == swapped[a]:
        return (a, b)
    return None


def _adjacent_swap_reconstruct(x: torch.Tensor,
                               width_length: Tuple[int, ...],
                               height_length: Tuple[int, ...],
                               swapped_width_length: Tuple[int, ...],
                               swapped_height_length: Tuple[int, ...]) -> torch.Tensor:
    """按文档的 T_AS（选项1）实现“内容映射”，不做 crop/pad。

    约束：每次仅在 L_w 或 L_h 中发生一次相邻交换。

    实现方式：
    - 若 L_w 发生交换：在 dim=2 切出的条带列表上交换对应条带；条带内再按 height_length 切块。
    - 若 L_h 发生交换：在 dim=3 切出的条带列表上交换对应条带；再按 width_length 切条带。
    """
    w_swap = _perm_from_adjacent_swap(width_length, swapped_width_length)
    h_swap = _perm_from_adjacent_swap(height_length, swapped_height_length)

    # 选项1下应该最多一个方向发生交换
    if w_swap is not None and h_swap is not None:
        # 防御性：若意外同时交换，退化为不交换（避免语义/尺寸异常）
        return x

    if w_swap is not None:
        j0, j1 = w_swap
        strips = list(torch.split(x, width_length, dim=2))
        strips[j0], strips[j1] = strips[j1], strips[j0]
        return torch.cat(strips, dim=2)

    if h_swap is not None:
        j0, j1 = h_swap
        strips = list(torch.split(x, height_length, dim=3))
        strips[j0], strips[j1] = strips[j1], strips[j0]
        return torch.cat(strips, dim=3)

    return x


def _rand_perspective_params_batch(batch_size: int, block_h: int, block_w: int, distortion_scale: float, device, dtype=torch.float32):
    """为一整个 batch 生成随机透视变换的(归一化)四角坐标。

    返回：src_n, dst_n: (B,4,2) in [-1,1]
    """
    src = torch.tensor(
        [[0.0, 0.0], [block_w - 1.0, 0.0], [block_w - 1.0, block_h - 1.0], [0.0, block_h - 1.0]],
        device=device,
        dtype=dtype,
    )
    src = src.unsqueeze(0).expand(batch_size, -1, -1).contiguous()  # (B,4,2)

    dx = distortion_scale * float(block_w)
    dy = distortion_scale * float(block_h)
    noise = torch.empty((batch_size, 4, 2), device=device, dtype=dtype)
    noise[..., 0].uniform_(-dx, dx)
    noise[..., 1].uniform_(-dy, dy)

    dst = src + noise
    dst[..., 0].clamp_(0.0, block_w - 1.0)
    dst[..., 1].clamp_(0.0, block_h - 1.0)

    # normalize to [-1,1]
    def normalize_xy(pts):
        x = pts[..., 0] / max(1.0, (block_w - 1.0)) * 2.0 - 1.0
        y = pts[..., 1] / max(1.0, (block_h - 1.0)) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    return normalize_xy(src), normalize_xy(dst)


def _homography_dlt_batch(src_n: torch.Tensor, dst_n: torch.Tensor) -> torch.Tensor:
    """批量 DLT 求解单应性 H（B,3,3），满足 dst ~ H * src。

    src_n/dst_n: (B,4,2) 归一化坐标（[-1,1]）。
    """
    if src_n.dim() != 3 or dst_n.dim() != 3 or src_n.shape != dst_n.shape or src_n.shape[1:] != (4, 2):
        raise ValueError(f"Expected src_n/dst_n shape (B,4,2), got {tuple(src_n.shape)} / {tuple(dst_n.shape)}")

    B = src_n.shape[0]
    device = src_n.device
    dtype = src_n.dtype

    x = src_n[..., 0]
    y = src_n[..., 1]
    u = dst_n[..., 0]
    v = dst_n[..., 1]

    zeros = torch.zeros((B, 4), device=device, dtype=dtype)
    ones = torch.ones((B, 4), device=device, dtype=dtype)

    # 构造 A: (B,8,9)
    # 第一行组：[-x, -y, -1, 0,0,0, u*x, u*y, u]
    a1 = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=-1)  # (B,4,9)
    # 第二行组：[0,0,0, -x,-y,-1, v*x, v*y, v]
    a2 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=-1)  # (B,4,9)

    A = torch.cat([a1, a2], dim=1)  # (B,8,9)

    # 批量 SVD，取最小奇异值对应向量
    # torch.linalg.svd 支持 batch
    _, _, Vh = torch.linalg.svd(A)
    h = Vh[..., -1, :]  # (B,9)
    H = h.view(B, 3, 3)
    return H / (H[..., 2:3, 2:3] + 1e-8)


def _batch_warp_perspective(block: torch.Tensor, Hs: torch.Tensor) -> torch.Tensor:
    """对 batch 内每张图用各自单应矩阵做透视变换（并行）。

    block: (B,C,H,W)
    Hs: (B,3,3)  src->dst

    为了“输出为 dst，采样 src”，内部使用 inv(Hs)。
    """
    if block.dim() != 4:
        raise ValueError(f"block must be (B,C,H,W), got {tuple(block.shape)}")
    if Hs.dim() != 3 or Hs.shape[1:] != (3, 3) or Hs.shape[0] != block.shape[0]:
        raise ValueError(f"Hs must be (B,3,3) aligned with block batch, got {tuple(Hs.shape)}")

    B, C, Hh, Ww = block.shape
    device = block.device
    dtype = block.dtype

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, Hh, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, Ww, device=device, dtype=torch.float32),
        indexing='ij',
    )
    ones = torch.ones_like(xs)
    grid_h = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).t().unsqueeze(0).expand(B, -1, -1)  # (B,3,H*W)

    H_inv = torch.linalg.inv(Hs)  # (B,3,3)
    src_h = torch.bmm(H_inv, grid_h)  # (B,3,H*W)
    src_h = src_h / (src_h[:, 2:3, :] + 1e-8)

    x_src = src_h[:, 0, :].view(B, Hh, Ww)
    y_src = src_h[:, 1, :].view(B, Hh, Ww)

    grid = torch.stack([x_src, y_src], dim=-1).to(dtype=dtype)  # (B,H,W,2)
    return F.grid_sample(block, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def _rotation_affine_matrices(angles: torch.Tensor) -> torch.Tensor:
    """由 angles (B,) 构造 (B,2,3) 的旋转仿射矩阵（绕中心，平移为0）。"""
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    zeros = torch.zeros_like(cos_a)

    row1 = torch.stack([cos_a, -sin_a, zeros], dim=-1)
    row2 = torch.stack([sin_a, cos_a, zeros], dim=-1)
    return torch.stack([row1, row2], dim=1)


def _vectorized_swap_lengths(width: int, height: int, num_blocks_h: int, num_blocks_w: int, p_swap: float, batch_size: int, device):
    """生成一次『仅在一个方向发生一次相邻交换』的交换指令。

    重要：由于 width_length/height_length 是随机不等分的，条带宽/高通常不同。
    因此**无法**在同一个 batch 内为不同样本选择不同的交换位置，否则会产生
    “把宽度 w1 的条带写入宽度 w2 的位置”的形状冲突。

    为保持与原实现一致且保证可并行，这里让整个 batch（含展开后的 copies）共享同一
    次交换：是否交换、交换方向、以及交换起点 j 都是全 batch 共享。

    返回:
      width_length, height_length: tuple
      swap_dim: int 0 表示 dim=2（W条带）交换，1 表示 dim=3（H条带）交换，2 表示不交换
      swap_idx: int 交换起点 j（交换 j 与 j+1）
    """
    width_length = get_block_lengths_bsr(width, num_blocks_h)
    height_length = get_block_lengths_bsr(height, num_blocks_w)

    do_swap = (torch.rand((), device=device) < float(p_swap)).item()
    if not do_swap:
        return width_length, height_length, 2, 0

    swap_dim = int((torch.rand((), device=device) < 0.5).item())  # 0->W(dim=2), 1->H(dim=3)

    if swap_dim == 0:
        max_w = max(1, num_blocks_h - 1)
        swap_idx = int(torch.randint(0, max_w, (), device=device).item()) if num_blocks_h > 1 else 0
    else:
        max_h = max(1, num_blocks_w - 1)
        swap_idx = int(torch.randint(0, max_h, (), device=device).item()) if num_blocks_w > 1 else 0

    return width_length, height_length, swap_dim, swap_idx


def _adjacent_swap_reconstruct_batch(x: torch.Tensor, *, width_length: Tuple[int, ...], height_length: Tuple[int, ...], swap_dim: int, swap_idx: int) -> torch.Tensor:
    """批量版本 T_AS（共享一次相邻交换）。

    - swap_dim==0: 在 dim=2 按 width_length 切条带并交换 swap_idx 与 swap_idx+1
    - swap_dim==1: 在 dim=3 按 height_length 切条带并交换 swap_idx 与 swap_idx+1
    - swap_dim==2: 不交换

    该实现完全消除 batch 维循环，且不会出现不等宽/不等高条带的形状冲突。
    """
    if swap_dim == 2:
        return x

    if swap_dim == 0:
        if len(width_length) <= 1:
            return x
        strips = list(torch.split(x, width_length, dim=2))
        j = max(0, min(int(swap_idx), len(strips) - 2))
        strips[j], strips[j + 1] = strips[j + 1], strips[j]
        return torch.cat(strips, dim=2)

    if swap_dim == 1:
        if len(height_length) <= 1:
            return x
        strips = list(torch.split(x, height_length, dim=3))
        j = max(0, min(int(swap_idx), len(strips) - 2))
        strips[j], strips[j + 1] = strips[j + 1], strips[j]
        return torch.cat(strips, dim=3)

    return x


def BSR_transform(x: torch.Tensor, num_blocks_h=2, num_blocks_w=2, max_angle=0.2, num_copies=20, *,
                  distortion_scale: float = 0.06,
                  p_swap: float = 0.3):
    """并行 BSR 变换：接收 x (B*num_copies,C,W,H)，一次性完成 T_AS + T_persp + R。"""
    Bn, C, W, H = x.shape
    device = x.device

    width_length, height_length, swap_dim, swap_idx = _vectorized_swap_lengths(
        W, H, num_blocks_h, num_blocks_w, p_swap, Bn, device
    )

    x_swapped = _adjacent_swap_reconstruct_batch(
        x,
        width_length=width_length,
        height_length=height_length,
        swap_dim=swap_dim,
        swap_idx=swap_idx,
    )

    blocks = _split_blocks(x_swapped, width_length, height_length)

    # per-block perspective + rotation（batch 并行）
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = blocks[i][j]  # (Bn,C,bW,bH)
            bW = block.shape[2]
            bH = block.shape[3]
            if bW <= 1 or bH <= 1:
                continue

            # perspective params + H per image
            src_n, dst_n = _rand_perspective_params_batch(Bn, bW, bH, distortion_scale, device=device, dtype=torch.float32)
            Hs = _homography_dlt_batch(src_n, dst_n)  # (Bn,3,3)
            block = _batch_warp_perspective(block, Hs)

            # rotation：一次生成所有角
            angles = torch.clamp(torch.randn(Bn, device=device) * 0.05, -max_angle, max_angle)
            theta = _rotation_affine_matrices(angles).to(device=device, dtype=torch.float32)
            grid = F.affine_grid(theta, block.size(), align_corners=False)
            block = F.grid_sample(block, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            blocks[i][j] = block

    return _merge_blocks(blocks)


def _perspective_permutation_transform(
    x: torch.Tensor,
    *,
    num_blocks_h: int,
    num_blocks_w: int,
    num_copies: int,
    p_swap: float,
    distortion_scale: float,
) -> torch.Tensor:
    """不使用 BSR 时：并行做『相邻置换 + 透视』（不做旋转）。

     约定：外部已经 repeat_interleave 展开到 (B*num_copies,C,W,H)
    """
    Bn, C, W, H = x.shape
    device = x.device

    width_length, height_length, swap_dim, swap_idx = _vectorized_swap_lengths(
        W, H, num_blocks_h, num_blocks_w, p_swap, Bn, device
    )

    x_swapped = _adjacent_swap_reconstruct_batch(
        x,
        width_length=width_length,
        height_length=height_length,
        swap_dim=swap_dim,
        swap_idx=swap_idx,
    )

    blocks = _split_blocks(x_swapped, width_length, height_length)
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = blocks[i][j]
            bW = block.shape[2]
            bH = block.shape[3]
            if bW <= 1 or bH <= 1:
                continue

            src_n, dst_n = _rand_perspective_params_batch(Bn, bW, bH, distortion_scale, device=device, dtype=torch.float32)
            Hs = _homography_dlt_batch(src_n, dst_n)
            block = _batch_warp_perspective(block, Hs)
            blocks[i][j] = block

    return _merge_blocks(blocks)


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


def mifgsm_attack_BSR(x, y, model, eps=16 / 255, iterations=10, mu=1.0,
                      num_blocks_h=2, num_blocks_w=2, max_angle=0.2,
                      num_copies=20, use_diversity=True, use_admix=False,
                      portion=0.2, admix_size=3, diversity_prob=0.5,
                      p_swap=0.3, distortion_scale=0.06, *, bsr: bool = False,
                      save_iter: int | None = None,
                      save_trans_dir: str | None = None,
                      filenames: Optional[list[str]] = None):
    """MI-FGSM + (BSR 或 透视+置换)。

    说明：
    - 这里对 num_copies 做完全并行：先 repeat_interleave 展开 batch，再一次性变换。
    """
    alpha = eps / iterations
    x_adv = x.clone().requires_grad_(True)
    momentum = torch.zeros_like(x).to(x.device)

    for i in range(iterations):
        if use_diversity:
            x_transformed = input_diversity(x_adv, prob=diversity_prob)
        else:
            x_transformed = x_adv

        # 先做 admix（batch 维扩展），再做 num_copies 扩展
        if use_admix:
            x_transformed = admix(x_transformed, portion=portion, size=admix_size)
            y_base = y.repeat_interleave(admix_size, dim=0)
        else:
            y_base = y

        # 扩展到 (B * admix_size * num_copies)
        x_expanded = x_transformed.repeat_interleave(num_copies, dim=0)
        y_expanded = y_base.repeat_interleave(num_copies, dim=0)

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
                x_expanded,
                num_blocks_h,
                num_blocks_w,
                max_angle,
                num_copies,
                distortion_scale=distortion_scale,
                p_swap=p_swap,
            )
        else:
            x_aug = _perspective_permutation_transform(
                x_expanded,
                num_blocks_h=num_blocks_h,
                num_blocks_w=num_blocks_w,
                num_copies=num_copies,
                p_swap=p_swap,
                distortion_scale=distortion_scale,
            )

        if (
            save_iter is not None
            and save_trans_dir is not None
            and filenames is not None
            and iter_1based == save_iter
        ):
            # 这里保存的是每张原始图的多个 copy；如果 use_admix=True，保存语义会变（标签也已扩展），
            # 为避免误导，仅在 non-admix 时启用严格 copy-major 可视化。
            if not use_admix:
                _save_transformed_copies(
                    x_aug,
                    filenames,
                    out_dir=os.path.join(save_trans_dir, 'copies'),
                    iter_idx_1based=iter_1based,
                    num_copies=num_copies,
                )

        output = model(x_aug)
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

        x_adv_batch = mifgsm_attack_BSR(
            x_batch,
            y_batch,
            source_model,
            eps=args.eps,
            iterations=args.iterations,
            mu=args.mu,
            num_blocks_h=args.num_blocks_h,
            num_blocks_w=args.num_blocks_w,
            max_angle=args.max_angle,
            num_copies=args.num_copies,
            use_diversity=args.use_diversity,
            use_admix=args.use_admix,
            portion=args.portion,
            admix_size=args.admix_size,
            diversity_prob=args.diversity_prob,
            p_swap=args.p_swap,
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

        # 调试：只 attack 一个 batch 就退出（保存会在 mifgsm_attack_BSR 内部触发）
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

    eval_transferability(source_results, adv_images_storage, args.output_adv_dir,model_repo, device)

if __name__ == '__main__':
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    main()