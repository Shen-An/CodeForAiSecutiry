import os

import torch
import torch.nn.functional as F
import numpy as np
import math

from ppsp.perspective_rotate import _rand_perspective_params, _homography_dlt, _warp_perspective_grid_sample
from ppsp.perspective_rotate import warp_perspective_then_rotate

from torchvision.utils import save_image

def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def shuffle_rotate_bsr(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2):
    """BSR的分块旋转和打乱变换"""
    batch_size, channels, w, h = x.shape

    # 获取分块长度
    width_length = get_block_lengths_bsr(w, num_blocks_h)
    height_length = get_block_lengths_bsr(h, num_blocks_w)

    # 生成随机排列
    width_perm = np.random.permutation(np.arange(num_blocks_h))
    height_perm = np.random.permutation(np.arange(num_blocks_w))

    # 宽度方向分块和打乱
    x_split_w = torch.split(x, width_length, dim=2)
    x_w_perm = torch.cat([x_split_w[i] for i in width_perm], dim=2)

    # 高度方向分块、旋转和打乱
    x_split_h_blocks = []
    for w_block in x_split_w:
        h_blocks = torch.split(w_block, height_length, dim=3)
        x_split_h_blocks.append(h_blocks)

    # 应用旋转并重新组合
    rotated_blocks = []
    for strip_idx, strip in enumerate(x_split_h_blocks):
        rotated_strip = []
        for block_idx, block in enumerate(strip):
            # 为每个块生成随机旋转角度
            angles = torch.clamp(
                torch.randn(batch_size, device=x.device) * 0.05,
                -max_angle, max_angle
            )

            # 应用旋转
            rotated_block = block.clone()
            for i in range(batch_size):
                if block.shape[2] > 1 and block.shape[3] > 1:  # 确保块足够大
                    angle_matrix = torch.tensor([
                        [math.cos(angles[i]), -math.sin(angles[i]), 0],
                        [math.sin(angles[i]), math.cos(angles[i]), 0]
                    ], dtype=torch.float32, device=x.device).unsqueeze(0)

                    grid = F.affine_grid(angle_matrix, block[i:i + 1].size(), align_corners=False)
                    rotated_block[i:i + 1] = F.grid_sample(block[i:i + 1], grid, mode='bilinear',
                                                           padding_mode='zeros', align_corners=False)

            rotated_strip.append(rotated_block)

        # 按高度排列组合
        rotated_strip_perm = [rotated_strip[i] for i in height_perm]
        rotated_blocks.append(torch.cat(rotated_strip_perm, dim=3))

    # 最终组合
    x_h_perm = torch.cat(rotated_blocks, dim=2)
    return x_h_perm


def BSR_transform(
    x,
    num_blocks_h=2,
    num_blocks_w=2,
    max_angle=0.2,
    num_copies=20,
    *,
    distortion_scale: float = 0.06,
    flip_prob: float = 0.0,
):
    """PPSP +BSR 路径：

    分块 -> 打乱 -> 旋转(BSR) -> 透视 -> 我们的旋转(第二次) （-> 可选：块内最后水平翻转），复制 num_copies 次。
    """

    def _apply_block_perspective_and_post_rotate(x_in: torch.Tensor) -> torch.Tensor:
        """对输入按『当前随机分块』逐块做随机透视，然后再逐块做一次旋转（我们的旋转）。"""
        B, C, w, h = x_in.shape

        width_length = get_block_lengths_bsr(w, num_blocks_h)
        height_length = get_block_lengths_bsr(h, num_blocks_w)

        # split blocks（保持顺序）
        x_split_w = torch.split(x_in, width_length, dim=2)
        blocks_2d = []
        for w_block in x_split_w:
            blocks_2d.append(list(torch.split(w_block, height_length, dim=3)))

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = blocks_2d[i][j]
                bH, bW = block.shape[2], block.shape[3]
                if bH <= 1 or bW <= 1:
                    continue

                out_block = block.clone()

                # 第二次旋转角（我们的旋转）：对该块的每张图独立采样，且与 BSR 中第一次旋转无关
                post_angles = torch.clamp(torch.randn(B, device=x_in.device) * 0.05, -max_angle, max_angle)

                for bi in range(B):
                    single = block[bi : bi + 1]

                    # 透视 + 我们的旋转 +（可选）块内最后翻转
                    single = warp_perspective_then_rotate(
                        single,
                        distortion_scale=distortion_scale,
                        angle=post_angles[bi],
                        flip_prob=flip_prob,
                    )

                    out_block[bi : bi + 1] = single

                blocks_2d[i][j] = out_block

        merged_strips = [torch.cat(blocks_2d[i], dim=3) for i in range(num_blocks_h)]
        return torch.cat(merged_strips, dim=2)

    transformed_copies = []
    for _ in range(num_copies):
        # 1) 分块+打乱+旋转（第一次旋转）
        x_sr = shuffle_rotate_bsr(x, num_blocks_h, num_blocks_w, max_angle)
        # 2) 透视 + 我们的旋转（第二次旋转）
        x_srp_r2 = _apply_block_perspective_and_post_rotate(x_sr)
        transformed_copies.append(x_srp_r2)

    return torch.cat(transformed_copies, dim=0)


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


def shuffle_rotate_bsr_PR(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2, *,
                       distortion_scale: float = 0.06):
    """PPSP.

    PPSP  +BSR 置换逻辑：宽/高方向分别做随机置换（完全打乱），然后对每个块做透视 + 旋转。

    注意：
    - 仅包含置换 + 透视 + 旋转；
    - params 保留为兼容外部调用签名（当前未使用）。
    """
    batch_size, channels, w, h = x.shape

    # 获取分块长度（与原实现一致：随机长度分块）
    width_length = get_block_lengths_bsr(w, num_blocks_h)
    height_length = get_block_lengths_bsr(h, num_blocks_w)

    # === 置换：替换为 notebook 版本（宽/高分别随机置换）===
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

    # 透视 + 旋转，并按 height_perm 重排每个 strip 的 h 块
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

            rotated_strip.append(out_block)

        rotated_strip_perm = [rotated_strip[i] for i in height_perm]
        rotated_blocks.append(torch.cat(rotated_strip_perm, dim=3))

    x_h_perm = torch.cat(rotated_blocks, dim=2)
    return x_h_perm
