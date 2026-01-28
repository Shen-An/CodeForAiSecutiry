import torch
import torch.nn.functional as F

from typing import Tuple
import math




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


def warp_perspective_then_rotate(
    single: torch.Tensor,
    *,
    distortion_scale: float,
    angle: torch.Tensor | float,
    flip_prob: float = 0.0,
    stretch_factor: float = 0.0,
) -> torch.Tensor:
    """对单张图片(1,C,H,W)执行：随机透视 -> 指定角度旋转 ->（可选）块内水平翻转 ->（可选）块内拉伸。

    约定：
    - flip 发生在块内透视/旋转之后；
    - stretch 发生在 flip 之后（块内最后一步）；
    - stretch 参数使用随机采样：lambda_w, lambda_h ~ U(-stretch_factor, stretch_factor)
      然后对该 block 做非等比例缩放并插值回原尺寸。
    """
    if single.dim() != 4 or single.size(0) != 1:
        raise ValueError(f"single must be (1,C,H,W), got {tuple(single.shape)}")

    bH, bW = int(single.shape[2]), int(single.shape[3])
    if bH <= 1 or bW <= 1:
        return single

    # 1) perspective（保持与原实现一致：每次调用都重新采样四角点）
    src_n, dst_n = _rand_perspective_params(bH, bW, float(distortion_scale), device=single.device)
    H_mat = _homography_dlt(src_n, dst_n)
    single = _warp_perspective_grid_sample(single, H_mat)

    # 2) rotation（保持与原实现一致：align_corners=False）
    a = float(angle) if isinstance(angle, (int, float)) else float(angle.item())
    angle_matrix = torch.tensor(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]],
        dtype=torch.float32,
        device=single.device,
    ).unsqueeze(0)
    grid = F.affine_grid(angle_matrix, single.size(), align_corners=False)
    single = F.grid_sample(single, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

   

    # 4) per-block aspect-ratio stretching (after flip)
    if stretch_factor and float(stretch_factor) != 0.0:
        sf = float(stretch_factor)
        # lambda_w, lambda_h ~ U(-sf, sf)
        lw = float(torch.empty((), device=single.device).uniform_(-sf, sf).item())
        lh = float(torch.empty((), device=single.device).uniform_(-sf, sf).item())

        # 目标尺寸（至少为 1，避免插值报错）
        new_h = max(1, int(round(bH * (1.0 + lh))))
        new_w = max(1, int(round(bW * (1.0 + lw))))

        # 先缩放到 (new_h, new_w)，再插值回 (bH, bW)
        resized = F.interpolate(single, size=(new_h, new_w), mode='bilinear', align_corners=False)
        single = F.interpolate(resized, size=(bH, bW), mode='bilinear', align_corners=False)
     # 3) per-block horizontal flip (after perspective+rotation)
    if flip_prob and float(flip_prob) > 0.0:
        if torch.rand(1, device=single.device).item() < float(flip_prob):
            single = torch.flip(single, dims=[3])

    return single


def _perspective_permutation_transform(
    x: torch.Tensor,
    *,
    num_blocks_h: int,
    num_blocks_w: int,
    num_copies: int,
    distortion_scale: float,
    max_angle: float,
    flip_prob: float = 0.0,
    stretch_factor: float = 0.0,
) -> torch.Tensor:
    """不使用 BSR 时：只做『分块透视 + 分块旋转』（不做相邻置换），复制 num_copies 次。

    说明：

    - 旋转与 BSR 中保持一致：对每个块做小角度随机旋转。
    """
    copies = []
    B, C, w, h = x.shape
    from ppsp.BSR import get_block_lengths_bsr
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

                    # perspective + rotation
                    single = warp_perspective_then_rotate(
                        single,
                        distortion_scale=distortion_scale,
                        angle=angles[bi],
                        flip_prob=flip_prob,
                        stretch_factor=stretch_factor,
                    )

                    out_block[bi:bi + 1] = single

                blocks[i][j] = out_block

        copies.append(_merge_blocks(blocks))

    return torch.cat(copies, dim=0)

