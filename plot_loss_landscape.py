import argparse
import os
import math
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ModelRepository
from preprocess import AdvPNGDataset, get_model_output


def parse_args():
    p = argparse.ArgumentParser(description='Geometric invariance: perspective-loss scan')
    p.add_argument('--model', default='tf2torch_inception_v3', type=str, help='model name')
    p.add_argument('--input_dir', default='./data', type=str, help='data dir containing labels.csv and images/')

    # 输出
    p.add_argument('--output_csv', default='./results/loss_perspective.csv', type=str, help='where to save loss csv')

    # 透视扰动网格（写入 csv 的 x/y 透视比例）
    # 两种方式：
    # 1) 显式列表：--dx_list 0,0.1,0.2
    # 2) 自动网格：--dx_start 0 --dx_end 0.5 --dx_step 0.05（同理 dy）
    # 如果指定了 *_start/_end/_step，则优先使用自动网格；否则使用 *_list。
    p.add_argument('--dx_list', default='0,0.1,0.2', type=str, help='comma-separated dx ratios')
    p.add_argument('--dy_list', default='0,0.1,0.2', type=str, help='comma-separated dy ratios')

    p.add_argument('--dx_start', default=None, type=float)
    p.add_argument('--dx_end', default=None, type=float)
    p.add_argument('--dx_step', default=None, type=float)
    p.add_argument('--dy_start', default=None, type=float)
    p.add_argument('--dy_end', default=None, type=float)
    p.add_argument('--dy_step', default=None, type=float)

    # 单次采样次数：同一个 (dx,dy) 可随机多次取 corner noise 再平均，减少随机性
    p.add_argument('--samples_per_point', default=1, type=int, help='random samples per (dx,dy) and image')

    p.add_argument('--batchsize', default=8, type=int)
    p.add_argument('--seed', default=0, type=int)

    return p.parse_args()


def _homography_dlt(src_n: torch.Tensor, dst_n: torch.Tensor) -> torch.Tensor:
    """DLT 求解单应性 H(3x3), dst ~ H * src；坐标为 [-1,1] 归一化。"""
    A_rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        x, y, u, v = x.item(), y.item(), u.item(), v.item()
        A_rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A_rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = torch.tensor(A_rows, device=src_n.device, dtype=torch.float32)
    _, _, Vh = torch.linalg.svd(A)
    h = Vh[-1, :]
    H = h.view(3, 3)
    return H / (H[2, 2] + 1e-8)


def _warp_perspective_grid_sample(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """对 batch 图像 x(B,C,H,W) 做透视变换；H: (B,3,3) 或 (3,3)。"""
    if x.dim() != 4:
        raise ValueError(f'x must be (B,C,H,W), got {tuple(x.shape)}')

    B, C, Hh, Ww = x.shape
    device = x.device
    dtype = x.dtype

    if H.dim() == 2:
        H = H.unsqueeze(0).expand(B, -1, -1)

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, Hh, device=device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, Ww, device=device, dtype=torch.float32),
        indexing='ij',
    )
    ones = torch.ones_like(xs)
    grid_h = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).t()  # (3, H*W)

    # 逐 batch 求逆并采样
    out = torch.empty_like(x)
    for i in range(B):
        try:
            H_inv = torch.linalg.inv(H[i])
        except (RuntimeError, torch._C._LinAlgError):
            out[i] = x[i]
            continue

        src_h = H_inv @ grid_h
        src_h = src_h / (src_h[2:3, :] + 1e-8)
        x_src = src_h[0, :].view(Hh, Ww)
        y_src = src_h[1, :].view(Hh, Ww)
        grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0).to(dtype)  # (1,H,W,2)
        out[i:i + 1] = F.grid_sample(x[i:i + 1], grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return out


def perspective_transform_batch(
    x: torch.Tensor,
    *,
    dx_ratio: float,
    dy_ratio: float,
) -> torch.Tensor:
    """对 batch (B,C,H,W) 做随机透视；dx_ratio/dy_ratio 控制四角扰动幅度（相对比例）。"""
    B, C, Hh, Ww = x.shape
    device = x.device

    # 源点：四个角（归一化坐标）
    src = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    # 目的点：添加噪声（归一化域里直接加），幅度由比例控制
    # 这里按“像素比例”近似映射到归一化：
    # x_norm = x_pix/(W-1)*2 - 1 => Δx_norm ≈ 2*Δx_pix/(W-1)
    dx_norm = 2.0 * float(dx_ratio)
    dy_norm = 2.0 * float(dy_ratio)

    H_list = []
    for _ in range(B):
        noise = torch.empty((4, 2), device=device, dtype=torch.float32)
        noise[:, 0].uniform_(-dx_norm, dx_norm)
        noise[:, 1].uniform_(-dy_norm, dy_norm)
        dst = (src + noise).clamp(-1.0, 1.0)
        H_mat = _homography_dlt(src, dst)
        H_list.append(H_mat)

    H_batch = torch.stack(H_list, dim=0)  # (B,3,3)
    return _warp_perspective_grid_sample(x, H_batch)


@torch.no_grad()
def batch_ce_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """返回 shape=(B,) 的逐样本 CE loss。"""
    logits = get_model_output(model, x)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    loss = F.cross_entropy(logits, y, reduction='none')
    return loss


def _make_grid_from_range(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError('step must be > 0')
    if end < start:
        raise ValueError('end must be >= start')

    n = int(math.floor((end - start) / step + 1e-12)) + 1
    vals = [start + k * step for k in range(n)]
    # 强制包含 end（避免浮点误差）
    if vals[-1] < end - 1e-9:
        vals.append(end)
    # 截断到 end
    vals = [min(v, end) for v in vals]
    # 去重（防止 end 被 append 两次）
    out = []
    for v in vals:
        if not out or abs(out[-1] - v) > 1e-9:
            out.append(v)
    return out


def _resolve_dxdy_lists(args) -> Tuple[List[float], List[float]]:
    use_dx_range = args.dx_start is not None and args.dx_end is not None and args.dx_step is not None
    use_dy_range = args.dy_start is not None and args.dy_end is not None and args.dy_step is not None

    if use_dx_range:
        dx_list = _make_grid_from_range(float(args.dx_start), float(args.dx_end), float(args.dx_step))
    else:
        dx_list = [float(s.strip()) for s in str(args.dx_list).split(',') if str(s).strip() != '']

    if use_dy_range:
        dy_list = _make_grid_from_range(float(args.dy_start), float(args.dy_end), float(args.dy_step))
    else:
        dy_list = [float(s.strip()) for s in str(args.dy_list).split(',') if str(s).strip() != '']

    return dx_list, dy_list


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_repo = ModelRepository(device)
    model_info = model_repo.get_source_model(args.model)
    model = model_info['model']
    model.eval()

    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    label_df = pd.read_csv(label_csv_path)
    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    ds = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(ds, batch_size=int(args.batchsize), shuffle=False)

    dx_list, dy_list = _resolve_dxdy_lists(args)
    samples_per_point = int(args.samples_per_point)

    rows = []

    total_batches = len(loader)
    total_points = len(dx_list) * len(dy_list)

    pbar = tqdm(loader, total=total_batches, desc=f'Scanning batches (grid={len(dx_list)}x{len(dy_list)}={total_points})')
    for batch_idx, (x, y, fn) in enumerate(pbar, start=1):
        # 更新进度条后缀
        pbar.set_postfix_str(f'batch={batch_idx}/{total_batches}')

        x = x.to(device)
        # 你工程里攻击时用 y+1，这里为了“和训练/标签对齐”也保持一致
        y = (y + 1).to(device)

        # base loss（不透视）也记录一下，便于对照
        base_loss = batch_ce_loss(model, x, y).detach().cpu()
        for i in range(x.size(0)):
            rows.append({'filename': fn[i], 'dx': 0.0, 'dy': 0.0, 'loss': float(base_loss[i].item())})

        for dx in dx_list:
            for dy in dy_list:
                if dx == 0.0 and dy == 0.0:
                    continue

                acc = torch.zeros((x.size(0),), dtype=torch.float32, device=device)
                for _ in range(samples_per_point):
                    x_p = perspective_transform_batch(x, dx_ratio=dx, dy_ratio=dy)
                    acc += batch_ce_loss(model, x_p, y)
                acc = (acc / float(samples_per_point)).detach().cpu()

                for i in range(x.size(0)):
                    rows.append({'filename': fn[i], 'dx': float(dx), 'dy': float(dy), 'loss': float(acc[i].item())})

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f'saved: {args.output_csv} (rows={len(rows)})')


if __name__ == '__main__':
    main()
