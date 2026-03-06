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
from ppsp.BSR import get_block_lengths_bsr
from ppsp.perspective_rotate import _homography_dlt, _warp_perspective_grid_sample


def parse_args():
    p = argparse.ArgumentParser(description='Loss under block-wise perspective (dx,dy grid) + block-wise horizontal flip')
    p.add_argument('--model', default='tf2torch_inception_v3', type=str)
    p.add_argument('--input_dir', default='./data', type=str)
    p.add_argument('--batchsize', default=8, type=int)

    # block split
    p.add_argument('--num_blocks_h', default=2, type=int)
    p.add_argument('--num_blocks_w', default=2, type=int)

    # dx/dy grid in ratio domain
    p.add_argument('--dx_start', default=0.1, type=float)
    p.add_argument('--dx_end', default=0.5, type=float)
    p.add_argument('--dx_step', default=0.1, type=float)
    p.add_argument('--dy_start', default=0.1, type=float)
    p.add_argument('--dy_end', default=0.5, type=float)
    p.add_argument('--dy_step', default=0.1, type=float)

    # block flip
    p.add_argument('--flip_prob', default=0.5, type=float, help='per-block horizontal flip probability')

    # per point sampling
    p.add_argument('--samples_per_point', default=1, type=int)

    # output
    p.add_argument('--output_csv', default='./results/loss_block_per.csv', type=str)
    p.add_argument('--output_mean_csv', default='./results/loss_block_per_mean.csv', type=str)

    p.add_argument('--seed', default=0, type=int)
    return p.parse_args()


def _make_grid(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError('step must be > 0')
    if end < start:
        raise ValueError('end must be >= start')
    n = int(math.floor((end - start) / step + 1e-12)) + 1
    vals = [start + k * step for k in range(n)]
    if vals[-1] < end - 1e-9:
        vals.append(end)
    vals = [min(v, end) for v in vals]
    out: List[float] = []
    for v in vals:
        if not out or abs(out[-1] - v) > 1e-9:
            out.append(float(v))
    return out


def _split_blocks(x: torch.Tensor, width_length: Tuple[int, ...], height_length: Tuple[int, ...]):
    x_split_w = torch.split(x, width_length, dim=2)
    blocks = []
    for w_block in x_split_w:
        blocks.append(list(torch.split(w_block, height_length, dim=3)))
    return blocks


def _merge_blocks(blocks):
    strips = [torch.cat(row, dim=3) for row in blocks]
    return torch.cat(strips, dim=2)


def _rand_perspective_params_ratio(block_h: int, block_w: int, dx_ratio: float, dy_ratio: float, device):
    """生成单个 block 的(src_n, dst_n)，噪声幅度由 dx_ratio/dy_ratio（相对比例）控制。"""
    src = torch.tensor(
        [[0.0, 0.0], [block_w - 1.0, 0.0], [block_w - 1.0, block_h - 1.0], [0.0, block_h - 1.0]],
        device=device,
        dtype=torch.float32,
    )

    dx = float(dx_ratio) * float(block_w)
    dy = float(dy_ratio) * float(block_h)
    noise = torch.empty((4, 2), device=device, dtype=torch.float32)
    noise[:, 0].uniform_(-dx, dx)
    noise[:, 1].uniform_(-dy, dy)

    dst = src + noise
    dst[:, 0].clamp_(0.0, block_w - 1.0)
    dst[:, 1].clamp_(0.0, block_h - 1.0)

    def normalize_xy(pts):
        x = pts[:, 0] / max(1.0, (block_w - 1.0)) * 2.0 - 1.0
        y = pts[:, 1] / max(1.0, (block_h - 1.0)) * 2.0 - 1.0
        return torch.stack([x, y], dim=1)

    return normalize_xy(src), normalize_xy(dst)


def block_perspective_flip(
    x: torch.Tensor,
    *,
    num_blocks_h: int,
    num_blocks_w: int,
    dx_ratio: float,
    dy_ratio: float,
    flip_prob: float,
) -> torch.Tensor:
    """对 batch 做：分块透视(dx,dy 控制) + 分块水平翻转。"""
    B, C, H, W = x.shape
    width_length = get_block_lengths_bsr(H, int(num_blocks_h))
    height_length = get_block_lengths_bsr(W, int(num_blocks_w))

    blocks = _split_blocks(x, width_length, height_length)

    for i in range(int(num_blocks_h)):
        for j in range(int(num_blocks_w)):
            block = blocks[i][j]
            bH, bW = int(block.shape[2]), int(block.shape[3])
            if bH <= 1 or bW <= 1:
                continue

            out_block = block.clone()
            for bi in range(B):
                single = block[bi:bi + 1]
                # perspective
                src_n, dst_n = _rand_perspective_params_ratio(bH, bW, float(dx_ratio), float(dy_ratio), device=single.device)
                H_mat = _homography_dlt(src_n, dst_n)
                single = _warp_perspective_grid_sample(single, H_mat)

                # flip
                if flip_prob and float(flip_prob) > 0.0:
                    if torch.rand(1, device=single.device).item() < float(flip_prob):
                        single = torch.flip(single, dims=[3])

                out_block[bi:bi + 1] = single

            blocks[i][j] = out_block

    return _merge_blocks(blocks)


@torch.no_grad()
def batch_ce_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = get_model_output(model, x)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    return F.cross_entropy(logits, y, reduction='none')


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_repo = ModelRepository(device)
    model = model_repo.get_source_model(args.model)['model']
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

    dx_list = _make_grid(float(args.dx_start), float(args.dx_end), float(args.dx_step))
    dy_list = _make_grid(float(args.dy_start), float(args.dy_end), float(args.dy_step))
    spp = int(args.samples_per_point)

    rows = []

    total_batches = len(loader)
    total_points = len(dx_list) * len(dy_list)
    pbar = tqdm(loader, total=total_batches, desc=f'BlockPF loss scan (grid={len(dx_list)}x{len(dy_list)}={total_points}, flip_prob={args.flip_prob}, spp={spp})')

    for batch_idx, (x, y, fn) in enumerate(pbar, start=1):
        pbar.set_postfix_str(f'batch={batch_idx}/{total_batches}')

        x = x.to(device)
        y = (y + 1).to(device)

        # baseline
        base_loss = batch_ce_loss(model, x, y).detach().cpu()
        for i in range(x.size(0)):
            rows.append({'filename': fn[i], 'dx': 0.0, 'dy': 0.0, 'flip_prob': float(args.flip_prob), 'loss': float(base_loss[i].item()), 'samples_per_point': 0})

        for dx in dx_list:
            for dy in dy_list:
                acc = torch.zeros((x.size(0),), dtype=torch.float32, device=device)
                for _ in range(spp):
                    x_aug = block_perspective_flip(
                        x,
                        num_blocks_h=int(args.num_blocks_h),
                        num_blocks_w=int(args.num_blocks_w),
                        dx_ratio=float(dx),
                        dy_ratio=float(dy),
                        flip_prob=float(args.flip_prob),
                    )
                    acc += batch_ce_loss(model, x_aug, y)

                acc = (acc / float(spp)).detach().cpu()
                for i in range(x.size(0)):
                    rows.append({'filename': fn[i], 'dx': float(dx), 'dy': float(dy), 'flip_prob': float(args.flip_prob), 'loss': float(acc[i].item()), 'samples_per_point': spp})

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    mean_df = (
        df.groupby(['dx', 'dy'], as_index=False)
        .agg(loss_mean=('loss', 'mean'), loss_std=('loss', 'std'), count=('loss', 'size'))
        .sort_values(['dx', 'dy'], ascending=True)
        .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(args.output_mean_csv) or '.', exist_ok=True)
    mean_df.to_csv(args.output_mean_csv, index=False)

    print(f'saved per-image csv: {args.output_csv} (rows={len(df)})')
    print(f'saved mean csv: {args.output_mean_csv} (rows={len(mean_df)})')


if __name__ == '__main__':
    main()
