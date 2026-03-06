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
from ppsp.perspective_rotate import _perspective_permutation_transform


def parse_args():
    p = argparse.ArgumentParser(description='Loss under block-wise perspective + horizontal flip')
    p.add_argument('--model', default='tf2torch_inception_v3', type=str)
    p.add_argument('--input_dir', default='./data', type=str)
    p.add_argument('--batchsize', default=8, type=int)

    # 分块参数
    p.add_argument('--num_blocks_h', default=2, type=int)
    p.add_argument('--num_blocks_w', default=2, type=int)

    # 变换强度
    p.add_argument('--distortion_list', default='0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5', type=str,
                   help='comma-separated distortion_scale values (same meaning as ppsp/perspective_rotate.py)')
    p.add_argument('--flip_prob_list', default='0,0.25,0.5,0.75,1.0', type=str,
                   help='comma-separated flip_prob values')

    # 每个点采样次数：同一张图、同一个 (distortion, flip_prob) 下随机采样 N 次取平均
    p.add_argument('--samples_per_point', default=1, type=int)

    # 输出
    p.add_argument('--output_csv', default='./results/loss_block_pf.csv', type=str,
                   help='per-image per-point loss csv')
    p.add_argument('--output_mean_csv', default='./results/loss_block_pf_mean.csv', type=str,
                   help='aggregated mean loss per (distortion_scale, flip_prob)')

    p.add_argument('--seed', default=0, type=int)

    return p.parse_args()


@torch.no_grad()
def batch_ce_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = get_model_output(model, x)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    return F.cross_entropy(logits, y, reduction='none')


def _parse_floats(s: str) -> List[float]:
    out = []
    for part in str(s).split(','):
        part = str(part).strip()
        if part == '':
            continue
        out.append(float(part))
    return out


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

    distortion_list = _parse_floats(args.distortion_list)
    flip_prob_list = _parse_floats(args.flip_prob_list)
    spp = int(args.samples_per_point)

    rows = []

    total_batches = len(loader)
    grid_points = len(distortion_list) * len(flip_prob_list)
    pbar = tqdm(loader, total=total_batches, desc=f'BPF loss scan (grid={grid_points}, spp={spp})')

    for batch_idx, (x, y, fn) in enumerate(pbar, start=1):
        pbar.set_postfix_str(f'batch={batch_idx}/{total_batches}')

        x = x.to(device)
        y = (y + 1).to(device)  # 与你的其它脚本保持一致

        # baseline（不做变换）
        base_loss = batch_ce_loss(model, x, y).detach().cpu()
        for i in range(x.size(0)):
            rows.append({
                'filename': fn[i],
                'distortion_scale': 0.0,
                'flip_prob': 0.0,
                'loss': float(base_loss[i].item()),
                'samples_per_point': 0,
            })

        for d in distortion_list:
            for fp in flip_prob_list:
                # baseline 已经记录过 0/0
                if float(d) == 0.0 and float(fp) == 0.0:
                    continue

                acc = torch.zeros((x.size(0),), dtype=torch.float32, device=device)
                for _ in range(spp):
                    # 关键：复用工程里的“分块透视 + 分块水平翻转”实现
                    x_aug = _perspective_permutation_transform(
                        x,
                        num_blocks_h=int(args.num_blocks_h),
                        num_blocks_w=int(args.num_blocks_w),
                        num_copies=1,
                        distortion_scale=float(d),
                        max_angle=0.0,          # 这里按你的需求：只做透视 + flip，不做旋转
                        flip_prob=float(fp),
                        stretch_factor=0.0,
                    )
                    acc += batch_ce_loss(model, x_aug, y)

                acc = (acc / float(spp)).detach().cpu()
                for i in range(x.size(0)):
                    rows.append({
                        'filename': fn[i],
                        'distortion_scale': float(d),
                        'flip_prob': float(fp),
                        'loss': float(acc[i].item()),
                        'samples_per_point': spp,
                    })

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    # 聚合：同一 (distortion_scale, flip_prob) 下对所有图片取均值
    mean_df = (
        df.groupby(['distortion_scale', 'flip_prob'], as_index=False)
        .agg(loss_mean=('loss', 'mean'), loss_std=('loss', 'std'), count=('loss', 'size'))
        .sort_values(['distortion_scale', 'flip_prob'], ascending=True)
        .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(args.output_mean_csv) or '.', exist_ok=True)
    mean_df.to_csv(args.output_mean_csv, index=False)

    print(f'saved per-image csv: {args.output_csv} (rows={len(df)})')
    print(f'saved mean csv: {args.output_mean_csv} (rows={len(mean_df)})')


if __name__ == '__main__':
    main()
