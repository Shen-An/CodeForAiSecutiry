import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Aggregate (dx,dy) loss from loss_perspective.csv')
    p.add_argument('--input_csv', default='./results/loss_perspective.csv', type=str, help='input csv path')
    p.add_argument('--output_csv', default='./results/loss_perspective_mean.csv', type=str, help='output aggregated csv path')

    # 浮点分组容易因为 0.30000000004 这种误差聚不到一块；可用 round_digits 先做四舍五入再 groupby
    p.add_argument('--round_digits', default=6, type=int, help='round dx/dy to N digits before grouping')

    return p.parse_args()


def aggregate_loss_csv(input_csv: str, output_csv: str, *, round_digits: int = 6) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required = {'dx', 'dy', 'loss'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Input csv missing columns: {sorted(missing)}. Got: {list(df.columns)}')

    # 转数值并清理
    df['dx'] = pd.to_numeric(df['dx'], errors='coerce')
    df['dy'] = pd.to_numeric(df['dy'], errors='coerce')
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    df = df.dropna(subset=['dx', 'dy', 'loss']).copy()

    if round_digits is not None and int(round_digits) >= 0:
        df['dx'] = df['dx'].round(int(round_digits))
        df['dy'] = df['dy'].round(int(round_digits))

    agg = (
        df.groupby(['dx', 'dy'], as_index=False)
        .agg(
            loss_mean=('loss', 'mean'),
            loss_std=('loss', 'std'),
            count=('loss', 'size'),
        )
        .sort_values(['dx', 'dy'], ascending=True)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    agg.to_csv(output_csv, index=False)
    return agg


def main():
    args = parse_args()
    agg = aggregate_loss_csv(args.input_csv, args.output_csv, round_digits=int(args.round_digits))
    print(f'saved: {args.output_csv} (rows={len(agg)})')


if __name__ == '__main__':
    main()
