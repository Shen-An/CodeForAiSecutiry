import argparse
import gc
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models import ModelRepository
from preprocess import AdvPNGDataset, load_source_model, get_model_prediction, get_model_output


# -----------------------------
# BSR transform (parameterized)
# -----------------------------

def get_block_lengths_bsr(length: int, num_blocks: int):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def generate_bsr_params(batch_size, num_blocks_h, num_blocks_w, max_angle, device, width, height):
    """生成一组BSR参数组合, 包括随机分块长度、置换和旋转角度"""
    return {
        'w_perm': np.random.permutation(np.arange(num_blocks_h)),
        'h_perm': np.random.permutation(np.arange(num_blocks_w)),
        # 每个块独立的旋转角度 (Batch, H_blocks, W_blocks)
        'angles': torch.clamp(
            torch.randn(batch_size, num_blocks_h, num_blocks_w, device=device) * 0.05,
            -max_angle,
            max_angle,
        ),
        # 固定的分块长度，保证评估和梯度的变换一致性
        'width_length': get_block_lengths_bsr(width, num_blocks_h),
        'height_length': get_block_lengths_bsr(height, num_blocks_w),
    }


def apply_bsr_with_params(x, params, num_blocks_h, num_blocks_w):
    """使用预设参数应用BSR变换"""
    batch_size, channels, w, h = x.shape

    width_length = params.get('width_length', get_block_lengths_bsr(w, num_blocks_h))
    height_length = params.get('height_length', get_block_lengths_bsr(h, num_blocks_w))

    # 按宽度分块
    x_split_w = torch.split(x, width_length, dim=2)

    rotated_blocks = []
    for w_idx in range(num_blocks_h):
        w_block = x_split_w[w_idx]
        h_blocks = torch.split(w_block, height_length, dim=3)

        rotated_strip = []
        for h_idx in range(num_blocks_w):
            block = h_blocks[h_idx]

            angles_param = params['angles']
            current_angles = angles_param[:, w_idx, h_idx] if angles_param.dim() == 3 else angles_param

            rotated_block = block.clone()
            for i in range(batch_size):
                if block.shape[2] > 1 and block.shape[3] > 1:
                    angle = current_angles[i]
                    angle_matrix = torch.tensor(
                        [
                            [math.cos(angle), -math.sin(angle), 0],
                            [math.sin(angle), math.cos(angle), 0],
                        ],
                        dtype=torch.float32,
                        device=x.device,
                    ).unsqueeze(0)

                    grid = F.affine_grid(angle_matrix, block[i:i + 1].size(), align_corners=False)
                    rotated_block[i:i + 1] = F.grid_sample(
                        block[i:i + 1],
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False,
                    )

            rotated_strip.append(rotated_block)

        rotated_strip_perm = [rotated_strip[i] for i in params['h_perm']]
        rotated_blocks.append(torch.cat(rotated_strip_perm, dim=3))

    return torch.cat([rotated_blocks[i] for i in params['w_perm']], dim=2)


def mutate_bsr_params(params, num_blocks_h, num_blocks_w, max_angle, beta):
    """对BSR参数进行变异"""
    noise = torch.randn_like(params['angles']) * beta
    new_angles = torch.clamp(params['angles'] + noise, -max_angle, max_angle)

    def mutate_perm(perm, n):
        new_perm = perm.copy()
        if n > 1:
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
        return new_perm

    new_params = {
        'w_perm': mutate_perm(params['w_perm'], num_blocks_h),
        'h_perm': mutate_perm(params['h_perm'], num_blocks_w),
        'angles': new_angles,
        # 保持分块长度固定（关键：保证虚拟更新和潜力loss中的变换一致）
        'width_length': params.get('width_length'),
        'height_length': params.get('height_length'),
    }
    return new_params


# -----------------------------
# helpers
# -----------------------------

def _clip_by_eps(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


def _l1_normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    return grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)


# -----------------------------
# EATA v2.0: potential-loss guided ES + MI-FGSM
# -----------------------------

def evaluate_potential_loss_and_grad(model,
                                    x_adv: torch.Tensor,
                                    y: torch.Tensor,
                                    params,
                                    alpha: float,
                                    eps: float,
                                    x_orig: torch.Tensor,
                                    num_blocks_h: int,
                                    num_blocks_w: int):
    """计算 g_pi 与 potential loss。

    g_pi = ∇_{x_adv} CE(f(T(x_adv;pi)), y)
    x_tilde = Clip(x_adv + alpha*sign(g_pi))
    L_pot(pi) = CE(f(T(x_tilde;pi)), y)

    返回: (L_pot_scalar, g_pi_tensor)
    """
    x_leaf = x_adv.detach().clone().requires_grad_(True)

    x_trans = apply_bsr_with_params(x_leaf, params, num_blocks_h, num_blocks_w)
    out = get_model_output(model, x_trans)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    elif out.dim() > 2:
        out = out.view(out.size(0), -1)

    loss = F.cross_entropy(out, y)
    g_pi = torch.autograd.grad(loss, [x_leaf], retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        x_tilde = _clip_by_eps(x_leaf + alpha * torch.sign(g_pi), x_orig, eps)

    with torch.no_grad():
        x_tilde_trans = apply_bsr_with_params(x_tilde, params, num_blocks_h, num_blocks_w)
        out2 = get_model_output(model, x_tilde_trans)
        if out2.dim() == 1:
            out2 = out2.unsqueeze(0)
        elif out2.dim() > 2:
            out2 = out2.view(out2.size(0), -1)
        pot_loss = F.cross_entropy(out2, y)

    return float(pot_loss.detach().item()), g_pi.detach()


def eata_v2_mifgsm_attack(x,
                         y,
                         model,
                         eps=16 / 255.0,
                         iterations=10,
                         mu: float = 1.0,
                         num_blocks_h: int = 2,
                         num_blocks_w: int = 2,
                         max_angle: float = 0.2,
                         n_population: int = 10,
                         k_elite: int = 5,
                         beta: float = 0.1,
                         mutation_rate: float = 0.7):
    """EATA v2.0：用 potential loss 做精英筛选，精英梯度聚合后按 MI-FGSM 更新。"""

    model.eval()
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)

    alpha = eps / max(1, iterations)

    prev_elites = None

    for _ in range(iterations):
        # 1) 初始化/继承种群
        population = []
        if prev_elites:
            for p in prev_elites:
                if np.random.random() < mutation_rate:
                    population.append(mutate_bsr_params(p, num_blocks_h, num_blocks_w, max_angle, beta))
                else:
                    population.append(p)

        if len(population) < n_population:
            for _k in range(n_population - len(population)):
                population.append(
                    generate_bsr_params(
                        batch_size=x_adv.size(0),
                        num_blocks_h=num_blocks_h,
                        num_blocks_w=num_blocks_w,
                        max_angle=max_angle,
                        device=x_adv.device,
                        width=x_adv.size(2),
                        height=x_adv.size(3),
                    )
                )

        # 2) 对每个 pi 计算 potential loss (需要梯度一次)
        pot_losses = []
        grads = []
        for p in population:
            pot_loss, g_pi = evaluate_potential_loss_and_grad(
                model=model,
                x_adv=x_adv,
                y=y,
                params=p,
                alpha=alpha,
                eps=eps,
                x_orig=x,
                num_blocks_h=num_blocks_h,
                num_blocks_w=num_blocks_w,
            )
            pot_losses.append(pot_loss)
            grads.append(g_pi)

        # 3) 选择 Top-K (按 potential loss 降序)
        idxs = np.argsort(np.array(pot_losses))[::-1][:max(1, k_elite)]
        elites = [population[i] for i in idxs]
        elite_grads = [grads[i] for i in idxs]
        prev_elites = elites

        # 4) 梯度聚合（简单均值；如需余弦权重可继续加）
        grad_bar = torch.stack(elite_grads, dim=0).mean(dim=0)
        grad_bar = _l1_normalize_grad(grad_bar)

        # 5) MI-FGSM 更新
        momentum = mu * momentum + grad_bar
        x_adv = _clip_by_eps(x_adv + alpha * torch.sign(momentum), x, eps)

    return x_adv.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="EATA v2.0 attack (Potential-loss guided ES)")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/EATA_v2/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/EATA_v2/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')

    # BSR params
    parser.add_argument("--num_blocks_h", default=2, type=int, help="number of blocks h")
    parser.add_argument("--num_blocks_w", default=2, type=int, help="number of blocks w")
    parser.add_argument("--max_angle", default=0.2, type=float, help="maximum angle (radians)")

    # ES params
    parser.add_argument("--n_population", default=10, type=int, help="population size (M)")
    parser.add_argument("--k_elite", default=5, type=int, help="elite size (K)")
    parser.add_argument("--beta", default=0.1, type=float, help="mutation scale")
    parser.add_argument("--mutation_rate", default=0.7, type=float, help="inherit elites mutation rate")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model_repo = ModelRepository(device)

    source_model = load_source_model(args.model, device)

    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    label_df = pd.read_csv(label_csv_path)

    label_df['exists'] = label_df['filename'].apply(lambda fn: os.path.isfile(os.path.join(img_root, fn)))
    label_df = label_df[label_df['exists']].drop(columns=['exists'])

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    orig_dataset = AdvPNGDataset(img_root, label_df, transform)
    loader = DataLoader(orig_dataset, batch_size=args.batchsize, shuffle=False)

    source_results = []
    adv_images_storage = []

    print(f"\n[Step 1/3] Attacking in Memory...")

    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)

        x_adv_batch = eata_v2_mifgsm_attack(
            x=x_batch,
            y=y_batch,
            model=source_model,
            eps=args.eps,
            iterations=args.iterations,
            mu=args.mu,
            num_blocks_h=args.num_blocks_h,
            num_blocks_w=args.num_blocks_w,
            max_angle=args.max_angle,
            n_population=args.n_population,
            k_elite=args.k_elite,
            beta=args.beta,
            mutation_rate=args.mutation_rate,
        )

        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item()) + 1
            s_adv_idx = int(source_adv_preds[i]) + 1
            s_orig_idx = int(source_orig_preds[i]) + 1
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label,
            })

    del source_model
    torch.cuda.empty_cache()

    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")
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

        del model
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[Step 3/3] Saving results and images to disk...")

    os.makedirs(args.output_adv_dir, exist_ok=True)
    for idx, res in enumerate(source_results):
        fn = res["filename"]
        save_image(all_adv_tensors[idx], os.path.join(args.output_adv_dir, fn))

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

    total_samples = len(source_results)
    source_rate = sum(1 for r in source_results if r['source_attack_success']) / total_samples * 100
    print(f"\nSource Model ({args.model}) Success Rate: {source_rate:.1f}%")
    for name, count in model_success_counts.items():
        print(f"  {name}: {count}/{total_samples} ({count / total_samples * 100:.1f}%)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(final_rows).to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
