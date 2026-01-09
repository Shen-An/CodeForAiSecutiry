import argparse
import gc
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from models import ModelRepository
from torch.utils.data import DataLoader, TensorDataset
import math

from preprocess import AdvPNGDataset, get_model_output, load_source_model, get_model_prediction


# ==========================================
# 0. 辅助函数 (TPX 交叉)
# ==========================================

def perform_tpx(target, mutant):
    """
    Two-Point Crossover (TPX) specifically for Permutations.
    保留 target 的片段，剩余部分由 mutant 按顺序填充。
    """
    size = len(target)
    c1, c2 = sorted(random.sample(range(size), 2))

    child = np.full(size, -1, dtype=target.dtype)
    child[c1:c2] = target[c1:c2]

    filled_set = set(target[c1:c2])
    fill_positions = list(range(0, c1)) + list(range(c2, size))
    fill_idx = 0

    for val in mutant:
        if val not in filled_set:
            if fill_idx < len(fill_positions):
                child[fill_positions[fill_idx]] = val
                fill_idx += 1
            else:
                break
    return child


def expand_block_indices(perm_batch, bounds, total_size, device):
    """
    将块级别的置换索引扩展为像素级别的索引
    perm_batch: [B, K] 块的置换
    bounds: [K+1] 块的边界
    total_size: H or W
    return: [B, total_size]
    """
    B, K = perm_batch.shape
    full_indices = torch.zeros((B, total_size), dtype=torch.long, device=device)

    # 预先生成所有块的索引片段并移动到设备
    block_ranges = []
    for k in range(K):
        start = bounds[k]
        end = bounds[k + 1]
        block_ranges.append(torch.arange(start, end, device=device))

    for b in range(B):
        p = perm_batch[b]
        gathered_ranges = [block_ranges[i] for i in p]
        full_indices[b] = torch.cat(gathered_ranges)

    return full_indices


# ==========================================
# 1. 单样本差分进化 (Per-image DE)
# ==========================================

def run_per_image_de(x_batch, y_batch, model, pop_size=20, generations=10, Pm=0.5, Pc=0.3, blocks=0):
    """
    针对 Batch 中的每一张图片独立学习最优置换排列。
    blocks: 如果 > 0，则将图片划分为 blocks * blocks 的块进行置换。
    """
    device = x_batch.device
    B, C, H, W = x_batch.shape

    # 确定基因长度 (Permutation Size)
    use_blocks = (blocks > 0 and blocks < min(H, W))

    dim_H = blocks if use_blocks else H
    dim_W = blocks if use_blocks else W

    # 预计算块边界 (如果使用分块)
    if use_blocks:
        h_bounds = np.linspace(0, H, blocks + 1, dtype=int)
        w_bounds = np.linspace(0, W, blocks + 1, dtype=int)
    else:
        h_bounds = None
        w_bounds = None

    # 初始化种群 (CPU Numpy 用于进化操作)
    pop_sigma = np.stack([np.stack([np.random.permutation(dim_H) for _ in range(pop_size)]) for _ in range(B)])
    pop_xi = np.stack([np.stack([np.random.permutation(dim_W) for _ in range(pop_size)]) for _ in range(B)])

    # 辅助函数：评估种群适应度
    def evaluate_population(curr_sigma, curr_xi):
        fitness = torch.zeros((B, pop_size), device=device)
        for p in range(pop_size):
            sigma_p_raw = torch.from_numpy(curr_sigma[:, p, :]).to(device).long()
            xi_p_raw = torch.from_numpy(curr_xi[:, p, :]).to(device).long()

            if use_blocks:
                sigma_p = expand_block_indices(sigma_p_raw, h_bounds, H, device)
                xi_p = expand_block_indices(xi_p_raw, w_bounds, W, device)
            else:
                sigma_p = sigma_p_raw
                xi_p = xi_p_raw

            R = torch.zeros((B, H, H), device=device)
            C_mat = torch.zeros((B, W, W), device=device)
            batch_indices = torch.arange(B, device=device).view(-1, 1)

            R[batch_indices, torch.arange(H, device=device), sigma_p] = 1.0
            C_mat[batch_indices, xi_p, torch.arange(W, device=device)] = 1.0

            x_p = torch.matmul(R.unsqueeze(1), x_batch)
            x_p = torch.matmul(x_p, C_mat.unsqueeze(1))

            with torch.no_grad():
                logits = model(x_p)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                fitness[:, p] = F.cross_entropy(logits, y_batch, reduction='none')
        return fitness

    pop_fitness = evaluate_population(pop_sigma, pop_xi)

    for g in range(generations):
        trial_sigma = pop_sigma.copy()
        trial_xi = pop_xi.copy()

        for b in range(B):
            for i in range(pop_size):
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = random.sample(candidates, 3)

                mutant_sigma = pop_sigma[b, r1].copy()
                if random.random() < Pm:
                    diff_indices = np.where(pop_sigma[b, r2] != pop_sigma[b, r3])[0]
                    if len(diff_indices) >= 2:
                        pos_a, pos_b = random.sample(list(diff_indices), 2)
                        mutant_sigma[pos_a], mutant_sigma[pos_b] = mutant_sigma[pos_b], mutant_sigma[pos_a]

                if random.random() < Pc:
                    trial_sigma[b, i] = perform_tpx(pop_sigma[b, i], mutant_sigma)
                else:
                    trial_sigma[b, i] = pop_sigma[b, i]

                mutant_xi = pop_xi[b, r1].copy()
                if random.random() < Pm:
                    diff_indices = np.where(pop_xi[b, r2] != pop_xi[b, r3])[0]
                    if len(diff_indices) >= 2:
                        pos_a, pos_b = random.sample(list(diff_indices), 2)
                        mutant_xi[pos_a], mutant_xi[pos_b] = mutant_xi[pos_b], mutant_xi[pos_a]

                if random.random() < Pc:
                    trial_xi[b, i] = perform_tpx(pop_xi[b, i], mutant_xi)
                else:
                    trial_xi[b, i] = pop_xi[b, i]

        trial_fitness = evaluate_population(trial_sigma, trial_xi)

        improve_mask = (trial_fitness > pop_fitness).cpu().numpy()

        for b in range(B):
            for i in range(pop_size):
                if improve_mask[b, i]:
                    pop_sigma[b, i] = trial_sigma[b, i]
                    pop_xi[b, i] = trial_xi[b, i]

        pop_fitness = torch.where(trial_fitness > pop_fitness, trial_fitness, pop_fitness)

    return torch.from_numpy(pop_sigma).to(device), torch.from_numpy(pop_xi).to(device), pop_fitness


# ==========================================
# 2. 提取并应用单样本 P_list
# ==========================================

def get_per_image_p_lists(pop_sigma, pop_xi, fitness_scores, K=5, blocks=0, img_size=(299, 299)):
    """
    为 Batch 中的每一张图片提取专属的 Top-K 矩阵集合
    返回格式: [B, K, 2 (R, C)]
    """
    B, pop_size, _ = pop_sigma.shape
    H, W = img_size
    device = pop_sigma.device

    use_blocks = (blocks > 0 and blocks < min(H, W))
    if use_blocks:
        h_bounds = np.linspace(0, H, blocks + 1, dtype=int)
        w_bounds = np.linspace(0, W, blocks + 1, dtype=int)

    top_k_indices = torch.topk(fitness_scores, K, dim=1).indices

    batch_p_lists = []
    for b in range(B):
        img_p_list = []
        for k in range(K):
            idx = top_k_indices[b, k]

            sigma_raw = pop_sigma[b, idx]
            xi_raw = pop_xi[b, idx]

            if use_blocks:
                sigma = expand_block_indices(sigma_raw.unsqueeze(0), h_bounds, H, device).squeeze(0)
                xi = expand_block_indices(xi_raw.unsqueeze(0), w_bounds, W, device).squeeze(0)
            else:
                sigma = sigma_raw
                xi = xi_raw

            R = torch.zeros((H, H), device=device)
            C = torch.zeros((W, W), device=device)
            R[torch.arange(H), sigma] = 1.0
            C[xi, torch.arange(W)] = 1.0
            img_p_list.append((R, C))
        batch_p_lists.append(img_p_list)

    return batch_p_lists


# ==========================================
# 3. 集成攻击逻辑 (单样本集成)
# ==========================================

def ldr_ensemble_attack(x, y, model, batch_p_lists, eps=16 / 255.0, iterations=20, mu=1.0):
    """
    每个样本使用其专属的 K 个矩阵进行梯度集成
    """
    model.eval()
    B, C, H, W = x.shape
    device = x.device
    K = len(batch_p_lists[0])
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)
    alpha_step = eps / iterations

    for i in range(iterations):
        x_adv.requires_grad = True
        total_grad = torch.zeros_like(x_adv)

        for k in range(K):
            R_batch = torch.stack([batch_p_lists[b][k][0] for b in range(B)]).to(device)
            C_batch = torch.stack([batch_p_lists[b][k][1] for b in range(B)]).to(device)

            x_permuted = torch.matmul(R_batch.unsqueeze(1), x_adv)
            x_permuted = torch.matmul(x_permuted, C_batch.unsqueeze(1))

            output = model(x_permuted)[0] if isinstance(model(x_permuted), (tuple, list)) else model(x_permuted)
            loss = F.cross_entropy(output, y)

            model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data
            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            total_grad += grad
            x_adv.grad.zero_()

        avg_grad = total_grad / K
        momentum = mu * momentum + avg_grad
        x_adv = x_adv.detach() + alpha_step * torch.sign(momentum)

        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="LDR Batch-level Attack")
    parser.add_argument("--model", default='inception_v3', type=str)
    parser.add_argument('--output_adv_dir', default='./results/LDR/images', type=str)
    parser.add_argument('--output_csv', default='./results/LDR/results.csv', type=str)
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=20, type=int)
    parser.add_argument('--mu', default=1.0, type=float)
    parser.add_argument('--k_ensemble', default=5, type=int, help='集成矩阵数量')
    parser.add_argument('--blocks', default=2, type=int, help='分块置换的块数量 (NxN), 默认2x2. 0或1表示全像素置换')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

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

        pop_sigma, pop_xi, scores = run_per_image_de(x_batch, y_batch, source_model,
                                                     pop_size=15, generations=5,
                                                     blocks=args.blocks)

        p_list = get_per_image_p_lists(pop_sigma, pop_xi, scores,
                                       K=args.k_ensemble,
                                       blocks=args.blocks,
                                       img_size=(299, 299))

        x_adv_batch = ldr_ensemble_attack(x_batch, y_batch, source_model, p_list,
                                          eps=args.eps, iterations=args.iterations, mu=args.mu)
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item()) + 1
            s_adv_idx = int(source_adv_preds[i]) + 1
            s_orig_idx = int(source_orig_preds[i]) + 1
            print(f"{s_orig_idx}\t{s_adv_idx}\t{true_label}")
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

    del source_model
    torch.cuda.empty_cache()

    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    model_repo = ModelRepository(device)

    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")

        current_model_info = model_repo.get_model_info(model_name)
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
    torch.cuda.empty_cache();
    main()
