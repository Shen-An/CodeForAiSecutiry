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
from models import ModelRepository
from torch.utils.data import DataLoader,  TensorDataset
import math

from preprocess import AdvPNGDataset, get_model_output, load_source_model, get_model_prediction

def generate_bsr_params(batch_size, num_blocks_h, num_blocks_w, max_angle, device, width, height):
    """生成一组BSR参数组合, 包括随机分块长度、置换和旋转角度"""
    return {
        'w_perm': np.random.permutation(np.arange(num_blocks_h)),
        'h_perm': np.random.permutation(np.arange(num_blocks_w)),
        # 每个块独立的旋转角度 (Batch, H_blocks, W_blocks)
        'angles': torch.clamp(torch.randn(batch_size, num_blocks_h, num_blocks_w, device=device) * 0.05, -max_angle, max_angle),
        # 固定的分块长度，保证评估和梯度的变换一致性
        'width_length': get_block_lengths_bsr(width, num_blocks_h),
        'height_length': get_block_lengths_bsr(height, num_blocks_w)
    }

def apply_bsr_with_params(x, params, num_blocks_h, num_blocks_w):
    """使用预设参数应用BSR变换"""
    batch_size, channels, w, h = x.shape
    
    # 使用参数中固定的分块长度 (如果存在)，否则重新生成 (兼容旧逻辑)
    width_length = params.get('width_length', get_block_lengths_bsr(w, num_blocks_h))
    height_length = params.get('height_length', get_block_lengths_bsr(h, num_blocks_w))

    # 应用宽度打乱
    x_split_w = torch.split(x, width_length, dim=2)
    
    # 按照记录的参数重组
    rotated_blocks = []
    for w_idx in range(num_blocks_h):
        w_block = x_split_w[w_idx]
        h_blocks = torch.split(w_block, height_length, dim=3)
        
        rotated_strip = []
        for h_idx in range(num_blocks_w):
            block = h_blocks[h_idx]
            
            # 获取当前块的角度
            # params['angles'] 可能是 (B,) 或 (B, H, W)
            angles_param = params['angles']
            if len(angles_param.shape) == 3:
                # EATA standard: independent angle per block
                current_angles = angles_param[:, w_idx, h_idx]
            else:
                 # Fallback: shared angle
                current_angles = angles_param

            # 旋转逻辑
            rotated_block = block.clone()
            for i in range(batch_size):
                if block.shape[2] > 1 and block.shape[3] > 1: # 确保块足够大
                    angle = current_angles[i]
                    angle_matrix = torch.tensor([
                        [math.cos(angle), -math.sin(angle), 0],
                        [math.sin(angle), math.cos(angle), 0]
                    ], dtype=torch.float32, device=x.device).unsqueeze(0)
                    
                    grid = F.affine_grid(angle_matrix, block[i:i + 1].size(), align_corners=False)
                    rotated_block[i:i + 1] = F.grid_sample(block[i:i + 1], grid, mode='bilinear',
                                                           padding_mode='zeros', align_corners=False)
            
            rotated_strip.append(rotated_block)
            
        # 使用传入的 h_perm
        rotated_strip_perm = [rotated_strip[i] for i in params['h_perm']]
        rotated_blocks.append(torch.cat(rotated_strip_perm, dim=3))
    
    # 使用传入的 w_perm
    return torch.cat([rotated_blocks[i] for i in params['w_perm']], dim=2)

def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def parse_args():
    parser = argparse.ArgumentParser(description="EATA attack")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/EATA/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/EATA/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=2, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--diversity_prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument("--num_blocks_h",default=2,type=int,help="number of blocks h")
    parser.add_argument("--num_blocks_w",default=2,type=int,help="number of blocks w")
    parser.add_argument("--max_angle",default=2,type=int,help="maximum angle")
    parser.add_argument("--num_samples",default=10,type=int,help="population size (M)")
    parser.add_argument("--num_keep",default=5,type=int,help="elite size (K)")
    parser.add_argument("--beta", default=0.1, type=float, help="mutation scale")
    return parser.parse_args()


def admix(x, portion=0.2, size=3):
    """混合输入变换"""
    indices = torch.randperm(x.size(0))
    admixed = []
    for _ in range(size):
        admixed_x = x + portion * x[indices]
        admixed.append(admixed_x)
    return torch.cat(admixed, dim=0)


def input_diversity(x, image_width=299, image_resize=331, prob=0.5):
    """输入多样性变换"""
    if torch.rand(1).item() < prob:
        # 随机调整大小
        rnd = torch.randint(image_width, image_resize + 1, (1,)).item()
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='nearest')

        # 随机填充
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem + 1, (1,)).item()
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        padded = F.interpolate(padded, size=(image_width, image_width), mode='nearest')
        return padded
    else:
        return x


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


def BSR_transform(x, num_blocks_h=2, num_blocks_w=2, max_angle=0.2, num_copies=20):
    """BSR变换：创建多个分块旋转打乱的副本"""
    transformed_copies = []
    for _ in range(num_copies):
        transformed_copy = shuffle_rotate_bsr(x, num_blocks_h, num_blocks_w, max_angle)
        transformed_copies.append(transformed_copy)

    return torch.cat(transformed_copies, dim=0)


def mutate_bsr_params(params, num_blocks_h, num_blocks_w, max_angle, beta, device):
    """对BSR参数进行变异"""
    # 变异角度: 添加高斯噪声
    noise = torch.randn_like(params['angles']) * beta
    new_angles = torch.clamp(params['angles'] + noise, -max_angle, max_angle)
    
    # 变异置换: 随机交换
    def mutate_perm(perm, n):
        new_perm = perm.copy()
        if n > 1:
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
        return new_perm

    new_params = {
        'w_perm': mutate_perm(params['w_perm'], num_blocks_h),
        'h_perm': mutate_perm(params['h_perm'], num_blocks_w),
        'angles': new_angles
    }
    # 继承分块长度 (不做变异，保持一致性)
    if 'width_length' in params:
        new_params['width_length'] = params['width_length']
    if 'height_length' in params:
        new_params['height_length'] = params['height_length']
        
    return new_params

def mifgsm_attack_EATA(x, y, model, eps=16 / 255, iterations=10, mu=1.0,
                       num_blocks_h=2, num_blocks_w=2, max_angle=0.2,
                       num_samples=10, num_keep=5, diversity_prob=0.5, beta=0.1):
    """
    EATA: 结合演化筛选与梯度对齐的攻击
    num_samples: 每一轮生成的随机变换种子总数 (演化池 M)
    num_keep: 最终用于梯度对齐的最优变换数 (精英种群 K)
    """
    alpha = eps / iterations
    x_adv = x.clone().requires_grad_(True)
    momentum = torch.zeros_like(x).to(x.device)

    for i in range(iterations):
        # 1. 演化筛选阶段 (仅推理，不计算梯度)
        model.eval()
        
        # A. 采样与评估 (Initialize Population)
        current_params = []
        current_losses = []
        
        with torch.no_grad():
            # 生成初始种群 M (或者上一代的继承? EATA通常每步重新生成或部分继承，这里按每步重新生成简化)
            for _ in range(num_samples):
                p = generate_bsr_params(x.size(0), num_blocks_h, num_blocks_w, max_angle, x.device, x.size(2), x.size(3))
                x_tmp = apply_bsr_with_params(input_diversity(x_adv, prob=diversity_prob), p, num_blocks_h, num_blocks_w)
                loss_val = F.cross_entropy(model(x_tmp), y, reduction='none') # 保持batch维度以便后续筛选?
                # 简化：这里假设batch_size=1或者针对整个batch统一评估。
                # 原始代码使用了 F.cross_entropy(..., y) 默认 mean reduction, 导致无法区分batch内个体的变换效果。
                # 但 generate_bsr_params 是针对 batch 的 angles，perm 是共享的
                # 为了简单起见，且遵循原始代码结构，我们假设 param 是针对整个 Image Batch 共享的结构参数
                # (注意: generate_bsr_params生成一个 scalar 的 angles 列表? 不, angles是 (batch_size))
                # 仔细看 generate_bsr_params: angles shape is (batch_size). 
                # 所以一组 param 实际上定义了整个 batch 的变换。
                
                loss_scalar = loss_val.mean().item()
                current_params.append(p)
                current_losses.append(loss_scalar)
            
            # B. 演化更新 (Evolutionary Update)
            # 选取 Top K 精英
            best_indices = np.argsort(current_losses)[-num_keep:]
            elite_params = [current_params[idx] for idx in best_indices]
            
            # 变异产生新后代补充到 M
            # 实际上 EATA 论文中是: 选 K 个 -> 变异产生 M-K 个 -> 合并 -> 再选 K 个
            mutated_params = []
            for _ in range(num_samples - num_keep):
                # 随机选一个精英进行变异
                parent = elite_params[np.random.randint(len(elite_params))]
                child = mutate_bsr_params(parent, num_blocks_h, num_blocks_w, max_angle, beta, x.device)
                mutated_params.append(child)
            
            # 评估变异后代
            mutated_losses = []
            for p in mutated_params:
                x_tmp = apply_bsr_with_params(input_diversity(x_adv, prob=diversity_prob), p, num_blocks_h, num_blocks_w)
                loss_val = F.cross_entropy(model(x_tmp), y).item()
                mutated_losses.append(loss_val)
            
            # 合并种群 (精英 + 变异)
            total_params = elite_params + mutated_params
            total_losses = [current_losses[idx] for idx in best_indices] + mutated_losses
            
            # 再次选取 Top K (最终精英)
            final_indices = np.argsort(total_losses)[-num_keep:]
            final_elite_params = [total_params[idx] for idx in final_indices]

        # 2. 梯度对齐阶段
        model.zero_grad()
        grads = []
        
        for p in final_elite_params:
            x_input = x_adv.clone().requires_grad_(True)
            x_transformed = apply_bsr_with_params(input_diversity(x_input, prob=diversity_prob), p, num_blocks_h, num_blocks_w)
            output = model(x_transformed)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            grads.append(x_input.grad.data)
            
        # 计算对齐权重 (Softmax)
        if i > 0 and len(grads) > 0:
            # 计算与动量的余弦相似度
            cos_sims = []
            flat_momentum = momentum.view(1, -1)
            for g in grads:
                flat_g = g.view(1, -1)
                sim = F.cosine_similarity(flat_g, flat_momentum).item()
                cos_sims.append(sim)
            
            # Softmax 权重
            cos_sims = torch.tensor(cos_sims, device=x.device)
            weights = F.softmax(cos_sims, dim=0)
        else:
            weights = torch.ones(len(grads), device=x.device) / len(grads)
            
        # 聚合梯度
        combined_grad = torch.zeros_like(x)
        for idx, g in enumerate(grads):
            combined_grad += weights[idx] * g
        
        # 3. 动量更新与对抗样本生成
        # 归一化 L1 [cite: 3, 100]
        grad_norm = torch.mean(torch.abs(combined_grad), dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = mu * momentum + (combined_grad / grad_norm)

        with torch.no_grad():
            x_adv = x_adv + alpha * torch.sign(momentum)
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 初始化模型仓库
    model_repo = ModelRepository(device)

    # --- 1. 攻击阶段：在内存中生成 ---
    source_model = load_source_model(args.model, device)

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

    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 记录原始预测
        source_orig_preds = get_model_prediction(source_model, x_batch)

        # 生成对抗样本
        x_adv_batch = mifgsm_attack_EATA(x_batch, y_batch, source_model,
                                iterations=args.iterations,
                                mu=args.mu,
                                num_blocks_h=args.num_blocks_h,
                                num_blocks_w=args.num_blocks_w,
                                max_angle=args.max_angle,
                                num_samples=args.num_samples,
                                num_keep=args.num_keep,
                                diversity_prob=args.diversity_prob,
                                beta=args.beta)

        # 记录攻击后预测
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        # 将生成的对抗样本移至 CPU 并存储，释放显存
        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item()) + 1
            s_adv_idx = int(source_adv_preds[i]) + 1
            s_orig_idx = int(source_orig_preds[i]) + 1
            # print(f"{s_orig_idx}\t{s_adv_idx}\t{true_label}")
            source_results.append({
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": s_orig_idx,
                "source_adv_pred": s_adv_idx,
                "source_attack_success": s_adv_idx != true_label
            })

    # 彻底释放源模型显存
    del source_model
    torch.cuda.empty_cache()

    # --- 2. 验证阶段：直接使用内存中的 Tensor ---
    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    # 将 List 转换为单个大 Tensor，并构建简单的 TensorDataset
    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False,pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")
        # 这里建议你根据之前讨论的，实现一个 load_single_model 或者 load_source_model
        # 假设这里依然通过 repo 加载
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
    torch.cuda.empty_cache();
    main()