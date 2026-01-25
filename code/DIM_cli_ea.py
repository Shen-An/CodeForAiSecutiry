# filepath: e:\TransferAttack\RuiGuoCode\DIM_cli_ea.py
import argparse
import gc
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from models import ModelRepository, Normalize
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os

# 指定.pth文件的根下载目录（自定义修改）
target_pth_dir = "E:\\TransferAttack\\RuiGuoCode\\weight"  # Windows（注意转义符）

# 设置TORCH_HOME环境变量
os.environ["TORCH_HOME"] = target_pth_dir

# --- 自定义数据集：从本地加载生成的 PNG 对抗样本 ---
class AdvPNGDataset(Dataset):
    def __init__(self, img_dir, label_df, transform):
        self.img_dir = img_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        # 兼容文件名获取
        fn = row['filename']
        img_path = os.path.join(self.img_dir, fn)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 返回 image, label, filename
        return img, row['label'], fn

# --- EATA-DIM 核心组件 ---

class EATAGene:
    """
    仅用于优化 DIM 参数的基因个体
    参数: scale (缩放尺寸), pad_top, pad_left
    """
    def __init__(self):
        self.resize_scale = 310 # 初始中值
        self.pad_top = 0
        self.pad_left = 0
        self.fitness = -float('inf')

    def random_init(self):
        # 对应 mask = np.random.randint(299, 330)
        self.resize_scale = random.randint(299, 330) 
        rem = 330 - self.resize_scale
        self.pad_top = random.randint(0, rem) if rem > 0 else 0
        self.pad_left = random.randint(0, rem) if rem > 0 else 0

    def mutate(self, mutation_rate=0.5):
        new_gene = copy.deepcopy(self)
        if random.random() < mutation_rate:
            # 变异：尺寸微调
            diff = random.choice([-2, -1, 1, 2])
            new_gene.resize_scale = max(299, min(329, new_gene.resize_scale + diff))
            
            # 根据新尺寸调整 Padding 约束
            rem = 330 - new_gene.resize_scale
            # 变异：Padding 微调
            p_diff = random.choice([-2, -1, 1, 2])
            new_gene.pad_top = max(0, min(rem, new_gene.pad_top + p_diff))
            new_gene.pad_left = max(0, min(rem, new_gene.pad_left + p_diff))
            
        return new_gene

def apply_gene_batch_process(x_batch, genes_list):
    """
    针对 Batch 中的每一张图片应用其对应的 Gene 变换
    x_batch: [B, C, H, W]
    genes_list: List length B (每张图片对应的最佳 Gene)
    """
    B = x_batch.shape[0]
    outputs = []
    
    for i in range(B):
        img = x_batch[i:i+1] # keep 4D: [1, C, H, W]
        gene = genes_list[i]
        
        scale = gene.resize_scale
        # 1. Resize
        resized = F.interpolate(img, size=(scale, scale), mode='nearest')
        
        # 2. Pad
        h_rem = 330 - scale
        w_rem = 330 - scale
        
        pad_top = gene.pad_top
        pad_bottom = h_rem - pad_top
        pad_left = gene.pad_left
        pad_right = w_rem - pad_left
        
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # 3. Resize back to 299
        final = F.interpolate(padded, size=(299, 299), mode='nearest')
        outputs.append(final)
        
    return torch.cat(outputs, dim=0)

def eata_dim_attack(x, y, model, eps=16 / 255, iterations=10, mu=1.0, 
                    pop_size=10, elite_size=2, mutation_rate=0.5):
    """
    EATA-DIM 攻击：每一张图片独立演化最优参数 (Batchsize 仅用于并行加速)
    """
    device = x.device
    B = x.shape[0]
    x_adv = x.clone().detach()
    alpha = eps / max(1, iterations)
    momentum = torch.zeros_like(x_adv, device=device)

    # 初始化种群 (每个样本都有自己的种群)
    populations = [[EATAGene() for _ in range(pop_size)] for _ in range(B)]
    for sub_pop in populations:
        for g in sub_pop: g.random_init()
    
    all_elites = [[] for _ in range(B)]

    for t in range(iterations):
        x_adv.requires_grad_(True)
        
        # --- 1. 演化步骤 (Evolution) - 针对每个样本独立进行 ---
        current_generation_candidates = [] 
        candidate_map = [] 
        
        for b_idx in range(B):
            sub_pop = []
            elites = all_elites[b_idx]
            
            if t > 0 and elites:
                sub_pop.extend(elites)
                while len(sub_pop) < pop_size:
                    parent = random.choice(elites)
                    sub_pop.append(parent.mutate(mutation_rate))
            else:
                if t == 0:
                    sub_pop = populations[b_idx]
                else: 
                    sub_pop = [g.mutate(mutation_rate) for g in populations[b_idx]]
            
            populations[b_idx] = sub_pop
            
            for gene in sub_pop:
                current_generation_candidates.append((gene, b_idx))

        # --- 2. 批量评估 (Batch Evaluation) ---
        temp_genes = [item[0] for item in current_generation_candidates]
        temp_indices = [item[1] for item in current_generation_candidates]
        
        eval_bs = 32 
        total_candidates = len(current_generation_candidates)
        all_fitness_scores = np.zeros(total_candidates)
        
        with torch.no_grad():
            for start_idx in range(0, total_candidates, eval_bs):
                end_idx = min(start_idx + eval_bs, total_candidates)
                
                batch_imgs = []
                batch_genes = []
                
                for k in range(start_idx, end_idx):
                    g, b_i = current_generation_candidates[k]
                    img_piece = x_adv[b_i:b_i+1] 
                    batch_imgs.append(img_piece)
                    batch_genes.append(g)
                
                trans_imgs = []
                for k in range(len(batch_imgs)):
                    g = batch_genes[k]
                    img = batch_imgs[k]
                    scale = g.resize_scale
                    resized = F.interpolate(img, size=(scale, scale), mode='nearest')
                    h_rem, w_rem = 330 - scale, 330 - scale
                    pad_top, pad_left = g.pad_top, g.pad_left
                    padded = F.pad(resized, (pad_left, w_rem-pad_left, pad_top, h_rem-pad_top), value=0)
                    final = F.interpolate(padded, size=(299, 299), mode='nearest')
                    trans_imgs.append(final)
                
                x_mini_batch = torch.cat(trans_imgs, dim=0)
                
                out = get_model_output(model, x_mini_batch)
                if out.dim() > 2: out = out.view(out.size(0), -1)
                
                target_indices = temp_indices[start_idx : end_idx]
                y_mini = y[target_indices]
                
                losses = F.cross_entropy(out, y_mini, reduction='none')
                all_fitness_scores[start_idx:end_idx] = losses.cpu().numpy()

        # --- 3. 筛选精英 (Selection) - 归位 ---
        current_best_genes = [] 
        
        cursor = 0
        for b_idx in range(B):
            sample_scores = all_fitness_scores[cursor : cursor + pop_size]
            sample_genes = populations[b_idx]
            
            for i, score in enumerate(sample_scores):
                sample_genes[i].fitness = score
            
            sample_genes.sort(key=lambda g: g.fitness, reverse=True)
            
            all_elites[b_idx] = copy.deepcopy(sample_genes[:elite_size])
            
            current_best_genes.append(sample_genes[0])
            
            cursor += pop_size

        # --- 4. 梯度计算 (Gradient Calculation) ---
        # 使用每张图片的所有精英变换计算平均梯度 (Gradient Alignment)
        elite_genes_flat = []
        counts = []
        for b_idx in range(B):
            elites = all_elites[b_idx]
            # 防御性编程：如果精英列表为空（理论上不应该发生），使用当前最佳
            if not elites: 
                elites = [current_best_genes[b_idx]]
            elite_genes_flat.extend(elites)
            counts.append(len(elites))

        # 扩展 x_adv 和 y 以匹配 flattened genes
        # x_adv: [B, C, H, W] -> 扩展为 [Total_Elites, C, H, W]
        # 通过 repeat_interleave 保持 autograd 图连接，梯度会自动累加回 x_adv
        counts_tensor = torch.tensor(counts, device=device)
        x_expanded = torch.repeat_interleave(x_adv, counts_tensor, dim=0)
        y_expanded = torch.repeat_interleave(y, counts_tensor, dim=0)
        
        # 应用变换
        x_elite_batch = apply_gene_batch_process(x_expanded, elite_genes_flat)
        
        # 前向传播
        output = get_model_output(model, x_elite_batch)
        if output.dim() == 1: output = output.unsqueeze(0)
        elif output.dim() > 2: output = output.view(output.size(0), -1)
            
        # 计算 Loss
        # CrossEntropy 默认 reduction='mean'，会对所有样本的 Loss 求平均
        # 由于 autograd 的机制，对于重复的 x_adv 节点，其梯度会自动累加
        # 最终得到的 x_adv.grad 实际上就是也就是对 Elite 变换后的梯度取了期望（平均）方向
        loss = F.cross_entropy(output, y_expanded)
        
        grad = torch.autograd.grad(loss, [x_adv])[0]

        # --- 5. 动量更新 (Update) ---
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = mu * momentum + grad
        x_adv = x_adv.detach() + alpha * torch.sign(momentum)

        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()


def get_model_output(model, x):
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")
    return output


def get_model_prediction(model, x):
    output = get_model_output(model, x)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)
    preds = torch.argmax(output, dim=1).cpu().numpy()
    return np.atleast_1d(preds)


def parse_args():
    parser = argparse.ArgumentParser(description="DIM Attack with Evolutionary Strategy (Per-Image Independent)")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/dim_ea/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/dim_ea/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--pop_size', default=10, type=int, help='EA population size')
    parser.add_argument('--elite_size', default=5, type=int, help='EA elite size to keep')
    parser.add_argument('--mutation_rate', default=0.5, type=float, help='Mutation probability')
    
    return parser.parse_args()


def load_source_model(model_name, device):
    if model_name == 'inception_v3':
        net = torchvision.models.inception_v3(pretrained=True)
    elif model_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
    elif model_name == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
    elif model_name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception("Invalid model name" + model_name)

    net = net.to(device)
    net.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model = torch.nn.Sequential(Normalize(mean=mean, std=std), net)
    return model


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 初始化模型仓库
    model_repo = ModelRepository(device)

    # --- 1. 攻击阶段 ---
    source_model = load_source_model(args.model, device)

    label_csv_path = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    label_df = pd.read_csv(label_csv_path)

    # 过滤文件
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

    print(f"\n[Step 1/3] Attacking in Memory (EA-DIM-Parallel)...")
    print(f"EA Config: Pop={args.pop_size}, Elite={args.elite_size}, Mut={args.mutation_rate}")

    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 原始预测
        source_orig_preds = get_model_prediction(source_model, x_batch)

        # 生成对抗样本 (使用 EATA-DIM Parallel)
        x_adv_batch = eata_dim_attack(
            x_batch, y_batch, source_model,
            eps=args.eps, iterations=args.iterations, mu=args.mu,
            pop_size=args.pop_size, elite_size=args.elite_size, mutation_rate=args.mutation_rate
        )

        # 攻击后预测
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
                "source_attack_success": s_adv_idx != true_label
            })

    del source_model
    torch.cuda.empty_cache()

    # --- 2. 验证阶段 ---
    print(f"\n[Step 2/3] Testing Transferability...")

    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")
        # 兼容性加载
        try:
             current_model_info = model_repo.load_single_model(model_name)
             model = current_model_info['model']
        except:
             # Fallback
             model = model_repo._load_model(model_name)
             
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

    # --- 3. 结果保存 ---
    print(f"\n[Step 3/3] Saving results...")

    os.makedirs(args.output_adv_dir, exist_ok=True)
    for idx, res in enumerate(source_results):
        fn = res["filename"]
        try:
            save_image(all_adv_tensors[idx], os.path.join(args.output_adv_dir, fn))
        except Exception as e:
            print(f"Error saving {fn}: {e}")

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
    # torch.manual_seed(42) # 可选：固定随机种子
    main()
