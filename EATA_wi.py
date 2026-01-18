import argparse
import copy
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
from torch.utils.data import DataLoader,  TensorDataset
import math

from preprocess import AdvPNGDataset, get_model_output, load_source_model, get_model_prediction
import torchvision

# 指定.pth文件的根下载目录（自定义修改）
target_pth_dir = "E:\\TransferAttack\\RuiGuoCode\\weight"  # Windows（注意转义符）

# 设置TORCH_HOME环境变量
os.environ["TORCH_HOME"] = target_pth_dir

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, input):
        # Broadcasting handles the resizing automatically
        return (input - self.mean) / self.std

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
        raise Exception("Invalid model name " + model_name)

    net.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model = torch.nn.Sequential(Normalize(mean=mean, std=std), net)
    model = model.to(device)
    return model

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
    """使用预设参数应用BSR变换 (batch 版 grid_sample 优化)"""
    batch_size, channels, w, h = x.shape

    width_length = params.get('width_length', get_block_lengths_bsr(w, num_blocks_h))
    height_length = params.get('height_length', get_block_lengths_bsr(h, num_blocks_w))

    x_split_w = torch.split(x, width_length, dim=2)

    rotated_blocks = []
    for w_idx in range(num_blocks_h):
        w_block = x_split_w[w_idx]
        h_blocks = torch.split(w_block, height_length, dim=3)

        rotated_strip = []
        for h_idx in range(num_blocks_w):
            block = h_blocks[h_idx]

            angles_param = params['angles']
            if len(angles_param.shape) == 3:
                current_angles = angles_param[:, w_idx, h_idx]
            else:
                current_angles = angles_param

            # batch-wise rotation via affine_grid + grid_sample
            if block.shape[2] > 1 and block.shape[3] > 1:
                cos_a = torch.cos(current_angles).to(block.dtype)
                sin_a = torch.sin(current_angles).to(block.dtype)
                theta = torch.zeros((batch_size, 2, 3), device=block.device, dtype=block.dtype)
                theta[:, 0, 0] = cos_a
                theta[:, 0, 1] = -sin_a
                theta[:, 1, 0] = sin_a
                theta[:, 1, 1] = cos_a

                grid = F.affine_grid(theta, block.size(), align_corners=False)
                rotated_block = F.grid_sample(block, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            else:
                rotated_block = block

            rotated_strip.append(rotated_block)

        rotated_strip_perm = [rotated_strip[i] for i in params['h_perm']]
        rotated_blocks.append(torch.cat(rotated_strip_perm, dim=3))

    return torch.cat([rotated_blocks[i] for i in params['w_perm']], dim=2)

def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

# --- 新增：DIM 演化基因类 (参考 DIM_cli_ea.py) ---
class EATAGene:
    """
    用于优化 DIM 参数的基因个体
    参数: scale (缩放尺寸), pad_top, pad_left
    """
    def __init__(self, image_width=299, image_resize=331):
        self.image_width = image_width
        self.image_resize = image_resize
        self.resize_scale = image_width # 初始值
        self.pad_top = 0
        self.pad_left = 0

    def random_init(self):
        # 随机初始化缩放尺寸 [image_width, image_resize)
        self.resize_scale = random.randint(self.image_width, self.image_resize - 1) 
        rem = self.image_resize - self.resize_scale
        self.pad_top = random.randint(0, rem) if rem > 0 else 0
        self.pad_left = random.randint(0, rem) if rem > 0 else 0

    def mutate(self, mutation_rate=0.5):
        new_gene = copy.deepcopy(self)
        if random.random() < mutation_rate:
            # 变异：尺寸微调
            diff = random.choice([-2, -1, 1, 2])
            # 限制范围 [image_width, image_resize - 1]
            new_gene.resize_scale = max(self.image_width, min(self.image_resize - 1, new_gene.resize_scale + diff))
            
            # 根据新尺寸调整 Padding 约束
            rem = self.image_resize - new_gene.resize_scale
            # 变异：Padding 微调
            p_diff = random.choice([-2, -1, 1, 2])
            new_gene.pad_top = max(0, min(rem, new_gene.pad_top + p_diff))
            new_gene.pad_left = max(0, min(rem, new_gene.pad_left + p_diff))
            
        return new_gene
    
    def apply(self, x):
        """应用 DIM 变换到 tensor x (Batch or Single)"""
        scale = self.resize_scale
        # 1. Resize
        resized = F.interpolate(x, size=(scale, scale), mode='nearest')
        
        # 2. Pad
        h_rem = self.image_resize - scale
        w_rem = self.image_resize - scale
        
        pad_top = self.pad_top
        pad_bottom = h_rem - pad_top
        pad_left = self.pad_left
        pad_right = w_rem - pad_left
        
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # 3. Resize back to original width
        final = F.interpolate(padded, size=(self.image_width, self.image_width), mode='nearest')
        return final

# --- 修改：联合参数生成与应用 ---

def generate_combined_params(batch_size, num_blocks_h, num_blocks_w, max_angle, device, width, height):
    """生成 BSR 和 DIM 的联合参数"""
    # 1. BSR Params
    bsr_params = {
        'w_perm': np.random.permutation(np.arange(num_blocks_h)),
        'h_perm': np.random.permutation(np.arange(num_blocks_w)),
        'angles': torch.clamp(torch.randn(batch_size, num_blocks_h, num_blocks_w, device=device) * 0.05, -max_angle, max_angle),
        'width_length': get_block_lengths_bsr(width, num_blocks_h),
        'height_length': get_block_lengths_bsr(height, num_blocks_w)
    }
    
    # 2. DIM Params (EATAGene)
    # create a gene
    dim_gene = EATAGene(image_width=299, image_resize=331) # 使用 EATA 默认参数
    dim_gene.random_init()
    
    return {'bsr': bsr_params, 'dim': dim_gene}

def mutate_combined_params(params, num_blocks_h, num_blocks_w, max_angle, beta, device):
    """联合变异"""
    # 1. Mutate BSR
    bsr_p = params['bsr']
    noise = torch.randn_like(bsr_p['angles']) * beta
    new_angles = torch.clamp(bsr_p['angles'] + noise, -max_angle, max_angle)
    
    def mutate_perm(perm, n):
        new_perm = perm.copy()
        if n > 1:
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
        return new_perm

    new_bsr = {
        'w_perm': mutate_perm(bsr_p['w_perm'], num_blocks_h),
        'h_perm': mutate_perm(bsr_p['h_perm'], num_blocks_w),
        'angles': new_angles
    }
    if 'width_length' in bsr_p: new_bsr['width_length'] = bsr_p['width_length']
    if 'height_length' in bsr_p: new_bsr['height_length'] = bsr_p['height_length']
    
    # 2. Mutate DIM
    new_dim = params['dim'].mutate(mutation_rate=0.5) # 使用 DIM_cli_ea 的默认 mutation_rate
    
    return {'bsr': new_bsr, 'dim': new_dim}

def apply_combined_transform(x, params, num_blocks_h, num_blocks_w):
    """先后应用 DIM 和 BSR"""
    # 1. Apply DIM (using Gene)
    x_dim = params['dim'].apply(x)
    
    # 2. Apply BSR
    # apply_bsr_with_params 需要 x, params(bsr部分)
    return apply_bsr_with_params(x_dim, params['bsr'], num_blocks_h, num_blocks_w)

def parse_args():
    parser = argparse.ArgumentParser(description='EATA Attack')
    parser.add_argument('--input_dir', default='./data', type=str, help='input images dir')
    parser.add_argument('--output_adv_dir', default='./results/eata/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/eata/results.csv', type=str, help='output CSV path')
    parser.add_argument('--ensemble_models', type=str, default=['inception_v3', 'resnet50', 'vgg16'], nargs='+', help='List of ensemble model names.')
    parser.add_argument('--batchsize', type=int, default=16, help='Batch size for processing.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of attack iterations.')
    parser.add_argument('--mu', type=float, default=1.0, help='Momentum factor.')
    parser.add_argument('--num_blocks_h', type=int, default=2, help='Number of blocks in height for BSR.')
    parser.add_argument('--num_blocks_w', type=int, default=2, help='Number of blocks in width for BSR.')
    parser.add_argument('--max_angle', type=float, default=0.2, help='Maximum rotation angle for BSR.')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for evolutionary search.')
    parser.add_argument('--num_keep', type=int, default=5, help='Number of top samples to keep for evolutionary search.')
    parser.add_argument('--diversity_prob', type=float, default=0.5, help='Probability of applying input diversity.')
    parser.add_argument('--beta', type=float, default=0.1, help='Mutation noise factor for evolutionary search.')
    return parser.parse_args()

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

def eata_single_attack(x, y, models, eps, iterations, mu,
                       num_blocks_h, num_blocks_w, max_angle,
                       num_samples, num_keep, diversity_prob, beta,
                       admix_gamma=1.0, admix_eta=0.2,
                       align_temperature=1.0):
    """
    Batch 版 EATA 攻击（带梯度对齐加权）。

    梯度对齐：
      w_k = Softmax( (g_k · m_{t-1}) / (||g_k||_2 ||m_{t-1}||_2) / temperature )
      g_bar = sum_k w_k * g_k

    说明：这里在 batch 维度上对梯度做 flatten 后计算余弦相似度，得到每个 elite 的单个权重。
    """
    assert models is not None and len(models) > 0, "models (ensemble) must be a non-empty list"

    alpha = eps / iterations
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)

    batch_size = x_adv.size(0)
    previous_elites_per_image = None

    for t in range(iterations):
        for m in models:
            m.eval()

        # ---- Evolutionary Search (vectorized over batch for losses) ----
        current_params = []
        if t == 0 or previous_elites_per_image is None:
            for _ in range(num_samples):
                p = generate_combined_params(batch_size, num_blocks_h, num_blocks_w, max_angle,
                                             x_adv.device, x_adv.size(2), x_adv.size(3))
                current_params.append(p)
        else:
            pool = []
            for elites in previous_elites_per_image:
                pool.extend(elites)
            current_params.extend(pool[:min(len(pool), num_samples)])

            num_needed = num_samples - len(current_params)
            if num_needed > 0:
                for _ in range(num_needed):
                    img_idx = np.random.randint(batch_size)
                    parent = previous_elites_per_image[img_idx][np.random.randint(num_keep)]
                    child = mutate_combined_params(parent, num_blocks_h, num_blocks_w, max_angle, beta, x_adv.device)
                    if child['bsr']['angles'].shape[0] != batch_size:
                        child['bsr']['angles'] = child['bsr']['angles'].expand(batch_size, -1, -1).contiguous()
                    current_params.append(child)

        losses_matrix = []
        with torch.no_grad():
            for p in current_params:
                x_tmp = apply_combined_transform(x_adv, p, num_blocks_h, num_blocks_w)
                per_img_loss = None
                for model in models:
                    logits = model(x_tmp)
                    l = F.cross_entropy(logits, y, reduction='none')
                    per_img_loss = l if per_img_loss is None else (per_img_loss + l)
                assert per_img_loss is not None
                per_img_loss = per_img_loss / float(len(models))
                losses_matrix.append(per_img_loss)

        losses_matrix = torch.stack(losses_matrix, dim=0)  # (num_samples, B)
        _, topk_idx = torch.topk(losses_matrix, k=num_keep, dim=0, largest=True, sorted=False)

        elites_per_image = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            for k in range(num_keep):
                elites_per_image[b].append(current_params[int(topk_idx[k, b].item())])
        previous_elites_per_image = elites_per_image

        # ---- Gradient Alignment (STRICT per-image: image b only uses its own K elites) ----
        # 1) 先对每个 unique elite 计算一次 batch 梯度，缓存下来
        unique_elites = {}
        for b in range(batch_size):
            for p in elites_per_image[b]:
                unique_elites[id(p)] = p
        unique_elites = list(unique_elites.values())

        elite_grad_cache = []  # list of (grad_tensor[B,C,H,W]) aligned with unique_elites
        elite_id_to_index = {}

        for idx, p in enumerate(unique_elites):
            elite_id_to_index[id(p)] = idx

            for m in models:
                m.zero_grad(set_to_none=True)

            x_input = x_adv.detach().clone().requires_grad_(True)

            # Admix
            perm = torch.randperm(batch_size, device=x_adv.device)
            x_rand = x_input[perm].detach()
            x_admix = admix_gamma * (x_input + admix_eta * x_rand)

            x_transformed = apply_combined_transform(x_admix, p, num_blocks_h, num_blocks_w)

            loss_vec = None
            for model in models:
                logits = model(x_transformed)
                l = F.cross_entropy(logits, y, reduction='none')
                loss_vec = l if loss_vec is None else (loss_vec + l)
            assert loss_vec is not None
            loss_vec = loss_vec / float(len(models))

            batch_loss = loss_vec.mean()
            grad = torch.autograd.grad(batch_loss, x_input, retain_graph=False, create_graph=False)[0]
            elite_grad_cache.append(grad.detach())

        if len(elite_grad_cache) == 0:
            combined_grad = torch.zeros_like(x_adv)
        else:
            grads_stack = torch.stack(elite_grad_cache, dim=0)  # (U, B, C, H, W)

            # 2) 对每张图 b：只取它自己的 K 个 elite 梯度，计算余弦相似度并 softmax 加权
            combined_grad = torch.zeros_like(x_adv)

            # flatten 供余弦计算
            grads_flat = grads_stack.flatten(2)  # (U, B, D)
            mom_flat = momentum.detach().flatten(1)  # (B, D)

            for b in range(batch_size):
                elite_indices_b = [elite_id_to_index[id(p)] for p in elites_per_image[b]]
                elite_indices_b = torch.as_tensor(elite_indices_b, device=x_adv.device, dtype=torch.long)  # (K,)

                g_b = grads_flat.index_select(0, elite_indices_b)[:, b, :]  # (K, D)
                m_b = mom_flat[b]  # (D,)

                g_norm = g_b.norm(p=2, dim=1).clamp_min(1e-12)  # (K,)
                m_norm = m_b.norm(p=2).clamp_min(1e-12)

                cos_sim = (g_b @ m_b) / (g_norm * m_norm)  # (K,)
                w = torch.softmax(cos_sim / float(align_temperature), dim=0).to(grads_stack.dtype)  # (K,)

                # 加权聚合回 (C,H,W)
                g_img = grads_stack.index_select(0, elite_indices_b)[:, b]  # (K, C, H, W)
                combined_grad[b] = (w[:, None, None, None] * g_img).sum(dim=0)

        # ---- Momentum + update ----
        grad_norm = torch.mean(torch.abs(combined_grad), dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = mu * momentum + (combined_grad / grad_norm)

        with torch.no_grad():
            x_adv = x_adv + alpha * torch.sign(momentum)
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def mifgsm_attack_EATA(x, y, models, eps=16 / 255, iterations=10, mu=1.0,
                       num_blocks_h=2, num_blocks_w=2, max_angle=0.2,
                       num_samples=10, num_keep=5, diversity_prob=0.5, beta=0.1,
                       admix_gamma=1.0, admix_eta=0.2):
    """EATA Wrapper (batch vectorized)"""
    return eata_single_attack(x, y, models, eps, iterations, mu,
                              num_blocks_h, num_blocks_w, max_angle,
                              num_samples, num_keep, diversity_prob, beta,
                              admix_gamma=admix_gamma, admix_eta=admix_eta)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 初始化模型仓库 (Target models)
    model_repo = ModelRepository(device)

    # --- 1. 攻击阶段：在内存中生成 ---
    # Load Ensemble Models (Updated Logic)
    ensemble_models = []
    print(f"Loading ensemble models: {args.ensemble_models}")
    
    # 使用新的 load_source_model 函数加载所有模型
    for model_name in args.ensemble_models:
        model = load_source_model(model_name, device)
        ensemble_models.append(model)
    
    # We use the first model in ensemble as a reference for logging "Source Prediction"
    ref_model = ensemble_models[0]

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

    print(f"\n[Step 1/3] Attacking in Memory using Ensemble EATA...")

    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Attacking"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 记录原始预测 (Reference Model)
        source_orig_preds = get_model_prediction(ref_model, x_batch)

        # 生成对抗样本 (Ensemble Attack)
        x_adv_batch = mifgsm_attack_EATA(x_batch, y_batch, ensemble_models,
                                iterations=args.iterations,
                                mu=args.mu,
                                num_blocks_h=args.num_blocks_h,
                                num_blocks_w=args.num_blocks_w,
                                max_angle=args.max_angle,
                                num_samples=args.num_samples,
                                num_keep=args.num_keep,
                                diversity_prob=args.diversity_prob,
                                beta=args.beta)

        # 记录攻击后预测 (Reference Model)
        source_adv_preds = get_model_prediction(ref_model, x_adv_batch)

        # 将生成的对抗样本移至 CPU 并存储，释放显存
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

    # 彻底释放源模型显存
    del ensemble_models
    del ref_model
    torch.cuda.empty_cache()

    # --- 2. 验证阶段：直接使用内存中的 Tensor ---
    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    # 将 List 转换为单个大 Tensor，并构建简单的 TensorDataset
    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False,pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    # Exclude all ensemble models from target list to properly measure transferability
    target_names = [name for name in all_model_names if name not in args.ensemble_models]
    
    # Warning: Standard torchvision models (inception_v3, resnet50 etc) might not be in ModelRepository 
    # if ModelRepository only contains tf2torch models.
    # The user asked to update "EATA.py" but kept 'models.py' intact.
    # If the user wants to test transferability against *other* models in ModelRepository, 
    # we should check if they exist.
    # We will iterate available models in ModelRepository.
    
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
    
    # Updated Summary output
    print(f"\nEnsemble Source Attack Success Rate (evaluated on primary model):")
    source_rate = sum(1 for r in source_results if r['source_attack_success']) / total_samples * 100
    print(f"  Reference Model ({args.ensemble_models[0]}) Fooling Rate: {source_rate:.1f}%")
    
    print("\nTransfer Success Rates (Target Models):")
    for name, count in model_success_counts.items():
        print(f"  {name}: {count}/{total_samples} ({count / total_samples * 100:.1f}%)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(final_rows).to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")

if __name__ == '__main__':
    # 清空CUDA缓存
    torch.cuda.empty_cache();
    main()

