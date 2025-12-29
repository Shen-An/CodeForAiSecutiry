import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from models import ModelRepository, Normalize

class AttackDataset(Dataset):
    def __init__(self, label_df, img_root, transform):
        self.label_df = label_df
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        img_filename = row["filename"]
        true_label = int(row["label"]) 
        img_path = os.path.join(self.img_root, img_filename)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        # 返回 label 的张量以避免后续 torch.tensor(...) 的警告
        return x, torch.tensor(true_label, dtype=torch.long), img_filename

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

def load_source_model(model_name, device, model_repo):
    """
    加载源模型。如果是 'tv_' 开头，加载 torchvision 官方模型；
    否则从 ModelRepository 加载。
    """
    import torch.nn as nn  # 在函数内部显式导入，确保作用域内可用
    if model_name.startswith('tv_'):
        import torchvision.models as tv_models
        name = model_name.replace('tv_', '')
        print(f"Loading torchvision model: {name}")
        
        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if name == 'inception_v3':
            net = tv_models.inception_v3(pretrained=True)
        elif name == 'resnet50':
            net = tv_models.resnet50(pretrained=True)
        elif name == 'vgg16':
            net = tv_models.vgg16(pretrained=True)
        elif name == 'densenet121':
            net = tv_models.densenet121(pretrained=True)
        elif name == 'resnet101':
            net = tv_models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unknown torchvision model: {name}")
            
        net = net.to(device)
        net.eval()
        
        model = nn.Sequential(
            Normalize(mean=mean, std=std),
            net
        )
        return model
    else:
        print(f"Loading model from repository: {model_name}")
        return model_repo.get_source_model(model_name)['model']

# --- Block-level Transformation Helper ---

def apply_block_transformation(x, idx_r, idx_c, block_size, k_map=None, enable_rotation=False):
    """
    Apply block-level permutation and optional rotation using PyTorch native operators.
    Ensures differentiability for autograd.
    """
    B, C, H, W = x.shape
    device = x.device
    
    # 1. Pad to multiple of block_size
    h_pad = (block_size - H % block_size) % block_size
    w_pad = (block_size - W % block_size) % block_size
    
    if h_pad > 0 or w_pad > 0:
        x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
    else:
        x_padded = x
        
    _, _, H_p, W_p = x_padded.shape
    n_h = H_p // block_size
    n_w = W_p // block_size
    
    # 2. Reshape to blocks: (B, C, n_h, block_size, n_w, block_size)
    x_blocks = x_padded.view(B, C, n_h, block_size, n_w, block_size)
    
    # 3. Permute blocks
    
    # Prepare indices for broadcasting
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)
    channel_idx = torch.arange(C, device=device).view(1, C, 1)
    
    # Row permutation (dim 2)
    if idx_r.dim() == 1:
        row_idx = idx_r.view(1, 1, -1).expand(B, C, -1)
    else:
        row_idx = idx_r.view(B, 1, n_h).expand(B, C, -1)
        
    # x_blocks: (B, C, n_h, block_size, n_w, block_size)
    # Indexing first 3 dims
    x_row_perm = x_blocks[batch_idx, channel_idx, row_idx]
    
    # Col permutation (dim 4)
    # Permute dim 4 to dim 2 to reuse indexing logic
    # (B, C, n_h, block_size, n_w, block_size) -> (B, C, n_w, n_h, block_size, block_size)
    x_temp = x_row_perm.permute(0, 1, 4, 2, 3, 5)
    
    if idx_c.dim() == 1:
        col_idx = idx_c.view(1, 1, -1).expand(B, C, -1)
    else:
        col_idx = idx_c.view(B, 1, n_w).expand(B, C, -1)
        
    x_col_perm = x_temp[batch_idx, channel_idx, col_idx]
    
    # Permute back: (B, C, n_w, n_h, block_size, block_size) -> (B, C, n_h, block_size, n_w, block_size)
    # Target dims: 0, 1, 3, 4, 2, 5
    x_perm = x_col_perm.permute(0, 1, 3, 4, 2, 5)
    
    # 4. Optional Rotation
    if enable_rotation:
        # (B, C, n_h, block_size, n_w, block_size) -> (B, C, n_h, n_w, block_size, block_size)
        x_rot_in = x_perm.permute(0, 1, 2, 4, 3, 5)
        
        if k_map is None:
            k_map = torch.randint(0, 4, (B, n_h, n_w), device=device)
        elif k_map.shape[0] != B:
            k_map = k_map.expand(B, -1, -1)
            
        # Compute all 4 rotations
        # x_rot_in: (B, C, n_h, n_w, H_b, W_b)
        # rot90 rotates dims 4 and 5
        r0 = x_rot_in
        r1 = torch.rot90(x_rot_in, 1, [4, 5])
        r2 = torch.rot90(x_rot_in, 2, [4, 5])
        r3 = torch.rot90(x_rot_in, 3, [4, 5])
        
        # Stack: (B, C, n_h, n_w, H_b, W_b, 4)
        r_stack = torch.stack([r0, r1, r2, r3], dim=-1)
        
        # Expand k_map for gathering
        # k_map: (B, n_h, n_w) -> (B, C, n_h, n_w, H_b, W_b, 1)
        k_map_expanded = k_map.view(B, 1, n_h, n_w, 1, 1, 1).expand(B, C, n_h, n_w, block_size, block_size, 1)
        
        # Gather along last dim
        x_rotated = torch.gather(r_stack, 6, k_map_expanded).squeeze(6)
        
        # Permute back to (B, C, n_h, block_size, n_w, block_size)
        x_out_blocks = x_rotated.permute(0, 1, 2, 4, 3, 5)
    else:
        x_out_blocks = x_perm

    # 5. Reshape back to image
    x_out = x_out_blocks.reshape(B, C, H_p, W_p)
    
    # 6. Crop padding if necessary
    if h_pad > 0 or w_pad > 0:
        x_out = x_out[:, :, :H, :W]
        
    return x_out

def init_population_batch(pop_size, length, batch_size, device='cpu'):
    """Initialize population with random permutations for each batch sample."""
    # Use torch for faster generation on GPU if available, then convert to numpy
    # Returns list of (B, length) arrays
    population = []
    # Generate random values: (pop_size, batch_size, length)
    rand_vals = torch.rand(pop_size, batch_size, length, device=device)
    # Argsort to get permutations
    perms_tensor = torch.argsort(rand_vals, dim=-1)
    
    # Convert to numpy list of arrays
    perms_np = perms_tensor.cpu().numpy()
    for i in range(pop_size):
        population.append(perms_np[i])
        
    return population

def init_rotation_population_batch(pop_size, n_h, n_w, batch_size, device='cpu'):
    """Initialize population of rotation maps for each batch sample."""
    # Returns list of (B, n_h, n_w) arrays
    # Generate random integers 0-3
    rot_tensor = torch.randint(0, 4, (pop_size, batch_size, n_h, n_w), device=device)
    
    rot_np = rot_tensor.cpu().numpy()
    population = []
    for i in range(pop_size):
        population.append(rot_np[i])
        
    return population

def mutation_perm(r1, r2, r3, prob_m):
    """
    Discrete Mutation for Permutations based on differences.
    V = r1 + F * (r2 - r3)
    We interpret (r2 - r3) as swaps.
    """
    mutant = r1.copy()
    
    # Number of changes determined by prob_m (acting as F/strength)
    # We perform swaps on r1 based on where r2 and r3 differ.
    
    # Find indices where r2 and r3 differ
    diff_indices = np.where(r2 != r3)[0]
    
    if len(diff_indices) > 0:
        # Attempt to apply difference logic
        # We iterate a few times proportional to prob_m
        num_ops = max(1, int(len(diff_indices) * prob_m))
        
        chosen_indices = np.random.choice(diff_indices, size=min(num_ops, len(diff_indices)), replace=False)
        
        for idx in chosen_indices:
            val_r2 = r2[idx]
            val_r3 = r3[idx]
            
            # In r2, value is val_r2. In r3, value is val_r3.
            # The "difference" vector suggests moving from r3 state to r2 state.
            # We apply this relative move to r1.
            # We find where val_r2 and val_r3 are in mutant (r1) and swap them.
            
            idx_a = np.where(mutant == val_r2)[0][0]
            idx_b = np.where(mutant == val_r3)[0][0]
            
            mutant[idx_a], mutant[idx_b] = mutant[idx_b], mutant[idx_a]
            
    return mutant

def mutation_perm_batch(r1_batch, r2_batch, r3_batch, prob_m):
    B = r1_batch.shape[0]
    mutants = []
    for b in range(B):
        mutants.append(mutation_perm(r1_batch[b], r2_batch[b], r3_batch[b], prob_m))
    return np.array(mutants)

def mutation_rot(r1, r2, r3, prob_m):
    """
    Mutation for Rotation Maps (Integers 0-3).
    V = (r1 + F * (r2 - r3)) % 4
    """
    # prob_m acts as F. Since we are discrete, we use it as probability to apply diff.
    diff = (r2 - r3)
    
    # Mask where we apply the difference
    mask = np.random.rand(*r1.shape) < prob_m
    
    mutant = r1.copy()
    # Apply difference modulo 4
    mutant[mask] = (mutant[mask] + diff[mask]) % 4
    
    return mutant

def crossover(parent1, parent2, prob):
    """Two-point crossover (Order Crossover - OX1) for permutations."""
    if np.random.rand() >= prob:
        return parent1.copy()
    
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(np.random.choice(size, 2, replace=False))
    
    child = -1 * np.ones(size, dtype=int)
    # Copy segment from parent1
    child[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    
    # Fill remaining from parent2
    current_p2_idx = 0
    for i in range(size):
        if i >= cxpoint1 and i < cxpoint2:
            continue
        while parent2[current_p2_idx] in child:
            current_p2_idx += 1
        child[i] = parent2[current_p2_idx]
    
    return child

def crossover_perm_batch(parent1_batch, parent2_batch, prob):
    B = parent1_batch.shape[0]
    children = []
    for b in range(B):
        children.append(crossover(parent1_batch[b], parent2_batch[b], prob))
    return np.array(children)

def crossover_rot(parent1, parent2, prob):
    """Uniform crossover for rotation maps."""
    if np.random.rand() >= prob:
        return parent1.copy()
    
    mask = np.random.rand(*parent1.shape) < 0.5
    child = parent1.copy()
    child[mask] = parent2[mask]
    return child

def evaluate_fitness(models_list, x, y, pop_r, pop_c, pop_k, block_size, enable_rotation=False):
    """
    Evaluate fitness of the population (Ensemble CrossEntropyLoss) per sample.
    Implements multi-model ensemble feedback with variance penalty.
    Phi = Sum(CE) - Sum((CE - Mean(CE))^2)
    """
    fitness_scores = []
    device = x.device
    
    # Ensure models_list is a list to support single model or ensemble
    if not isinstance(models_list, list):
        models_list = [models_list]
    
    # We evaluate each individual in the population
    for i in range(len(pop_r)):
        r = pop_r[i] # (B, n_h)
        c = pop_c[i] # (B, n_w)
        k = pop_k[i] if enable_rotation else None # (B, n_h, n_w)
        
        idx_r = torch.from_numpy(r).long().to(device)
        idx_c = torch.from_numpy(c).long().to(device)
        
        if k is not None:
            k_map = torch.from_numpy(k).long().to(device)
        else:
            k_map = None
        
        with torch.no_grad():
            # Apply block permutation with FIXED k_map for this individual
            x_perm = apply_block_transformation(x, idx_r, idx_c, block_size, k_map, enable_rotation)
            
            # Calculate losses for all models
            losses_list = []
            for model in models_list:
                output = get_model_output(model, x_perm)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                elif output.dim() > 2:
                    output = output.view(output.size(0), -1)
                
                # Use reduction='none' to get loss per sample
                loss = F.cross_entropy(output, y, reduction='none')
                losses_list.append(loss)
            
            # Stack losses: (num_models, B)
            losses = torch.stack(losses_list, dim=0)
            
            # Calculate fitness according to formula: Phi = Sum(CE) - Sum((CE - Mean(CE))^2)
            # This encourages high loss across all models (Transferability) and penalizes variance
            
            sum_loss = torch.sum(losses, dim=0)
            mean_loss = torch.mean(losses, dim=0)
            
            # Variance term
            diff = losses - mean_loss.unsqueeze(0)
            variance_term = torch.sum(diff ** 2, dim=0)
            
            # Final score
            score = sum_loss - variance_term
            
            fitness_scores.append(score.cpu().numpy())
            
    return np.array(fitness_scores) # (pop_size, B)

def input_diversity(x, prob=0.5):
    """
    New input_diversity Logic:
    * Divide the input tensor x (size 299x299) into a 2x2 grid (4 patches).
    * For each patch, apply a random rescale factor between 0.9 and 1.1.
    * Use F.interpolate with mode='bilinear' and align_corners=False to resize each patch back to its original quadrant size.
    * Reassemble the 4 patches back into a single 299x299 image.
    * Apply a final global F.interpolate (bilinear) to the reassembled image to smooth the seams between blocks.
    """
    if prob <= 0.0:
        return x
    
    if np.random.rand() > prob:
        return x
        
    B, C, H, W = x.shape
    
    # Calculate split points (approx middle)
    h_split = H // 2 + (1 if H % 2 != 0 else 0)
    w_split = W // 2 + (1 if W % 2 != 0 else 0)
    
    # Extract quadrants
    q1 = x[:, :, :h_split, :w_split]
    q2 = x[:, :, :h_split, w_split:]
    q3 = x[:, :, h_split:, :w_split]
    q4 = x[:, :, h_split:, w_split:]
    
    quadrants = [q1, q2, q3, q4]
    processed = []
    
    for q in quadrants:
        qh, qw = q.shape[2], q.shape[3]
        # Random rescale factor 0.9 - 1.1
        factor = 0.9 + (1.1 - 0.9) * torch.rand(1, device=x.device).item()
        
        target_h = int(qh * factor)
        target_w = int(qw * factor)
        
        # Resize to target and back
        # mode='bilinear', align_corners=False
        q_scaled = F.interpolate(q, size=(target_h, target_w), mode='bilinear', align_corners=False)
        q_back = F.interpolate(q_scaled, size=(qh, qw), mode='bilinear', align_corners=False)
        processed.append(q_back)
        
    # Reassemble
    top = torch.cat([processed[0], processed[1]], dim=3)
    bottom = torch.cat([processed[2], processed[3]], dim=3)
    reassembled = torch.cat([top, bottom], dim=2)
    
    # Final global smoothing
    final = F.interpolate(reassembled, size=(H, W), mode='bilinear', align_corners=False)
    
    return final

def ldr_attack(x, y, source_models, eps=16/255, iterations=10, mu=1.0, 
               de_pop_size=5, de_generations=5, de_prob_m=0.5, de_prob_c=0.8,
               block_size=1, enable_rotation=False, prob=0.5):

    """
    LDR Attack (Block-level) using Differential Evolution.
    """
    device = x.device
    B, C, H, W = x.shape
    
    # Ensure source_models is a list
    if not isinstance(source_models, list):
        source_models = [source_models]
    
    # Ensure x is detached for DE phase to avoid memory leaks
    x_de = x.detach()
    
    # Determine grid size
    h_pad = (block_size - H % block_size) % block_size
    w_pad = (block_size - W % block_size) % block_size
    n_h = (H + h_pad) // block_size
    n_w = (W + w_pad) // block_size
    
    # --- Step 1: DE Optimization for Permutations & Rotation ---
    # Initialize population (Batch-aware) - Use GPU for initialization
    pop_r = init_population_batch(de_pop_size, n_h, B, device=device)
    pop_c = init_population_batch(de_pop_size, n_w, B, device=device)
    pop_k = init_rotation_population_batch(de_pop_size, n_h, n_w, B, device=device) if enable_rotation else [None]*de_pop_size
    
    # Initial fitness: (pop_size, B)
    fitness = evaluate_fitness(source_models, x_de, y, pop_r, pop_c, pop_k, block_size, enable_rotation)
    
    # Track best per sample
    best_indices = np.argmax(fitness, axis=0) # (B,)
    best_fitness = np.max(fitness, axis=0) # (B,)
    
    # Construct initial bests
    best_r = np.zeros((B, n_h), dtype=int)
    best_c = np.zeros((B, n_w), dtype=int)
    best_k = np.zeros((B, n_h, n_w), dtype=int) if enable_rotation else None
    
    for b in range(B):
        idx = best_indices[b]
        best_r[b] = pop_r[idx][b]
        best_c[b] = pop_c[idx][b]
        if enable_rotation and best_k is not None and pop_k[idx] is not None:
            best_k[b] = pop_k[idx][b]
    
    for g in range(de_generations):
        new_pop_r = []
        new_pop_c = []
        new_pop_k = []
        
        for i in range(de_pop_size):
            # DE/rand/1 strategy
            idxs = [idx for idx in range(de_pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            # Mutation (Batch)
            v_r = mutation_perm_batch(pop_r[r1], pop_r[r2], pop_r[r3], de_prob_m)
            v_c = mutation_perm_batch(pop_c[r1], pop_c[r2], pop_c[r3], de_prob_m)
            
            if enable_rotation:
                # mutation_rot works with (B, ...) arrays directly
                v_k = mutation_rot(pop_k[r1], pop_k[r2], pop_k[r3], de_prob_m)
            else:
                v_k = None
            
            # Crossover (Batch)
            u_r = crossover_perm_batch(pop_r[i], v_r, de_prob_c)
            u_c = crossover_perm_batch(pop_c[i], v_c, de_prob_c)
            
            if enable_rotation:
                # crossover_rot works with (B, ...) arrays directly
                u_k = crossover_rot(pop_k[i], v_k, de_prob_c)
            else:
                u_k = None
            
            new_pop_r.append(u_r)
            new_pop_c.append(u_c)
            new_pop_k.append(u_k)
            
        # Evaluate candidates
        new_fitness = evaluate_fitness(source_models, x_de, y, new_pop_r, new_pop_c, new_pop_k, block_size, enable_rotation)
        
        # Selection (Greedy per sample)
        for i in range(de_pop_size):
            # improved: (B,) boolean mask
            improved = new_fitness[i] > fitness[i]
            
            if improved.any():
                pop_r[i][improved] = new_pop_r[i][improved]
                pop_c[i][improved] = new_pop_c[i][improved]
                if enable_rotation and pop_k[i] is not None and new_pop_k[i] is not None:
                    pop_k[i][improved] = new_pop_k[i][improved]
                
                fitness[i][improved] = new_fitness[i][improved]
        
        # Update global best
        current_best_indices = np.argmax(fitness, axis=0) # (B,)
        current_best_fitness = np.max(fitness, axis=0) # (B,)
        
        improved_best = current_best_fitness > best_fitness
        if improved_best.any():
            best_fitness[improved_best] = current_best_fitness[improved_best]
            # Update best_r, best_c, best_k for improved samples
            for b in np.where(improved_best)[0]:
                idx = current_best_indices[b]
                best_r[b] = pop_r[idx][b]
                best_c[b] = pop_c[idx][b]
                if enable_rotation and best_k is not None and pop_k[idx] is not None:
                    best_k[b] = pop_k[idx][b]
        # --- 插入监控代码开始 ---
        if (g + 1) % 1 == 0:  # 每一代都打印，方便观察
            avg_best_loss = best_fitness.mean()
            # 这里的 5.0 是一个经验阈值（Inception通常Loss到3-5以上就很稳了）
            print(f"    [DE Gen {g+1}/{de_generations}] Avg Best Loss: {avg_best_loss:.4f}")
        # --- 插入监控代码结束 ---

    # 循环结束后，可以加一个最终统计
    print(f"  >> DE Optimization Finished. Final Avg Best Loss: {best_fitness.mean():.4f}")            

    # --- Step 2: MI-FGSM with Learned Permutation & Rotation ---
    x_adv = x.clone().detach()
    alpha = eps / max(iterations, 1)
    momentum = torch.zeros_like(x_adv, device=device)

    # Convert best perms to torch indices
    idx_r_best = torch.from_numpy(best_r).long().to(device)
    idx_c_best = torch.from_numpy(best_c).long().to(device)
    
    if enable_rotation and best_k is not None:
        k_map_best = torch.from_numpy(best_k).long().to(device)
    else:
        k_map_best = None

    # Generate transformed image for saving
    with torch.no_grad():
        x_best_trans = apply_block_transformation(x, idx_r_best, idx_c_best, block_size, k_map_best, enable_rotation)

    # Number of augmentations for gradient averaging (BSR style)
    num_augmentations = 1

    for _ in range(iterations):
        x_adv.requires_grad_(True)

        # Apply Input Diversity
        x_div = input_diversity(x_adv, prob=prob)

        
        x_inputs = x_div       # 不再 repeat

        # 索引部分也只取最优的
        idx_r_all = idx_r_best
        idx_c_all = idx_c_best
        if enable_rotation:
            k_map_all = k_map_best
        else:
            k_map_all = None

        # Apply transformation: Tau(x) = R * x * C
        # x_trans will be (B*N, C, H, W)
        x_trans = apply_block_transformation(x_inputs, idx_r_all, idx_c_all, block_size, k_map_all, enable_rotation)

        # Targets need to be repeated to match batch size
        y_repeated = y.repeat(num_augmentations)
        
        total_loss = 0.0
        for model in source_models:
            output = get_model_output(model, x_trans)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                output = output.view(output.size(0), -1)

            total_loss += F.cross_entropy(output, y_repeated)
        
        loss = total_loss / len(source_models)

        # Compute gradient
        # autograd will aggregate gradients from all copies back to x_adv
        grad = torch.autograd.grad(loss, [x_adv])[0]
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

        momentum = mu * momentum + grad
        x_adv = x_adv.detach() + alpha * torch.sign(momentum)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach(), x_best_trans.detach()

def parse_args():
    parser = argparse.ArgumentParser(description='LDR Attack')
    parser.add_argument('--input_dir', default='./data', type=str, help='input directory')
    parser.add_argument('--model', default='tf2torch_inception_v3', type=str, help='source model name')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--eps', default=16/255, type=float, help='epsilon for attack')
    parser.add_argument('--iterations', default=10, type=int, help='number of iterations for attack')
    parser.add_argument('--mu', default=1.0, type=float, help='momentum for attack')
    parser.add_argument('--de_pop_size', default=5, type=int, help='DE population size')
    parser.add_argument('--de_generations', default=5, type=int, help='DE generations')
    parser.add_argument('--de_prob_m', default=0.5, type=float, help='DE mutation probability')
    parser.add_argument('--de_prob_c', default=0.8, type=float, help='DE crossover probability')
    parser.add_argument('--block_size', default=1, type=int, help='block size for LDR')
    parser.add_argument('--enable_rotation', action='store_true', help='enable rotation in LDR')
    parser.add_argument('--prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument('--output_csv', default='./attack_results_LDR.csv', type=str, help='output CSV path')
    parser.add_argument('--output_dir', default='./output_adv', type=str, help='directory to save adversarial images')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA device id, e.g., 0')
    return parser.parse_args()

def main_cli():
    args = parse_args()
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_repo = ModelRepository(device)
    
    # 使用新的加载逻辑
    try:
        source_model = load_source_model(args.model, device, model_repo)
    except ValueError as e:
        print(f"Error loading model: {e}")
        return

    # Prepare Ensemble
    ensemble_models = [source_model]
    # Load auxiliary models for ensemble feedback
    aux_models = ['tv_resnet50', 'tv_vgg16']
    print(f"Loading auxiliary models for ensemble: {aux_models}")
    for aux_name in aux_models:
        # Avoid duplicating the main source model if it's one of the aux models
        if aux_name == args.model:
             continue
             
        try:
            aux_model = load_source_model(aux_name, device, model_repo)
            ensemble_models.append(aux_model)
        except Exception as e:
            print(f"Warning: Could not load auxiliary model {aux_name}: {e}")
            
    print(f"Ensemble size: {len(ensemble_models)}")

    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)
    print(f"Available models: {all_models}")
    print(f"Selected {len(target_models)} target models for testing")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    import pandas as pd
    label_csv = os.path.join(args.input_dir, 'labels.csv')
    img_root = os.path.join(args.input_dir, 'images')
    if not os.path.exists(label_csv) or not os.path.isdir(img_root):
        raise FileNotFoundError(f"Data not found. labels: {label_csv}, images dir: {img_root}")
    label_df = pd.read_csv(label_csv)
    
    label_df['abs_path'] = label_df['filename'].apply(lambda fn: os.path.join(img_root, fn))
    missing = label_df[~label_df['abs_path'].apply(os.path.isfile)]
    if not missing.empty:
        print(f"Warning: Found {len(missing)} missing files, skipping. E.g.: {missing['filename'].iloc[0]}")
    label_df = label_df[label_df['abs_path'].apply(os.path.isfile)].drop(columns=['abs_path'])
    if label_df.empty:
        raise FileNotFoundError("No valid images found in data/images matching labels.csv")

    dataset = AttackDataset(label_df, img_root, transform)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    print(f"Total {len(label_df)} valid images, using batch_size={args.batchsize}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    attack_params = {
        "eps": args.eps, 
        "iterations": args.iterations, 
        "mu": args.mu,
        "de_pop_size": args.de_pop_size,
        "de_generations": args.de_generations,
        "de_prob_m": args.de_prob_m,
        "de_prob_c": args.de_prob_c,
        "block_size": args.block_size,
        "enable_rotation": args.enable_rotation,
        "prob": args.prob
    }

    results = []
    print("\nStarting LDR Attack...")
    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Running LDR"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)
        print(f"True Label: {y_batch[0].item()}, Model Pred: {source_orig_preds[0]}")
        # Run LDR Attack
        x_adv_batch, x_trans_batch = ldr_attack(x_batch, y_batch, ensemble_models, **attack_params)
        
        # Save adversarial images
        trans_dir = os.path.join(args.output_dir, 'transformed')
        if not os.path.exists(trans_dir):
            os.makedirs(trans_dir)

        for i in range(len(filename_batch)):
            save_path = os.path.join(args.output_dir, filename_batch[i])
            save_image(x_adv_batch[i].cpu(), save_path)

            save_path_trans = os.path.join(trans_dir, filename_batch[i])
            save_image(x_trans_batch[i].cpu(), save_path_trans)

        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        target_batch_preds = {}
        for model_name, model_info in target_models.items():
            model = model_info['model']
            target_batch_preds[model_name] = get_model_prediction(model, x_adv_batch)

        for i in range(x_batch.size(0)):
            true_label = int(y_batch[i].item()) +1
            entry = {
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": int(source_orig_preds[i]) +1,
                "source_adv_pred": int(source_adv_preds[i]),
                "source_attack_success": int(source_adv_preds[i]) != true_label,
                "target_results": {}
            }
            for model_name, preds in target_batch_preds.items():
                pred_i = int(preds[i])
                entry["target_results"][model_name] = {
                    "prediction": pred_i,
                    "fooled": pred_i != true_label
                }
            results.append(entry)

    print("\n" + "="*80)
    print(f"Summary of LDR Attack Results (Source: {args.model})")
    print("="*80)

    if results:
        source_success = sum(1 for r in results if r['source_attack_success'])
        rate = source_success / len(results) * 100
        print(f"Source model attack success rate: {source_success}/{len(results)} ({rate:.1f}%)")

        model_names = list(target_models.keys())
        model_success_counts = {name: 0 for name in model_names}
        for r in results:
            for name, tr in r['target_results'].items():
                if tr['fooled']:
                    model_success_counts[name] += 1
        print("\nTransfer attack success rates for each target model:")
        for name, count in model_success_counts.items():
            print(f"  {name}: {count}/{len(results)} ({count/len(results)*100:.1f}%)")
        avg_rate = np.mean(list(model_success_counts.values())) / len(results) * 100
        print(f"\nAverage transfer success rate: {avg_rate:.1f}%")

        flat_rows = []
        for r in results:
            row = {
                "filename": r["filename"],
                "true_label": r["true_label"],
                "source_original_pred": r["source_original_pred"],
                "source_adv_pred": r["source_adv_pred"],
                "source_attack_success": r["source_attack_success"],
            }
            for name, tr in r["target_results"].items():
                row[f"{name}_pred"] = tr["prediction"]
                row[f"{name}_fooled"] = tr["fooled"]
            flat_rows.append(row)
        
        pd.DataFrame(flat_rows).to_csv(args.output_csv, index=False)
        print(f"\nDetailed results saved to {args.output_csv}")
    else:
        print("No results to show.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_cli()
