import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        true_label = int(row["label"]) + 1  # 与模型标签对齐
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
    Apply block-level permutation and optional rotation.
    x: (B, C, H, W)
    idx_r: (B, Nh) or (Nh,) tensor of row block indices
    idx_c: (B, Nw) or (Nw,) tensor of col block indices
    block_size: int
    k_map: (B, Nh, Nw) tensor of rotation types (0-3), optional.
    enable_rotation: bool
    """
    B, C, H, W = x.shape
    
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
    # Handle idx_r and idx_c broadcasting
    if idx_r.dim() == 1:
        idx_r = idx_r.unsqueeze(0).expand(B, -1)
    if idx_c.dim() == 1:
        idx_c = idx_c.unsqueeze(0).expand(B, -1)
        
    # Use Advanced Indexing for efficient permutation
    
    # 1. Row permutation (dim 2)
    # x_blocks: (B, C, n_h, block_size, n_w, block_size)
    # We want to permute dim 2 based on idx_r (B, n_h)
    
    ib = torch.arange(B, device=x.device).view(B, 1, 1)
    ic = torch.arange(C, device=x.device).view(1, C, 1)
    ir = idx_r.view(B, 1, n_h)
    
    # Result shape: (B, C, n_h, block_size, n_w, block_size)
    x_perm = x_blocks[ib, ic, ir]
    
    # 2. Col permutation (dim 4)
    # We want to permute dim 4 based on idx_c (B, n_w)
    # To avoid complex 6D indexing, we permute dim 4 to dim 1, index, then permute back.
    
    # (B, C, n_h, block_size, n_w, block_size) -> (B, n_w, C, n_h, block_size, block_size)
    x_perm = x_perm.permute(0, 4, 1, 2, 3, 5)
    
    ib = torch.arange(B, device=x.device).view(B, 1)
    ic_col = idx_c.view(B, n_w)
    
    # Indexing dim 1 (n_w) with (B, n_w)
    # Result: (B, n_w, C, n_h, block_size, block_size)
    x_perm = x_perm[ib, ic_col]
    
    # Permute back: (B, n_w, C, n_h, block_size, block_size) -> (B, C, n_h, block_size, n_w, block_size)
    # Dims: 0, 2, 3, 4, 1, 5
    x_perm = x_perm.permute(0, 2, 3, 4, 1, 5)
    
    # 4. Optional Rotation
    if enable_rotation:
        # Permute to (B, C, n_h, n_w, block_size, block_size) to handle rotation
        x_reshaped = x_perm.permute(0, 1, 2, 4, 3, 5)
        
        # Create output tensor
        x_rotated = torch.zeros_like(x_reshaped)
        
        # Use provided k_map or generate random one
        if k_map is None:
            # If k_map is not provided, we generate one. 
            # Note: For MI-FGSM stability, k_map should be provided (fixed).
            k_map = torch.randint(0, 4, (B, n_h, n_w), device=x.device)
        else:
            # Ensure k_map matches batch size if it was generated for a single individual
            if k_map.shape[0] != B:
                k_map = k_map.expand(B, -1, -1)
        
        for k in range(4):
            mask = (k_map == k)
            if not mask.any():
                continue
            
            # Expand mask: (B, C, n_h, n_w, block_size, block_size)
            mask_exp = mask.unsqueeze(1).unsqueeze(4).unsqueeze(5).expand_as(x_reshaped)
            
            # Rotate dims 4 and 5 (block_size, block_size)
            rotated_k = torch.rot90(x_reshaped, k, [4, 5])
            
            x_rotated[mask_exp] = rotated_k[mask_exp]
            
        # Permute back to (B, C, n_h, block_size, n_w, block_size)
        x_out_blocks = x_rotated.permute(0, 1, 2, 4, 3, 5)
    else:
        x_out_blocks = x_perm

    # 5. Reshape back to image
    x_out = x_out_blocks.contiguous().view(B, C, H_p, W_p)
    
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

def evaluate_fitness(model, x, y, pop_r, pop_c, pop_k, block_size, enable_rotation=False):
    """Evaluate fitness of the population (CrossEntropyLoss) per sample."""
    fitness_scores = []
    device = x.device
    
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
            
            output = get_model_output(model, x_perm)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                output = output.view(output.size(0), -1)
            
            # Use reduction='none' to get loss per sample
            loss = F.cross_entropy(output, y, reduction='none')
            fitness_scores.append(loss.cpu().numpy())
            
    return np.array(fitness_scores) # (pop_size, B)

def ldr_attack(x, y, model, eps=16/255, iterations=10, mu=1.0, 
               de_pop_size=5, de_generations=5, de_prob_m=0.5, de_prob_c=0.8,
               block_size=1, enable_rotation=False):
    """
    LDR Attack (Block-level) using Differential Evolution.
    """
    device = x.device
    B, C, H, W = x.shape
    
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
    fitness = evaluate_fitness(model, x_de, y, pop_r, pop_c, pop_k, block_size, enable_rotation)
    
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
        if enable_rotation:
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
        new_fitness = evaluate_fitness(model, x_de, y, new_pop_r, new_pop_c, new_pop_k, block_size, enable_rotation)
        
        # Selection (Greedy per sample)
        for i in range(de_pop_size):
            # improved: (B,) boolean mask
            improved = new_fitness[i] > fitness[i]
            
            if improved.any():
                pop_r[i][improved] = new_pop_r[i][improved]
                pop_c[i][improved] = new_pop_c[i][improved]
                if enable_rotation:
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
                if enable_rotation:
                    best_k[b] = pop_k[idx][b]

    # --- Step 2: MI-FGSM with Learned Permutation & Rotation ---
    x_adv = x.clone().detach()
    alpha = eps / max(iterations, 1)
    momentum = torch.zeros_like(x_adv, device=device)
    
    # Convert best perms to torch indices
    idx_r = torch.from_numpy(best_r).long().to(device)
    idx_c = torch.from_numpy(best_c).long().to(device)
    
    if enable_rotation and best_k is not None:
        k_map = torch.from_numpy(best_k).long().to(device)
    else:
        k_map = None
    
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        
        # Apply transformation: Tau(x) = R * x * C
        # idx_r, idx_c are (B, n_h) and (B, n_w)
        x_trans = apply_block_transformation(x_adv, idx_r, idx_c, block_size, k_map, enable_rotation)
        
        output = get_model_output(model, x_trans)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            output = output.view(output.size(0), -1)
            
        loss = F.cross_entropy(output, y)
        
        grad = torch.autograd.grad(loss, [x_adv])[0]
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        
        momentum = mu * momentum + grad
        x_adv = x_adv.detach() + alpha * torch.sign(momentum)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
        
    return x_adv.detach()

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
    parser.add_argument('--output_csv', default='./attack_results_LDR.csv', type=str, help='output CSV path')
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

    attack_params = {
        "eps": args.eps, 
        "iterations": args.iterations, 
        "mu": args.mu,
        "de_pop_size": args.de_pop_size,
        "de_generations": args.de_generations,
        "de_prob_m": args.de_prob_m,
        "de_prob_c": args.de_prob_c,
        "block_size": args.block_size,
        "enable_rotation": args.enable_rotation
    }

    results = []
    print("\nStarting LDR Attack...")
    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Running LDR"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)
        
        # Run LDR Attack
        x_adv_batch = ldr_attack(x_batch, y_batch, source_model, **attack_params)
        
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        target_batch_preds = {}
        for model_name, model_info in target_models.items():
            model = model_info['model']
            target_batch_preds[model_name] = get_model_prediction(model, x_adv_batch)

        for i in range(x_batch.size(0)):
            true_label = int(y_batch[i].item())
            entry = {
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": int(source_orig_preds[i]),
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
