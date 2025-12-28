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
from scipy.optimize import linear_sum_assignment
from models import ModelRepository, Normalize

# --- Helper Functions (Shared) ---

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
    if model_name.startswith('tv_'):
        import torchvision.models as tv_models
        name = model_name.replace('tv_', '')
        print(f"Loading torchvision model: {name}")
        
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

def get_label_offset(model_name):
    if model_name.startswith('tv_'):
        return 0
    return 1

class AttackDataset(Dataset):
    def __init__(self, label_df, img_root, transform, label_offset=0):
        self.label_df = label_df
        self.img_root = img_root
        self.transform = transform
        self.label_offset = label_offset

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        img_filename = row["filename"]
        true_label = int(row["label"]) + self.label_offset
        img_path = os.path.join(self.img_root, img_filename)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(true_label, dtype=torch.long), img_filename

# --- NDSA Core Functions ---

def sinkhorn(log_alpha, n_iters=20, tau=1.0):
    """
    Log-space Sinkhorn iteration to project log_alpha onto the Birkhoff polytope.
    """
    log_alpha = log_alpha / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)

def extract_blocks(x, block_size):
    """
    Splits image into blocks.
    """
    B, C, H, W = x.shape
    h_pad = (block_size - H % block_size) % block_size
    w_pad = (block_size - W % block_size) % block_size
    
    if h_pad > 0 or w_pad > 0:
        x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
    else:
        x_padded = x
        
    _, _, H_p, W_p = x_padded.shape
    n_h = H_p // block_size
    n_w = W_p // block_size
    N = n_h * n_w
    
    # (B, C, n_h, block_size, n_w, block_size)
    x_blocks = x_padded.view(B, C, n_h, block_size, n_w, block_size)
    # (B, C, n_h, n_w, block_size, block_size)
    x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5)
    # (B, N, C, block_size, block_size)
    x_blocks = x_blocks.contiguous().view(B, N, C, block_size, block_size)
    
    return x_blocks, (h_pad, w_pad, n_h, n_w, H, W)

def assemble_blocks(blocks, pad_info, block_size):
    """
    Reassembles blocks into image.
    """
    h_pad, w_pad, n_h, n_w, H, W = pad_info
    B, N, C, _, _ = blocks.shape
    
    # (B, C, n_h, n_w, block_size, block_size)
    x_reshaped = blocks.view(B, C, n_h, n_w, block_size, block_size)
    # (B, C, n_h, block_size, n_w, block_size)
    x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
    # (B, C, H_p, W_p)
    x_out = x_permuted.contiguous().view(B, C, n_h * block_size, n_w * block_size)
    
    if h_pad > 0 or w_pad > 0:
        x_out = x_out[:, :, :H, :W]
        
    return x_out

def soft_shuffle(x, P, block_size):
    """
    Differentiable block shuffling using soft permutation matrix P.
    """
    blocks, pad_info = extract_blocks(x, block_size)
    B, N, C, h, w = blocks.shape
    blocks_flat = blocks.view(B, N, -1)
    
    # (B, N, N) @ (B, N, D) -> (B, N, D)
    shuffled_flat = torch.bmm(P, blocks_flat)
    
    shuffled_blocks = shuffled_flat.view(B, N, C, h, w)
    return assemble_blocks(shuffled_blocks, pad_info, block_size)

def hard_shuffle(x, indices, block_size):
    """
    Discrete block shuffling using indices.
    """
    blocks, pad_info = extract_blocks(x, block_size)
    B, N, C, h, w = blocks.shape
    
    batch_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, N)
    shuffled_blocks = blocks[batch_idx, indices]
    
    return assemble_blocks(shuffled_blocks, pad_info, block_size)

def ndsa_attack(x, y, model, eps=16/255, iterations=10, mu=1.0, 
                block_size=32, opt_iters=20, ensemble_size=5, tau=0.1):
    """
    Neural Differentiable Shuffling Attack (NDSA)
    """
    device = x.device
    B, C, H, W = x.shape
    
    # --- Phase 1: Manifold Optimization ---
    _, (_, _, n_h, n_w, _, _) = extract_blocks(x, block_size)
    N = n_h * n_w
    
    # Initialize Logits M for permutation
    M = torch.randn(B, N, N, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([M], lr=0.1)
    
    # Optimization loop to find best soft permutation
    for _ in range(opt_iters):
        optimizer.zero_grad()
        P = sinkhorn(M, tau=tau)
        x_shuffled = soft_shuffle(x, P, block_size)
        
        output = get_model_output(model, x_shuffled)
        if output.dim() == 1: output = output.unsqueeze(0)
        elif output.dim() > 2: output = output.view(output.size(0), -1)
            
        # Maximize CrossEntropy (make it adversarial)
        loss = -F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        
    # --- Phase 2: Discretization & Ensemble ---
    perm_ensemble = [] 
    
    with torch.no_grad():
        M_final = M.detach()
        for _ in range(ensemble_size):
            # Gumbel-Sinkhorn sampling / Noise injection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(M_final) + 1e-20) + 1e-20)
            M_noisy = M_final + gumbel_noise
            
            batch_indices = []
            for b in range(B):
                cost_matrix = -M_noisy[b].cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                batch_indices.append(torch.from_numpy(col_ind).long())
            
            perm_ensemble.append(torch.stack(batch_indices).to(device))
            
    # --- Phase 3: Synchronous MI-FGSM ---
    x_adv = x.clone().detach()
    alpha = eps / max(iterations, 1)
    momentum = torch.zeros_like(x_adv, device=device)
    
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        
        total_loss = 0
        for indices in perm_ensemble:
            x_trans = hard_shuffle(x_adv, indices, block_size)
            output = get_model_output(model, x_trans)
            if output.dim() == 1: output = output.unsqueeze(0)
            elif output.dim() > 2: output = output.view(output.size(0), -1)
            
            total_loss += F.cross_entropy(output, y)
            
        loss = total_loss / ensemble_size
        
        grad = torch.autograd.grad(loss, [x_adv])[0]
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        
        momentum = mu * momentum + grad
        x_adv = x_adv.detach() + alpha * torch.sign(momentum)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
        
    return x_adv.detach()

def parse_args():
    parser = argparse.ArgumentParser(description='NDSA Attack')
    parser.add_argument('--input_dir', default='./data', type=str, help='input directory')
    parser.add_argument('--model', default='tf2torch_inception_v3', type=str, help='source model name')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--eps', default=16/255, type=float, help='epsilon for attack')
    parser.add_argument('--iterations', default=10, type=int, help='number of iterations for attack')
    parser.add_argument('--mu', default=1.0, type=float, help='momentum for attack')
    parser.add_argument('--block_size', default=32, type=int, help='block size for NDSA')
    parser.add_argument('--opt_iters', default=20, type=int, help='optimization iterations for permutation')
    parser.add_argument('--ensemble_size', default=5, type=int, help='number of permutations in ensemble')
    parser.add_argument('--tau', default=0.1, type=float, help='temperature for Sinkhorn')
    parser.add_argument('--output_csv', default='./results/ndsa/attack_results_NDSA.csv', type=str, help='output CSV path')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA device id')
    return parser.parse_args()

def main_cli():
    args = parse_args()
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_repo = ModelRepository(device)
    
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
        print(f"Warning: Found {len(missing)} missing files, skipping.")
    label_df = label_df[label_df['abs_path'].apply(os.path.isfile)].drop(columns=['abs_path'])
    if label_df.empty:
        raise FileNotFoundError("No valid images found.")

    source_offset = get_label_offset(args.model)
    dataset = AttackDataset(label_df, img_root, transform, label_offset=source_offset)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    print(f"Total {len(label_df)} valid images, using batch_size={args.batchsize}")

    attack_params = {
        "eps": args.eps, 
        "iterations": args.iterations, 
        "mu": args.mu,
        "block_size": args.block_size,
        "opt_iters": args.opt_iters,
        "ensemble_size": args.ensemble_size,
        "tau": args.tau
    }

    results = []
    print("\nStarting NDSA Attack...")
    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Running NDSA"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)
        
        x_adv_batch = ndsa_attack(x_batch, y_batch, source_model, **attack_params)
        
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        target_batch_preds = {}
        for model_name, model_info in target_models.items():
            model = model_info['model']
            target_batch_preds[model_name] = get_model_prediction(model, x_adv_batch)

        for i in range(x_batch.size(0)):
            true_label_src = int(y_batch[i].item())
            canonical_label = true_label_src - source_offset
            
            entry = {
                "filename": filename_batch[i],
                "true_label": canonical_label,
                "source_original_pred": int(source_orig_preds[i]),
                "source_adv_pred": int(source_adv_preds[i]),
                "source_attack_success": int(source_adv_preds[i]) != true_label_src,
                "target_results": {}
            }
            for model_name, preds in target_batch_preds.items():
                pred_i = int(preds[i])
                target_offset = get_label_offset(model_name)
                true_label_tgt = canonical_label + target_offset
                
                entry["target_results"][model_name] = {
                    "prediction": pred_i,
                    "fooled": pred_i != true_label_tgt
                }
            results.append(entry)

    print("\n" + "="*80)
    print(f"Summary of NDSA Attack Results (Source: {args.model})")
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

        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
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
