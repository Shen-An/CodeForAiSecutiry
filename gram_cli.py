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
    import torch.nn as nn
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
        return model_repo._load_model(model_name)

# --- G-FIA Attack Implementation ---

class FeatureExtractor(nn.Module):
    """
    Wraps a model to extract features from a specific layer.
    """
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.feature_maps = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        target_layer = None
        # Recursive search for layer
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            # If wrapped in Sequential (Normalize, Net), check Net
            if isinstance(self.model, nn.Sequential):
                for name, _ in self.model[1].named_modules():
                    if name == self.layer_name:
                        target_layer = self.model[1].get_submodule(name)
                        break
            
            if target_layer is None:
                print(f"Error: Layer '{self.layer_name}' not found in model.")
                print("Available top-level modules:")
                for name, _ in self.model.named_children():
                    print(f" - {name}")
                raise ValueError(f"Layer {self.layer_name} not found.")

        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)
        print(f"Hook registered on layer: {self.layer_name}")

    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    def remove_hooks(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

    def forward(self, x):
        # We only care about running the model to trigger the hook
        # But we also need the final output for Loss calculation (CrossEntropy)
        return self.model(x)

def get_gram(f_map):
    """
    Compute Gram Matrix: G_ij = sum(F_ik * F_jk) / (C * H * W)
    """
    n, c, h, w = f_map.shape
    f = f_map.view(n, c, h * w)
    # Batch matrix multiplication
    gram = torch.bmm(f, f.transpose(1, 2)) 
    # Normalize
    return gram / (c * h * w)

def input_diversity(x, prob=0.5):
    """
    Standard Input Diversity (resize and pad).
    """
    # User requested to disable input diversity temporarily
    return x

def g_fia_attack(x, y, extractor, eps=16/255, iterations=10, mu=1.0, num_ens=30, prob=0.7):
    """
    G-FIA Attack Implementation.
    """
    device = x.device
    model = extractor.model
    B, C, H, W = x.shape
    
    # 1. Wrap model to extract features
    # Note: We assume 'model' is already in eval mode
    # extractor is passed as argument
    
    # 2. Calculate FIA Weights (Alpha)
    # We need gradients of Loss w.r.t features
    # Ensemble over random transforms
    
    agg_grad = 0
    x_temp = x.clone().detach()
    
    for _ in range(num_ens):
        x_in = x_temp # Disable input diversity
        x_in.requires_grad = True
        
        # Forward
        output = extractor(x_in)
        features = extractor.feature_maps
        
        # We need to capture gradient at 'features'
        # Since 'features' is an intermediate tensor, we retain grad
        features.retain_grad()
        
        # Loss (CrossEntropy) - we want to find features important for classification
        # We use the true label y
        loss = nn.CrossEntropyLoss()(output, y)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Accumulate gradients
        if features.grad is not None:
            agg_grad += features.grad.data
        else:
            # If layer is not connected to output or something is wrong
            pass
            
    # Average gradients
    agg_grad /= num_ens
    
    # Calculate Alpha: ReLU(GAP(Mean(Grad)))
    # agg_grad: (B, C, H, W)
    # GAP over H, W
    fia_weight_vec = torch.mean(torch.abs(agg_grad), dim=(2, 3)) # (B, C)
    # Normalize to [0, 1]
    fia_weight_vec = fia_weight_vec / (fia_weight_vec.max(dim=1, keepdim=True)[0] + 1e-7)
    
    # Construct Importance Matrix W_matrix: (B, C, C)
    # W_ij = alpha_i * alpha_j
    W_matrix = torch.bmm(fia_weight_vec.unsqueeze(2), fia_weight_vec.unsqueeze(1))
    W_matrix = W_matrix.detach() # Fixed weights
    
    # 3. Attack Loop
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)
    alpha_step = eps / iterations
    
    # Pre-calculate Gram of original image
    with torch.no_grad():
        _ = extractor(x)
        features_orig = extractor.feature_maps.detach()
        gram_orig = get_gram(features_orig)
    
    for i in range(iterations):
        x_adv.requires_grad = True
        
        x_in = x_adv # Disable input diversity
        
        # Forward
        _ = extractor(x_in)
        features_adv = extractor.feature_maps
        
        # Calculate Gram
        gram_adv = get_gram(features_adv)
        
        # Calculate Loss: Weighted MSE
        # We want to MAXIMIZE the difference, so we minimize -Loss or use gradient ascent
        # The formula in gram.md: loss = (W_matrix * (gram_adv - gram_orig) ** 2).sum() / W_matrix.sum()
        # This is a distance. We want to maximize this distance to destroy correlations.
        # So we should ASCEND on this loss.
        
        diff_sq = (gram_adv - gram_orig) ** 2
        loss = (W_matrix * diff_sq).sum() / (W_matrix.sum() + 1e-7)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad
        
        # Normalize gradient
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        
        # Update momentum
        momentum = mu * momentum + grad
        
        # Update x_adv (Gradient Ascent because we want to maximize the Gram distance)
        x_adv = x_adv.detach() + alpha_step * torch.sign(momentum)
        
        # Projection
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
        
    return x_adv.detach()

def parse_args():
    parser = argparse.ArgumentParser(description='G-FIA Attack')
    parser.add_argument('--input_dir', default='./data', type=str, help='input directory')
    parser.add_argument('--model', default='tv_inception_v3', type=str, help='source model name')
    parser.add_argument('--layer_name', default='layer3', type=str, help='layer to attack (e.g. layer3 for ResNet)')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--eps', default=16/255, type=float, help='epsilon for attack')
    parser.add_argument('--iterations', default=20, type=int, help='number of iterations for attack')
    parser.add_argument('--mu', default=0.9, type=float, help='momentum for attack')
    parser.add_argument('--num_ens', default=30, type=int, help='number of ensemble for FIA weights')
    parser.add_argument('--prob', default=0, type=float, help='input diversity probability')
    parser.add_argument('--output_csv', default='./attack_results_GFIA.csv', type=str, help='output CSV path')
    parser.add_argument('--output_dir', default='./output_adv_gfia', type=str, help='directory to save adversarial images')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA device id')
    return parser.parse_args()

def main_cli():
    args = parse_args()
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize ModelRepository
    model_repo = ModelRepository.__new__(ModelRepository)
    model_repo.device = device
    model_repo.model_dir = 'torch_nets_weight/'
    model_repo.models = {}
    
    # Load Source Model
    try:
        source_model = load_source_model(args.model, device, model_repo)
    except ValueError as e:
        print(f"Error loading model: {e}")
        return

    # Initialize FeatureExtractor
    extractor = FeatureExtractor(source_model, args.layer_name)

    # Target models for evaluation
    all_models = [
        'tf2torch_inception_v3',
        'tf2torch_resnet_v2_50',
        'tv_resnet50',
        'tv_vgg16'
    ]
    
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
    
    # Filter existing
    label_df['abs_path'] = label_df['filename'].apply(lambda fn: os.path.join(img_root, fn))
    label_df = label_df[label_df['abs_path'].apply(os.path.isfile)].drop(columns=['abs_path'])

    dataset = AttackDataset(label_df, img_root, transform)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = []
    print("\nStarting G-FIA Attack...")
    
    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Running G-FIA"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Get original prediction
        source_orig_preds = get_model_prediction(source_model, x_batch)

        # Run Attack
        x_adv_batch = g_fia_attack(
            x_batch, y_batch, extractor, 
            eps=args.eps, 
            iterations=args.iterations, 
            mu=args.mu,
            num_ens=args.num_ens,
            prob=args.prob
        )
        
        # Save images
        for i in range(len(filename_batch)):
            save_image(x_adv_batch[i], os.path.join(args.output_dir, filename_batch[i]))

        # Evaluation
        # 1. Source model prediction
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)
        
        # 2. Target models prediction
        target_batch_preds = {}
        
        for model_name in all_models:
            if model_name == args.model:
                target_batch_preds[model_name] = source_adv_preds
                continue
            
            try:
                # Load temp model
                temp_model = model_repo._load_model(model_name)
                target_batch_preds[model_name] = get_model_prediction(temp_model, x_adv_batch)
                
                # Cleanup
                del temp_model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error evaluating on {model_name}: {e}")
                target_batch_preds[model_name] = np.zeros(len(x_batch)) - 2 # Error code

        # 3. Collect results
        for i in range(len(filename_batch)):
            # Use 1-based indexing for consistency
            true_label = int(y_batch[i].item()) + 1
            orig_pred = int(source_orig_preds[i]) + 1
            adv_pred = int(source_adv_preds[i]) + 1
            
            entry = {
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": orig_pred,
                "source_adv_pred": adv_pred,
                "source_attack_success": adv_pred != true_label,
                "target_results": {}
            }
            
            for model_name, preds in target_batch_preds.items():
                p_val = int(preds[i])
                if p_val < 0:
                    # Error case
                    entry["target_results"][model_name] = {
                        "prediction": -1,
                        "fooled": False
                    }
                else:
                    p_val += 1 # 1-based
                    entry["target_results"][model_name] = {
                        "prediction": p_val,
                        "fooled": p_val != true_label
                    }
            
            results.append(entry)

    extractor.remove_hooks()

    # Summary and Save
    print("\n" + "="*80)
    print(f"Summary of G-FIA Attack Results (Source: {args.model})")
    print("="*80)

    if results:
        source_success = sum(1 for r in results if r['source_attack_success'])
        rate = source_success / len(results) * 100
        print(f"Source model attack success rate: {source_success}/{len(results)} ({rate:.1f}%)")

        model_success_counts = {name: 0 for name in all_models}
        for r in results:
            for name, tr in r['target_results'].items():
                if tr['fooled']:
                    model_success_counts[name] += 1
        
        print("\nTransfer attack success rates for each target model:")
        for name, count in model_success_counts.items():
            print(f"  {name}: {count}/{len(results)} ({count/len(results)*100:.1f}%)")
            
        avg_rate = np.mean(list(model_success_counts.values())) / len(results) * 100
        print(f"\nAverage transfer success rate: {avg_rate:.1f}%")

        # Flatten for CSV
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
    main_cli()
``` 