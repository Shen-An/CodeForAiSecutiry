import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as torchvision_models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models import ModelRepository, Normalize

#M-DI²-FGSM
# 输入多样性增强 (DIM)
def input_diversity(x, prob=0.5):
    if prob <= 0:
        return x
    if np.random.random() > prob:
        return x
    rnd = np.random.randint(299, 330)
    resized = F.interpolate(x, size=(rnd, rnd), mode='nearest')
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem + 1)
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem + 1)
    pad_right = w_rem - pad_left
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return F.interpolate(padded, size=(299, 299), mode='nearest')


def get_model_output(model, x):
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")
    return output


def get_model_prediction(model, x):
    # 始终返回一维 numpy 数组（长度为 batch）
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


def dim_attack(x, y, model, eps=16/255, iterations=10, mu=1.0, prob=0.5):
    """DIM + MI-FGSM，支持批量。"""
    device = x.device
    x_adv = x.clone().detach()
    alpha = eps / max(iterations, 1)
    momentum = torch.zeros_like(x_adv, device=device)

    for _ in range(iterations):
        x_adv.requires_grad_(True)
        x_div = input_diversity(x_adv, prob) if prob > 0 else x_adv
        output = get_model_output(model, x_div)
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
        return x, torch.tensor(true_label, dtype=torch.long), img_filename


def parse_args():
    parser = argparse.ArgumentParser(description='DIM (Diverse Input Method) attack (batch + CLI)')
    parser.add_argument('--model', default='tf2torch_inception_v3', type=str, help='source model')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--eps', default=16/255, type=float, help='L_inf epsilon')
    parser.add_argument('--iterations', default=10, type=int, help='attack iterations')
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument('--input_dir', default='./data', type=str, help='data root containing images/ and labels.csv')
    parser.add_argument('--output_csv', default='./results/dim/attack_results_DIM.csv', type=str, help='output CSV path')
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
        print(f"警告：发现 {len(missing)} 个缺失文件，将跳过。例如：{missing['filename'].iloc[0]}")
    label_df = label_df[label_df['abs_path'].apply(os.path.isfile)].drop(columns=['abs_path'])
    if label_df.empty:
        raise FileNotFoundError("所有标注对应的图像文件均不存在，请检查 data/images 与 labels.csv")

    dataset = AttackDataset(label_df, img_root, transform)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    print(f"Total {len(label_df)} valid images, using batch_size={args.batchsize}")

    attack_params = {"eps": args.eps, "iterations": args.iterations, "mu": args.mu, "prob": args.prob}

    results = []
    print("\nStarting DIM Attack...")
    for x_batch, y_batch, filename_batch in tqdm(loader, desc="Running DIM"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)
        x_adv_batch = dim_attack(x_batch, y_batch, source_model, **attack_params)
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
    print(f"Summary of DIM Attack Results (Source: {args.model})")
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

        import pandas as pd
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
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        import pandas as pd
        pd.DataFrame(flat_rows).to_csv(args.output_csv, index=False)
        print(f"\nDetailed results saved to {args.output_csv}")
    else:
        print("No results to show.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_cli()