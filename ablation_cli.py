#消融实验
# 新建了 ablation_cli.py，支持一键切换与关闭：

# --method fgsm|mi|dim|tim|sim
# 关闭项：
# --disable_momentum 关闭动量项
# --disable_diversity 关闭 DIM 的输入多样性
# --disable_tim 关闭 TIM 的平移不变卷积
# --disable_scale 关闭 SIM 的缩放不变性
# 其余常用参数：--eps --iterations --mu --batchsize --input_dir --output_csv
# 可选保存对抗样本：--save_adv --adv_dir
# 示例：

# DIM 全量：python ablation_cli.py --method dim --model tf2torch_inception_v3 --batchsize 32 --eps 0.062745 --iterations 10 --mu 1.0 --prob 0.5 --input_dir ./data --output_csv ./results/ablation/dim_full.csv --GPU_ID 0
# DIM 去多样性：python ablation_cli.py --method dim --disable_diversity --prob 0.0 --output_csv ./results/ablation/dim_no_div.csv
# DIM 去动量：python ablation_cli.py --method dim --disable_momentum --output_csv ./results/ablation/dim_no_mom.csv
# TIM 去卷积：python ablation_cli.py --method tim --disable_tim --output_csv ./results/ablation/tim_no_conv.csv
# SIM 去缩放：python ablation_cli.py --method sim --disable_scale --output_csv ./results/ablation/sim_no_scale.csv
# 保存对抗样本：python ablation_cli.py --method dim --save_adv --adv_dir ./results/ablation/adv

# 用法示例-图片展示（单张图包含迭代次数与步长刻度）：

# python ablation_cli.py --method dim --model tf2torch_inception_v3 --batchsize 32 --eps 0.062745 --mu 1.0 --prob 0.5 --input_dir ./data --GPU_ID 0 --sweep_iters 30,25,20,15,10,5 --plot_png ./results/ablation/dim_iters_vs_alpha.png
# 参数含义：
# --sweep_iters 逗号分隔的迭代次数列表；脚本依次运行不同 iterations，统计平均迁移成功率
# --plot_png 输出一张图；横轴为迭代次数，顶部副轴为步长 alpha=eps/iterations
# 若只做单次运行并绘图：去掉 --sweep_iters，保留 --plot_png 即可输出按目标模型的迁移成功率曲线。
# python ablation_cli.py --method dim --model tf2torch_inception_v3 --batchsize 32 --eps 0.062745 --mu 1.0 --prob 0.5 --input_dir ./data --GPU_ID 0 --sweep_iters 30,25,20,15,10,5 --plot_iters_png ./results/ablation/dim_iters_curve.png --plot_alpha_png ./results/ablation/dim_alpha_curve.png
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy import stats
from models import ModelRepository, Normalize
import torch.nn as nn
import matplotlib.pyplot as plt

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

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = stats.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)

TIM_KERNEL = torch.from_numpy(np.stack([gkern(), gkern(), gkern()])).unsqueeze(1).float()

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

def attack_fgsm(x, y, model, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = get_model_output(model, x_adv)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x_adv.detach() + eps * torch.sign(grad)
    delta = torch.clamp(x_adv - x, min=-eps, max=eps)
    return torch.clamp(x + delta, 0, 1)

def attack_mi(x, y, model, eps, iterations, mu, alpha=None):
    alpha = alpha or (eps / max(iterations, 1))
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv, device=x.device)
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        output = get_model_output(model, x_adv)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            output = output.view(output.size(0), -1)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = mu * momentum + grad
        x_adv = x_adv.detach() + alpha * torch.sign(momentum if mu > 0 else grad)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()

def attack_dim(x, y, model, eps, iterations, mu, prob, use_momentum=True):
    alpha = eps / max(iterations, 1)
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv, device=x.device)
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
        if use_momentum:
            momentum = mu * momentum + grad
            step = torch.sign(momentum)
        else:
            step = torch.sign(grad)
        x_adv = x_adv.detach() + alpha * step
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()

def attack_tim(x, y, model, eps, iterations, mu, kernel, use_momentum=True, use_tim=True):
    alpha = eps / max(iterations, 1)
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv, device=x.device)
    k = kernel.to(x.device)
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        output = get_model_output(model, x_adv)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            output = output.view(output.size(0), -1)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        if use_tim:
            grad = F.conv2d(grad, k, padding=7, groups=3)
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        if use_momentum:
            momentum = mu * momentum + grad
            step = torch.sign(momentum)
        else:
            step = torch.sign(grad)
        x_adv = x_adv.detach() + alpha * step
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
        true_label = int(row["label"]) + 1
        img_path = os.path.join(self.img_root, img_filename)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(true_label, dtype=torch.long), img_filename

def parse_args():
    parser = argparse.ArgumentParser(description='Ablation CLI for transfer attacks (FGSM, MI, DIM, TIM, SIM)')
    parser.add_argument('--method', choices=['fgsm', 'mi', 'dim', 'tim', 'sim'], default='dim', help='attack method')
    parser.add_argument('--model', default='tf2torch_inception_v3', type=str, help='source model')
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--eps', default=16/255, type=float, help='L_inf epsilon')
    parser.add_argument('--iterations', default=10, type=int, help='attack iterations')
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--prob', default=0.5, type=float, help='DIM input diversity probability')
    parser.add_argument('--disable_momentum', action='store_true', help='disable momentum term (mu ignored)')
    parser.add_argument('--disable_diversity', action='store_true', help='disable input diversity in DIM')
    parser.add_argument('--disable_tim', action='store_true', help='disable translation-invariant convolution in TIM')
    parser.add_argument('--disable_scale', action='store_true', help='disable scale-invariance in SIM (use scale=1.0)')
    parser.add_argument('--save_adv', action='store_true', help='save adversarial images')
    parser.add_argument('--adv_dir', default='./results/ablation/adv', type=str, help='dir to save adversarial images')
    parser.add_argument('--input_dir', default='./data', type=str, help='data root containing images/ and labels.csv')
    parser.add_argument('--output_csv', default='./results/ablation/ablation_results.csv', type=str, help='output CSV path')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA device id')
    parser.add_argument('--plot_png', default='', type=str, help='保存单次或联合可视化的图片路径（PNG），为空则不生成')
    parser.add_argument('--sweep_iters', default='', type=str, help='以逗号分隔的迭代次数列表，例如 30,25,20,15,10,5；若为空则不做批量实验')
    parser.add_argument('--plot_iters_png', default='', type=str, help='在 sweep 模式下保存 迭代次数-成功率 的曲线图路径')
    parser.add_argument('--plot_alpha_png', default='', type=str, help='在 sweep 模式下保存 步长(alpha)-成功率 的曲线图路径')
    return parser.parse_args()

def run_one_experiment(args, device, source_model, target_models, transform, label_df, img_root):
    dataset = AttackDataset(label_df, img_root, transform)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    results = []
    for x_batch, y_batch, filename_batch in tqdm(loader, desc=f"Running {args.method.upper()} (iters={args.iterations})"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        source_orig_preds = get_model_prediction(source_model, x_batch)
        if args.method == 'fgsm':
            x_adv_batch = attack_fgsm(x_batch, y_batch, source_model, args.eps)
        elif args.method == 'mi':
            mu = 0.0 if args.disable_momentum else args.mu
            x_adv_batch = attack_mi(x_batch, y_batch, source_model, args.eps, args.iterations, mu)
        elif args.method == 'dim':
            mu_enabled = not args.disable_momentum
            prob = 0.0 if args.disable_diversity else args.prob
            x_adv_batch = attack_dim(x_batch, y_batch, source_model, args.eps, args.iterations, args.mu, prob, use_momentum=mu_enabled)
        elif args.method == 'tim':
            mu_enabled = not args.disable_momentum
            use_tim = not args.disable_tim
            x_adv_batch = attack_tim(x_batch, y_batch, source_model, args.eps, args.iterations, args.mu, TIM_KERNEL, use_momentum=mu_enabled, use_tim=use_tim)
        elif args.method == 'sim':
            alpha = args.eps / max(args.iterations, 1)
            x_adv = x_batch.clone().detach()
            momentum = torch.zeros_like(x_adv, device=device)
            scales = [1.0] if args.disable_scale else [1.0, 0.5, 0.25, 0.125, 0.0625]
            for _ in range(args.iterations):
                x_adv.requires_grad_(True)
                grad = torch.zeros_like(x_adv, device=device)
                for s in scales:
                    x_scaled = x_adv * s
                    out = get_model_output(source_model, x_scaled)
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                    elif out.dim() > 2:
                        out = out.view(out.size(0), -1)
                    loss = F.cross_entropy(out, y_batch)
                    grad_s = torch.autograd.grad(loss, [x_adv])[0]
                    grad += grad_s
                grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
                if not args.disable_momentum:
                    momentum = args.mu * momentum + grad
                    step = torch.sign(momentum)
                else:
                    step = torch.sign(grad)
                x_adv = x_adv.detach() + alpha * step
                delta = torch.clamp(x_adv - x_batch, min=-args.eps, max=args.eps)
                x_adv = torch.clamp(x_batch + delta, 0, 1)
            x_adv_batch = x_adv.detach()
        else:
            raise ValueError(f"Unknown method: {args.method}")
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)
        target_batch_preds = {}
        for model_name, model_info in target_models.items():
            model = model_info['model']
            target_batch_preds[model_name] = get_model_prediction(model, x_adv_batch)
        for i in range(x_batch.size(0)):
            true_label = int(y_batch[i].item())
            entry = {
                "source_attack_success": int(source_adv_preds[i]) != true_label,
                "target_results": {name: int(preds[i]) != true_label for name, preds in target_batch_preds.items()}
            }
            results.append(entry)
    if not results:
        return 0.0, 0
    model_names = list(target_models.keys())
    counts = {n: 0 for n in model_names}
    for r in results:
        for n in model_names:
            if r['target_results'][n]:
                counts[n] += 1
    avg_rate = np.mean([counts[n] / len(results) * 100 for n in model_names])
    return avg_rate, len(results)

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

    if args.sweep_iters:
        iters_list = [int(x.strip()) for x in args.sweep_iters.split(',') if x.strip()]
        iters_list = [i for i in iters_list if i > 0]
        if not iters_list:
            print('sweep_iters 为空或非法，跳过批量实验。')
        else:
            rates = []
            alphas = []
            for it in iters_list:
                print(f"\n>>> Sweep run: iterations={it}")
                args_iter = argparse.Namespace(**vars(args))
                args_iter.iterations = it
                avg_rate, nres = run_one_experiment(args_iter, device, source_model, target_models, transform, label_df, img_root)
                rates.append(avg_rate)
                alphas.append(args.eps / it)
                print(f"Avg transfer success: {avg_rate:.2f}% (N={nres})")
            # 分别保存两张图
            def ensure_dir(path):
                d = os.path.dirname(path)
                if d:
                    os.makedirs(d, exist_ok=True)
            if args.plot_iters_png:
                ensure_dir(args.plot_iters_png)
                plt.figure(figsize=(10,6))
                plt.plot(iters_list, rates, marker='o')
                plt.xlabel('Iterations')
                plt.ylabel('Avg Transfer Success (%)')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.title(f'Ablation: Iterations vs Success ({args.method.upper()})')
                plt.tight_layout()
                plt.savefig(args.plot_iters_png)
                print(f"Iterations plot saved to {args.plot_iters_png}")
            if args.plot_alpha_png:
                ensure_dir(args.plot_alpha_png)
                plt.figure(figsize=(10,6))
                plt.plot(alphas, rates, marker='o')
                plt.xlabel('Step size (alpha = eps / iterations)')
                plt.ylabel('Avg Transfer Success (%)')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.title(f'Ablation: Step Size vs Success ({args.method.upper()})')
                plt.tight_layout()
                plt.savefig(args.plot_alpha_png)
                print(f"Step size plot saved to {args.plot_alpha_png}")
            # 保留原联合图（可选）
            if args.plot_png:
                fig, ax1 = plt.subplots(figsize=(10,6))
                color1 = 'tab:blue'
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Avg Transfer Success (%)', color=color1)
                ax1.plot(iters_list, rates, marker='o', color=color1, label='Iterations')
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.grid(True, linestyle='--', alpha=0.4)
                ax2 = ax1.twiny()
                ax2.set_xlabel('Step size (alpha = eps / iterations)')
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_xticks(iters_list)
                ax2.set_xticklabels([f"{a:.4f}" for a in alphas])
                plt.title(f"Ablation: Iterations vs Step Size ({args.method.upper()})")
                out_dir = os.path.dirname(args.plot_png)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.plot_png)
                print(f"Visualization saved to {args.plot_png}")
            return

    dataset = AttackDataset(label_df, img_root, transform)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    print(f"Total {len(label_df)} valid images, using batch_size={args.batchsize}")

    if args.save_adv:
        os.makedirs(args.adv_dir, exist_ok=True)

    results = []
    print("\nStarting Ablation Attack...")
    for x_batch, y_batch, filename_batch in tqdm(loader, desc=f"Running {args.method.upper()}"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        source_orig_preds = get_model_prediction(source_model, x_batch)

        if args.method == 'fgsm':
            x_adv_batch = attack_fgsm(x_batch, y_batch, source_model, args.eps)
        elif args.method == 'mi':
            mu = 0.0 if args.disable_momentum else args.mu
            x_adv_batch = attack_mi(x_batch, y_batch, source_model, args.eps, args.iterations, mu)
        elif args.method == 'dim':
            mu_enabled = not args.disable_momentum
            prob = 0.0 if args.disable_diversity else args.prob
            x_adv_batch = attack_dim(x_batch, y_batch, source_model, args.eps, args.iterations, args.mu, prob, use_momentum=mu_enabled)
        elif args.method == 'tim':
            mu_enabled = not args.disable_momentum
            use_tim = not args.disable_tim
            x_adv_batch = attack_tim(x_batch, y_batch, source_model, args.eps, args.iterations, args.mu, TIM_KERNEL, use_momentum=mu_enabled, use_tim=use_tim)
        elif args.method == 'sim':
            alpha = args.eps / max(args.iterations, 1)
            x_adv = x_batch.clone().detach()
            momentum = torch.zeros_like(x_adv, device=device)
            scales = [1.0] if args.disable_scale else [1.0, 0.5, 0.25, 0.125, 0.0625]
            for _ in range(args.iterations):
                x_adv.requires_grad_(True)
                grad = torch.zeros_like(x_adv, device=device)
                for s in scales:
                    x_scaled = x_adv * s
                    out = get_model_output(source_model, x_scaled)
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                    elif out.dim() > 2:
                        out = out.view(out.size(0), -1)
                    loss = F.cross_entropy(out, y_batch)
                    grad_s = torch.autograd.grad(loss, [x_adv])[0]
                    grad += grad_s
                grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
                if not args.disable_momentum:
                    momentum = args.mu * momentum + grad
                    step = torch.sign(momentum)
                else:
                    step = torch.sign(grad)
                x_adv = x_adv.detach() + alpha * step
                delta = torch.clamp(x_adv - x_batch, min=-args.eps, max=args.eps)
                x_adv = torch.clamp(x_batch + delta, 0, 1)
            x_adv_batch = x_adv.detach()
        else:
            raise ValueError(f"Unknown method: {args.method}")

        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        if args.save_adv:
            from torchvision.utils import save_image
            for i, fname in enumerate(filename_batch):
                out_path = os.path.join(args.adv_dir, fname)
                save_image(x_adv_batch[i].clamp(0, 1), out_path)

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
    print(f"Summary of {args.method.upper()} Attack Results (Source: {args.model})")
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

        if args.plot_png:
            names = list(model_success_counts.keys())
            rates = [model_success_counts[n] / len(results) * 100 for n in names]
            plt.figure(figsize=(10, 6))
            plt.plot(names, rates, marker='o')
            plt.ylim(0, 100)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title(f'Transfer Success Rate by Target Model ({args.method.upper()})')
            plt.ylabel('Success Rate (%)')
            plt.xticks(rotation=30, ha='right')
            out_dir = os.path.dirname(args.plot_png)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.plot_png)
            print(f"Visualization saved to {args.plot_png}")
    else:
        print("No results to show.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main_cli()