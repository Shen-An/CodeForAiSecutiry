import argparse
import gc
import os
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



def mi_fgsm_attack(x, y, model, eps=16 / 255.0, iterations=20, mu=1.0, **kwargs):
    """
    MI-FGSM 攻击 (Momentum Iterative FGSM)
    目标：仅通过带动量的分类损失梯度进行攻击
    """
    model.eval()
    device = x.device

    # 1. 初始化
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)
    alpha_step = eps / iterations

    for i in range(iterations):
        x_adv.requires_grad = True

        # 2. 前向传播计算交叉熵损失
        output = model(x_adv)
        if isinstance(output, (tuple, list)):
            output = output[0]

        loss = F.cross_entropy(output, y)

        # 3. 反向传播获取梯度
        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.data

        # 4. 梯度的 L1 归一化 (MI-FGSM 标准操作)
        # 将当前步梯度除以其平均绝对值，使梯度在同一量级
        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)

        # 5. 动量累积：g_{t+1} = mu * g_t + grad_t
        momentum = mu * momentum + grad

        # 6. 符号更新：x_{t+1} = x_t + alpha * sign(momentum)
        x_adv = x_adv.detach() + alpha_step * torch.sign(momentum)

        # 7. 投影约束 (Clip)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()



# 输入多样性增强 (DIM) - Batched Version (Independent per image)
def input_diversity(x, prob=0.5):
    """
    对 Batch 中的每一张图像独立应用随机 Resize 和 Padding (DIM)。
    保证 Batchsize 仅仅是加速，每一张图的变换参数是独立的。
    """
    if prob <= 0:
        return x
    
    device = x.device
    B, C, H, W = x.shape
    
    # 我们需要在 [299, 329] 之间为每一张图随机选择一个尺寸
    # 生成 B 个随机目标尺寸
    rnd_scales = np.random.randint(299, 330, size=B) 
    
    # 决定每一张图是否应用变换 (Probability check)
    apply_flags = np.random.random(B) > (1 - prob) # True means apply
    
    outputs = []
    
    for i in range(B):
        img = x[i:i+1] # keep 4dims: [1, C, H, W]
        
        if not apply_flags[i]:
             outputs.append(img)
             continue
             
        scale = rnd_scales[i]
        
        # 1. Resize
        resized = F.interpolate(img, size=(scale, scale), mode='nearest')
        
        # 2. Pad
        h_rem = 330 - scale
        w_rem = 330 - scale
        
        pad_top = np.random.randint(0, h_rem + 1)
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem + 1)
        pad_right = w_rem - pad_left
        
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # 3. Resize back to 299
        final = F.interpolate(padded, size=(299, 299), mode='nearest')
        outputs.append(final)

    return torch.cat(outputs, dim=0)


# -----------------------------
# PEA-DIM: Posterior Evolutionary Anticipation + DIM
# -----------------------------

def _clip_by_eps(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


def _l1_normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    return grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)


def apply_dim_transform_with_params(x: torch.Tensor,
                                   scale: int,
                                   pad_top: int,
                                   pad_bottom: int,
                                   pad_left: int,
                                   pad_right: int) -> torch.Tensor:
    """对 batch 应用同一组 DIM 参数（用于 EA 精英参数回放/回溯梯度）。"""
    resized = F.interpolate(x, size=(scale, scale), mode='nearest')
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    final = F.interpolate(padded, size=(299, 299), mode='nearest')
    return final


def sample_dim_params(n: int, low: int = 299, high: int = 330):
    """采样 n 组 DIM 参数。high 为开区间风格，与 np.random.randint 一致。"""
    params = []
    for _ in range(n):
        scale = int(np.random.randint(low, high))  # [299, 329]
        rem = 330 - scale
        pad_top = int(np.random.randint(0, rem + 1))
        pad_left = int(np.random.randint(0, rem + 1))
        pad_bottom = int(rem - pad_top)
        pad_right = int(rem - pad_left)
        params.append((scale, pad_top, pad_bottom, pad_left, pad_right))
    return params


def evaluate_fitness_loss(model, x_tmp: torch.Tensor, y: torch.Tensor, params) -> float:
    """在预演点上评估单个变换参数的适应度（loss 越大越好）。不回传梯度。"""
    x_t = apply_dim_transform_with_params(x_tmp, *params)
    output = get_model_output(model, x_t)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)
    loss = F.cross_entropy(output, y)
    return float(loss.detach().item())


def mutate_dim_params(params, sigma_scale: int = 6, sigma_shift: int = 10):
    """对 DIM 参数做轻量变异：scale 和 padding 随机扰动后再投影回合法范围。"""
    scale, pad_top, pad_bottom, pad_left, pad_right = params

    # mutate scale
    scale = int(scale + np.random.randint(-sigma_scale, sigma_scale + 1))
    scale = int(np.clip(scale, 299, 329))

    rem = 330 - scale

    # reconstruct a consistent padding after mutating shifts
    pad_top = int(pad_top + np.random.randint(-sigma_shift, sigma_shift + 1))
    pad_left = int(pad_left + np.random.randint(-sigma_shift, sigma_shift + 1))
    pad_top = int(np.clip(pad_top, 0, rem))
    pad_left = int(np.clip(pad_left, 0, rem))
    pad_bottom = int(rem - pad_top)
    pad_right = int(rem - pad_left)

    return (scale, pad_top, pad_bottom, pad_left, pad_right)


def select_elite_params(model,
                        x_tmp: torch.Tensor,
                        y: torch.Tensor,
                        n_population: int,
                        k_elite: int,
                        prev_elites=None,
                        mutation_rate: float = 0.7):
    """第二步：在预演点做 EA/Top-K 选择。返回当步精英参数列表（长度 K）。"""
    # 初始化/继承
    population = []
    if prev_elites:
        # 继承并变异
        for p in prev_elites:
            if np.random.random() < mutation_rate:
                population.append(mutate_dim_params(p))
            else:
                population.append(p)

    # 补足随机个体
    if len(population) < n_population:
        population.extend(sample_dim_params(n_population - len(population)))

    # 适应度评估（loss 越大越好）
    losses = []
    with torch.no_grad():
        for p in population:
            losses.append(evaluate_fitness_loss(model, x_tmp, y, p))

    # Top-K
    idxs = np.argsort(losses)[::-1][:max(1, k_elite)]
    elites = [population[i] for i in idxs]
    return elites


def pea_dim_attack(x,
                   y,
                   model,
                   eps=16 / 255.0,
                   iterations=10,
                   mu=1.0,
                   n_population: int = 20,
                   k_elite: int = 5,
                   mutation_rate: float = 0.7):
    """
    PEA-DIM 攻击：
    1) 位移预演(look-ahead)：x_tmp = Clip(x, eps){ x_t + mu * alpha * sign(g_{t-1}) }
    2) 精英演化筛选：在 x_tmp 上评估 N 组 DIM 参数的 loss，选 Top-K
    3) 回溯梯度计算：将精英参数应用到 x_t，对 x_t 求平均梯度
    4) 动量更新 + 投影
    """
    device = x.device
    x_t = x.clone().detach()
    momentum = torch.zeros_like(x_t, device=device)
    alpha = eps / max(1, iterations)

    prev_elites = None

    for _ in range(iterations):
        # Step 1: look-ahead / probe
        if torch.count_nonzero(momentum).item() == 0:
            x_tmp = x_t.detach()
        else:
            x_tmp = _clip_by_eps(x_t.detach() + (mu * alpha) * torch.sign(momentum.detach()), x, eps)

        # Step 2: elite selection on x_tmp (no grads)
        elites = select_elite_params(
            model=model,
            x_tmp=x_tmp,
            y=y,
            n_population=n_population,
            k_elite=k_elite,
            prev_elites=prev_elites,
            mutation_rate=mutation_rate,
        )
        prev_elites = elites

        # Step 3: back-projection gradients on x_t using elites
        x_t.requires_grad_(True)
        grad_sum = torch.zeros_like(x_t)

        for p in elites:
            x_view = apply_dim_transform_with_params(x_t, *p)
            output = get_model_output(model, x_view)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                output = output.view(output.size(0), -1)
            loss = F.cross_entropy(output, y)
            grad_i = torch.autograd.grad(loss, [x_t], retain_graph=True, create_graph=False)[0]
            grad_sum = grad_sum + grad_i

        grad_bar = grad_sum / max(1, len(elites))
        grad_bar = _l1_normalize_grad(grad_bar)

        # Step 4: momentum + update
        momentum = mu * momentum + grad_bar.detach()
        x_t = _clip_by_eps(x_t.detach() + alpha * torch.sign(momentum), x, eps)

    return x_t.detach()


# MI-FGSM +dim
def dim_attack(x, y, model, eps=16 / 255, iterations=10, mu=1.0, prob=0.5, **kwargs):
    """兼容旧接口：prob 参数保留但在 PEA-DIM 中不再使用（由 N/K/变异控制多样性）。"""
    return pea_dim_attack(
        x=x,
        y=y,
        model=model,
        eps=eps,
        iterations=iterations,
        mu=mu,
        n_population=int(kwargs.get('n_population', 20)),
        k_elite=int(kwargs.get('k_elite', 5)),
        mutation_rate=float(kwargs.get('mutation_rate', 0.7)),
    )


def get_model_output(model, x):
    output = model(x);
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")
    return output;


def get_model_prediction(model, x):
    output = get_model_output(model, x)

    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)

    preds = torch.argmax(output, dim=1).cpu().numpy()

    return np.atleast_1d(preds)


def parse_args():
    parser = argparse.ArgumentParser(description="DIM attack")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/dim/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/dim/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--prob', default=0.5, type=float, help='(legacy) input diversity probability, unused in PEA-DIM')
    # PEA-DIM / EA params
    parser.add_argument('--n_population', default=20, type=int, help='EA population size N (per iteration)')
    parser.add_argument('--k_elite', default=5, type=int, help='EA elite size K (Top-K by loss)')
    parser.add_argument('--mutation_rate', default=0.7, type=float, help='EA mutation rate when inheriting elites')
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
        raise Exception("Invalid model name" + model_name);

    net = net.to(device);
    net.eval();
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model = torch.nn.Sequential(Normalize(mean=mean, std=std), net);
    return model;


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model_repo = ModelRepository(device)

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

        x_adv_batch = dim_attack(x_batch, y_batch, source_model,
                                 eps=args.eps, iterations=args.iterations,
                                 mu=args.mu, prob=args.prob,
                                 n_population=args.n_population,
                                 k_elite=args.k_elite,
                                 mutation_rate=args.mutation_rate)

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

    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(adv_mem_dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
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
