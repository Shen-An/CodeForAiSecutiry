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
import math


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

def get_model_output(model, x):
    output = model(x);
    if isinstance(output, (tuple, list)):
        # 若output是元组或列表，取其第一个元素重新赋值给output
        # 目的是提取列表/元组中存储的核心张量数据（部分模型会返回多元素输出，第一个元素通常是预测结果）
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")
    return output;

    # 函数功能：获取模型预测结果，并始终返回一个一维numpy数组（数组长度等于批量大小batch）

def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

def get_model_prediction(model, x):
    # 1. 调用自定义函数获取模型原始输出（可能是不同维度的张量，如1维/2维/3维及以上）
    output = get_model_output(model, x)

    # 2. 若模型输出是一维张量（形状：[C]，无批量维度）
    if output.dim() == 1:
        # 在第0维（最前面）插入批量维度，转换为二维张量（形状：[1, C]），统一后续处理格式
        output = output.unsqueeze(0)
    # 3. 若模型输出维度大于2（如3维[B, C, 1]、4维[B, C, H, W]等）
    elif output.dim() > 2:
        # 保持批量维度（第0维）不变，将剩余所有维度展平为一维向量
        # output.size(0)：获取批量大小B
        # -1：自动计算展平后的维度尺寸，保证张量总元素数不变
        # 转换后形状为[B, N]（N为展平后的特征数/类别数）
        output = output.view(output.size(0), -1)

    # 4. 计算预测标签：在维度1上取最大值对应的索引（即模型预测的类别）
    #    .cpu()：将张量从GPU移至CPU（避免numpy不支持GPU张量）
    #    .numpy()：将PyTorch张量转换为numpy数组
    preds = torch.argmax(output, dim=1).cpu().numpy()

    # 5. 确保返回值是至少一维的numpy数组，最终强制转为一维数组（长度等于批量大小batch）
    return np.atleast_1d(preds)


def parse_args():
    parser = argparse.ArgumentParser(description="DIM attack")
    parser.add_argument("--model", default='inception_v3', type=str, help="source model")
    parser.add_argument('--output_adv_dir', default='./results/dim/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/dim/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--diversity_prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument("--num_blocks_h",default=2,type=int,help="number of blocks h")
    parser.add_argument("--num_blocks_w",default=2,type=int,help="number of blocks w")
    parser.add_argument("--max_angle",default=2,type=int,help="maximum angle")
    parser.add_argument("--num_copies",default=20,type=int,help="number of copies")
    parser.add_argument("--use_diversity",default=True,type=bool,help="use diversity");
    parser.add_argument("--use_admix",default=False,type=bool,help="use admix");
    parser.add_argument("--portion",default=0.2,help="portion admix");
    parser.add_argument("--admix_size",default=3,help="admix size");
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


def mifgsm_attack_BSR(x, y, model, eps=16 / 255, iterations=10, mu=1.0,
                      num_blocks_h=2, num_blocks_w=2, max_angle=0.2,
                      num_copies=20, use_diversity=True, use_admix=False,
                      portion=0.2, admix_size=3, diversity_prob=0.5):
    """BSR版本的MI-FGSM攻击"""
    alpha = eps / iterations
    x_adv = x.clone().requires_grad_(True)
    momentum = torch.zeros_like(x).to(x.device)

    for i in range(iterations):
        # 应用输入变换
        if use_diversity:
            x_transformed = input_diversity(x_adv, prob=diversity_prob)
        else:
            x_transformed = x_adv

        if use_admix:
            x_transformed = admix(x_transformed, portion=portion, size=admix_size)
            y_expanded = y.repeat(admix_size * num_copies)
        else:
            y_expanded = y.repeat(num_copies)

        # 应用BSR变换
        x_bsr = BSR_transform(x_transformed, num_blocks_h, num_blocks_w, max_angle, num_copies)

        # 前向传播
        output = model(x_bsr)

        # 处理不同输出类型
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, list):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected output type: {type(output)}")

        # 计算损失
        loss = F.cross_entropy(output, y_expanded)

        # 反向传播
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        # 梯度归一化和动量更新
        grad = x_adv.grad.data
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8
        grad_normalized = grad / grad_norm

        momentum = mu * momentum + grad_normalized

        # 更新对抗样本
        with torch.no_grad():
            x_adv = x_adv + alpha * torch.sign(momentum)
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1)

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
        x_adv_batch = mifgsm_attack_BSR(x_batch, y_batch, source_model,
                                iterations=args.iterations,
                                 mu=args.mu,num_blocks_h=args.num_blocks_h,
                                        num_blocks_w=args.num_blocks_w,
                                        max_angle=args.max_angle,
                                        num_copies=args.num_copies,
                                        use_diversity=args.use_diversity,
                                        use_admix=args.use_admix,
                                        portion=args.portion,
                                        admix_size=args.admix_size,
                                        diversity_prob=args.diversity_prob)

        # 记录攻击后预测
        source_adv_preds = get_model_prediction(source_model, x_adv_batch)

        # 将生成的对抗样本移至 CPU 并存储，释放显存
        adv_images_storage.append(x_adv_batch.cpu())

        for i in range(x_adv_batch.size(0)):
            true_label = int(y_batch[i].item()) + 1
            s_adv_idx = int(source_adv_preds[i]) + 1
            s_orig_idx = int(source_orig_preds[i]) + 1
            print(f"{s_orig_idx}\t{s_adv_idx}\t{true_label}")
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