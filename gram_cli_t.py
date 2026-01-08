import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from models import ModelRepository, Normalize
from torch.utils.data import DataLoader, Dataset, TensorDataset


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


class FeatureExtractor:
    def __init__(self):
        self.feature_maps = None

    def hook_fn(self, module, input, output):
        self.feature_maps = output

    def register(self, model, layer_idx):
        # 递归获取所有子模块（比 children() 更深一层）
        all_layers = list(model.modules())
        # 排除掉模型本身(index 0)
        layers = [l for l in all_layers if not isinstance(l, (nn.Sequential, type(model)))]

        available_count = len(layers)

        # 自动调整索引：如果索引越界或设为 -1，取最后一层卷积/模块
        if layer_idx >= available_count or layer_idx < 0:
            print(f"Warning: layer_idx {layer_idx} out of range (max {available_count - 1}). Using last layer.")
            layer_idx = available_count - 1

        target_layer = layers[layer_idx]
        print(f"Targeting layer: {type(target_layer).__name__} at index {layer_idx}")

        return target_layer.register_forward_hook(self.hook_fn)


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


import torch
import torch.nn as nn
import torch.nn.functional as F


def get_weighted_centered_gram(f_map):
    """
    [WSG-FGSM 核心模块 A]: 加权中心化 Gram 矩阵
    模拟 FIA 的通道重要性意识，同时消除均值干扰。
    """
    n, c, h, w = f_map.shape
    # 1. 计算通道权重 (GAP): 激活越强的通道承载越多语义
    weights = torch.mean(f_map, dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)

    # 2. 中心化处理
    f = f_map.view(n, c, -1)
    f = f - f.mean(dim=2, keepdim=True)

    # 3. 施加权重并计算相关性
    f_weighted = f * weights.view(n, c, 1)
    gram = torch.bmm(f_weighted, f_weighted.transpose(1, 2))

    return gram / (c * h * w)


def get_saliency_mask(f_map, tau=1.5):
    """
    [WSG-FGSM 核心模块 B]: 空间显著性掩码
    定位特征图中响应最强的区域 (ResNet 的关键判别区域)
    """
    # 计算空间维度的激活图
    saliency_map = torch.mean(f_map, dim=1, keepdim=True)  # (N, 1, H, W)
    # 提取超过均值 tau 倍的区域
    threshold = torch.mean(saliency_map, dim=(2, 3), keepdim=True) * tau
    mask = (saliency_map > threshold).float()
    return mask

def get_standard_gram(f_map):
    """标准 Gram 矩阵计算"""
    n, c, h, w = f_map.shape
    f = f_map.view(n, c, -1)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)

def apply_tv_regularization(x_adv, x_orig, lambda_tv=0.01):
    """特征平滑约束"""
    diff = x_adv - x_orig
    tv_loss = torch.sum(torch.abs(diff[:, :, :, :-1] - diff[:, :, :, 1:])) + \
              torch.sum(torch.abs(diff[:, :, :-1, :] - diff[:, :, 1:, :]))
    return lambda_tv * tv_loss


def wsg_fgsm_attack(x, y, model, eps=16 / 255.0, iterations=20, mu=1.0,
                    use_weighted=True,  # 启用通道加权 (对标 FIA)
                    use_saliency=True,  # 启用空间显著性压制
                    use_multi_scale=True,  # 启用多尺度 Hook
                    use_tv=False):  # 启用 TV 平滑
    model.eval()
    device = x.device

    # --- 1. 动态 Hook 配置 ---
    all_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    # 黄金采样深度：0.65 (中层迁移性最好) 和 0.85 (语义层)
    layers_idx = [int(len(all_convs) * 0.65)]
    if use_multi_scale:
        layers_idx.append(int(len(all_convs) * 0.85))

    extractors = []
    hooks = []
    for idx in layers_idx:
        ext = FeatureExtractor()
        h = all_convs[idx].register_forward_hook(ext.hook_fn)
        extractors.append(ext)
        hooks.append(h)

    # --- 2. 获取原始参考特征 (Reference) ---
    with torch.no_grad():
        _ = model(x)
        gram_refs = []
        saliency_masks = []
        orig_features = []
        for ext in extractors:
            f = ext.feature_maps.detach()
            orig_features.append(f)
            # 计算 Gram 参考
            g = get_weighted_centered_gram(f) if use_weighted else get_standard_gram(f)
            gram_refs.append(g)
            # 计算显著性掩码
            if use_saliency:
                saliency_masks.append(get_saliency_mask(f))

    # --- 3. 迭代攻击循环 ---
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x_adv)
    alpha_step = eps / iterations

    for i in range(iterations):
        x_adv.requires_grad = True
        output = model(x_adv)
        if isinstance(output, (tuple, list)): output = output[0]

        # A. 分类损失
        loss_ce = F.cross_entropy(output, y)

        # B. 特征损失 (Gram + Saliency)
        loss_feat = 0
        for j, ext in enumerate(extractors):
            f_adv = ext.feature_maps
            # 加权相关性损失
            g_adv = get_weighted_centered_gram(f_adv) if use_weighted else get_standard_gram(f_adv)
            loss_feat += torch.abs(g_adv - gram_refs[j]).mean()

            # 空间显著性阻断损失 (让 ResNet 的高响应区域消失)
            if use_saliency:
                loss_feat += torch.mean(torch.abs(f_adv * saliency_masks[j])) * 0.01

        # --- 4. 核心改进：梯度对齐 (Gradient Alignment) ---
        model.zero_grad()
        grad_ce = torch.autograd.grad(loss_ce, x_adv, retain_graph=True)[0]
        grad_feat = torch.autograd.grad(loss_feat, x_adv, retain_graph=True)[0]

        # 归一化 CE 梯度作为基准尺度
        n_ce = torch.mean(torch.abs(grad_ce))
        n_feat = torch.mean(torch.abs(grad_feat))

        # 混合梯度：CE 引导破防，Feat 引导迁移
        # 权重 0.5 确保不会因为特征破坏太狠而导致源模型不破防
        combined_grad = grad_ce + 0.5 * (n_ce / (n_feat + 1e-8)) * grad_feat

        if use_tv:
            loss_tv = apply_tv_regularization(x_adv, x)
            combined_grad += torch.autograd.grad(loss_tv, x_adv)[0]

        # --- 5. 动量更新与投影 ---
        combined_grad = combined_grad / (torch.mean(torch.abs(combined_grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = mu * momentum + combined_grad
        x_adv = x_adv.detach() + alpha_step * torch.sign(momentum)

        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    for h in hooks: h.remove()
    return x_adv.detach()

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


# 输入多样性增强 (DIM)
def input_diversity(x, prob=0.5):
    if prob <= 0:
        return x;
    # 让攻击过程不是对每一张图像、每一次迭代都强制执行变换，而是随机跳过，增加对抗样本的多样性；
    if np.random.random() > prob:
        return x
    # 生成一个在 [299, 329] 区间内的随机整数（左闭右开，randint的上限330不包含，因此取值范围是299≤rnd≤329）
    # 该随机整数将作为图像缩放后的目标宽高尺寸（正方形尺寸，宽=高=rnd）
    rnd = np.random.randint(299, 330)

    # 使用PyTorch的函数式插值方法，对输入图像张量x进行尺寸缩放
    # x：输入的图像张量，形状通常为 [B, C, H, W]（批量数, 通道数, 高, 宽）或 [C, H, W]（单张图像）
    # size=(rnd, rnd)：指定缩放后的目标尺寸为 (rnd, rnd)，即生成正方形图像
    # mode='nearest'：插值模式为“最近邻插值”，该方法直接选取距离目标像素最近的原始像素值作为插值结果，计算速度快且无像素值平滑过渡
    resized = F.interpolate(x, size=(rnd, rnd), mode='nearest')
    # 计算图像高度方向需要补充的像素数（填充量）
    # 330是目标基准尺寸，rnd是之前随机生成的图像缩放尺寸，h_rem为高度方向需填充的总像素
    h_rem = 330 - rnd;
    # 计算图像宽度方向需要补充的像素数（填充量）
    # w_rem为宽度方向需填充的总像素，由于rnd是正方形尺寸，h_rem与w_rem数值相等
    w_rem = 330 - rnd;

    # 随机生成高度方向上方的填充像素数
    # 取值范围是 [0, h_rem]（左闭右闭，因为randint上限是h_rem+1，左闭右开特性使得最大值为h_rem）
    # 即上方填充量可以是0到h_rem之间的任意整数
    pad_top = np.random.randint(0, h_rem + 1)
    # 高度方向下方的填充像素数 = 高度总填充量 - 上方填充量
    # 保证高度方向上下填充量之和等于h_rem，最终填充后高度恢复到330
    pad_bottom = h_rem - pad_top

    # 随机生成宽度方向左侧的填充像素数
    # 取值范围是 [0, w_rem]（左闭右闭，randint上限w_rem+1确保最大值为w_rem）
    pad_left = np.random.randint(0, w_rem + 1)
    # 宽度方向右侧的填充像素数 = 宽度总填充量 - 左侧填充量
    # 保证宽度方向左右填充量之和等于w_rem，最终填充后宽度恢复到330
    pad_right = w_rem - pad_left
    # 对缩放后的图像张量进行常量填充（零填充）
    # F.pad：PyTorch函数式填充接口，按指定填充量对张量边缘进行填充
    # resized：输入张量（前文缩放后的图像，形状为 [B, C, rnd, rnd] 或 [C, rnd, rnd]）
    # (pad_left, pad_right, pad_top, pad_bottom)：填充顺序为「左、右、上、下」（PyTorch F.pad的固定顺序，遵循W/H维度优先）
    # mode='constant'：填充模式为常量填充，即所有填充像素使用同一个固定值
    # value=0：常量填充的数值为0，即对图像进行零填充（黑色填充，对应图像像素值为0）
    # 填充后张量形状变为 [B, C, 330, 330] 或 [C, 330, 330]，恢复到基准正方形尺寸
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # 将填充后的330×330图像张量重新插值缩放回目标尺寸299×299
    # F.interpolate：PyTorch函数式插值接口，用于张量尺寸调整
    # padded：输入的330×330填充后图像张量
    # size=(299, 299)：指定目标缩放尺寸为299×299（与原始图像尺寸一致）
    # mode='nearest'：插值模式为最近邻插值，计算快速且保留像素的离散特性，无平滑模糊效果
    # 返回值：形状为 [B, C, 299, 299] 或 [C, 299, 299] 的图像张量，与初始输入尺寸一致
    return F.interpolate(padded, size=(299, 299), mode='nearest')


# MI-FGSM
def dim_attack(x, y, model, eps=16 / 255.0, iterations=10, mu=1.0, prob=0.5):
    device = x.device
    x_adv = x.clone().detach()  # 停止梯度追踪detach
    alpha = eps / max(1, iterations);
    # 创建一个与对抗样本张量x_adv形状完全一致、数据类型一致的全零张量，命名为momentum（动量张量）
    momentum = torch.zeros_like(x_adv, device=device);

    for _ in range(iterations):
        x_adv.requires_grad_(True)  # 启用对抗样本张量x_adv的梯度追踪功能
        x_div = input_diversity(x_adv, prob) if prob > 0 else x_adv
        output = get_model_output(model, x_div)
        if output.dim() == 1:
            # 若为一维张量，在第0维（最前面）增加一个批量维度（size=1），将其转换为二维张量[C]->[1,c]
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            # 若张量output的维度数>2（二维张量）将剩余所有维度展平为一个一维向量,[B, H, W] → [B, H*W]
            # output.size(0)：获取第0维的尺寸（批量大小B）-1：自动计算该维度的尺寸，确保张量总元素数不变
            output = output.view(output.size(0), -1)
        loss = F.cross_entropy(output, y)
        # [x_adv]：需要求解梯度的目标张量列表（此处仅求解x_adv的梯度）
        # 返回值：一个梯度张量列表，[0] 表示取出列表中第一个元素（即x_adv对应的梯度张量，形状与x_adv一致）

        grad = torch.autograd.grad(loss, [x_adv])[0]  # grad [B,C,H,w],B 张样本、C通道、H×W 分辨率
        # 对梯度张量进行归一化（标准化）处理，消除梯度幅值差异的影响
        # 第一步：torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        #   - torch.abs(grad)：对梯度取绝对值，避免正负梯度相互抵消
        #   - dim=(1,2,3)：在通道维度（1）、高度维度（2）、宽度维度（3）上计算均值（保留批量维度0） chw
        #   - keepdim=True：保持梯度张量的维度数不变（避免广播机制出错），计算后形状为 [B, 1, 1, 1]（B为批量大小）
        # 第二步：+1e-8：添加极小值防止分母为0，避免除法报错或数值爆炸
        # 第三步：grad / (...)：将原始梯度除以通道-高-宽维度的平均绝对梯度，实现梯度幅值的归一化

        grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = mu * momentum + grad;

        x_adv = x_adv.detach() + alpha * torch.sign(momentum)

        # 裁剪函数,把扰动范围限制在 eps 内
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()


def main():
    return 0;


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
    parser.add_argument('--output_adv_dir', default='./results/gram/images', type=str, help='adv images dir')
    parser.add_argument('--output_csv', default='./results/gram/results.csv', type=str, help='output CSV path')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--eps', default=16 / 255.0, type=float)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--mu', default=1.0, type=float, help='momentum factor')
    parser.add_argument('--prob', default=0.5, type=float, help='input diversity probability')
    parser.add_argument('--layer_name', default='layer3', type=str, help='layer to attack (e.g. layer3 for ResNet)')
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
        x_adv_batch = wsg_fgsm_attack(x_batch, y_batch, source_model,
                                 eps=args.eps, iterations=args.iterations,
                                 mu=args.mu)

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
