import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models import *
# ======================================================================
# 依赖模型定义 (请确保 torch_nets 库和权重文件存在)
# ======================================================================

# 假设 ModelRepository 依赖的 torch_nets 库已导入
from torch_nets import (
    tf2torch_inception_v3, tf2torch_inception_v4, tf2torch_resnet_v2_50,
    tf2torch_resnet_v2_101, tf2torch_resnet_v2_152, tf2torch_inc_res_v2,
    tf2torch_adv_inception_v3, tf2torch_ens3_adv_inc_v3, tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)


# 简化定义，确保代码可以运行
class Normalize(nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, input):
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)
        return (input - self.mean) / self.std





class AttackDataset(Dataset):
    """用于加载图像及其标签的自定义数据集"""

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

        try:
            img = Image.open(img_path).convert("RGB")
            x = self.transform(img)
            return x, true_label, img_filename
        except Exception as e:
            # 允许加载失败并返回 None，DataLoader 可能会跳过或报错
            return torch.zeros(3, 299, 299), -1, img_filename  # 推荐返回占位符以便于追踪


# ----------------------------------------------------------------------
# 辅助函数 (保持不变)
# ----------------------------------------------------------------------

def load_source_model(model_name, device, model_repo):
    """
    加载源模型。如果是 'tv_' 开头，加载 torchvision 官方模型；
    否则从 ModelRepository 加载。
    """
    if model_name.startswith('tv_'):
        import torchvision.models as tv_models
        name = model_name.replace('tv_', '')
        print(f"Loading torchvision model: {name}")
        
        input_size = 224
        if name == 'inception_v3':
            net = tv_models.inception_v3(pretrained=True)
            input_size = 299
        elif name == 'resnet50':
            net = tv_models.resnet50(pretrained=True)
        elif name == 'vgg16':
            net = tv_models.vgg16(pretrained=True)
        elif name == 'densenet121':
            net = tv_models.densenet121(pretrained=True)
        else:
            raise ValueError(f"Unknown torchvision model: {name}")
            
        net.eval().to(device)
        
        # ImageNet normalization
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = nn.Sequential(normalization, net).to(device)
        
        return {
            'model': model,
            'input_size': input_size,
            'type': 'torchvision',
            'normalization': 'imagenet'
        }
    else:
        return model_repo.get_source_model(model_name)

def get_model_output(model, x):
    """统一处理模型输出，确保返回张量"""
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")
    return output


def get_model_prediction(model, x):
    """从模型输出中获取预测结果，始终返回一维 numpy 数组（长度为 batch 大小）。"""
    output = get_model_output(model, x)
    # 统一为 [N, C]
    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)
    preds = torch.argmax(output, dim=1).cpu().numpy()
    # 保证即使 batch=1 也不是标量，而是一维数组
    return np.atleast_1d(preds)


# ----------------------------------------------------------------------
# FGSM 攻击函数 (核心)
# ----------------------------------------------------------------------

def attack(x, y, model, eps, iterations, mu, alpha=None):
    """
    FGSM 攻击实现。
    注意：为了兼容，接收 iterations 和 mu，但 FGSM 逻辑中它们将被忽略。
    """
    # 1. 准备输入：确保输入需要梯度
    x.requires_grad = True

    # 2. 前向传播和计算损失
    output = get_model_output(model, x)

    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)

    loss = F.cross_entropy(output, y)

    # 3. 反向传播：计算梯度
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()

    # 4. 获取梯度符号
    grad_sign = x.grad.data.sign()

    # 5. 生成对抗样本 (x_adv = x + epsilon * sign(gradient))
    x_perturbed = x.data + eps * grad_sign

    # 6. 像素值裁剪
    x_adv = torch.clamp(x_perturbed, 0, 1)

    return x_adv.detach()


# ----------------------------------------------------------------------
# 参数解析 (采用用户指定的格式)
# ----------------------------------------------------------------------

def parse_args():
    """解析命令行参数"""

    # 假定 transferattack.attack_zoo 至少包含 'fgsm' 和 'mifgsm'
    class DummyAttackZoo:
        def __init__(self):
            self.keys = lambda: ['fgsm', 'mifgsm', 'cw', 'pgd']

    # 临时定义一个对象来模拟用户环境中的 attack_zoo
    transferattack = type('transferattack', (object,), {'attack_zoo': DummyAttackZoo()})()

    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')

    # 功能和通用参数
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='fgsm', type=str, help='the attack algorithm',
                        choices=transferattack.attack_zoo.keys())
    parser.add_argument('--model', default='tf2torch_inception_v3', type=str,
                        help='the source surrogate model (e.g., resnet50)')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')

    # 攻击参数
    parser.add_argument('--epoch', default=1, type=int,
                        help='the iterations for updating the adversarial patch (FGSM should be 1)')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the L-inf norm constraint (epsilon)')
    parser.add_argument('--alpha', default=1.6 / 255, type=float,
                        help='the stepsize to update the perturbation (FGSM ignores this)')
    parser.add_argument('--momentum', default=0., type=float,
                        help='the decay factor for momentum based attack (FGSM ignores this)')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')

    # I/O 参数
    parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    parser.add_argument('--input_dir', default='./data', type=str,
                        help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results/fgsm', type=str,
                        help='the path to store the adversarial patches and results CSV')
    parser.add_argument('--GPU_ID', default='0', type=str)

    return parser.parse_args()


# ----------------------------------------------------------------------
# main 主函数 (使用新的参数名称)
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    # 强制 FGSM 的特定参数
    if args.attack.lower() == 'fgsm':
        args.epoch = 1
        args.momentum = 0.0
        print("\n--- ATTENTION: Running FGSM, forcing epoch=1 and momentum=0.0 ---\n")

    # 设备设置
    # 使用 GPU_ID 参数，但 Python 中我们只需要判断是否有 CUDA
    device = torch.device(f"cuda:{args.GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (as requested by --GPU_ID {args.GPU_ID} or falling back to CPU)")

    # 模型加载
    try:
        model_repo = ModelRepository(device)
    except Exception as e:
        print(f"Error initializing ModelRepository: {e}")
        return

    source_model_name = args.model
    try:
        source_model_info = load_source_model(source_model_name, device, model_repo)
        source_model = source_model_info['model']
        source_input_size = source_model_info.get('input_size', 299)
    except ValueError as e:
        print(f"Error: {e}")
        return

    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)
    print(f"Source model: {source_model_name} (Input Size: {source_input_size})")
    print(f"Selected {len(target_models)} target models for testing")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((source_input_size, source_input_size)),
        transforms.ToTensor(),
    ])

    attack_params = {
        "eps": args.eps,
        "iterations": args.epoch,  # 使用 epoch 作为迭代次数
        "mu": args.momentum,  # 使用 momentum 作为动量
        "alpha": args.alpha  # 虽然 FGSM 不用，但传入保持接口一致
    }

    # I/O 检查
    data_root = args.input_dir
    label_csv_path = os.path.join(data_root, 'labels.csv')
    img_root = os.path.join(data_root, 'images')
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, f"{args.attack}_{source_model_name}_results.csv")  # 动态生成文件名

    if not os.path.exists(label_csv_path) or not os.path.exists(img_root):
        print(f"Error: Data directory not correctly found. Expected images folder at: {img_root}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    label_df = pd.read_csv(label_csv_path)

    # 创建 DataLoader
    dataset = AttackDataset(label_df, img_root, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batchsize,  # 使用 args.batchsize
        shuffle=False,
        num_workers=0
    )
    print(f"Total {len(label_df)} images, using batch_size={args.batchsize}")

    results = []
    print("\nStarting Attack...")

    total_samples = len(label_df)

    for x_batch, y_batch, filename_batch in tqdm(data_loader, desc=f"Running {args.attack}"):

        # 过滤掉加载失败的样本
        valid_indices = (y_batch != -1)
        x_batch = x_batch[valid_indices].to(device)
        y_batch = y_batch[valid_indices].to(device)
        filename_batch = [filename_batch[i] for i, valid in enumerate(valid_indices) if valid]

        if x_batch.size(0) == 0:
            continue

        # 1. 原始预测
        source_original_pred_batch = get_model_prediction(source_model, x_batch)

        # 2. 生成对抗样本
        x_adv_batch = attack(x_batch, y_batch, source_model, **attack_params)

        # 3. 对抗样本在源模型上的预测
        source_adv_pred_batch = get_model_prediction(source_model, x_adv_batch)

        # 4. 测试在所有目标模型上的迁移性
        target_predictions_batch = {}
        for model_name, model_info in target_models.items():
            target_model = model_info['model']
            target_input_size = model_info.get('input_size', 299)
            
            if target_input_size != source_input_size:
                # Resize x_adv_batch to match target model input size
                x_adv_input = F.interpolate(x_adv_batch, size=(target_input_size, target_input_size), mode='bilinear', align_corners=False)
            else:
                x_adv_input = x_adv_batch
                
            target_pred = get_model_prediction(target_model, x_adv_input)
            target_predictions_batch[model_name] = target_pred

        # 5. 逐个样本保存结果
        true_labels = y_batch.cpu().numpy()

        for i in range(len(x_batch)):
            true_label = true_labels[i]+1#  source_original_pred =source_original_pred -1;#对齐标签
            source_original_pred = source_original_pred_batch[i]


            source_adv_pred = source_adv_pred_batch[i]
            source_attack_success = source_adv_pred != true_label

            result_entry = {
                "filename": filename_batch[i],
                "true_label": true_label,
                "source_original_pred": source_original_pred,
                "source_adv_pred": source_adv_pred,
                "source_attack_success": source_attack_success,
                "target_results": {}
            }

            for model_name, preds in target_predictions_batch.items():
                target_pred = preds[i]
                result_entry["target_results"][model_name] = {
                    "prediction": target_pred,
                    "fooled": target_pred != true_label
                }

            results.append(result_entry)

    # 汇总统计
    print("\n" + "=" * 80)
    print(f"Summary of {args.attack.upper()} Attack Results (Source: {source_model_name})")
    print("=" * 80)

    if results:
        # ... (统计逻辑与之前保持一致) ...
        source_success = sum(1 for r in results if r['source_attack_success'])
        source_success_rate = source_success / len(results) * 100
        print(f"Source Model Attack Success Rate: {source_success_rate:.1f}% (N={total_samples})")

        model_success_counts = {name: 0 for name in target_models.keys()}
        for result in results:
            for model_name, target_result in result['target_results'].items():
                if target_result['fooled']:
                    model_success_counts[model_name] += 1

        print("\nTransfer attack success rates for each target model:")
        valid_rates = []
        for model_name, count in model_success_counts.items():
            success_rate = count / len(results) * 100
            print(f"  {model_name}: {success_rate:.1f}%")
            valid_rates.append(success_rate)

        avg_success_rate = np.mean(valid_rates)
        print(f"\nAverage Transfer Success Rate: {avg_success_rate:.1f}%")

        # 保存详细结果到CSV
        detailed_results = []
        for result in results:
            row = {
                "filename": result["filename"],
                "true_label": result["true_label"],
                "source_original_pred": result["source_original_pred"],
                "source_adv_pred": result["source_adv_pred"],
                "source_attack_success": result["source_attack_success"]
            }
            for model_name, target_result in result["target_results"].items():
                row[f"{model_name}_pred"] = target_result["prediction"]
                row[f"{model_name}_fooled"] = target_result["fooled"]
            detailed_results.append(row)

        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
    else:
        print("No results to show.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()