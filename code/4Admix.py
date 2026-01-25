import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from models import *
from tqdm.notebook import tqdm
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from typing import List, Dict

class EnsembleModel(nn.Module):
    """
    集成模型，将多个模型的输出进行平均
    """
    def __init__(self, models: List[nn.Module], device: torch.device):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models).to(device)
        self.device = device
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            
            # 处理不同类型的输出
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, list):
                output = output[0]
            
            # 确保所有输出形状一致
            if output.dim() == 1:
                # 一维张量，添加batch维度
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                # 展平除了batch维度外的所有维度
                output = output.view(output.size(0), -1)
            
            outputs.append(output)
        
        # 对输出进行平均
        avg_output = torch.stack(outputs).mean(dim=0)
        return avg_output
    
    def eval(self):
        for model in self.models:
            model.eval()
        return self

class AdmixSIM:
    """
    Admix + SIM（Scale-Invariant Method）攻击实现
    结合Admix混合和多尺度梯度计算
    """
    
    def __init__(self, model, eps=16/255, iterations=10, mu=1.0, 
                 portion=0.2, admix_size=3, scale_factor=0.5, 
                 num_scales=5, device='cuda'):
        """
        初始化攻击参数
        
        Args:
            model: 目标模型
            eps: 最大扰动限制
            iterations: 迭代次数
            mu: 动量系数
            portion: 混合比例
            admix_size: Admix采样图像数
            scale_factor: 尺度缩放因子
            num_scales: 尺度数量
            device: 计算设备
        """
        self.model = model
        self.eps = eps
        self.iterations = iterations
        self.mu = mu
        self.portion = portion
        self.admix_size = admix_size
        self.scale_factor = scale_factor
        self.num_scales = num_scales
        self.device = device
        
        # 计算迭代步长
        self.alpha = eps / iterations
        
        # 计算尺度因子列表
        self.scale_factors = [scale_factor**i for i in range(num_scales)]
    
    def get_model_output(self, x):
        """
        统一处理模型输出，确保返回二维张量 (batch_size, num_classes)
        """
        output = self.model(x)
        
        # 处理不同类型的输出
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, list):
            output = output[0]
        
        # 确保输出是张量
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected output type: {type(output)}")
        
        # 统一输出形状为二维 (batch_size, num_classes)
        if output.dim() == 1:
            # 一维张量，添加batch维度
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            # 展平除了batch维度外的所有维度
            output = output.view(output.size(0), -1)
        
        return output
    
    def create_admix_copies(self, x, dataset_images, available_indices):
        """
        创建Admix混合副本
        将原始图像复制admix_size份，每份与不同的数据集图像混合
        
        Args:
            x: 原始图像 [1, 3, H, W]
            dataset_images: 数据集图像张量 [N, 3, H, W]
            available_indices: 可用的图像索引列表
            
        Returns:
            mixed_batch: 混合后的图像批 [admix_size, 3, H, W]
        """
        # 确保有足够的可用图像
        if len(available_indices) < self.admix_size:
            # 如果没有足够的图像，重复使用可用的
            selected_indices = []
            for i in range(self.admix_size):
                selected_indices.append(available_indices[i % len(available_indices)])
        else:
            # 随机选择admix_size个不同的图像索引
            selected_indices = random.sample(available_indices, self.admix_size)
        
        # 获取选中的图像
        selected_images = dataset_images[selected_indices]  # [admix_size, 3, H, W]
        
        # 将原始图像复制admix_size份
        x_copies = x.repeat(self.admix_size, 1, 1, 1)  # [admix_size, 3, H, W]
        
        # 执行Admix混合: x_copy + portion * selected_image
        mixed_batch = x_copies + self.portion * selected_images
        
        return mixed_batch
    
    def scale_images(self, images):
        """
        将图像缩放到不同尺度
        基于SIM（Scale-Invariant Method）的多尺度梯度计算
        
        Args:
            images: 输入图像 [batch_size, 3, H, W]
            
        Returns:
            scaled_batch: 多尺度图像批 [batch_size * num_scales, 3, H, W]
        """
        scaled_images = []
        
        # 对每个尺度因子进行缩放
        for scale in self.scale_factors:
            # 缩放图像
            scaled = images * scale
            scaled_images.append(scaled)
        
        # 拼接所有尺度图像
        scaled_batch = torch.cat(scaled_images, dim=0)  # [batch_size * num_scales, 3, H, W]
        
        return scaled_batch
    
    def compute_admix_sim_gradient(self, x, y, dataset_images, available_indices):
        """
        计算Admix + SIM的梯度
        步骤：
        1. 创建admix_size个Admix混合副本
        2. 对每个副本进行num_scales个尺度的缩放
        3. 计算所有尺度下所有混合图像的梯度
        4. 加权平均梯度
        
        Args:
            x: 当前对抗样本 [1, 3, H, W]
            y: 真实标签 [1]
            dataset_images: 数据集图像张量 [N, 3, H, W]
            available_indices: 可用的图像索引列表
            
        Returns:
            avg_grad: 平均梯度 [1, 3, H, W]
        """
        # 1. 创建Admix混合副本
        admix_batch = self.create_admix_copies(x, dataset_images, available_indices)
        
        # 2. 进行多尺度缩放
        scaled_batch = self.scale_images(admix_batch)  # [admix_size * num_scales, 3, H, W]
        
        # 3. 扩展标签以匹配批大小
        total_samples = scaled_batch.size(0)
        y_expanded = y.repeat(total_samples)
        
        # 4. 计算损失
        output = self.get_model_output(scaled_batch)
        loss = F.cross_entropy(output, y_expanded)
        
        # 5. 计算梯度
        grad = torch.autograd.grad(loss, [x])[0]
        
        return grad
    
    def attack(self, x, y, dataset_images, available_indices):
        """
        执行Admix + SIM攻击
        
        Args:
            x: 原始图像 [1, 3, H, W]
            y: 真实标签 [1]
            dataset_images: 数据集图像张量 [N, 3, H, W]
            available_indices: 可用的图像索引列表
            
        Returns:
            x_adv: 对抗样本 [1, 3, H, W]
        """
        # 初始化对抗样本和动量
        x_adv = x.clone().detach()
        momentum = torch.zeros_like(x).to(self.device)
        
        for i in range(self.iterations):
            x_adv.requires_grad = True
            
            # 计算Admix + SIM梯度
            grad = self.compute_admix_sim_gradient(x_adv, y, dataset_images, available_indices)
            
            # 梯度归一化
            grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-12)
            
            # 动量更新
            momentum = self.mu * momentum + grad
            
            # 更新对抗样本
            x_adv = x_adv.detach() + self.alpha * torch.sign(momentum)
            
            # 裁剪到允许范围内
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        return x_adv


def load_dataset_for_admix(data_dir, label_csv, transform, device, max_images=200):
    """
    加载数据集图像用于Admix操作
    
    Args:
        data_dir: 图像目录
        label_csv: 标签CSV文件
        transform: 图像变换
        device: 计算设备
        max_images: 最大加载图像数
        
    Returns:
        dataset_tensor: 数据集图像张量 [N, 3, H, W]
        filename_to_idx: 文件名到索引的映射字典
    """
    # 读取标签文件
    label_df = pd.read_csv(label_csv)
    
    # 限制加载的图像数量
    if max_images > 0 and max_images < len(label_df):
        label_df = label_df.sample(n=max_images, random_state=42)
    
    images = []
    filenames = []
    
    print(f"Loading {len(label_df)} dataset images for Admix...")
    
    for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Loading dataset"):
        img_filename = row["filename"]
        img_path = os.path.join(data_dir, img_filename)
        
        try:
            # 加载并预处理图像
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0)
            images.append(x)
            filenames.append(img_filename)
        except Exception as e:
            print(f"Warning: Could not load {img_filename}: {e}")
    
    if not images:
        raise ValueError("No images loaded from dataset!")
    
    # 拼接所有图像并移动到设备
    dataset_tensor = torch.cat(images, dim=0).to(device)
    
    # 创建文件名到索引的映射
    filename_to_idx = {filename: idx for idx, filename in enumerate(filenames)}
    
    print(f"Successfully loaded {len(images)} images for Admix dataset.")
    
    return dataset_tensor, filename_to_idx


def get_model_prediction(model, x):
    """
    获取模型预测结果
    """
    with torch.no_grad():
        output = model(x)
        
        # 处理不同类型的输出
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, list):
            output = output[0]
        
        # 统一输出形状为二维 (batch_size, num_classes)
        if output.dim() == 1:
            # 一维张量，添加batch维度
            output = output.unsqueeze(0)
        elif output.dim() > 2:
            # 展平除了batch维度外的所有维度
            output = output.view(output.size(0), -1)
        
        # 现在输出应该是二维张量 (batch, num_classes)
        return torch.argmax(output, dim=1).item()


def test_admix_sim_attack():
    """
    测试Admix + SIM攻击效果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    
    # 路径配置
    data_dir = "./data/images"
    label_csv = "./data/labels.csv"
    
    # 初始化模型仓库
    model_repo = ModelRepository(device)
    
    # 获取集成模型所需的各个模型
    model_names = ['tf2torch_inception_v3', 'tf2torch_inception_v4', 'tf2torch_resnet_v2_101', 'tf2torch_inc_res_v2']
    
    # 检查模型是否可用
    all_available_models = model_repo.get_all_model_names()
    print(f"Available models: {all_available_models}")
    
    ensemble_models = []
    for model_name in model_names:
        if model_name in all_available_models:
            model_info = model_repo.get_source_model(model_name)
            ensemble_models.append(model_info['model'])
            print(f"Added {model_name} to ensemble")
        else:
            print(f"Warning: {model_name} not found in available models")
    
    if not ensemble_models:
        raise ValueError("No models found for ensemble!")
    
    # 创建集成模型
    ensemble_model = EnsembleModel(ensemble_models, device).eval()
    print(f"Created ensemble model with {len(ensemble_models)} models")
    
    # 选择目标模型（用于迁移性测试）
    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)
    
    print(f"\nSelected {len(target_models)} target models for testing")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # 加载数据集图像用于Admix
    print("\nLoading dataset images for Admix operation...")
    dataset_images, filename_to_idx = load_dataset_for_admix(
        data_dir, label_csv, transform, device, max_images=200
    )
    
    # 攻击参数
    attack_params = {
        "eps": 16 / 255,
        "iterations": 10,
        "mu": 1.0,
        "portion": 0.2,
        "admix_size": 3,  # Admix采样图像数
        "scale_factor": 0.5,  # SIM尺度因子
        "num_scales": 5,  # SIM尺度数量
        "device": device
    }
    
    print(f"\nAttack parameters: {attack_params}")
    print(f"\nMethod details:")
    print(f"  1. Take 1 original image")
    print(f"  2. Create {attack_params['admix_size']} Admix copies (each mixed with different dataset image)")
    print(f"  3. Scale each copy with {attack_params['num_scales']} scales: {[attack_params['scale_factor']**i for i in range(attack_params['num_scales'])]}")
    print(f"  4. Total images per iteration: {attack_params['admix_size']} × {attack_params['num_scales']} = {attack_params['admix_size'] * attack_params['num_scales']}")
    print(f"  5. Compute gradients from all images and average")
    print(f"  6. Update with momentum (μ={attack_params['mu']}) for {attack_params['iterations']} iterations")
    
    # 初始化攻击
    attack = AdmixSIM(ensemble_model, **attack_params)
    
    # 读取测试图像
    label_df = pd.read_csv(label_csv)
    
    # 为了快速测试，可以使用前几张图像
    # label_df = label_df.head(10)
    
    results = []
    total_images = len(label_df)
    
    print(f"\nStarting Admix + SIM attack on {total_images} images...")
    
    for idx, row in tqdm(label_df.iterrows(), total=total_images, desc="Attacking images"):
        img_filename = row["filename"]
        true_label = int(row["label"])
        img_path = os.path.join(data_dir, img_filename)
        
        try:
            # 加载并预处理当前图像
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            y = torch.tensor([true_label]).to(device)
            
            # 获取当前图像在数据集中的索引（用于排除）
            current_idx = filename_to_idx.get(img_filename)
            
            # 准备可用的图像索引列表（排除当前图像）
            available_indices = list(range(len(dataset_images)))
            if current_idx is not None and current_idx in available_indices:
                available_indices.remove(current_idx)
            
            # 确保有足够的图像用于Admix
            if len(available_indices) < attack_params["admix_size"]:
                print(f"Warning: Not enough images for Admix for {img_filename} "
                      f"(available: {len(available_indices)}, needed: {attack_params['admix_size']})")
            
            # 获取原始预测
            orig_pred = get_model_prediction(ensemble_model, x)
            
            # 执行攻击
            x_adv = attack.attack(x, y, dataset_images, available_indices)
            
            # 获取对抗样本预测
            adv_pred = get_model_prediction(ensemble_model, x_adv)
            
            # 测试在所有目标模型上的迁移性
            target_results = {}
            for model_name, model_info in target_models.items():
                target_model = model_info['model']
                target_pred = get_model_prediction(target_model, x_adv)
                
                target_results[model_name] = {
                    "prediction": target_pred,
                    "fooled": target_pred != true_label
                }
            
            # 保存结果
            results.append({
                "filename": img_filename,
                "true_label": true_label,
                "source_original_pred": orig_pred,
                "source_adv_pred": adv_pred,
                "source_attack_success": adv_pred != true_label,
                "target_results": target_results
            })
            
        except Exception as e:
            print(f"\nError processing {img_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 分析结果
    analyze_admix_sim_results(results, target_models, attack_params, total_images)


def analyze_admix_sim_results(results, target_models, attack_params, total_images):
    """
    分析Admix + SIM攻击结果
    """
    if not results:
        print("No results to analyze.")
        return
    
    print("\n" + "="*80)
    print("Admix + SIM Attack Results Summary (with Ensemble Model)")
    print("="*80)
    
    # 源模型（集成模型）攻击成功率
    source_success = sum(1 for r in results if r['source_attack_success'])
    source_success_rate = source_success / len(results) * 100
    print(f"Ensemble model attack success rate: {source_success}/{len(results)} ({source_success_rate:.2f}%)")
    
    # 各目标模型的迁移成功率
    model_success_counts = {model_name: 0 for model_name in target_models.keys()}
    
    for result in results:
        for model_name, target_result in result['target_results'].items():
            if target_result['fooled']:
                model_success_counts[model_name] += 1
    
    print("\nTransfer attack success rates for each target model:")
    for model_name, count in model_success_counts.items():
        success_rate = count / len(results) * 100
        print(f"  {model_name}: {count}/{len(results)} ({success_rate:.2f}%)")
    
    # 平均迁移成功率
    avg_success_rate = np.mean(list(model_success_counts.values())) / len(results) * 100
    print(f"\nAverage transfer success rate: {avg_success_rate:.2f}%")
    
    # 计算攻击强度指标
    if results:
        print(f"\nAttack strength indicators:")
        print(f"  Admix copies per iteration: {attack_params['admix_size']}")
        print(f"  Scales per copy: {attack_params['num_scales']}")
        print(f"  Total images per iteration: {attack_params['admix_size'] * attack_params['num_scales']}")
        print(f"  Total forward passes per image: {attack_params['iterations'] * attack_params['admix_size'] * attack_params['num_scales']}")
    
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
    
    # 生成结果文件名，包含参数信息
    eps_str = f"eps{int(attack_params['eps']*255)}"
    iter_str = f"iter{attack_params['iterations']}"
    mu_str = f"mu{attack_params['mu']}"
    portion_str = f"portion{int(attack_params['portion']*100)}"
    admix_str = f"admix{attack_params['admix_size']}"
    scale_str = f"scale{attack_params['scale_factor']}"
    numscale_str = f"numscale{attack_params['num_scales']}"
    
    filename = f"admix_sim_ensemble_{eps_str}_{iter_str}_{mu_str}_{portion_str}_{admix_str}_{scale_str}_{numscale_str}"
    
    results_df.to_csv(f"./{filename}_results.csv", index=False)
    print(f"\nDetailed results saved to ./{filename}_results.csv")
    
    # 保存攻击参数和统计信息
    with open(f"./{filename}_stats.txt", "w") as f:
        f.write("Admix + SIM Attack Statistics (with Ensemble Model)\n")
        f.write("="*50 + "\n\n")
        
        f.write("Attack Parameters:\n")
        for key, value in attack_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nAttack Method:\n")
        f.write(f"  1. Take 1 original image\n")
        f.write(f"  2. Create {attack_params['admix_size']} Admix copies (mix with different dataset images)\n")
        f.write(f"  3. Scale each copy with {attack_params['num_scales']} scales: {[attack_params['scale_factor']**i for i in range(attack_params['num_scales'])]}\n")
        f.write(f"  4. Total images per iteration: {attack_params['admix_size']} × {attack_params['num_scales']} = {attack_params['admix_size'] * attack_params['num_scales']}\n")
        f.write(f"  5. Compute gradients from all images and average\n")
        f.write(f"  6. Update with momentum (μ={attack_params['mu']})\n")
        f.write(f"  7. Repeat for {attack_params['iterations']} iterations\n")
        
        f.write("\nResults Summary:\n")
        f.write(f"  Total images processed: {len(results)}\n")
        f.write(f"  Ensemble model success rate: {source_success_rate:.2f}%\n")
        f.write(f"  Average transfer success rate: {avg_success_rate:.2f}%\n")
        
        f.write("\nIndividual Model Success Rates:\n")
        for model_name, count in model_success_counts.items():
            success_rate = count / len(results) * 100
            f.write(f"  {model_name}: {success_rate:.2f}%\n")
    
    print(f"Statistics saved to ./{filename}_stats.txt")


def main():
    """
    主函数
    """
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("="*80)
    print("Admix + SIM (Scale-Invariant Method) Attack with Ensemble Model")
    print("="*80)
    print("Method: Combines Admix data mixing with multi-scale gradient computation")
    print("        using an ensemble of 4 models (incV3, incV4, res101, incresv2)")
    print("="*80)
    
    # 运行测试
    test_admix_sim_attack()


if __name__ == "__main__":
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()