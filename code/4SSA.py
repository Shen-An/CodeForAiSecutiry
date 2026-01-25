import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import *
from torch.autograd import Variable as V  # 引入Variable兼容旧逻辑
from dct import dct_2d, idct_2d  # 导入DCT变换函数
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

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = stats.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def clip_by_tensor(t, t_min, t_max):
    """复用attack.py中的裁剪函数"""
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def get_model_output(model, x):
    """
    统一处理模型输出，确保返回二维张量 (batch_size, num_classes)
    """
    output = model(x)
    
    # 处理不同类型的输出
    if isinstance(output, tuple):
        # 如果是元组，取第一个元素（通常是主输出）
        output = output[0]
    elif isinstance(output, list):
        # 如果是列表，取第一个元素
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

def get_model_prediction(model, x):
    """
    从模型输出中获取预测结果
    处理不同形状的输出
    """
    output = get_model_output(model, x)
    
    # 现在输出应该是二维张量 (batch, num_classes)
    return torch.argmax(output, dim=1).item()

def attack(x, y, model, eps=16/255, iterations=10, mu=1.0, N=20, rho=0.5, sigma=16.0):
    """
    修改为attack.py中的Spectrum_Simulation_Attack方法
    参数说明：
    - x: 输入图像
    - y: 真实标签
    - model: 模型
    - eps: 最大扰动
    - iterations: 迭代次数
    - mu: 动量系数
    - N: 谱变换次数
    - rho: 调节因子
    - sigma: 随机噪声的标准差
    """
    device = x.device
    image_width = x.shape[-1]  # 图像宽度
    alpha = eps / iterations  # 步长
    x_adv = x.clone().detach()  # 初始化为原始图像
    
    # 初始化动量
    grad = torch.zeros_like(x).to(device)
    
    # 计算边界
    images_min = clip_by_tensor(x - eps, 0.0, 1.0)
    images_max = clip_by_tensor(x + eps, 0.0, 1.0)
    
    # 获取高斯核（用于TI-FGSM，可选）
    # T_kernel = gkern(7, 3)
    
    for i in range(iterations):
        noise = 0
        
        # N次谱变换梯度累加
        for n in range(N):
            # 添加高斯噪声
            gauss = torch.randn(x_adv.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.to(device)
            
            # DCT变换
            x_dct = dct_2d(x_adv + gauss).to(device)
            
            # 随机掩码
            mask = (torch.rand_like(x_adv) * 2 * rho + 1 - rho).to(device)
            
            # 反DCT变换
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)
            
            # 模型前向传播
            output = get_model_output(model, x_idct)
            
            # 计算损失
            loss = F.cross_entropy(output, y)
            
            # 反向传播
            loss.backward()
            
            # 累加梯度
            noise += x_idct.grad.data
            
            # 分离计算图
            x_idct = x_idct.detach()
        
        # 梯度平均
        noise = noise / N
        
        # 可选：TI-FGSM（平移不变性）
        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        
        # MI-FGSM（动量更新）
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = mu * grad + noise
        grad = noise
        
        # 更新对抗样本
        x_adv = x_adv + alpha * torch.sign(noise)
        x_adv = clip_by_tensor(x_adv, images_min, images_max)
    
    return x_adv.detach()

def test_single_image(img_path, model, true_label, device, transform, attack_params=None):
    """
    测试单张图像的攻击效果
    """
    if attack_params is None:
        attack_params = {
            "eps": 16 / 255,
            "iterations": 10,
            "mu": 1.0,
            "N": 20,
            "rho": 0.5,
            "sigma": 16.0
        }
    
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        orig_pred = get_model_prediction(model, x)
        y_tensor = torch.tensor([true_label]).to(device)
        x_adv = attack(x, y_tensor, model, **attack_params)
        adv_pred = get_model_prediction(model, x_adv)
        
        return {
            "success": adv_pred != true_label,
            "original_pred": orig_pred,
            "adv_pred": adv_pred,
            "true_label": true_label,
            "adv_image": x_adv
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    主函数：测试攻击效果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # 选择目标模型（测试迁移性）
    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)
    
    print(f"\nSelected {len(target_models)} target models for testing")
    
    # 图像预处理
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # 攻击参数
    attack_params = {
        "eps": 16 / 255,
        "iterations": 10,
        "mu": 1.0,
        "N": 20,
        "rho": 0.5,
        "sigma": 16.0
    }
    
    # 读取标签文件
    import pandas as pd
    import os
    label_df = pd.read_csv('./data/labels.csv')
    
    # 导入tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        # 如果tqdm未安装，创建一个简单的替代
        class SimpleTqdm:
            def __init__(self, iterable, total=None):
                self.iterable = iterable
                self.total = total if total is not None else len(iterable)
                self.i = 0
                
            def __iter__(self):
                for item in self.iterable:
                    self.i += 1
                    print(f"Processing {self.i}/{self.total}", end='\r')
                    yield item
                print()
                
            def __len__(self):
                return self.total
        tqdm = SimpleTqdm
    
    results = []
    
    for idx, row in tqdm(label_df.iterrows(), total=len(label_df)):
        img_filename = row["filename"]
        true_label = int(row["label"])
        img_path = os.path.join('./data/images', img_filename)
        
        # 在集成模型上生成对抗样本
        result = test_single_image(
            img_path, ensemble_model, true_label, 
            device, transform, attack_params
        )
        
        if result is None:
            continue
        
        # 测试在所有目标模型上的迁移性
        target_results = {}
        for model_name, model_info in target_models.items():
            target_model = model_info['model']
            
            # 测试对抗样本在目标模型上的效果
            target_pred = get_model_prediction(target_model, result['adv_image'])
            
            target_results[model_name] = {
                "prediction": target_pred,
                "fooled": target_pred != true_label
            }
        
        # 保存结果
        results.append({
            "filename": img_filename,
            "true_label": true_label,
            "source_original_pred": result['original_pred'],
            "source_adv_pred": result['adv_pred'],
            "source_attack_success": result['success'],
            "target_results": target_results
        })
    
    # 汇总统计
    print("\n" + "="*80)
    print("Summary of Attack Results (SSA with Ensemble)")
    print("="*80)
    
    if results:
        # 源模型（集成模型）攻击成功率
        source_success = sum(1 for r in results if r['source_attack_success'])
        source_success_rate = source_success / len(results) * 100
        print(f"Ensemble model attack success rate: {source_success}/{len(results)} ({source_success_rate:.1f}%)")
        
        # 各目标模型的平均迁移成功率
        model_success_counts = {}
        for model_name in target_models.keys():
            model_success_counts[model_name] = 0
        
        for result in results:
            for model_name, target_result in result['target_results'].items():
                if target_result['fooled']:
                    model_success_counts[model_name] += 1
        
        print("\nTransfer attack success rates for each target model:")
        for model_name, count in model_success_counts.items():
            success_rate = count / len(results) * 100
            print(f"  {model_name}: {count}/{len(results)} ({success_rate:.1f}%)")
        
        # 平均迁移成功率
        avg_success_rate = np.mean(list(model_success_counts.values())) / len(results) * 100
        print(f"\nAverage transfer success rate: {avg_success_rate:.1f}%")
        
        # 保存详细结果到CSV
        import pandas as pd
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
        results_df.to_csv("./ssa_ensemble_attack_results.csv", index=False)
        print(f"\nDetailed results saved to ./ssa_ensemble_attack_results.csv")
    else:
        print("No results to show.")

if __name__ == "__main__":
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    main()