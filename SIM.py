import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import *
from tqdm import tqdm

def get_model_output(model, x):
    """
    统一处理模型输出，确保返回张量
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
    
    return output

def get_model_prediction(model, x):
    """
    从模型输出中获取预测结果
    处理不同形状的输出
    """
    output = get_model_output(model, x)
    
    # 处理不同维度的输出
    if output.dim() == 1:
        # 一维张量，直接取argmax
        return torch.argmax(output).item()
    elif output.dim() == 2:
        # 二维张量 (batch, num_classes)
        return torch.argmax(output, dim=1).item()
    else:
        # 更高维度，展平为二维
        if output.dim() > 2:
            # 展平除了batch维度外的所有维度
            output = output.view(output.size(0), -1)
            return torch.argmax(output, dim=1).item()
        else:
            raise ValueError(f"Unexpected output dimension: {output.dim()}, shape: {output.shape}")

def attack(x, y, model, eps=16/255, iterations=10, mu=1.0):
    """
    纯净的SI-MI-FGSM攻击：只包含缩放不变性和动量迭代
    
    Args:
        x: 原始图像 [1, 3, H, W]
        y: 真实标签
        model: 目标模型
        eps: 最大扰动
        iterations: 迭代次数
        mu: 动量系数
    Returns:
        x_adv: 对抗样本
    """
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True)
    
    # 计算迭代步长
    alpha = eps / iterations 
    
    # 初始化动量
    momentum = torch.zeros_like(x).to(device)
    
    for i in range(iterations):
        x_adv.requires_grad = True
        
        # 缩放不变性：在不同缩放尺度上计算梯度
        grad = torch.zeros_like(x).to(device)
        scales = [1.0, 1/2, 1/4, 1/8, 1/16]
        
        for scale in scales:
            # 缩放输入
            x_scaled = x_adv * scale
            
            # 获取模型输出
            output = get_model_output(model, x_scaled)
            
            # 处理输出形状以确保可以计算交叉熵损失
            if output.dim() == 1:
                # 如果是一维输出，添加batch维度
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                # 如果是更高维度的输出，展平为二维
                output = output.view(output.size(0), -1)
            
            # 计算损失
            loss = F.cross_entropy(output, y)
            
            # 计算梯度并累加
            grad_scale = torch.autograd.grad(loss, [x_adv])[0]
            grad += grad_scale
        
        # 归一化梯度（根据SI-MI-FGSM原文）
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        
        # 动量更新
        momentum = mu * momentum + grad
        
        # 更新对抗样本
        x_adv = x_adv.detach() + alpha * torch.sign(momentum)
        
        # 裁剪到允许范围内
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0, 1)
    
    return x_adv.detach()

def test_single_image(img_path, model, true_label, device, transform, attack_params=None):
    """
    测试单张图像的攻击效果
    
    Args:
        img_path: 图像路径
        model: 目标模型
        true_label: 真实标签
        device: 设备
        transform: 图像变换
        attack_params: 攻击参数
    Returns:
        result: 攻击结果字典
    """
    if attack_params is None:
        attack_params = {
            "eps": 16 / 255,
            "iterations": 10,
            "mu": 1.0
        }
    
    try:
        # 加载并预处理图像
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        # 原始预测
        orig_pred = get_model_prediction(model, x)
        
        # 生成对抗样本
        y_tensor = torch.tensor([true_label]).to(device)
        x_adv = attack(x, y_tensor, model, **attack_params)
        
        # 对抗样本预测
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
    
    # 选择源模型（攻击基于此模型）
    source_model_info = model_repo.get_source_model('tf2torch_inc_res_v2')
    source_model = source_model_info['model']
    
    # 选择目标模型（测试迁移性）
    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)
    
    print(f"Available models: {all_models}")
    print(f"Selected {len(target_models)} target models for testing")
    
    # 图像预处理
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # 攻击参数 - 只使用缩放不变性和动量迭代
    attack_params = {
        "eps": 16 / 255,
        "iterations": 10,
        "mu": 1.0
    }
    
    # 读取标签文件
    import pandas as pd
    import os
    label_df = pd.read_csv('./data/labels.csv')
    
    results = []
    
    for idx, row in tqdm(label_df.iterrows(), total=len(label_df)):
        img_filename = row["filename"]
        true_label = int(row["label"])
        img_path = os.path.join('./data/images', img_filename)
        
        # 在源模型上生成对抗样本
        result = test_single_image(
            img_path, source_model, true_label, 
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
    print("Summary of Attack Results (Pure SI-MI-FGSM)")
    print("="*80)
    
    if results:
        # 源模型攻击成功率
        source_success = sum(1 for r in results if r['source_attack_success'])
        source_success_rate = source_success / len(results) * 100
        print(f"Source model attack success rate: {source_success}/{len(results)} ({source_success_rate:.1f}%)")
        
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
        results_df.to_csv("./pure_si_mi_fgsm_results.csv", index=False)
        print(f"\nDetailed results saved to ./pure_si_mi_fgsm_results.csv")
    else:
        print("No results to show.")

if __name__ == "__main__":
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    main()