import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import pandas as pd
import os
from tqdm.notebook import tqdm
import math
import scipy.stats as st  # 添加缺失的导入

# 从PMRT导入模型和工具函数
from PMRT import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

# 预计算高斯核（虽然在这个版本中可能用不到，但保留以防万一）
kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

def get_block_lengths_bsr(length, num_blocks):
    """BSR版本的分块长度计算"""
    length = int(length)
    rand = np.random.uniform(size=num_blocks)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

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
                    
                    grid = F.affine_grid(angle_matrix, block[i:i+1].size(), align_corners=False)
                    rotated_block[i:i+1] = F.grid_sample(block[i:i+1], grid, mode='bilinear', 
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

def admix(x, portion=0.2, size=3):
    """混合输入变换"""
    indices = torch.randperm(x.size(0))
    admixed = []
    for _ in range(size):
        admixed_x = x + portion * x[indices]
        admixed.append(admixed_x)
    return torch.cat(admixed, dim=0)

def mifgsm_attack_BSR(x, y, model, eps=16/255, iterations=10, mu=1.0,
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
        grad_norm = torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8
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
    # 路径配置
    paths = {
        "img_root": r"./data/images",
        "label_csv": r"./data/labels.csv", 
        "result_save": r"./bsr_test_results"
    }
    
    # 初始化模型仓库
    model_repo = ModelRepository(device)
    print(f"Model repository initialized, total {len(model_repo.get_all_model_names())} models")
    print(f"Available models: {model_repo.get_all_model_names()}")
    
    # 设置源模型
    source_model_info = model_repo.get_source_model('tf2torch_inception_v3')
    source_model = source_model_info['model']
    source_input_size = source_model_info['input_size']
    
    # 获取所有目标模型
    all_models = model_repo.get_all_model_names()
    target_model_names = [model for model in all_models]
    target_models = model_repo.get_target_models(target_model_names)
    
    print(f"Selected {len(target_models)} target models: {list(target_models.keys())}")
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # 加载标签数据
    label_df = pd.read_csv(paths["label_csv"])
    print(f"Successfully read CSV, total {len(label_df)} images")
    
    # BSR攻击参数
    batch_results = []
    BSR_attack_params = {
        "eps": 16 / 255,
        "iterations": 10, 
        "mu": 1.0,
        "num_blocks_h": 2,
        "num_blocks_w": 2,
        "max_angle": 0.2,
        "num_copies": 20,
        "use_diversity": True,
        "use_admix": False,
        "diversity_prob": 0.5
    }
    print(f"Testing BSR attack with parameters: {BSR_attack_params}")
    
    for idx, row in tqdm(label_df.iterrows(), desc='Processing images', total=len(label_df), mininterval=1.0):
        img_filename = row["filename"]
        true_label = int(row["label"])
        img_path = os.path.join(paths["img_root"], img_filename)
        
        try:
            # 加载图像
            x = load_image_and_transform(img_path, transform, device)
            
            # 源模型原始预测
            source_original_pred = get_model_prediction(source_model, x)
            
            # BSR攻击
            x_adv_BSR = mifgsm_attack_BSR(
                x=x,
                y=torch.tensor([true_label]).to(device),
                model=source_model,
                **BSR_attack_params
            )
            
            # 在所有目标模型上测试
            target_predictions = {}
            for model_name, model_info in target_models.items():
                target_model = model_info['model']
                
                # 转换回PIL并重新变换以确保一致性
                x_adv_pil = transforms.ToPILImage()(x_adv_BSR.squeeze(0).cpu())
                x_adv_target = transform(x_adv_pil).unsqueeze(0).to(device)
                
                # 获取预测
                target_adv_pred = get_model_prediction(target_model, x_adv_target)
                target_predictions[model_name] = target_adv_pred
            
            # 存储结果
            result_entry = {
                "filename": img_filename,
                "true_label": true_label,
                "source_original_pred": source_original_pred,
                "status": "Success"
            }
            
            # 添加目标模型预测
            for model_name, pred in target_predictions.items():
                result_entry[f"target_{model_name}_pred"] = pred
            
            batch_results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing image {img_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果和统计信息
    if batch_results:
        save_path = f"{paths['result_save']}_bsr.csv"
        result_df = pd.DataFrame(batch_results)
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        
        # 计算成功率
        success_count = len(result_df)
        
        if success_count > 0:
            print(f"Successfully processed: {success_count} images")
            print(f"\n=== BSR attack results ===")
            
            # 目标模型成功率
            for model_name in target_models.keys():
                target_success = len(result_df[result_df[f"target_{model_name}_pred"] != result_df["true_label"]])
                target_success_rate = target_success / success_count * 100
                print(f"Target model {model_name} attack success rate: {target_success_rate:.2f}%")
    else:
        print("No successfully processed images")

if __name__ == "__main__":
    main()