from PHFRT import *
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()

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

def get_model_output(model, x):
    """
    统一处理模型输出，确保返回二维张量 (batch_size, num_classes)
    """
    output = model(x)
    
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

def get_model_prediction(model, x):
    """
    从模型输出中获取预测结果
    处理不同形状的输出
    """
    output = get_model_output(model, x)
    
    # 现在输出应该是二维张量 (batch, num_classes)
    return torch.argmax(output, dim=1).item()

# Path configuration
paths = {
    "img_root": r"./data/images",
    "label_csv": r"./data/labels.csv",
    "result_save": r"./PHFRT_ensemble/phfrt_ensemble"
}
# 注意：这里去掉了 .csv 扩展名，因为后面会添加参数信息

# Initialize model repository
model_repo = ModelRepository(device)
print(f"Model repository initialized, total {len(model_repo.get_all_model_names())} models")
print(f"Available models: {model_repo.get_all_model_names()}")

# 获取集成模型所需的各个模型
model_names = ['tf2torch_inception_v3', 'tf2torch_inception_v4', 'tf2torch_resnet_v2_101', 'tf2torch_inc_res_v2']

ensemble_models = []
for model_name in model_names:
    if model_name in model_repo.get_all_model_names():
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

# 获取所有目标模型
all_models = model_repo.get_all_model_names()
target_model_names = [model for model in all_models]
target_models = model_repo.get_target_models(target_model_names)

print(f"Selected {len(target_models)} target models: {list(target_models.keys())}")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

label_df = pd.read_csv(paths["label_csv"])
# label_df = label_df.sample(n=1)
print(f"Successfully read CSV, total {len(label_df)} images")


         
batch_results = []
PHFT_attack_params = {
    "eps": 16 / 255,
    "iterations": 10,
    "mu": 1.0,
    "num_blocks_h": 2,
    "num_blocks_w": 2,
    "flip_prob": 0.4,
    "max_angle": 1.6,
    "max_translate_ratio": 0.06,
    "num_copies": 20
}

for idx, row in tqdm(label_df.iterrows(), total=len(label_df)):
    img_filename = row["filename"]
    true_label = int(row["label"])
    img_path = os.path.join(paths["img_root"], img_filename)

    try:
        x = load_image_and_transform(img_path, transform, device)

        # Source model original prediction
        source_original_pred = get_model_prediction(ensemble_model, x)

        # PHFT attack (水平翻转+旋转版本)
        x_adv_PHFT = mifgsm_attack_PMRT_hflip_rotate(
            x=x,
            y=torch.tensor([true_label]).to(device),
            model=ensemble_model,
            **PHFT_attack_params
        )

        # Test on all target models
        target_predictions = {}
        target_fooled = {}
        for model_name, model_info in target_models.items():
            target_model = model_info['model']

            x_adv_pil = transforms.ToPILImage()(x_adv_PHFT.squeeze(0).cpu())
            x_adv_target = transform(x_adv_pil).unsqueeze(0).to(device)

            # Get prediction
            target_adv_pred = get_model_prediction(target_model, x_adv_target)
            target_predictions[model_name] = target_adv_pred
            target_fooled[model_name] = target_adv_pred != true_label

        # 获取对抗样本在源模型上的预测
        source_adv_pred = get_model_prediction(ensemble_model, x_adv_PHFT)
        source_attack_success = source_adv_pred != true_label

        # Store results
        result_entry = {
            "filename": img_filename,
            "true_label": true_label,
            "source_original_pred": source_original_pred,
            "source_adv_pred": source_adv_pred,
            "source_attack_success": source_attack_success
        }

        # Add target model predictions
        for model_name, pred in target_predictions.items():
            result_entry[f"target_{model_name}_pred"] = pred
            result_entry[f"target_{model_name}_fooled"] = target_fooled[model_name]

        batch_results.append(result_entry)

    except Exception as e:
        print(f"Error processing image {img_filename}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Save results and statistics
if batch_results:
    # 生成包含参数信息的保存路径
    save_path = f"{paths['result_save']}.csv"
    result_df = pd.DataFrame(batch_results)
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # Calculate success rate
    success_count = len(result_df)

    if success_count > 0:
        print(f"Successfully processed: {success_count} images")
        print(f"\n=== PHFT attack results with ensemble model")

        # 源模型（集成模型）攻击成功率
        source_success = result_df["source_attack_success"].sum()
        source_success_rate = source_success / success_count * 100
        print(f"Ensemble model attack success rate: {source_success}/{success_count} ({source_success_rate:.2f}%)")

        # Target model success rates (how often target models are fooled)
        target_success_rates = {}
        target_success_counts = {}
        for model_name in target_models.keys():
            target_success = result_df[f"target_{model_name}_fooled"].sum()
            target_success_counts[model_name] = target_success
            target_success_rate = target_success / success_count * 100
            target_success_rates[model_name] = target_success_rate
            print(f"Target model {model_name} attack success rate: {target_success}/{success_count} ({target_success_rate:.2f}%)")

        # 计算平均迁移成功率
        avg_success_rate = np.mean(list(target_success_rates.values()))
        print(f"Average transfer success rate: {avg_success_rate:.2f}%")

        # 保存统计信息
        stats_path = f"{paths['result_save']}.txt"
        with open(stats_path, 'w') as f:
            f.write(f"PHFT Attack Statistics (with Ensemble Model)\n")
            f.write(f"Ensemble models: {model_names}\n")
            f.write(f"Number of models in ensemble: {len(ensemble_models)}\n")
            f.write(f"Processed images: {success_count}\n")
            f.write(f"Ensemble model success rate: {source_success_rate:.2f}%\n")
            f.write(f"Average transfer success rate: {avg_success_rate:.2f}%\n")
            f.write("\nIndividual model success rates:\n")
            for model_name, rate in target_success_rates.items():
                f.write(f"{model_name}: {rate:.2f}%\n")

        # 保存详细结果到单独的文件
        detailed_results = []
        for result in batch_results:
            row = {
                "filename": result["filename"],
                "true_label": result["true_label"],
                "source_original_pred": result["source_original_pred"],
                "source_adv_pred": result["source_adv_pred"],
                "source_attack_success": result["source_attack_success"]
            }
            for model_name in target_models.keys():
                row[f"{model_name}_pred"] = result[f"target_{model_name}_pred"]
                row[f"{model_name}_fooled"] = result[f"target_{model_name}_fooled"]
            detailed_results.append(row)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = f"{paths['result_save']}.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")

        print(f"Statistics saved to: {stats_path}")
else:
    print("No successfully processed images")

print("-" * 80)