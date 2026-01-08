import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import math
from tqdm.notebook import tqdm
from torch_nets import (
    tf2torch_inception_v3,
    tf2torch_inception_v4,
    tf2torch_resnet_v2_50,
    tf2torch_resnet_v2_101,
    tf2torch_resnet_v2_152,
    tf2torch_inc_res_v2,
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
class Normalize(nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, input):
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)
        return (input - self.mean) / self.std


class ModelRepository:
    """Model repository class for managing source and target models"""

    def __init__(self, device, model_dir='torch_nets_weight/'):
        self.device = device
        self.model_dir = model_dir
        self.models = {}
        self._load_all_models()

    def _load_model(self, net_name):
        """Load converted model following torch_attack.py style"""
        model_path = os.path.join(self.model_dir, net_name + '.npy')

        if net_name == 'tf2torch_inception_v3':
            net = tf2torch_inception_v3
        elif net_name == 'tf2torch_inception_v4':
            net = tf2torch_inception_v4
        elif net_name == 'tf2torch_resnet_v2_50':
            net = tf2torch_resnet_v2_50
        elif net_name == 'tf2torch_resnet_v2_101':
            net = tf2torch_resnet_v2_101
        elif net_name == 'tf2torch_resnet_v2_152':
            net = tf2torch_resnet_v2_152
        elif net_name == 'tf2torch_inc_res_v2':
            net = tf2torch_inc_res_v2
        elif net_name == 'tf2torch_adv_inception_v3':
            net = tf2torch_adv_inception_v3
        elif net_name == 'tf2torch_ens3_adv_inc_v3':
            net = tf2torch_ens3_adv_inc_v3
        elif net_name == 'tf2torch_ens4_adv_inc_v3':
            net = tf2torch_ens4_adv_inc_v3
        elif net_name == 'tf2torch_ens_adv_inc_res_v2':
            net = tf2torch_ens_adv_inc_res_v2
        else:
            raise ValueError(f'Wrong model name: {net_name}!')

        if 'inc' in net_name:
            model = nn.Sequential(
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                net.KitModel(model_path, aux_logits=True).eval().to(self.device), )
        else:
            model = nn.Sequential(
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                net.KitModel(model_path).eval().to(self.device), )
        return model

    def _load_all_models(self):
        """Load all models into the repository following torch_attack.py style"""
        model_names = [
             'tf2torch_inception_v3',
             'tf2torch_inception_v4',
            # 'tf2torch_resnet_v2_50',不要
            'tf2torch_resnet_v2_101',
            # 'tf2torch_resnet_v2_152',不要
             'tf2torch_inc_res_v2',
             # 'tf2torch_adv_inception_v3',不要
             'tf2torch_ens3_adv_inc_v3',
             'tf2torch_ens4_adv_inc_v3',
             'tf2torch_ens_adv_inc_res_v2'
        ]

        for model_name in model_names:
            model = self._load_model(model_name)
            if model is not None:
                self.models[model_name] = {
                    'model': model,
                    'input_size': 299,
                    'type': 'both',
                    'normalization': 'tensorflow'
                }

    def get_source_model(self, model_name='tf2torch_inception_v3'):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} does not exist in repository")

    def get_target_models(self, model_names=None):
        """Get multiple target models"""
        if model_names is None:
            return {name: info for name, info in self.models.items()}
        else:
            target_models = {}
            for name in model_names:
                if name in self.models:
                    target_models[name] = self.models[name]
                else:
                    print(f"Warning: Model {name} does not exist in repository")
            return target_models

    def get_all_model_names(self):
        return list(self.models.keys())

    def get_model_info(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} does not exist in repository")

    def load_single_model(self, model_name):
        """
        根据模型名称加载单个模型，并返回包含模型的字典。
        直接调用 _load_model，它已经包含了 Normalize 层。
        """
        print(f"Loading {model_name}...")

        # 直接获取模型，_load_model 内部已经处理了 .to(device) 和 .eval()
        model = self._load_model(model_name)

        # 保持一致性，返回字典格式
        return {
            'model': model,
            'input_size': 299,
            'type': 'both',
            'normalization': 'tensorflow'
        }

def load_image_and_transform(img_path, transform, device):
    """Unified image loading and transformation function"""
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def calculate_success_rate(results_df, success_status="Success"):
    """Calculate attack success rate"""
    success_results = results_df[results_df["status"] == success_status]

    if len(success_results) == 0:
        return 0, 0, 0

    PHFRT_target_success = len(
        success_results[success_results["target_adv_pred_PHFRT"] != success_results["true_label"]])
    PHFRT_target_success_rate = PHFRT_target_success / len(success_results) * 100

    return len(success_results), PHFRT_target_success_rate


def get_model_prediction(model, x):
    """Unified model prediction function"""
    with torch.no_grad():
        output = model(x)

    if isinstance(output, tuple):
        output = output[0]
    elif isinstance(output, list):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Unexpected output type: {type(output)}")

    if output.dim() == 1:
        return torch.argmax(output).item()
    elif output.dim() == 2:
        return torch.argmax(output, dim=1).item()
    else:
        if output.dim() > 2:
            output = output.view(output.size(0), -1)
            return torch.argmax(output, dim=1).item()
        else:
            raise ValueError(f"Unexpected output dimension: {output.dim()}, shape: {output.shape}")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.cuda.empty_cache()
    # Path configuration
    paths = {
        "img_root": r"./data/images",
        "label_csv": r"./data/labels.csv",
        "result_save": r"./PHFRT/phfrt"
    }

    # Initialize model repository
    model_repo = ModelRepository(device)
    print(f"Model repository initialized, total {len(model_repo.get_all_model_names())} models")
    print(f"Available models: {model_repo.get_all_model_names()}")

    source_model_info = model_repo.get_source_model('tf2torch_inception_v3')
    source_model = source_model_info['model']
    source_input_size = source_model_info['input_size']

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
    attack_params = {
        "eps": 16 / 255,
        "iterations": 10,
        "mu": 1.0,
    }

    for idx, row in tqdm(label_df.iterrows(), desc='Processing images', total=len(label_df), mininterval=1.0):
        # torch.cuda.empty_cache()
        img_filename = row["filename"]
        true_label = int(row["label"])
        img_path = os.path.join(paths["img_root"], img_filename)

        try:
            x = load_image_and_transform(img_path, transform, device)

            # Source model original prediction
            source_original_pred = get_model_prediction(source_model, x)

            x_adv = attack(
                x=x,
                y=torch.tensor([true_label]).to(device),
                model=source_model,
                **attack_params
            )

            # Test on all target models
            target_predictions = {}
            for model_name, model_info in target_models.items():
                target_model = model_info['model']

                x_adv_pil = transforms.ToPILImage()(x_adv.squeeze(0).cpu())
                x_adv_target = transform(x_adv_pil).unsqueeze(0).to(device)

                # Get prediction
                target_adv_pred = get_model_prediction(target_model, x_adv_target)
                target_predictions[model_name] = target_adv_pred

            # Store results
            result_entry = {
                "filename": img_filename,
                "true_label": true_label,
                "source_original_pred": source_original_pred,
                "status": "Success"
            }

            # Add target model predictions
            for model_name, pred in target_predictions.items():
                result_entry[f"target_{model_name}_pred"] = pred

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
            print(f"\n=== attack results ===")

            # Target model success rates (how often target models are fooled)
            target_success_rates = {}
            for model_name in target_models.keys():
                target_success = len(result_df[result_df[f"target_{model_name}_pred"] != result_df["true_label"]])
                target_success_rate = target_success / success_count * 100
                target_success_rates[model_name] = target_success_rate
                print(f"Target model {model_name} attack success rate: {target_success_rate:.2f}%")

            # 计算平均成功率
            avg_success_rate = np.mean(list(target_success_rates.values()))
            print(f"Average attack success rate: {avg_success_rate:.2f}%")

            # 保存统计信息
            stats_path = f"{paths['result_save']}.txt"
            with open(stats_path, 'w') as f:
                f.write(f" Attack Statistics\n")
                f.write(f"Processed images: {success_count}\n")
                f.write(f"Average success rate: {avg_success_rate:.2f}%\n")
                f.write("\nIndividual model success rates:\n")
                for model_name, rate in target_success_rates.items():
                    f.write(f"{model_name}: {rate:.2f}%\n")
    else:
        print("No successfully processed images")

    print("-" * 80)
