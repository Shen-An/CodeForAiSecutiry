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


