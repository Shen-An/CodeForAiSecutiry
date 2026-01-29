import os
import torch
import torch.nn as nn
from typing import List
from PIL import Image

from torch_nets import (
    tf2torch_inception_v3,
    tf2torch_inception_v4,
    tf2torch_resnet_v2_50,
    tf2torch_resnet_v2_101,
    tf2torch_resnet_v2_152,
    tf2torch_inc_res_v2,
    # tf2torch_adv_inception_v3,
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
        # elif net_name == 'tf2torch_adv_inception_v3':
        #     net = tf2torch_adv_inception_v3
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
        model_names = [
            'tf2torch_inception_v3',
            'tf2torch_inception_v4',
            'tf2torch_resnet_v2_50',
            'tf2torch_resnet_v2_101',
            'tf2torch_resnet_v2_152',
            'tf2torch_inc_res_v2',
            # 'tf2torch_adv_inception_v3',
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
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} does not exist in repository")
        return self.models[model_name]

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


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], device: torch.device):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models).to(device)
        self.device = device

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)

            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, list):
                output = output[0]

            if output.dim() == 1:
                output = output.unsqueeze(0)
            elif output.dim() > 2:
                output = output.view(output.size(0), -1)
            outputs.append(output)

        avg_output = torch.stack(outputs).mean(dim=0)
        return avg_output

    def eval(self):
        for model in self.models:
            model.eval()
        return self

def load_image_and_transform(img_path, transform, device):
    """Load and transform image"""
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)