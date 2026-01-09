import os
import numpy as np

import torch

import torchvision.models
from PIL import Image

from models import Normalize
from torch.utils.data import Dataset


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
