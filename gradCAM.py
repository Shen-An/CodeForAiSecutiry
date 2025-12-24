import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO
import timm
import pandas as pd


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        # 前向传播
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # 计算权重 - 确保所有张量在同一设备
        gradients = self.gradients[0]  # 取batch中第一个
        activations = self.activations[0]

        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(1, 2))

        # 计算热力图 - 在activations相同设备上初始化
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU激活
        heatmap = torch.relu(heatmap)

        # 归一化
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()

        return heatmap.detach().cpu().numpy(), target_class


def get_inception_v3_torch():
    """获取本地InceptionV3模型（torch版本）"""
    model = models.inception_v3(pretrained=False)
    state_dict = torch.load(r"../models/inception_v3_google-0cc3c7bd.pth")
    model.load_state_dict(state_dict)
    model.eval()
    # InceptionV3的Mixed_7c层
    target_layer = model.Mixed_7c.branch_pool.conv
    return model, target_layer


def get_inception_v4():
    """获取本地InceptionV4模型"""
    model = timm.create_model('inception_v4', pretrained=False)
    state_dict = torch.load(r"../models/inceptionV4.bin")
    model.load_state_dict(state_dict)
    model.eval()

    # 修复InceptionV4的目标层选择
    try:
        # 使用最后一个混合模块
        target_layer = model.features[-1].branch_pool[1]
    except:
        try:
            # 使用最后一个卷积块
            target_layer = model.features[-1].conv
        except:
            try:
                # 使用模型的最后一个卷积层
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                print(f"使用最后一个卷积层: {target_layer}")
            except:
                # 方法4：使用全局平均池化层之前的层
                target_layer = model.global_pool
                print("使用全局平均池化层作为目标层")

    return model, target_layer


def get_inception_resnet_v2():
    """获取本地Inception-ResNetV2模型"""
    model = timm.create_model('inception_resnet_v2', pretrained=False)
    state_dict = torch.load(r"../models/incep_resnet_V2.bin")
    model.load_state_dict(state_dict)
    model.eval()

    # 修复Inception-ResNetV2的目标层选择
    try:
        # 使用最后一个卷积块
        target_layer = model.conv2d_7b
    except:
        try:
            # 使用最后一个混合模块
            target_layer = model.mixed_7a
        except:
            # 使用最后一个卷积层
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module

    return model, target_layer


def get_resnet101():
    """获取本地ResNet101模型"""
    model = models.resnet101(pretrained=False)
    state_dict = torch.load(r"../models/resnet101-63fe2227.pth")
    model.load_state_dict(state_dict)
    model.eval()
    # ResNet101的layer4的最后一个卷积层
    target_layer = model.layer4[-1].conv3
    return model, target_layer


def get_densenet121():
    """获取本地DenseNet121模型"""
    model = timm.create_model('densenet121', pretrained=False)
    state_dict = torch.load(r"../models/densenet121.bin")
    model.load_state_dict(state_dict)
    model.eval()

    # 修复DenseNet121的目标层选择
    try:
        # 使用最后一个密集块的最后一个卷积层
        target_layer = model.features.denseblock4.denselayer16.conv2
    except:
        try:
            # 使用最后一个卷积层
            target_layer = model.features.norm5
        except:
            # 使用最后一个批量归一化层
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    target_layer = module

    return model, target_layer


def get_adv_inception_v3():
    """获取本地Adversarial InceptionV3模型"""
    model = timm.create_model('inception_v3', pretrained=False)
    state_dict = torch.load(r"../models/adv_inceptionV3.bin")
    model.load_state_dict(state_dict)
    model.eval()

    # 修复Adversarial InceptionV3的目标层选择
    try:
        # Adversarial InceptionV3的Mixed_7c层
        target_layer = model.Mixed_7c.branch_pool.conv
    except:
        try:
            # 使用最后一个卷积层
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
        except:
            # 使用全局平均池化层
            target_layer = model.global_pool

    return model, target_layer


def preprocess_image(image_path, size=(299, 299)):
    """预处理图像 - 使用299x299以适应Inception系列模型"""
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')

    original_image = image.copy()
    input_tensor = transform(image).unsqueeze(0)

    return original_image, input_tensor


def overlay_heatmap(heatmap, original_image, alpha=0.5, gamma=3):
    """将热力图叠加到原始图像上，增强注意力区域并让背景更通透"""
    # 调整热力图大小以匹配原始图像
    heatmap = cv2.resize(heatmap, original_image.size)

    # 增强热力图对比度（使注意力区域更突出）
    heatmap = np.clip(heatmap, 0, 1)  # 确保在0-1范围内
    heatmap = np.power(heatmap, gamma)  # 伽马校正增强对比度
    heatmap = heatmap / (heatmap.max() + 1e-8)  # 重新归一化，避免除零

    # 转换为彩色热力图（使用JET映射，红色/黄色表示强注意力）
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # 转换颜色通道（BGR->RGB）
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 转换原始图像为numpy数组
    original_np = np.array(original_image)

    # 移除alpha通道（如果存在）
    if original_np.shape[2] == 4:
        original_np = original_np[:, :, :3]

    # 统一数据类型以避免叠加错误
    heatmap_colored = heatmap_colored.astype(np.float32)
    original_np = original_np.astype(np.float32)

    # 叠加热力图：alpha控制热力图强度（越大注意力越明显），1-alpha控制背景透明度
    superimposed = heatmap_colored * alpha + original_np * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)


# 加载完整的ImageNet标签映射
def load_imagenet_labels(csv_path):
    """从CSV文件加载完整的ImageNet标签映射"""
    df = pd.read_csv(csv_path)
    # 创建一个从标签索引到文件名的映射
    label_to_filename = {}
    for _, row in df.iterrows():
        label_to_filename[int(row['label'])] = row['filename']
    return label_to_filename


# 加载ImageNet类别名称
def load_imagenet_class_names():
    """加载ImageNet类别名称"""
    # 尝试从torchvision加载
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        class_names = weights.meta["categories"]
        return class_names
    except:
        pass

    # 如果torchvision方法失败，尝试从timm加载
    try:
        import timm
        model = timm.create_model('resnet50', pretrained=True)
        if hasattr(model, 'get_classifier'):
            # 对于timm模型，我们可能需要其他方式获取类别名称
            pass
    except:
        pass

    # 如果上述方法都失败，使用一个简单的类别名称字典
    class_names = {
        0: "tench, Tinca tinca",
        1: "goldfish, Carassius auratus",
        2: "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
        # 添加更多类别...
        281: "tabby, tabby cat",
        282: "tiger cat",
        386: "African elephant, Loxodonta africana",
        # 可以根据需要添加更多类别
    }

    return class_names


# 全局变量存储标签映射和类别名称
IMAGENET_LABELS = None
IMAGENET_CLASS_NAMES = None


def get_class_name(class_idx):
    """获取类别名称"""
    global IMAGENET_CLASS_NAMES

    if IMAGENET_CLASS_NAMES is None:
        IMAGENET_CLASS_NAMES = load_imagenet_class_names()

    # 如果类别名称字典中有这个索引，返回类别名称
    if class_idx in IMAGENET_CLASS_NAMES:
        return IMAGENET_CLASS_NAMES[class_idx]
    else:
        # 否则返回类别索引
        return f"Class {class_idx}"


def get_filename_from_label(class_idx):
    """根据标签索引获取文件名"""
    global IMAGENET_LABELS

    if IMAGENET_LABELS is None:
        try:
            IMAGENET_LABELS = load_imagenet_labels(r"../data/labels.csv")
        except Exception as e:
            print(f"加载标签文件失败: {e}")
            IMAGENET_LABELS = {}

    return IMAGENET_LABELS.get(class_idx, f"Unknown file for class {class_idx}")


def visualize_gradcam(model_name, model, target_layer, original_image, input_tensor, axs, col_idx):
    """可视化Grad-CAM结果"""
    try:
        # 如果是原图像，只显示原图像，不进行Grad-CAM处理
        if model_name == 'Original':
            axs[col_idx].imshow(original_image)
            axs[col_idx].set_title('Original Image')
            axs[col_idx].axis('off')
            return "Original", "Original"

        # 创建Grad-CAM实例
        gradcam = GradCAM(model, target_layer)

        # 生成热力图
        heatmap, class_idx = gradcam.generate_heatmap(input_tensor)

        # 获取类别名称和文件名
        class_name = get_class_name(class_idx)
        filename = get_filename_from_label(class_idx)

        # 叠加热力图
        superimposed_img = overlay_heatmap(heatmap, original_image)

        # 显示结果
        axs[col_idx].imshow(superimposed_img)
        axs[col_idx].set_title(f'{model_name}\nPredicted: {class_name}')
        axs[col_idx].axis('off')

        return class_name, filename
    except Exception as e:
        print(f"可视化 {model_name} 失败: {e}")
        # 显示错误信息
        if model_name == 'Original':
            axs[col_idx].imshow(original_image)
            axs[col_idx].set_title('Original Image')
        else:
            axs[col_idx].text(0.5, 0.5, f'{model_name}\nError:\n{str(e)}',
                              ha='center', va='center', transform=axs[col_idx].transAxes)
            axs[col_idx].set_title(f'{model_name} Error')

        axs[col_idx].axis('off')

        return f"Error: {e}", "Unknown"


def main(image_path):
    """主函数"""
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载图像
    original_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # 定义模型列表
    model_configs = [
        ('InceptionV3', get_inception_v3_torch),
        ('InceptionV4', get_inception_v4),
        ('Inception-ResNetV2', get_inception_resnet_v2),
        ('ResNet101', get_resnet101),
        ('DenseNet121', get_densenet121),
        ('Adv-InceptionV3', get_adv_inception_v3)
    ]

    # 加载所有模型
    models_list = []
    for model_name, model_loader in model_configs:
        try:
            model, target_layer = model_loader()
            model = model.to(device)
            models_list.append((model_name, model, target_layer))
            print(f"成功加载模型: {model_name}")
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")

    if not models_list:
        print("没有成功加载任何模型！")
        return

    # 创建可视化图表 - 横向排列：原图像 + 所有模型的Grad-CAM
    num_cols = len(models_list) + 1  # 原图像 + 所有模型
    fig, axs = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # 如果只有一个子图，调整axs为数组
    if num_cols == 1:
        axs = [axs]

    results = {}

    # 首先绘制原图像
    class_name_original, filename_original = visualize_gradcam(
        'Original', None, None, original_image, input_tensor, axs, 0
    )

    # 为每个模型生成Grad-CAM
    for idx, (model_name, model, target_layer) in enumerate(models_list):
        try:
            class_name, filename = visualize_gradcam(
                model_name, model, target_layer,
                original_image, input_tensor, axs, idx + 1
            )
            results[model_name] = {
                'class_name': class_name,
                'class_idx': None,  # 这里可以添加获取类别索引的逻辑
            }
            print(f"{model_name} 预测类别: {class_name}")
        except Exception as e:
            print(f"{model_name} 生成Grad-CAM失败: {e}")
            results[model_name] = {
                'class_name': f"Error: {e}",
                'class_idx': None,
            }

    plt.tight_layout()
    # 根据需要取消注释其中一个保存语句
    plt.savefig('gradcam_results_original.png', dpi=300, bbox_inches='tight')
    # plt.savefig('gradcam_results_transform.png', dpi=300, bbox_inches='tight')
    # plt.savefig('gradcam_results_adv.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印所有结果
    for model_name, result in results.items():
        print(f"{model_name}预测类别: {result['class_name']}")

    return results


if __name__ == "__main__":
    # 使用示例图像
    image_path = r"./data/adv/adv_ILSVRC2012_val_00001148.jpg"
    # image_path=r'D:\pycharm\python\Adversial\work_now\PMRT\transform_images_PHFRT\transformed_image.jpg'
    # image_path = r"../ours/adv_images/ILSVRC2012_val_00001148.png"
    # 运行完整版本
    results = main(image_path)