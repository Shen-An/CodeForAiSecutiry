import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置全局字体为新罗马
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # 使数学符号也接近新罗马风格

# ==========================================
# 1. GradCAM 类 (保持不变)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        heatmap = torch.relu(heatmap)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.detach().cpu().numpy(), target_class


# ==========================================
# 2. 变换逻辑 (保持不变)
# ==========================================

# --- A. BSR 变换 ---
def forward_bsr_viz(image_pil, split=2, max_angle=15):
    img_np = np.array(image_pil)
    h, w = img_np.shape[:2]
    h_mid, w_mid = h // split, w // split

    # 1. 切块
    blocks = [
        img_np[0:h_mid, 0:w_mid],  # 0: TL
        img_np[0:h_mid, w_mid:],  # 1: TR
        img_np[h_mid:, 0:w_mid],  # 2: BL
        img_np[h_mid:, w_mid:]  # 3: BR
    ]

    # 2. 旋转
    rotated_blocks = []
    rot_params = []
    for idx, blk in enumerate(blocks):
        angle = random.uniform(-max_angle, max_angle)
        h_b, w_b = blk.shape[:2]
        center = (w_b // 2, h_b // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_blk = cv2.warpAffine(blk, M, (w_b, h_b), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rotated_blocks.append(rot_blk)
        rot_params.append((M, (w_b, h_b)))

    # 3. 乱序
    perm_order = [3, 2, 1, 0]

    target_sizes = [
        (w_mid, h_mid), (w - w_mid, h_mid),
        (w_mid, h - h_mid), (w - w_mid, h - h_mid)
    ]

    bsr_img = np.zeros_like(img_np)
    bsr_img[0:h_mid, 0:w_mid] = cv2.resize(rotated_blocks[perm_order[0]], target_sizes[0])
    bsr_img[0:h_mid, w_mid:] = cv2.resize(rotated_blocks[perm_order[1]], target_sizes[1])
    bsr_img[h_mid:, 0:w_mid] = cv2.resize(rotated_blocks[perm_order[2]], target_sizes[2])
    bsr_img[h_mid:, w_mid:] = cv2.resize(rotated_blocks[perm_order[3]], target_sizes[3])

    return Image.fromarray(bsr_img), (perm_order, rot_params)


def inverse_bsr_viz(heatmap, params, original_size):
    perm_order, rot_params = params
    h, w = original_size
    heatmap = cv2.resize(heatmap, (w, h))
    h_mid, w_mid = h // 2, w // 2

    current_blocks = [
        heatmap[0:h_mid, 0:w_mid], heatmap[0:h_mid, w_mid:],
        heatmap[h_mid:, 0:w_mid], heatmap[h_mid:, w_mid:]
    ]

    restored_blocks_raw = [None] * 4
    for pit_idx, original_blk_idx in enumerate(perm_order):
        orig_w, orig_h = rot_params[original_blk_idx][1]
        restored_blocks_raw[original_blk_idx] = cv2.resize(current_blocks[pit_idx], (orig_w, orig_h))

    final_blocks = restored_blocks_raw

    restored_img = np.zeros_like(heatmap)
    restored_img[0:h_mid, 0:w_mid] = final_blocks[0]
    restored_img[0:h_mid, w_mid:] = final_blocks[1]
    restored_img[h_mid:, 0:w_mid] = final_blocks[2]
    restored_img[h_mid:, w_mid:] = final_blocks[3]

    return restored_img


# --- B. BGPF 变换 ---
def forward_bgpf_viz(image_pil, split=2, rho=0.2, flip_prob=1.0):
    img_np = np.array(image_pil)
    h, w = img_np.shape[:2]
    h_mid, w_mid = h // split, w // split

    coords = [
        (0, h_mid, 0, w_mid), (0, h_mid, w_mid, w),
        (h_mid, h, 0, w_mid), (h_mid, h, w_mid, w)
    ]

    bgpf_img = np.copy(img_np)
    trans_params = []

    for idx, (y1, y2, x1, x2) in enumerate(coords):
        block = img_np[y1:y2, x1:x2]
        h_b, w_b = block.shape[:2]

        src = np.float32([[0, 0], [w_b, 0], [0, h_b], [w_b, h_b]])
        limit_w, limit_h = int(w_b * rho), int(h_b * rho)
        delta = np.random.randint(-min(limit_w, limit_h), min(limit_w, limit_h), size=(4, 2)).astype(np.float32)
        dst = src + delta
        M = cv2.getPerspectiveTransform(src, dst)
        # NOTE: 之前用 BORDER_REFLECT 会把越界区域用反射像素“补齐”，视觉上几乎看不到黑边。
        # 为了让透视导致的空洞/黑边在图2(Transformed Input)中可见，改用常量黑色填充。
        warped_blk = cv2.warpPerspective(
            block,
            M,
            (w_b, h_b),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        is_flipped = False
        if flip_prob > 0 and idx >= 2 and random.random() < flip_prob:
            warped_blk = cv2.flip(warped_blk, 1)
            is_flipped = True

        bgpf_img[y1:y2, x1:x2] = warped_blk

        _, M_inv = cv2.invert(M)
        trans_params.append({'M_inv': M_inv, 'flipped': is_flipped, 'rect': (y1, y2, x1, x2), 'size': (w_b, h_b)})

    return Image.fromarray(bgpf_img), trans_params


def inverse_bgpf_viz(heatmap, params, original_size):
    h, w = original_size
    heatmap = cv2.resize(heatmap, (w, h))

    # 仅做图2对应操作的逆：flip -> inverse perspective
    restored = np.zeros_like(heatmap, dtype=np.float32)

    for p in params:
        y1, y2, x1, x2 = p['rect']
        w_b, h_b = p['size']
        blk = heatmap[y1:y2, x1:x2]

        # 逆 flip
        if p['flipped']:
            blk = cv2.flip(blk, 1)

        # 逆透视（不做任何补洞/填充）
        inv_blk = cv2.warpPerspective(
            blk,
            p['M_inv'],
            (w_b, h_b),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)

        restored[y1:y2, x1:x2] = inv_blk

    restored = np.clip(restored, 0.0, 1.0)
    return restored


# ==========================================
# 3. 主程序 (绘图逻辑修改为 2行3列 横排)
# ==========================================
def preprocess(image_pil):
    image_pil = image_pil.resize((299, 299), Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_pil, transform(image_pil).unsqueeze(0)


def overlay(heatmap, bg_img, alpha=0.6):
    heatmap = cv2.resize(heatmap, bg_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    bg = np.array(bg_img)
    return Image.fromarray(cv2.addWeighted(bg, 1 - alpha, heatmap, alpha, 0))


def main_fig1(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("Loading Model...")
    model = models.inception_v3(pretrained=True)
    model.eval().to(device)
    target_layer = model.Mixed_7c.branch_pool.conv
    gradcam = GradCAM(model, target_layer)

    # 准备原图
    original_pil = Image.open(image_path).convert('RGB')
    original_pil, ten_clean = preprocess(original_pil)
    ten_clean = ten_clean.to(device)

    # --- Step A: Original ---
    print("Processing Original...")
    map_clean, _ = gradcam.generate_heatmap(ten_clean)
    img_clean_overlay = overlay(map_clean, original_pil)

    # --- Step B: BSR ---
    print("Processing BSR...")
    img_bsr_pil, bsr_params = forward_bsr_viz(original_pil)
    _, ten_bsr = preprocess(img_bsr_pil)
    map_bsr_raw, _ = gradcam.generate_heatmap(ten_bsr.to(device))
    map_bsr_inv = inverse_bsr_viz(map_bsr_raw, bsr_params, original_pil.size)

    # --- Step C: BGPF ---
    print("Processing BGPF...")
    img_bgpf_pil, bgpf_params = forward_bgpf_viz(original_pil)
    _, ten_bgpf = preprocess(img_bgpf_pil)
    map_bgpf_raw, _ = gradcam.generate_heatmap(ten_bgpf.to(device))
    map_bgpf_inv = inverse_bgpf_viz(map_bgpf_raw, bgpf_params, original_pil.size)

    # --- 绘图 (2行3列) ---
    print("Plotting...")
    fig, axs = plt.subplots(2, 3, figsize=(4.9, 3))

    # 标题字体设置
    fs_title = 6
    fs_label = 6

    # === 第一行: BSR (Method: BSR) ===
    # Col 1: Original
    axs[0, 0].imshow(img_clean_overlay)
    axs[0, 0].set_ylabel("Method: BSR", fontsize=fs_label)
    axs[0, 0].set_title("Standard Attention\n(Ground Truth)", fontsize=fs_title,pad=0)

    # Col 2: BSR Transformed
    axs[0, 1].imshow(overlay(map_bsr_raw, img_bsr_pil))
    axs[0, 1].set_title("Transformed Input\n(Shuffled & Rotated)", fontsize=fs_title,pad=0)

    # Col 3: BSR Restored
    axs[0, 2].imshow(overlay(map_bsr_inv, original_pil))
    axs[0, 2].set_title("Projected Back\n(Fragmented Structure)", fontsize=fs_title,pad=0)

    # === 第二行: BGPF (Method: BGPF) ===
    # Col 1: Original
    axs[1, 0].imshow(img_clean_overlay)
    axs[1, 0].set_ylabel("Method: BGPF (Ours)", fontsize=fs_label)
    axs[1, 0].set_title("Standard Attention\n(Ground Truth)", fontsize=fs_title,pad=0)

    # Col 2: BGPF Transformed
    axs[1, 1].imshow(overlay(map_bgpf_raw, img_bgpf_pil))
    axs[1, 1].set_title("Transformed Input\n(Warped & Flipped)", fontsize=fs_title,pad=0)

    # Col 3: BGPF Restored
    axs[1, 2].imshow(overlay(map_bgpf_inv, original_pil))
    axs[1, 2].set_title("Projected Back\n(Preserved Structure)", fontsize=fs_title,pad=0)

    # 美化
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        # 移除边框线 (Spines) 但保留 ylabel
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout(pad=0.0)  # pad 控制图像与边缘的距离
    plt.subplots_adjust(hspace=0.15)
    plt.savefig('fig1_horizontal_layout.png', dpi=150, bbox_inches='tight')
    print("Done! Saved to fig1_horizontal_layout.png")

    # 2. 保存为 EPS (用于 ECCV 投稿)
    # EPS 是矢量图，LaTeX 会自动处理，不需要设置 dpi
    plt.savefig('fig1_horizontal_layout.eps', format='eps', bbox_inches='tight')
    # 在 plt.show() 之前添加，控制子图间的横向(wspace)和纵向(hspace)间距

    plt.show()


if __name__ == "__main__":
    img_path = r"data/images/ILSVRC2012_val_00013528.png"
    main_fig1(img_path)