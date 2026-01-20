from models import ModelRepository, load_image_and_transform
import os
import torch
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as transforms

def get_model_prediction(model, x):
    output = model(x)

    if isinstance(output, tuple):
        output = output[0]
    elif isinstance(output, list):
        output = output[0]

    if output.dim() == 1:
        output = output.unsqueeze(0)
    elif output.dim() > 2:
        output = output.view(output.size(0), -1)

    output = torch.argmax(output, dim=1)
    return output


import torch
import gc
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image


def eval_transferability(source_results, adv_images_storage, args, model_repo, device):
    """
    source_results: 包含原模型攻击结果的列表 (List of dicts)
    adv_images_storage: 存储对抗样本张量的列表 (List of Tensors)
    """
    print(f"\n[Step 2/3] Testing Transferability (Using Memory Storage)...")

    # 1. 准备数据加载器
    # 假设 adv_images_storage 里的 tensor 形状是 [1, C, H, W]，先合并
    all_adv_tensors = torch.cat(adv_images_storage, dim=0)
    adv_mem_dataset = TensorDataset(all_adv_tensors)
    adv_mem_loader = DataLoader(
        adv_mem_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        pin_memory=True
    )

    # 2. 获取目标模型列表 (排除掉生成对抗样本的源模型)
    all_model_names = model_repo.get_all_model_names()
    target_names = [name for name in all_model_names if name != args.model]
    target_predictions = {name: [] for name in target_names}

    # 3. 逐个模型进行推理
    for model_name in target_names:
        print(f"  --> Testing target model: {model_name}")

        # 加载单个模型
        current_model_info = model_repo.load_single_model(model_name)
        model = current_model_info['model']
        model.to(device)
        model.eval()

        model_preds = []
        with torch.no_grad():
            for [x_adv_batch] in tqdm(adv_mem_loader, desc=f"Scanning {model_name}"):
                x_adv_batch = x_adv_batch.to(device)
                # 获取预测结果 (假设 get_model_prediction 返回的是 tensor 或 list)
                preds = get_model_prediction(model, x_adv_batch)

                # 确保转为 list 以便存储
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy().tolist()
                model_preds.extend(preds)

        target_predictions[model_name] = model_preds

        # 关键步骤：显存清理
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    print(f"\n[Step 3/3] Saving results and images to disk...")

    # 4. 保存对抗样本图片
    os.makedirs(args.output_adv_dir, exist_ok=True)
    for idx, res in enumerate(source_results):
        fn = res["filename"]
        # 移除 batch 维度保存: [C, H, W]
        save_image(all_adv_tensors[idx], os.path.join(args.output_adv_dir, fn))

    # 5. 统计攻击成功率 (ASR) 并整合 CSV
    final_rows = []
    model_success_counts = {name: 0 for name in target_names}

    for idx, res in enumerate(source_results):
        row = res.copy()
        true_label = res["true_label"]

        for model_name in target_names:
            pred = int(target_predictions[model_name][idx])
            # 攻击成功判定：预测类别不等于真实类别
            fooled = (pred != true_label)

            row[f"{model_name}_pred"] = pred
            row[f"{model_name}_fooled"] = fooled

            if fooled:
                model_success_counts[model_name] += 1
        final_rows.append(row)

    # 6. 打印最终报告
    total_samples = len(source_results)
    source_rate = sum(1 for r in source_results if r['source_attack_success']) / total_samples * 100

    print("-" * 30)
    print(f"Source Model ({args.model}) Success Rate: {source_rate:.1f}%")
    for name, count in model_success_counts.items():
        asr = (count / total_samples) * 100
        print(f"  Target {name}: {count}/{total_samples} (ASR: {asr:.1f}%)")
    print("-" * 30)

    # 7. 保存 CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(final_rows).to_csv(args.output_csv, index=False)
    print(f"Detailed results saved to {args.output_csv}")

# if __name__=='__main__':
#     main()