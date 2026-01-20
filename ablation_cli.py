import numpy as np


# 加载文件
file_path = 'E:\\TransferAttack\\RuiGuoCode\\torch_nets_weight\\tf2torch_resnet_v2_50.npy'

data = np.load(file_path, allow_pickle=True).item()

# 目标层
target_key = 'resnet_v2_50/logits/Conv2D'

if target_key in data:
    item = data[target_key]
    print(f"正在检查层: {target_key}")

    # 情况 A: 如果该键指向的是一个字典 (包含 weights/biases)
    if isinstance(item, dict):
        print(f"该层内部包含的键: {list(item.keys())}")
        for sub_key in item.keys():
            val = item[sub_key]
            if hasattr(val, 'shape'):
                print(f"  -> 子项 '{sub_key}' 的维度为: {val.shape}")
                # 通常最后一个维度或第一个维度就是类别数
                # 寻找 1000 或 1001 这个数字
                for dim in val.shape:
                    if dim in [1000, 1001]:
                        print(f"\n结论：输出标签总数是 {dim}，最大索引是 {dim - 1}")
                        break

    # 情况 B: 如果该键直接指向一个 Numpy 数组
    elif hasattr(item, 'shape'):
        print(f"该层的维度为: {item.shape}")
        for dim in item.shape:
            if dim in [1000, 1001]:
                print(f"\n结论：输出标签总数是 {dim}，最大索引是 {dim - 1}")
                break
else:
    print("未在文件中找到"+ {target_key- 1}+"，请检查键名是否正确。")