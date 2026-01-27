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

def main():
    batch_size = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {
        "img_root": r"./adv",
        "label_csv": r"./data/labels.csv",
    }
    model_repo = ModelRepository(device)
    all_models = model_repo.get_all_model_names()
    target_models = model_repo.get_target_models(all_models)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    label_df = pd.read_csv(paths["label_csv"])
    total_samples = len(label_df)

    for model_name, model_info in target_models.items():
        target_model = model_info['model']
        attack_success = 0
        
        for batch_start in tqdm(range(0, total_samples, batch_size), desc=f'Predicting {model_name}'):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_df = label_df.iloc[batch_start:batch_end]
            batch_img_tensors = []
            batch_true_labels = []
            for idx, row in batch_df.iterrows():
                img_filename = row["filename"]
                true_label = int(row["label"]) + 1
                img_path = os.path.join(paths["img_root"], img_filename)

                img_tensor = load_image_and_transform(img_path, transform, device)
                batch_img_tensors.append(img_tensor)
                batch_true_labels.append(true_label)
            
            batch_img_tensor = torch.cat(batch_img_tensors, dim=0)
            pred_labels = get_model_prediction(target_model, batch_img_tensor)
            true_labels_tensor = torch.tensor(batch_true_labels, device=device)
            attack_success += torch.sum(pred_labels != true_labels_tensor).item()
            
        print("================================")
        print(f"{model_name}_ASR: {attack_success / total_samples * 100:.2f}%")
        print("================================")

if __name__=='__main__':
    main()