
# scripts/inference_single_model.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from lightning_model import CustomLightningModule
from src.lightning_model import CustomLightningModule
from datetime import datetime


def load_model(ckpt_path, device):
    # 모델 체크포인트 로드 후 평가 모드로 설정
    model = CustomLightningModule.load_from_checkpoint(ckpt_path).to(device)
    model.eval()
    return model


def predict_with_8_way_tta(model, image, img_size, device):
    # 8 방향 TTA: 회전(0~270도) + 좌우반전 후 회전
    tta_images = []

    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    flipped_img = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped_img, k=k).copy())

    # 전처리: Resize + Normalize + Tensor 변환
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)

    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)

    avg_preds = torch.mean(preds, dim=0)
    return avg_preds.cpu()


def run_single_model_inference(
    ckpt_path: str,
    img_size: int = 512,
    test_dir: str = "./data/raw/test/",
    sample_csv: str = "./data/raw/sample_submission.csv",
    output_dir: str = "./submissions"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.basename(ckpt_path).replace(".ckpt", "")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"submission_{model_name}_{timestamp}.csv")

    print(f"[🔄] Loading model: {model_name}")
    model = load_model(ckpt_path, device)

    submission_df = pd.read_csv(sample_csv)
    final_predictions = []

    print(f"[🧠] Running inference on test set...")
    for image_id in tqdm(submission_df["ID"], desc="Predicting"):
        image_path = os.path.join(test_dir, image_id)
        image = cv2.imread(image_path)

        if image is None:
            final_predictions.append(0)  # 예외처리: 이미지가 없으면 0 클래스
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preds = predict_with_8_way_tta(model, image, img_size, device)
        final_prediction = torch.argmax(preds, dim=0).item()
        final_predictions.append(final_prediction)

    submission_df["target"] = final_predictions
    submission_df.to_csv(output_filename, index=False)
    print(f"\n✅ Inference done. Saved to: {output_filename}")


'''
    •	0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt
	•	auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt
	•	auto-resnet101-lr1e-4-sz384-epoch=19-val_f1=0.9839.ckpt
	•	tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt 
'''
if __name__ == "__main__":
    # [사용자 수정 필요] 아래 경로만 바꿔서 실행하세요.
    # CKPT_PATH = "./models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt"
    # CKPT_PATH = "./models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt"
    # CKPT_PATH = "./models/auto-resnet101-lr1e-4-sz384-epoch=19-val_f1=0.9839.ckpt"
    CKPT_PATH = "./models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt"

    
    IMG_SIZE = 512

    run_single_model_inference(
        ckpt_path=CKPT_PATH,
        img_size=IMG_SIZE
    )