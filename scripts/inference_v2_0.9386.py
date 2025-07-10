# scripts/inference_v2.py
# 리더보드 점수 0.9386

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from src.lightning_model import CustomLightningModule
# from lightning_model import CustomLightningModule
from datetime import datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule


def predict_with_enhanced_tta(model, image, img_size, device):
    """개선된 TTA (회전, 반전 + blur/noise/elastic 등 다양한 질감 변형 포함)"""
    tta_images = []

    # 회전 및 좌우반전 이미지 8개 생성
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    flipped = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped, k=k).copy())

    # 다양한 질감 및 노이즈 변형 적용
    weak_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.3),  # 약한 흐림
            A.MotionBlur(blur_limit=3, p=0.3),    # 흔들린 이미지
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),  # 노이즈 추가
            A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=20, p=0.3),  # 탄성 변형
        ], p=0.6),  # 위 네 가지 중 하나 적용할 확률
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),  # 밝기/대비 조정
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    batch = torch.stack([
        weak_transforms(image=img)['image'] for img in tta_images
    ]).to(device)

    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)

    avg_preds = torch.mean(preds, dim=0)
    return avg_preds.cpu()

if __name__ == '__main__':
    # --- 설정 ---
    CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    # CKPT_PATH = './models/0708-convnext_base-sz384-epoch=09-val_f1=0.9928.ckpt'
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'

    # --- 모델 로드 ---
    print("Loading model...")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    model.eval()
    print("Model loaded.")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="TTA Inference v2"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = predict_with_enhanced_tta(model, image, IMG_SIZE, DEVICE)
        final_predictions.append(torch.argmax(pred).item())

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_tta_v2_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] TTA v2 추론 결과 저장 → {output_filename}")

