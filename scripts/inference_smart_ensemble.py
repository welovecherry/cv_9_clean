# scripts/inference_smart_ensemble.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule


def predict_with_smart_tta(model, image, img_size, device, n_tta=16):
    """
    [변경] TTA와 MC Dropout을 통합한 '스마트 TTA' 함수
    - 더 다양한 변환을 적당한 횟수만큼 적용
    """
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.Rotate(limit=15, p=0.4),
            A.GridDistortion(p=0.4),
            A.GaussianBlur(p=0.4),
        ], p=0.9), # 셋 중 하나를 90% 확률로 적용
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    tta_images = [transform(image=image)['image'] for _ in range(n_tta)]
    batch = torch.stack(tta_images).to(device)

    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)

    avg_preds = torch.mean(preds, dim=0)
    return avg_preds.cpu()

if __name__ == '__main__':
    # --- 설정 ---
    CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'

    # [추가] 후처리 규칙
    CORRECTION_RULES = {
        13: {12: 0.1, 14: 0.15},
        14: {13: 0.1, 4: 0.05},
        12: {13: 0.05, 14: 0.05},
        10: {14: 0.1},
        3: {4: 0.1},
        4: {3: 0.1},
    }

    # --- 모델 로드 ---
    print("Loading model...")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    
    # [추가] MC Dropout을 위해 train 모드로 설정
    model.train()
    print("Model loaded in train mode for MC Dropout.")


    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="Smart Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            # 이미지가 없는 경우, 가장 흔한 클래스 또는 0으로 예측
            final_predictions.append(0) 
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # '스마트 TTA' 함수로 예측값(softmax 확률) 얻기
        pred_vector = predict_with_smart_tta(model, image, IMG_SIZE, DEVICE)
        
        # [추가] 후처리 규칙 적용
        pred_class = torch.argmax(pred_vector).item()
        if pred_class in CORRECTION_RULES:
            correction_amount = pred_vector[pred_class]
            for target_class, weight in CORRECTION_RULES[pred_class].items():
                pred_vector[target_class] += correction_amount * weight
        
        # 최종 클래스 결정
        final_pred = torch.argmax(pred_vector).item()
        final_predictions.append(final_pred)

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_smart_ensemble_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] 스마트 앙상블 추론 결과 저장 → {output_filename}")