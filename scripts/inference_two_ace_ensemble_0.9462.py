# scripts/inference_two_ace_ensemble.py
# 리더보드 최고점 인퍼런스 파일
'''
1. 2개 모델 앙상블 (convnext_base (val_f1=0.9950) + tf_efficientnetv2_s (val_f1=0.9955))
2. 두 모델을 소프트맥스로 단순 평균함

'''

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

def predict_with_simple_tta(model, image, img_size, device):
    """
    [변경] 빠르고 간단한 TTA 함수
    """
    tta_images = []
    # 기본 회전 및 좌우 반전 (총 5개)
    # 이미지 회전 0, 90, 180, 270도(4개)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    # 좌우 반전 추가
    flipped = cv2.flip(image, 1)
    tta_images.append(flipped.copy())

    # 이미지 전처리
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 정규화
        ToTensorV2() # 텐서 변환
    ])
    # TTA 이미지들을 변환하여 배치로 묶기
    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)

    model.eval() # 평가 모드로 설정
    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1) # 소프트맥스 적용

    avg_preds = torch.mean(preds, dim=0)
    return avg_preds.cpu()

if __name__ == '__main__':
    # --- 설정 ---
    # 두 개의 최고 모델 경로 설정
    CKPT_PATH_A = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    CKPT_PATH_B = './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt'
    
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'

    # confusion matrix를 기반으로 한 후처리 규칙 정함.
    # 후처리 규칙 (혼동되는 클래스 간 신뢰도 미세 보정)
    CORRECTION_RULES = {
        13: {12: 0.1, 14: 0.15}, # 예측이 13일 때, 12와 14의 확률을 조정 (12에는 0.1, 14에는 0.15를 더함)
        14: {13: 0.1, 4: 0.05},
        12: {13: 0.05, 14: 0.05},
        10: {14: 0.1},
        3: {4: 0.1},
        4: {3: 0.1},
    }

    # --- 모델 로드 ---
    print("Loading Two Ace models...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    print("Models loaded.")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="Two-Ace Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 각 모델로 예측 수행
        pred_a = predict_with_simple_tta(model_a, image, IMG_SIZE, DEVICE)
        pred_b = predict_with_simple_tta(model_b, image, IMG_SIZE, DEVICE)
        
        # [핵심] 두 모델의 예측 확률을 평균
        ensembled_pred_vector = (pred_a + pred_b) / 2.0
        
        # 후처리 규칙 적용
        pred_class = torch.argmax(ensembled_pred_vector).item()
        if pred_class in CORRECTION_RULES:
            correction_amount = ensembled_pred_vector[pred_class]
            for target_class, weight in CORRECTION_RULES[pred_class].items():
                ensembled_pred_vector[target_class] += correction_amount * weight
        
        # 최종 클래스 결정
        final_pred = torch.argmax(ensembled_pred_vector).item()
        final_predictions.append(final_pred)

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_two_ace_ensemble_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] Two-Ace 앙상블 추론 결과 저장 → {output_filename}")