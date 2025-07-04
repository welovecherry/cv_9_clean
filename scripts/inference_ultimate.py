# scripts/inference_ultimate.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lightning_model import CustomLightningModule 
from skimage import io
from deskew import determine_skew

def predict_with_8_way_tta(model, image, img_size, device):
    """ 8방향 TTA를 적용하여 예측 확률의 평균을 반환하는 함수 """
    
    # 8가지 버전의 이미지를 담을 리스트
    tta_images = []
    
    # 0, 90, 180, 270도 회전
    for k in [0, 1, 2, 3]: # np.rot90의 k 파라미터는 90도 회전 횟수
        rotated_img = np.rot90(image, k=k).copy()
        tta_images.append(rotated_img)

    # 원본 이미지를 좌우반전 시킨 뒤, 4가지 방향으로 회전
    flipped_img = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        rotated_flipped_img = np.rot90(flipped_img, k=k).copy()
        tta_images.append(rotated_flipped_img)

    # 기본 변환기 (리사이즈 및 정규화)
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 8개의 이미지를 하나의 배치로 만들어 한 번에 예측 (GPU 효율성 극대화)
    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)
    
    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)
    
    # 8개 예측 결과의 평균을 계산
    avg_preds = torch.mean(preds, dim=0)
    return avg_preds.cpu()

if __name__ == '__main__':
    # --- 설정: 두 에이스 모델의 정보와 가중치 ---
    
    # A팀 (챔피언)
    CKPT_PATH_A = './models/tf_efficientnetv2_s-epoch=07-val_f1=0.9383.ckpt'
    IMG_SIZE_A = 512
    WEIGHT_A = 0.55 # A팀의 예측에 55% 가중치

    # B팀 (파트너)
    CKPT_PATH_B = './models/efficientnet_b1-epoch=09-val_f1=0.9224.ckpt'
    IMG_SIZE_B = 512
    WEIGHT_B = 0.45 # B팀의 예측에 45% 가중치
    
    # --- 공통 설정 ---
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 로드 ---
    print("Loading models for Ensemble...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    model_a.eval()
    model_b.eval()
    print("Models loaded successfully.")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="Ultimate Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue
        
        # 전처리는 적용하지 않음 (TTA가 회전을 처리하므로)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 각 모델에 대한 8방향 TTA 예측 수행
        preds_a = predict_with_8_way_tta(model_a, image, IMG_SIZE_A, DEVICE)
        preds_b = predict_with_8_way_tta(model_b, image, IMG_SIZE_B, DEVICE)
        
        # 가중치를 적용하여 두 모델의 예측 확률을 합침
        ensemble_preds = (preds_a * WEIGHT_A) + (preds_b * WEIGHT_B)
        final_prediction = torch.argmax(ensemble_preds, dim=0).item()
        final_predictions.append(final_prediction)

    # --- 제출 파일 생성 ---
    submission_df['target'] = final_predictions
    submission_df.to_csv('./submission_ultimate.csv', index=False)
    print("\nUltimate Inference finished! `submission_ultimate.csv` has been created.")