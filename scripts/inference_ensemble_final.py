# scripts/inference_ensemble_final.py

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

def get_tta_transforms(img_size):
    """ 8가지 TTA(Test-Time Augmentation)를 위한 변환기 목록을 반환하는 함수 """
    
    # 1. 기본 변환 (0도)
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 2. 좌우반전 변환
    hflip_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return [base_transform, hflip_transform] # 더 복잡한 TTA는 여기에 추가 가능

def predict_with_tta(model, image, transforms, device):
    """ 주어진 모든 변환(TTA)을 적용하여 예측 확률의 평균을 반환하는 함수 """
    
    model.eval()
    all_preds = []
    with torch.no_grad():
        for transform in transforms:
            transformed_image = transform(image=image)['image'].unsqueeze(0).to(device)
            preds = model(transformed_image)
            all_preds.append(torch.softmax(preds, dim=1).cpu())
            
    # 모든 TTA 예측의 평균 확률을 계산
    avg_preds = torch.mean(torch.stack(all_preds), dim=0)
    return avg_preds

if __name__ == '__main__':
    
    # A팀 모델 정보
    CKPT_PATH_A = './models/tf_efficientnetv2_s-epoch=07-val_f1=0.9383.ckpt'
    IMG_SIZE_A = 512

    # B팀 모델 정보
    CKPT_PATH_B = './models/efficientnet_b1-epoch=09-val_f1=0.9224.ckpt'
    IMG_SIZE_B = 512

    # --- 공통 설정 ---
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 로드 ---
    print("Loading models...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    print("Models loaded successfully.")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    
    final_predictions = []
    for image_id in tqdm(submission_df['ID'], desc="Ensemble Inference (TTA)"):
        image_path = os.path.join(TEST_DIR, image_id)
        
        # 1. 이미지 읽기 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0) # 실패 시 기본값 0으로 예측
            continue
        try:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            grayscale = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(grayscale)
            rotated = io.rotate(denoised, angle, resize=True, cval=255) * 255
            image = rotated.astype(np.uint8)
        except Exception:
            image = cv2.imread(image_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 각 모델에 대한 TTA 예측 수행
        tta_transforms_a = get_tta_transforms(IMG_SIZE_A)
        tta_transforms_b = get_tta_transforms(IMG_SIZE_B)
        
        preds_a = predict_with_tta(model_a, image, tta_transforms_a, DEVICE)
        preds_b = predict_with_tta(model_b, image, tta_transforms_b, DEVICE)
        
        # 3. 두 모델의 예측 확률을 평균내어 최종 답 결정
        ensemble_preds = (preds_a + preds_b) / 2
        final_prediction = torch.argmax(ensemble_preds, dim=1).item()
        final_predictions.append(final_prediction)

    submission_df['target'] = final_predictions
    submission_df.to_csv('./submission_ensemble.csv', index=False)

    print("\nEnsemble TTA Inference finished! `submission_ensemble.csv` has been created.")