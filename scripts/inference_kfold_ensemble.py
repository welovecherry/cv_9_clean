# scripts/inference_kfold_ensemble.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lightning_model import CustomLightningModule
from datetime import datetime

def predict_with_tta(model, image, img_size, device):
    """ TTA를 적용하여 예측 확률의 평균을 반환하는 함수 """
    tta_images = []
    # 간단하고 안정적인 2-way TTA (원본 + 좌우반전) 사용
    transform = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
    tta_transform = A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()])

    original_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    tta_tensor = tta_transform(image=image)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_original = torch.softmax(model(original_tensor), dim=1)
        pred_tta = torch.softmax(model(tta_tensor), dim=1)

    avg_preds = (pred_original + pred_tta) / 2
    return avg_preds.cpu()

if __name__ == '__main__':
    # --- ⚙️ 설정: K-Fold로 훈련된 5명의 에이스 ---
    
    MODEL_PATHS = [
        './models/kfold-convnext_base-fold0-epoch=13-val_f1=0.9657.ckpt',
        './models/kfold-convnext_base-fold1-epoch=12-val_f1=0.9681.ckpt',
        './models/kfold-convnext_base-fold2-epoch=12-val_f1=0.9623.ckpt',
        './models/kfold-convnext_base-fold3-epoch=17-val_f1=0.9502.ckpt',
        './models/kfold-convnext_base-fold4-epoch=16-val_f1=0.9424.ckpt'
    ]
    
    # 이 모델들은 모두 384 사이즈로 학습했음
    IMG_SIZE = 384 
    
    # --- 공통 설정 ---
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32 # 추론 시에는 배치 사이즈를 키워도 괜찮음

    # --- 모델 로드 ---
    print("Loading 5 K-Fold models...")
    models = [CustomLightningModule.load_from_checkpoint(path).to(DEVICE) for path in MODEL_PATHS]
    for model in models:
        model.eval()
    print("All models assembled for K-Fold Ensemble.")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="K-Fold Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 각 모델의 예측 확률을 담을 리스트
        all_model_preds = []
        for model in models:
            # 각 모델에 대해 TTA 예측 수행
            pred_probs = predict_with_tta(model, image, IMG_SIZE, DEVICE)
            all_model_preds.append(pred_probs)
        
        # 5개 모델의 예측 확률을 모두 합쳐서 평균
        ensemble_probs = torch.mean(torch.stack(all_model_preds), dim=0)
        final_prediction = torch.argmax(ensemble_probs, dim=1).item()
        final_predictions.append(final_prediction)

    # --- 제출 파일 생성 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_kfold_ensemble_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\nK-Fold Ensemble submission created: `{output_filename}`")