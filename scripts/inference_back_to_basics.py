# scripts/inference_back_to_basics.py

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

def predict_with_8_way_tta(model, image, img_size, device):
    """ 8방향 TTA를 적용하여 예측 확률의 평균을 반환하는 함수 """
    tta_images = []
    
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())

    flipped_img = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped_img, k=k).copy())

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

if __name__ == '__main__':
    
    # A팀 (tf_efficientnetv2_s 챔피언)
    # CKPT_PATH_A = './models/retrain-weighted-loss-tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt'
    CKPT_PATH_A = './models/convnext_base-epoch=06-val_f1=0.9585.ckpt'
    IMG_SIZE_A = 512

    # B팀 (ConvNeXt 챔피언)
    # CKPT_PATH_B = './models/convnext_base-epoch=17-val_f1=0.9564.ckpt'
    CKPT_PATH_B = './models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt'
    IMG_SIZE_B = 512
    
    # --- 공통 설정 ---
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 로드 ---
    print("Loading Back-to-Basics models...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    model_a.eval()
    model_b.eval()
    print("Models loaded successfully.")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="Simple Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue
        
        # [핵심] Denoise, Deskew 등 복잡한 전처리 없이, 원본 이미지를 바로 사용
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preds_a = predict_with_8_way_tta(model_a, image, IMG_SIZE_A, DEVICE)
        preds_b = predict_with_8_way_tta(model_b, image, IMG_SIZE_B, DEVICE)
        
        # 단순 평균 앙상블
        ensemble_preds = (preds_a + preds_b) / 2
        final_prediction = torch.argmax(ensemble_preds, dim=0).item()
        final_predictions.append(final_prediction)

    # --- 제출 파일 생성 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_simple_ensemble_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\nSimple Ensemble Inference finished! `{output_filename}` has been created.")