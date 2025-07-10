# scripts/inference_3musketeers_ensemble.py

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

if __name__ == '__main__':
    # --- 설정 ---
    CKPT_PATH_A = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    CKPT_PATH_B = './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt'
    CKPT_PATH_C = './models/auto-resnet101-lr1e-4-sz384-epoch=19-val_f1=0.9839.ckpt'
    
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    
    # --- 모델 로드 ---
    print("Loading Three Musketeers models...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    model_c = CustomLightningModule.load_from_checkpoint(CKPT_PATH_C).to(DEVICE)
    
    model_a.eval()
    model_b.eval()
    model_c.eval()
    
    print("Models loaded.")

    # TTA 없이 단순 추론을 위한 변환
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE), # [수정] 변수명을 IMG_SIZE (대문자)로 수정
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="3 Musketeers Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed_image = transform(image=image)['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_a = torch.softmax(model_a(transformed_image), dim=1)
            pred_b = torch.softmax(model_b(transformed_image), dim=1)
            pred_c = torch.softmax(model_c(transformed_image), dim=1)
        
        ensembled_pred_vector = (pred_a + pred_b + pred_c) / 3.0
        
        final_pred = torch.argmax(ensembled_pred_vector).item()
        final_predictions.append(final_pred)

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_3musketeers_ensemble_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] 3 Musketeers 앙상블 추론 결과 저장 → {output_filename}")