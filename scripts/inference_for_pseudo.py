# scripts/inference_for_pseudo.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lightning_model import CustomLightningModule 

if __name__ == '__main__':
    # --- ⚙️ 설정: 가장 신뢰도 높은 단일 챔피언 모델 ---
    
    # ConvNeXt가 가장 안정적이고 높은 점수를 기록했으니, 이 모델을 사용하자.
    CKPT_PATH = './models/convnext_base-epoch=06-val_f1=0.9585.ckpt' 
    IMG_SIZE = 512
    
    # --- 공통 설정 ---
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 로드 ---
    print(f"Loading model: {os.path.basename(CKPT_PATH)}")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # --- 추론 준비 ---
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    submission_df = pd.read_csv(SUBMISSION_FILE)
    
    predictions = []
    confidences = []

    for image_id in tqdm(submission_df['ID'], desc="Generating predictions with confidence"):
        image_path = os.path.join(TEST_DIR, image_id)
        
        image = cv2.imread(image_path)
        if image is None:
            predictions.append(0)
            confidences.append(0) # 실패 시 확신도 0
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_tensor = transform(image=image)['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(img_tensor)
            # [핵심] 소프트맥스를 적용해 각 클래스별 확률(확신도)을 계산
            probs = torch.softmax(preds, dim=1)
            
            # 가장 확률이 높은 값(확신도)과 그 인덱스(예측 클래스)를 가져옴
            confidence, prediction = torch.max(probs, dim=1)

            predictions.append(prediction.item())
            confidences.append(confidence.item())

    # --- '신뢰도 점수'가 포함된 결과 파일 생성 ---
    pseudo_label_df = pd.DataFrame({
        "ID": submission_df['ID'],
        "target": predictions,
        "confidence": confidences
    })
    
    output_filename = './pseudo_label_candidates.csv'
    pseudo_label_df.to_csv(output_filename, index=False)

    print(f"\nPseudo-labeling candidates created! File saved as `{output_filename}`")