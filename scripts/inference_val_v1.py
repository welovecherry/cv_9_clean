import pandas as pd
import torch
import cv2
import os
from tqdm import tqdm
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule

# 설정
CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'  # 모델 경로
IMG_SIZE = 384
VAL_CSV_PATH = './data/final_training_data/val.csv'  # 이전 단계에서 만든 파일
IMAGE_DIR = './data/final_training_data/train/'  # 이미지 폴더
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 전처리 정의
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 모델 로드
model = CustomLightningModule.load_from_checkpoint(CKPT_PATH)
model.to(DEVICE)
model.eval()

# val.csv 로드
df = pd.read_csv(VAL_CSV_PATH)

# 예측 결과 저장 리스트
predictions = []

# 예측 루프
for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting val images"):
    image_path = os.path.join(IMAGE_DIR, row['ID'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    
    predictions.append(pred)

# 결과 저장
df['pred'] = predictions
save_path = './data/final_training_data/val_preds.csv'
df.to_csv(save_path, index=False)
print(f"Saved prediction results to: {save_path}")