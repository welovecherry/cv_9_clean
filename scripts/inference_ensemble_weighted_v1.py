# 가중치 기반 앙상블하기 3:1:1

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule

# --- 설정 ---
CKPT_PATHS = [
    ("./models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt", 3),
    ("./models/finetuned-tf_efficientnetv2_s-epoch=15-val_f1=0.9551.ckpt", 1),
    ("./models/resnet101-epoch=17-val_f1=0.9535.ckpt", 1),
]
IMG_SIZE = 384
TEST_DIR = './data/raw/test/'
SUBMISSION_FILE = './data/raw/sample_submission.csv'
OUTPUT_CSV = './submission.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 이미지 전처리 ---
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def load_models(ckpt_paths):
    models = []
    weights = []
    for path, weight in ckpt_paths:
        model = CustomLightningModule.load_from_checkpoint(path)
        model.to(DEVICE)
        model.eval()
        models.append(model)
        weights.append(weight)
    return models, weights

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    return image.unsqueeze(0).to(DEVICE)

def predict(models, weights, image_tensor):
    with torch.no_grad():
        total_weight = sum(weights)
        weighted_sum = None
        for model, w in zip(models, weights):
            outputs = torch.softmax(model(image_tensor), dim=1)
            if weighted_sum is None:
                weighted_sum = w * outputs
            else:
                weighted_sum += w * outputs
        return torch.argmax(weighted_sum / total_weight, dim=1).item()

if __name__ == '__main__':
    submission_df = pd.read_csv(SUBMISSION_FILE)
    models, weights = load_models(CKPT_PATHS)

    predictions = []
    for image_id in tqdm(submission_df['ID'], desc="Weighted Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image_tensor = preprocess_image(image_path)
        pred = predict(models, weights, image_tensor)
        predictions.append(pred)

    submission_df['target'] = predictions
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Inference 완료! 결과 저장됨: {OUTPUT_CSV}")