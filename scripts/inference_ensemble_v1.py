# scripts/inference_ensemble_v1.py

import os
import torch
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

# 경로 설정
TEST_DIR = './data/raw/test/'
SUBMISSION_FILE = './data/raw/sample_submission.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384  # 공통 크기

# 모델 경로
CKPT_PATHS = [
    './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt',
    './models/finetuned-tf_efficientnetv2_s-epoch=15-val_f1=0.9551.ckpt',
    './models/resnet101-epoch=17-val_f1=0.9535.ckpt',
]

# 모델 클래스 불러오기
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule

# 이미지 전처리 정의
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 모델 로드 함수
def load_model(ckpt_path):
    model = CustomLightningModule.load_from_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval()
    return model

# 추론용 데이터셋 클래스
class TestDataset(Dataset):
    def __init__(self, image_dir, image_ids, transform):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image, image_id

if __name__ == '__main__':
    # 모델 여러 개 로드
    models = [load_model(path) for path in CKPT_PATHS]

    # 테스트 이미지 불러오기
    submission = pd.read_csv(SUBMISSION_FILE)
    image_ids = submission['ID'].tolist()
    test_dataset = TestDataset(TEST_DIR, image_ids, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    all_predictions = []

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Inference Ensemble"):
            images = images.to(DEVICE)

            # 모든 모델의 softmax 결과 합산
            total_preds = None
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                if total_preds is None:
                    total_preds = probs
                else:
                    total_preds += probs
            
            # 평균 확률에서 argmax
            final_preds = torch.argmax(total_preds, dim=1)
            all_predictions.extend(final_preds.cpu().numpy())

    # 예측 결과 저장
    submission['target'] = all_predictions
    submission.to_csv('./submission_ensemble_v1.csv', index=False)
    print("✅ 앙상블 제출 파일이 'submission_ensemble_v1.csv'로 저장되었습니다.")