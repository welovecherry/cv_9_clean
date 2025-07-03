# scripts/inference_tta.py
import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lightning_model import CustomLightningModule 

class TestDataset(Dataset):
    # ... (이전 inference.py와 동일) ...
    def __init__(self, data_root, filenames, transform=None):
        self.data_root = data_root
        self.filenames = filenames
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.filenames[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, self.filenames[idx]

if __name__ == '__main__':
    # --- 설정 ---
    CKPT_PATH = './models/efficientnet_b1-epoch=04-val_f1=0.9133.ckpt' # 우리의 챔피언 모델
    TEST_DIR = './data/processed/test_cleaned/' # 깨끗해진 테스트 데이터 폴더
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    IMG_SIZE = 224
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- TTA를 위한 변환기 ---
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0), # 100% 좌우반전
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- 모델 로드 ---
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH)
    model.to(DEVICE)
    model.eval()

    # --- 데이터로더 준비 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    test_filenames = submission_df['ID'].values
    
    dataset = TestDataset(TEST_DIR, filenames=test_filenames, transform=transform)
    tta_dataset = TestDataset(TEST_DIR, filenames=test_filenames, transform=tta_transform)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    tta_dataloader = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 추론 실행 (원본 + TTA) ---
    predictions = []
    with torch.no_grad():
        # 원본 이미지 예측
        original_preds = []
        for images, _ in tqdm(dataloader, desc="Inferencing Original"):
            images = images.to(DEVICE)
            preds = model(images)
            original_preds.append(torch.softmax(preds, dim=1).cpu())
        
        # TTA(좌우반전) 이미지 예측
        tta_preds = []
        for images, _ in tqdm(tta_dataloader, desc="Inferencing TTA"):
            images = images.to(DEVICE)
            preds = model(images)
            tta_preds.append(torch.softmax(preds, dim=1).cpu())

    # 예측 결과 합치기
    original_preds = torch.cat(original_preds, dim=0)
    tta_preds = torch.cat(tta_preds, dim=0)
    
    # 두 예측의 평균을 내고, 가장 확률이 높은 클래스를 최종 예측으로 선택
    final_preds = torch.argmax(original_preds + tta_preds, dim=1)

    # --- 제출 파일 생성 ---
    submission_df['target'] = final_preds.numpy()
    submission_df.to_csv('./submission_tta.csv', index=False)
    print("\nInference with TTA finished! `submission_tta.csv` file has been created.")