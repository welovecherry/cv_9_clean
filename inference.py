# 제출용 csv 파일을 생성하는 코드
# scripts/inference.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 학습 코드에 있는 모델 클래스를 그대로 가져와야 함
from src.lightning_model import CustomLightningModule 

# [수정] 테스트 데이터셋이 파일명 리스트를 직접 받도록 변경
class TestDataset(Dataset):
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
    # --- 설정 부분 ---
    CKPT_PATH = 'models/efficientnet_b1-epoch=04-val_f1=0.9133.ckpt' # 챔피언 모델 경로
    
    # [수정] raw 데이터 폴더와 sample_submission 파일 경로
    DATA_DIR = './data/raw/'
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
    
    IMG_SIZE = 224 # 최종 학습시킨 모델의 이미지 크기
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- 추론 과정 ---
    # 1. 모델 로드
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH)
    model.to(DEVICE)
    model.eval()

    # 2. [수정] sample_submission.csv를 읽어서 파일 순서를 가져옴
    submission_df = pd.read_csv(SUBMISSION_FILE)
    test_filenames = submission_df['ID'].values

    # 3. 데이터로더 준비 (파일명 리스트를 직접 전달)
    test_dataset = TestDataset(TEST_DIR, filenames=test_filenames, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # 4. 예측 실행
    predictions = []
    with torch.no_grad():
        for images, fns in tqdm(test_dataloader, desc="Inferencing"):
            images = images.to(DEVICE)
            preds = model(images)
            preds = torch.argmax(preds, dim=1)
            predictions.extend(preds.cpu().numpy())

    # 5. [수정] 읽어온 submission_df에 예측값을 채워넣음
    submission_df['target'] = predictions
    submission_df.to_csv('./submission.csv', index=False)

    print("\nInference finished! `submission.csv` file has been created with the correct order.")