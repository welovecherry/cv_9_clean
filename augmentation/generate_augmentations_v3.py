# scripts/generate_augmentations_v3.py

import pandas as pd
import os
import cv2
import shutil
from tqdm import tqdm
import albumentations as A
import numpy as np
from datetime import datetime

# --- ⚙️ 설정 ---
RAW_DATA_PATH = './data/raw/'
timestamp = datetime.now().strftime("%m%d")
AUG_DATA_PATH = f'./data/augmented_v3_{timestamp}/'
NUM_AUGMENTATIONS_PER_IMAGE = 7  # 증강 수

def create_augmentations():
    # 📁 기존 폴더 정리 후 새로 생성
    aug_train_dir = os.path.join(AUG_DATA_PATH, 'train')
    if os.path.exists(AUG_DATA_PATH):
        shutil.rmtree(AUG_DATA_PATH)
    os.makedirs(aug_train_dir)
    
    # 📄 원본 이미지 및 CSV 불러오기
    original_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train.csv'))
    original_image_dir = os.path.join(RAW_DATA_PATH, 'train')
    
    # 🧪 v3 증강 파이프라인 (OCR 친화적, 자연스러운 왜곡 중심)
    ocr_friendly_transform = A.Compose([
        # ✅ 방향 왜곡 - 회전 + 좌우반전
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.6),
        A.HorizontalFlip(p=0.5),

        # ✅ 잘린 이미지 흉내내기
        A.RandomResizedCrop(height=512, width=512, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.3),

        # ✅ 자연스러운 흐림 (OCR 가능한 수준)
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),

        # ✅ 가벼운 압축 효과
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),

        # ✅ 약한 색상 변화
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        
        # ❌ 글자가 뭉개지거나 지워지는 효과는 제외
    ])

    augmented_data = []
    print(f"Generating {NUM_AUGMENTATIONS_PER_IMAGE} OCR-friendly augmented images...")

    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Augmenting Data (v3)"):
        image_name, label = row['ID'], row['target']
        image_path = os.path.join(original_image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            augmented = ocr_friendly_transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['image']
            
            base_name, ext = os.path.splitext(image_name)
            new_image_name = f"{base_name}_aug_v3_{i}{ext}"
            save_path = os.path.join(aug_train_dir, new_image_name)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            
            augmented_data.append([new_image_name, label])

    # 📊 새 train.csv 저장
    augmented_df = pd.DataFrame(augmented_data, columns=['ID', 'target'])
    augmented_df.to_csv(os.path.join(AUG_DATA_PATH, 'train.csv'), index=False)

    print(f"\n[완료] 총 {len(augmented_df)}장의 v3 증강 이미지가 생성되었습니다 → '{aug_train_dir}'")

if __name__ == '__main__':
    create_augmentations()