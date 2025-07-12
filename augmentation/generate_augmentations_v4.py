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
AUG_DATA_PATH = f'./data/augmented_v4_{timestamp}/'
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

    # 🧪 v4 증강 파이프라인 (시각적 저하 + 다양한 왜곡)
    degraded_transform = A.Compose([
        # ✅ 흐림 및 안개
        A.OneOf([
            A.GaussianBlur(blur_limit=(5, 9), p=1.0),
            A.MotionBlur(blur_limit=(5, 9), p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.4),

        # ✅ 노이즈
        A.GaussNoise(var_limit=(20.0, 50.0), mean=0, p=0.3),

        # ✅ 색상 변화
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),

        # ✅ 왜곡 계열
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),
        ], p=0.4),

        # ✅ 회전 및 반전
        A.OneOf([
            A.Rotate(limit=10, p=1.0),
            A.Rotate(limit=30, p=1.0),
            A.Rotate(limit=60, p=1.0),
            A.Rotate(limit=90, p=1.0),
        ], p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # ✅ 랜덤 크롭 & 리사이즈
        A.RandomResizedCrop(height=512, width=512, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.3),
        
        # ✅ 압축
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.2),
    ])

    augmented_data = []
    print(f"Generating {NUM_AUGMENTATIONS_PER_IMAGE} degraded augmented images...")

    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Augmenting Data (v4)"):
        image_name, label = row['ID'], row['target']
        image_path = os.path.join(original_image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            augmented = degraded_transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['image']
            base_name, ext = os.path.splitext(image_name)
            new_image_name = f"{base_name}_aug_v4_{i}{ext}"
            save_path = os.path.join(aug_train_dir, new_image_name)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            augmented_data.append([new_image_name, label])

    # 📊 새 train.csv 저장
    augmented_df = pd.DataFrame(augmented_data, columns=['ID', 'target'])
    augmented_df.to_csv(os.path.join(AUG_DATA_PATH, 'train.csv'), index=False)

    print(f"\n[완료] 총 {len(augmented_df)}장의 v4 증강 이미지가 생성되었습니다 → '{aug_train_dir}'")

if __name__ == '__main__':
    create_augmentations()