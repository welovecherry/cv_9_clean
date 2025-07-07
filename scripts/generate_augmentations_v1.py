# scripts/generate_augmentations_v1.py

import pandas as pd
import os
import cv2
import shutil
from tqdm import tqdm
import albumentations as A
import augraphy
from augraphy import default_augraphy_pipeline
import numpy as np
from datetime import datetime

# --- ⚙️ 설정 ---
RAW_DATA_PATH = './data/raw/'
timestamp = datetime.now().strftime("%m%d")
AUG_DATA_PATH = f'./data/augmented_v1_{timestamp}/'
NUM_AUGMENTATIONS_PER_IMAGE = 7

def create_augmentations():
    aug_train_dir = os.path.join(AUG_DATA_PATH, 'train')
    if os.path.exists(aug_train_dir):
        shutil.rmtree(aug_train_dir)
    os.makedirs(aug_train_dir)
    
    # 원본 CSV와 이미지 폴더 읽기
    original_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train.csv'))
    original_image_dir = os.path.join(RAW_DATA_PATH, 'train')
    
    # 1. Albumentations 증강 설정
    albumentations_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.15, rotate_limit=20, p=0.6), # 이동, 크기조정, 회전
        A.Perspective(scale=(0.05, 0.1), p=0.4), # 원근감 왜곡
        A.HorizontalFlip(p=0.5), # 좌우 반전
        A.OneOf([ # 블러, 노이즈, 압축 중 하나 선택
            A.GaussianBlur(p=1.0),
            A.GaussNoise(p=1.0),
            A.ImageCompression(quality_lower=70, p=1.0),
        ], p=0.5), 
        A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5), # 밝기/대비 조절
    ])

    # 2. Augraphy 기본 파이프라인 (문서 이미지 특화 증강)
    augraphy_transform = default_augraphy_pipeline()
    augmented_data = []
    print(f"Generating {NUM_AUGMENTATIONS_PER_IMAGE} augmented images...")

    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Augmenting Data"):
        image_name = row['ID']
        label = row['target']
        image_path = os.path.join(original_image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None: continue
            
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1차 변환: Albumentations
            augmented_image_alb = albumentations_transform(image=image_rgb)['image']
            
            # 2차 변환: Augraphy
            augmented_result = augraphy_transform(image=augmented_image_alb)
            # augmented_image_aug = augmented_result["output"]
            augmented_image_aug = augraphy_transform(image=augmented_image_alb)

            # 저장
            base_name, ext = os.path.splitext(image_name)
            new_image_name = f"{base_name}_aug_{i}{ext}"
            save_path = os.path.join(aug_train_dir, new_image_name)
            cv2.imwrite(save_path, cv2.cvtColor(augmented_image_aug, cv2.COLOR_RGB2BGR))
            
            augmented_data.append([new_image_name, label])

    augmented_df = pd.DataFrame(augmented_data, columns=['ID', 'target'])
    augmented_df.to_csv(os.path.join(AUG_DATA_PATH, 'train.csv'), index=False)

    print(f"\nOffline data augmentation finished! A total of {len(augmented_df)} images created in '{aug_train_dir}'")

if __name__ == '__main__':
    create_augmentations()