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
AUG_DATA_PATH = f'./data/augmented_v5_{timestamp}/'
NUM_AUGMENTATIONS_PER_IMAGE = 12  # 증강 수

def create_augmentations():
    # 📁 기존 폴더 정리 후 새로 생성
    aug_train_dir = os.path.join(AUG_DATA_PATH, 'train')
    if os.path.exists(AUG_DATA_PATH):
        shutil.rmtree(AUG_DATA_PATH)
    os.makedirs(aug_train_dir)

    # 📄 원본 이미지 및 CSV 불러오기
    original_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train.csv'))
    original_image_dir = os.path.join(RAW_DATA_PATH, 'train')

    # 🎨 증강 조합 정의 (모든 변형에 줌인 효과 추가)
    transforms = [
        A.Compose([A.Rotate(limit=(180, 180), p=1.0), A.RandomFog(p=1.0), A.HorizontalFlip(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), A.ImageCompression(quality_lower=40, quality_upper=70, p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(30, 60), p=1.0), A.MotionBlur(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.7, 0.85), p=1.0)]),
        A.Compose([A.Rotate(limit=(45, 45), p=1.0), A.ElasticTransform(p=1.0), A.VerticalFlip(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(15, 15), p=1.0), A.GaussianBlur(p=1.0), A.HorizontalFlip(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(75, 75), p=1.0), A.GridDistortion(p=1.0), A.RandomFog(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(5, 5), p=1.0), A.GaussNoise(p=1.0), A.ImageCompression(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(12, 12), p=1.0), A.Blur(p=1.0), A.RandomBrightnessContrast(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(66, 66), p=1.0), A.GaussianBlur(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.75, 0.9), p=1.0)]),
        A.Compose([A.Rotate(limit=(23, 23), p=1.0), A.Blur(p=1.0), A.RandomFog(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(38, 38), p=1.0), A.GaussNoise(p=1.0), A.ElasticTransform(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.85, 1.0), p=1.0)]),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), A.Blur(p=1.0), A.GaussianBlur(p=1.0), A.RandomResizedCrop(512, 512, scale=(0.7, 0.85), p=1.0)]),
    ]

    augmented_data = []
    print(f"Generating {NUM_AUGMENTATIONS_PER_IMAGE} augmented images per input...")

    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Augmenting Data (v5)"):
        image_name, label = row['ID'], row['target']
        image_path = os.path.join(original_image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        for i, transform in enumerate(transforms):
            augmented = transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['image']
            base_name, ext = os.path.splitext(image_name)
            new_image_name = f"{base_name}_aug_v5_{i}{ext}"
            save_path = os.path.join(aug_train_dir, new_image_name)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            augmented_data.append([new_image_name, label])

    # 📊 새 train.csv 저장
    augmented_df = pd.DataFrame(augmented_data, columns=['ID', 'target'])
    augmented_df.to_csv(os.path.join(AUG_DATA_PATH, 'train.csv'), index=False)

    print(f"\n[완료] 총 {len(augmented_df)}장의 v5 증강 이미지가 생성되었습니다 → '{aug_train_dir}'")

if __name__ == '__main__':
    create_augmentations()