# scripts/generate_augmentations_v2.py

import pandas as pd
import os
import cv2
import shutil
from tqdm import tqdm
import albumentations as A
import augraphy
import numpy as np
from datetime import datetime

# 경로 설정
RAW_DATA_PATH = './data/raw/'
timestamp = datetime.now().strftime("%m%d")
AUG_DATA_PATH = f'./data/augmented_v2_{timestamp}/'
NUM_AUGMENTATIONS_PER_IMAGE = 7

# 간단한 Augraphy 파이프라인
# def apply_augraphy_effect(image):
#     pipeline = augraphy.default_augraphy_pipeline()
#     result = pipeline(image)
#     return result["output"]
def apply_augraphy_effect(image):
    pipeline = augraphy.default_augraphy_pipeline()
    result = pipeline(image)  # 결과는 np.ndarray
    return result

def create_augmentations():
    aug_train_dir = os.path.join(AUG_DATA_PATH, 'train')
    if os.path.exists(AUG_DATA_PATH):
        shutil.rmtree(AUG_DATA_PATH)
    os.makedirs(aug_train_dir)

    original_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'train.csv'))
    original_image_dir = os.path.join(RAW_DATA_PATH, 'train')

    albumentations_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Perspective(scale=(0.03, 0.08), p=0.2),
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    ])

    augmented_data = []
    print(f"Generating {NUM_AUGMENTATIONS_PER_IMAGE} v2 augmented images...")

    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Augmenting Data (v2)"):
        image_name, label = row['ID'], row['target']
        image_path = os.path.join(original_image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            alb_image = albumentations_transform(image=image_rgb)['image']
            aug_image = apply_augraphy_effect(alb_image)

            base_name, ext = os.path.splitext(image_name)
            new_image_name = f"{base_name}_aug_v2_{i}{ext}"
            save_path = os.path.join(aug_train_dir, new_image_name)
            cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            augmented_data.append([new_image_name, label])

    augmented_df = pd.DataFrame(augmented_data, columns=['ID', 'target'])
    augmented_df.to_csv(os.path.join(AUG_DATA_PATH, 'train.csv'), index=False)

    print(f"\n✅ v2 증강 완료! 총 {len(augmented_df)}개 이미지 생성됨 → '{aug_train_dir}'")

if __name__ == '__main__':
    create_augmentations()