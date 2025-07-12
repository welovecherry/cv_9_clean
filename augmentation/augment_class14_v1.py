# # scripts/augment_class14_v1.py 로 저장
# import os
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import uuid

# SRC_DIR = './data/augmented/class14_original'
# OUT_DIR = './data/augmented/class14_augmented'
# os.makedirs(OUT_DIR, exist_ok=True)

# # 증강 파이프라인 정의 (자연스러운 변형)
# transform = A.Compose([
#     A.RandomBrightnessContrast(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.Rotate(limit=15, p=0.7),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
#     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#     A.RandomShadow(p=0.3),
#     A.Resize(384, 384)
# ])

# for img_name in os.listdir(SRC_DIR):
#     img_path = os.path.join(SRC_DIR, img_name)
#     image = cv2.imread(img_path)

#     base_name = os.path.splitext(img_name)[0]
#     # 원본도 1장 복사
#     cv2.imwrite(os.path.join(OUT_DIR, f'{base_name}_orig.jpg'), image)

#     # 4장 증강 = 총 5장
#     for i in range(4):
#         augmented = transform(image=image)['image']
#         out_name = f'{base_name}_aug_{i}.jpg'
#         cv2.imwrite(os.path.join(OUT_DIR, out_name), augmented)


# scripts/augment_class14_v1.py

import os
import cv2
import random
from tqdm import tqdm
from glob import glob
from albumentations import (
    Compose, Rotate, GaussianBlur, RandomBrightnessContrast,
    ShiftScaleRotate, OneOf, RandomShadow, RandomFog, RandomSunFlare,
    GaussNoise, MotionBlur, ZoomBlur
)
# from albumentations.augmentations.transforms import RandomCrop

# 1. 경로 설정
INPUT_DIR = 'data/train_split/14_statement_of_opinion'
OUTPUT_DIR = 'data/augmented_only/class14_augmented'
NUM_AUGS = 5  # 한 이미지당 5배 증강

# 2. 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 증강 파이프라인 구성
transform = Compose([
    OneOf([
        Rotate(limit=40, p=0.7),         # 더 다양한 회전
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5)
    ], p=1.0),
    
    OneOf([
        GaussianBlur(blur_limit=(3, 7), p=0.5),   # 흐림 강도 ↑
        MotionBlur(p=0.5),
        ZoomBlur(max_factor=1.2, p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    ], p=0.7),

    RandomBrightnessContrast(p=0.5),

    OneOf([
        RandomShadow(p=0.3),
        RandomFog(p=0.3),
        RandomSunFlare(p=0.2)
    ], p=0.3),

    # 약간의 줌인으로 데칼코마니 방지
    ShiftScaleRotate(scale_limit=(0.1, 0.2), rotate_limit=0, shift_limit=0, p=0.5)
])

# 4. 이미지 증강 실행
image_paths = glob(os.path.join(INPUT_DIR, '*.jpg'))

for img_path in tqdm(image_paths, desc="Augmenting class 14 images"):
    img = cv2.imread(img_path)
    if img is None:
        continue

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(NUM_AUGS):
        augmented = transform(image=img)['image']
        save_path = os.path.join(OUTPUT_DIR, f'{base_name}_aug_{i}.jpg')
        cv2.imwrite(save_path, augmented)


# import os
# import cv2
# import random
# from glob import glob
# from tqdm import tqdm
# import albumentations as A

# # 1. 입력 디렉토리와 출력 디렉토리 설정
# INPUT_DIR = "data/train_split/14_statement_of_opinion"
# OUTPUT_DIR = "data/augmented/class14_augmented"
# NUM_AUG_PER_IMAGE = 5  # 이미지당 생성할 증강 수
# IMG_SIZE = (512, 512)  # resize 할 최종 이미지 크기

# # 2. 출력 디렉토리 비우기 및 새로 생성
# if os.path.exists(OUTPUT_DIR):
#     for f in glob(os.path.join(OUTPUT_DIR, '*')):
#         os.remove(f)
# else:
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 3. 증강 파이프라인 정의
# transform = A.Compose([
#     A.RandomResizedCrop(height=512, width=512, scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.7),
#     A.RandomBrightnessContrast(p=0.4),
#     A.Rotate(limit=20, p=0.6),
#     A.RandomScale(scale_limit=0.1, p=0.4),
#     A.Blur(blur_limit=3, p=0.2),  # 너무 흐리지 않게 blur_limit=3
#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
#     A.OneOf([
#         A.GaussNoise(var_limit=(5.0, 15.0), p=0.5),
#         A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5)
#     ], p=0.3),
#     A.Resize(*IMG_SIZE)
# ])

# # 4. 이미지 증강 실행
# image_paths = glob(os.path.join(INPUT_DIR, '*'))

# for img_path in tqdm(image_paths, desc="Augmenting class 14 images"):
#     image = cv2.imread(img_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     basename = os.path.splitext(os.path.basename(img_path))[0]

#     for i in range(NUM_AUG_PER_IMAGE):
#         augmented = transform(image=image)
#         aug_img = augmented['image']
#         aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
#         save_path = os.path.join(OUTPUT_DIR, f"{basename}_aug_{i}.jpg")
#         cv2.imwrite(save_path, aug_img)

# print(f"✅ {len(image_paths) * NUM_AUG_PER_IMAGE} images saved to {OUTPUT_DIR}")