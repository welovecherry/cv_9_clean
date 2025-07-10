# scripts/inference_conditional_ensemble.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule


def predict_with_enhanced_tta(model, image, img_size, device):
    """개선된 TTA (기존 코드와 동일)"""
    tta_images = []

    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    flipped = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped, k=k).copy())

    weak_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
            A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=20, p=0.3),
        ], p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    batch = torch.stack([
        weak_transforms(image=img)['image'] for img in tta_images
    ]).to(device)

    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)

    avg_preds = torch.mean(preds, dim=0)
    return avg_preds.cpu()


if __name__ == '__main__':
    # --- 설정 ---
    # 🌟 1. 모델 경로 설정 (기본 모델 + 전문가 모델)
    MAIN_CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    SPECIALIST_CKPT_PATH = './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt'
    
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    META_FILE = './data/raw/meta.csv' # 클래스 이름과 인덱스 매핑을 위해 meta.csv 사용

    # --- 모델 로드 ---
    print("Loading models...")
    main_model = CustomLightningModule.load_from_checkpoint(MAIN_CKPT_PATH).to(DEVICE)
    main_model.eval()
    
    # 🌟 2. 전문가 모델 추가 로드
    specialist_model = CustomLightningModule.load_from_checkpoint(SPECIALIST_CKPT_PATH).to(DEVICE)
    specialist_model.eval()
    print("Models loaded.")

    # 🌟 3. 혼동 클래스 인덱스 찾기
    meta_df = pd.read_csv(META_FILE)
    confused_class_names = ['resume', 'statement_of_opinion']
    confused_class_indices = meta_df[meta_df['class_name'].isin(confused_class_names)]['target'].tolist()
    print(f"Confused classes '{confused_class_names}' correspond to indices: {confused_class_indices}")

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="Conditional Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. 기본 모델로 예측
        main_pred_tensor = predict_with_enhanced_tta(main_model, image, IMG_SIZE, DEVICE)
        main_pred_class = torch.argmax(main_pred_tensor).item()

        # 🌟 4. 조건부 로직 적용
        # 만약 기본 모델이 '이력서' 또는 '소견서'로 예측했다면,
        if main_pred_class in confused_class_indices:
            # 전문가 모델에게 다시 예측을 맡김
            specialist_pred_tensor = predict_with_enhanced_tta(specialist_model, image, IMG_SIZE, DEVICE)
            final_pred = torch.argmax(specialist_pred_tensor).item()
        else:
            # 그 외에는 기본 모델의 예측을 그대로 사용
            final_pred = main_pred_class
            
        final_predictions.append(final_pred)

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_conditional_ensemble_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] 조건부 앙상블 추론 결과 저장 → {output_filename}")