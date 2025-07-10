# scripts/inference_final_ensemble.py

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

def simple_tta_predict(model, image, img_size, device):
    """'안정적인 에이스'를 위한 간단한 TTA 예측 함수"""
    tta_images = []
    # 기본 회전 및 반전
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    flipped = cv2.flip(image, 1)
    tta_images.append(flipped.copy())

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)
    
    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)
    
    return torch.mean(preds, dim=0).cpu()

def advanced_tta_predict(model, image, n_tta, img_size, device):
    """'공격적인 해결사'를 위한 강화된 TTA 예측 함수 (기존 1634 코드와 유사)"""
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.SomeOf([
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GridDistortion(p=0.5),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
        ], n=3, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tta_images = [transform(image=image)['image'] for _ in range(n_tta)]
    batch = torch.stack(tta_images).to(device)
    
    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)
        
    return torch.mean(preds, dim=0).cpu()


if __name__ == '__main__':
    # --- 설정 ---
    CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    
    # 고급 추론 설정
    N_INFERENCES = 5  # 다중 추론 횟수
    N_TTA_ADVANCED = 20 # 고급 TTA 이미지 개수

    # 최종판 후처리 규칙
    CORRECTION_RULES = {
        13: {12: 0.1, 14: 0.15},
        14: {13: 0.1, 4: 0.05},  # 14->4 혼동 케이스 추가
        12: {13: 0.05, 14: 0.05},# 12->13, 14 혼동 케이스 추가
        10: {14: 0.1},
        3: {4: 0.1},
        4: {3: 0.1},
    }

    # --- 모델 로드 ---
    print("Loading model...")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    
    # --- 추론 실행 ---
    all_stable_preds = []
    all_aggressive_preds_runs = []
    test_image_ids = pd.read_csv(SUBMISSION_FILE)['ID']

    # 1. '안정적인 에이스' 예측 (model.eval() 모드)
    print("\n--- Stage 1: Predicting with 'Stable Ace' model ---")
    model.eval()
    for image_id in tqdm(test_image_ids, desc="Stable Ace Inference"):
        # (이미지 로드 로직은 아래와 동일하므로 생략, 최종 코드에서 합침)
        pass # 실제로는 아래 루프와 합쳐서 실행

    # 2. '공격적인 해결사' 예측 (model.train() 모드)
    print("\n--- Stage 2: Predicting with 'Aggressive Problem-Solver' model ---")
    model.train() # MC Dropout
    for i in range(N_INFERENCES):
        run_preds = []
        for image_id in tqdm(test_image_ids, desc=f"Aggressive Run {i+1}/{N_INFERENCES}"):
            # (이미지 로드 로직은 아래와 동일하므로 생략, 최종 코드에서 합침)
            pass # 실제로는 아래 루프와 합쳐서 실행
        all_aggressive_preds_runs.append(torch.stack(run_preds))

    # --- 실제 통합 추론 로직 ---
    print("\n--- Starting Final Integrated Inference ---")
    final_ensembled_softmax = []

    for image_id in tqdm(test_image_ids, desc="Final Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_ensembled_softmax.append(torch.ones(17) / 17)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Stage 1: Stable Ace 예측
        model.eval()
        stable_pred = simple_tta_predict(model, image, IMG_SIZE, DEVICE)

        # Stage 2: Aggressive Solver 예측
        model.train()
        aggressive_preds_per_run = []
        for _ in range(N_INFERENCES):
            aggressive_preds_per_run.append(
                advanced_tta_predict(model, image, N_TTA_ADVANCED, IMG_SIZE, DEVICE)
            )
        aggressive_pred = torch.mean(torch.stack(aggressive_preds_per_run), dim=0)

        # 앙상블: 두 예측의 Softmax 평균
        ensembled_pred = (stable_pred + aggressive_pred) / 2.0
        final_ensembled_softmax.append(ensembled_pred)

    final_ensembled_softmax = torch.stack(final_ensembled_softmax)
    
    # 3. 후처리 적용
    corrected_preds = []
    for pred_vector in final_ensembled_softmax:
        pred_class = torch.argmax(pred_vector).item()
        if pred_class in CORRECTION_RULES:
            correction_amount = pred_vector[pred_class]
            for target_class, weight in CORRECTION_RULES[pred_class].items():
                pred_vector[target_class] += correction_amount * weight
        corrected_preds.append(pred_vector)

    final_labels = torch.argmax(torch.stack(corrected_preds), dim=1).tolist()

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_final_ensemble_{timestamp}.csv'
    submission_df = pd.read_csv(SUBMISSION_FILE)
    submission_df['target'] = final_labels
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] 최종 앙상블 추론 결과 저장 → {output_filename}")