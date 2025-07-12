# scripts/inference_advanced.py
# 리더보드 점수: 0.9188
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

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule

def generate_tta_batch(image, n_tta, img_size, device):
    """
    1. 강화된 TTA: RandomCrop, Rotate, Distortion 등 20개의 다양한 TTA 이미지를 생성
    """
    # Albumentations 변환 파이프라인 정의 (매번 다른 증강을 적용)
    transform = A.Compose([
        A.Resize(img_size, img_size),
        # 다양한 증강 기법 추가
        A.SomeOf([
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GridDistortion(p=0.5),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
        ], n=3, p=0.9), # 위 리스트 중 3개를 90% 확률로 적용
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tta_images = [transform(image=image)['image'] for _ in range(n_tta)]
    batch = torch.stack(tta_images).to(device)
    return batch


if __name__ == '__main__':
    # --- 설정 ---
    CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    
    # 고급 추론 설정
    N_INFERENCES = 5  # 2. 다중 추론 횟수
    N_TTA = 20        # 1. TTA 이미지 개수

    # 4. 클래스 민감도 보정 규칙 {예측된 클래스: {보정해줄 클래스: 보정 가중치}}
    # 예: 13번 클래스로 예측되면, 13번 점수의 10%를 12번과 14번에 더해줌
    CORRECTION_RULES = {
        13: {12: 0.1, 14: 0.15}, # resume -> driver_lisence, statement_of_opinion
        14: {13: 0.1},           # statement_of_opinion -> resume
        10: {14: 0.1}            # payment_confirmation -> statement_of_opinion
    }

    # --- 모델 로드 ---
    print("Loading model...")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    
    # 3. MC Dropout: model.eval() 대신 model.train()을 사용해 Dropout 활성화
    model.train()
    print("Model loaded in train mode for MC Dropout.")

    # --- 다중 추론 실행 ---
    all_run_preds = []
    test_image_ids = pd.read_csv(SUBMISSION_FILE)['ID']

    for i in range(N_INFERENCES):
        print(f"\n--- Starting Inference Run {i+1}/{N_INFERENCES} ---")
        current_run_preds = []
        
        for image_id in tqdm(test_image_ids, desc=f"Run {i+1} TTA Inference"):
            image_path = os.path.join(TEST_DIR, image_id)
            image = cv2.imread(image_path)
            
            if image is None:
                # 이미지를 못 읽을 경우, 17개 클래스에 대해 균등 확률 부여
                current_run_preds.append(torch.ones(17) / 17)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                # TTA 배치 생성 및 추론
                tta_batch = generate_tta_batch(image, n_tta=N_TTA, img_size=IMG_SIZE, device=DEVICE)
                preds = model(tta_batch)
                preds = torch.softmax(preds, dim=1)
                
                # TTA 결과 평균
                avg_pred = torch.mean(preds, dim=0)
                current_run_preds.append(avg_pred.cpu())
        
        all_run_preds.append(torch.stack(current_run_preds))

    # --- 최종 예측값 계산 ---
    print("\n--- Averaging all inference runs and applying post-processing ---")
    
    # 2. 다중 추론 평균: 모든 실행 결과의 Softmax 값 평균
    final_avg_preds = torch.mean(torch.stack(all_run_preds), dim=0)
    
    # 4. 클래스 민감도 후처리 적용
    corrected_preds = []
    for pred_vector in final_avg_preds:
        pred_class = torch.argmax(pred_vector).item()
        
        if pred_class in CORRECTION_RULES:
            correction_amount = pred_vector[pred_class]
            for target_class, weight in CORRECTION_RULES[pred_class].items():
                pred_vector[target_class] += correction_amount * weight
        
        corrected_preds.append(pred_vector)

    final_labels = torch.argmax(torch.stack(corrected_preds), dim=1).tolist()

    # --- 결과 저장 ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_advanced_{timestamp}.csv'
    submission_df = pd.read_csv(SUBMISSION_FILE)
    submission_df['target'] = final_labels
    submission_df.to_csv(output_filename, index=False)

    print(f"\n[완료] 고급 추론 결과 저장 → {output_filename}")