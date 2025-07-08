# scripts/generate_validation_preds.py
import torch
import pandas as pd
import os
from tqdm import tqdm
import pytorch_lightning as pl

# 우리 프로젝트의 '주방장'과 '학생'을 불러옴
from src.data import CustomDataModule
from src.lightning_model import CustomLightningModule

if __name__ == '__main__':
    # --- ⚙️ 설정 ---

    # 1. [수정!] 방금 '지옥 훈련'이 끝난 최고의 모델 경로를 여기에 정확히 적어줘!
    #    'ULTIMATE-TRAIN...'으로 시작하는 가장 최신 .ckpt 파일이야.
    CKPT_PATH = 'models/ULTIMATE-TRAIN-convnext_base-sz384-epoch=29-val_f1=0.9534.ckpt'

    # 2. 이 모델을 학습시켰던 설정과 동일하게 맞춰야 함
    DATA_PATH = './data/raw'
    IMG_SIZE = 384
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    AUG_LEVEL = 'strong' # 검증셋에는 영향을 주지 않지만, 학습 때와 동일하게 설정

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 데이터 모듈 준비 (모의고사 시험지만 필요) ---
    print("Preparing validation data (the 'mock exam')...")
    # 주방장을 불러서, 어떤 데이터로 모의고사를 볼지 알려줌
    data_module = CustomDataModule(
        data_path=DATA_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        augmentation_level=AUG_LEVEL
    )
    data_module.setup() # setup을 호출해야 val_dataset이 생성됨
    val_dataloader = data_module.val_dataloader()

    # --- 모델 로드 및 예측 ---
    print(f"Loading model from: {os.path.basename(CKPT_PATH)}")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    model.eval() # 모델을 '시험 모드'로 설정

    all_preds = []
    all_labels = []
    with torch.no_grad(): # 정답을 베끼지 못하도록 '학습 기능' 정지
        for images, labels in tqdm(val_dataloader, desc="Predicting on validation set"):
            images = images.to(DEVICE)
            preds = model(images)
            preds = torch.argmax(preds, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- '모의고사 채점표' 파일 생성 ---
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds
    })
    results_df.to_csv('./validation_results.csv', index=False)

    print("\nValidation results saved to `validation_results.csv`!")

    # --- 간단한 정확도 계산 및 출력 ---
    accuracy = (results_df['true_label'] == results_df['predicted_label']).mean()
    print(f"\nValidation Accuracy on this run: {accuracy:.4f}")