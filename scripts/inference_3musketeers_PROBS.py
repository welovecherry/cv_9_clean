# scripts/inference_3musketeers_PROBS.py

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

if __name__ == '__main__':
    CKPT_PATH_A = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'
    CKPT_PATH_B = './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt'
    CKPT_PATH_C = './models/auto-resnet101-lr1e-4-sz384-epoch=19-val_f1=0.9839.ckpt'
    IMG_SIZE = 384
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'

    print("Loading Three Musketeers models...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    model_c = CustomLightningModule.load_from_checkpoint(CKPT_PATH_C).to(DEVICE)
    model_a.eval(); model_b.eval(); model_c.eval()
    print("Models loaded.")

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    submission_df = pd.read_csv(SUBMISSION_FILE)
    all_softmax_vectors = []

    for image_id in tqdm(submission_df['ID'], desc="3 Musketeers Ensemble (Saving Probs)"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            all_softmax_vectors.append(torch.ones(17) / 17)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed_image = transform(image=image)['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_a = torch.softmax(model_a(transformed_image), dim=1)
            pred_b = torch.softmax(model_b(transformed_image), dim=1)
            pred_c = torch.softmax(model_c(transformed_image), dim=1)
        
        ensembled_pred_vector = (pred_a + pred_b + pred_c) / 3.0
        all_softmax_vectors.append(ensembled_pred_vector.squeeze(0))

    # [변경] 최종 라벨 대신 모든 확률을 저장
    probs_df = pd.DataFrame([tensor.cpu().numpy() for tensor in all_softmax_vectors])
    probs_df.columns = [f'prob_{i}' for i in range(17)]
    
    result_df = pd.concat([submission_df['ID'], probs_df], axis=1)
    
    output_filename = './submission_3musketeers_PROBS2222.csv'
    result_df.to_csv(output_filename, index=False)
    print(f"\n[완료] 3 Musketeers 앙상블 확률 저장 → {output_filename}")