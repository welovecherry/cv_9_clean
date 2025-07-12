import torch
import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule


def predict_with_tta(model, image, img_size, device):
    tta_images = []
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    flipped = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped, k=k).copy())

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)

    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)

    return torch.mean(preds, dim=0).cpu()

if __name__ == '__main__':
    MODEL_PATHS = {
        'conv': './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt',
        'eff_x': './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt',
        'eff_y': './models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt'
    }
    IMG_SIZE = 512
    TEST_DIR = './data/raw/test/'
    SUB_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("모델 로드 중...")
    conv_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['conv']).to(DEVICE).eval()
    eff_x_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_x']).to(DEVICE).eval()
    eff_y_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_y']).to(DEVICE).eval()
    print("모든 모델 로드 완료.")

    submission_df = pd.read_csv(SUB_FILE)
    final_preds = []

    # 클래스별 가장 잘 맞춘 모델
    best_model_by_class = {
        1: 'eff_x', 3: 'eff_x', 4: 'eff_x', 14: 'eff_x', 15: 'eff_x',
        5: 'conv', 6: 'conv', 8: 'conv', 13: 'conv', 16: 'conv',
        11: 'eff_x', 12: 'eff_x'
    }

    for image_id in tqdm(submission_df['ID']):
        img_path = os.path.join(TEST_DIR, image_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred_conv = predict_with_tta(conv_model, img, IMG_SIZE, DEVICE)
        pred_eff_x = predict_with_tta(eff_x_model, img, IMG_SIZE, DEVICE)
        pred_eff_y = predict_with_tta(eff_y_model, img, IMG_SIZE, DEVICE)

        # Soft Voting (기본값)
        soft_vote = (pred_conv * 0.4 + pred_eff_x * 0.4 + pred_eff_y * 0.2)
        soft_pred = torch.argmax(soft_vote).item()

        # 만약 해당 클래스가 사전에 정한 모델 분류 대상이면 해당 모델의 예측만 사용
        if soft_pred in best_model_by_class:
            selected = best_model_by_class[soft_pred]
            if selected == 'conv':
                final_prediction = torch.argmax(pred_conv).item()
            elif selected == 'eff_x':
                final_prediction = torch.argmax(pred_eff_x).item()
            else:
                final_prediction = torch.argmax(pred_eff_y).item()
        else:
            final_prediction = soft_pred

        final_preds.append(final_prediction)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    out_path = f'./submissions/submission_alternative_{timestamp}.csv'
    submission_df['target'] = final_preds
    submission_df.to_csv(out_path, index=False)
    print(f"✅ 대체 전략 제출 완료: {out_path}")