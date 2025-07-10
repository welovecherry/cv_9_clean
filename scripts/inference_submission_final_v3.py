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

# src 모듈 import 위해 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lightning_model import CustomLightningModule


# 기본 TTA (4회 회전 + 좌우반전 후 4회 회전 = 총 8장)
def basic_tta(image):
    tta_images = []
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())
    flipped = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped, k=k).copy())
    return tta_images

# 클래스 민감 TTA (보다 강한 회전 위주)
def strong_tta(image):
    tta_images = []
    flipped = cv2.flip(image, 1)
    for k in range(4):
        tta_images.append(np.rot90(image, k=k).copy())
        tta_images.append(np.rot90(flipped, k=k).copy())
    # 추가로 상하반전까지 적용
    flipped_ud = cv2.flip(image, 0)
    for k in range(4):
        tta_images.append(np.rot90(flipped_ud, k=k).copy())
    return tta_images


# 클래스 민감도 기반으로 TTA 방식 선택
def get_tta_images(image, target_sensitive=False):
    return strong_tta(image) if target_sensitive else basic_tta(image)


def predict_with_conditional_tta(model, image, img_size, device, use_strong_tta=False):
    tta_images = get_tta_images(image, target_sensitive=use_strong_tta)

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


# 클래스별 강한 TTA가 필요한 클래스 리스트
TTA_SENSITIVE_CLASSES = [3, 4, 12, 14]

# Softmax 예측값 threshold 아래이면 보정
SOFTMAX_THRESHOLD = 0.45

# 예측값이 13번일 때 불확실한 경우 보정 후보
DEFLECT_FROM_CLASS_13 = {10, 14, 12}


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

    for image_id in tqdm(submission_df['ID']):
        img_path = os.path.join(TEST_DIR, image_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 강한 TTA가 필요한 경우
        use_strong_tta = any(cls in image_id for cls in map(str, TTA_SENSITIVE_CLASSES))

        pred_conv = predict_with_conditional_tta(conv_model, img, IMG_SIZE, DEVICE, use_strong_tta)
        pred_eff_x = predict_with_conditional_tta(eff_x_model, img, IMG_SIZE, DEVICE, use_strong_tta)
        pred_eff_y = predict_with_conditional_tta(eff_y_model, img, IMG_SIZE, DEVICE, use_strong_tta)

        # 클래스별로 앙상블 weight 달리 줌 (예: eff_x > eff_y > conv)
        avg_pred = (pred_eff_x * 0.55 + pred_eff_y * 0.35 + pred_conv * 0.10)

        max_prob = avg_pred.max().item()
        pred_class = torch.argmax(avg_pred).item()

        # softmax 값 낮고 13번이면 흡수 의심 → 보정 시도
        if pred_class == 13 and max_prob < SOFTMAX_THRESHOLD:
            second_choice = torch.topk(avg_pred, 2).indices[1].item()
            if second_choice in DEFLECT_FROM_CLASS_13:
                pred_class = second_choice  # 보정

        final_preds.append(pred_class)

    timestamp = datetime.now().strftime('%m%d_%H%M')
    out_path = f'./submissions/{timestamp}.csv'
    submission_df['target'] = final_preds
    submission_df.to_csv(out_path, index=False)
    print(f"✅ 제출 파일 저장 완료: {out_path}")