# # 개선됨.  ./submissions/0710_1310.csv 제출은 안함
# import torch
# import pandas as pd
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.lightning_model import CustomLightningModule

# # 기본 TTA 함수
# # 클래스 민감도가 낮은 클래스용 (기존 TTA)
# def predict_with_default_tta(model, image, img_size, device):
#     tta_images = []
#     for k in [0, 1, 2, 3]:
#         tta_images.append(np.rot90(image, k=k).copy())
#     flipped = cv2.flip(image, 1)
#     for k in [0, 1, 2, 3]:
#         tta_images.append(np.rot90(flipped, k=k).copy())

#     transform = A.Compose([
#         A.Resize(img_size, img_size),
#         A.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

#     batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)

#     with torch.no_grad():
#         preds = model(batch)
#         preds = torch.softmax(preds, dim=1)

#     return torch.mean(preds, dim=0).cpu()

# # selective ensemble: conv + eff_x
# if __name__ == '__main__':
#     MODEL_PATHS = {
#         'conv': './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt',
#         'eff_x': './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt'
#     }
#     IMG_SIZE = 512
#     TEST_DIR = './data/raw/test/'
#     SUB_FILE = './data/raw/sample_submission.csv'
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("모델 로드 중...")
#     conv_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['conv']).to(DEVICE).eval()
#     eff_x_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_x']).to(DEVICE).eval()
#     print("모든 모델 로드 완료.")

#     submission_df = pd.read_csv(SUB_FILE)
#     final_preds = []

#     for image_id in tqdm(submission_df['ID']):
#         img_path = os.path.join(TEST_DIR, image_id)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         pred_conv = predict_with_default_tta(conv_model, img, IMG_SIZE, DEVICE)
#         pred_eff_x = predict_with_default_tta(eff_x_model, img, IMG_SIZE, DEVICE)

#         # soft voting
#         avg_pred = (pred_conv * 0.4 + pred_eff_x * 0.6)

#         # 예측 softmax 최댓값의 클래스
#         final_pred = torch.argmax(avg_pred).item()

#         # 예외 보정: 13번 과잉흡수 차단 (13으로 예측됐지만 13 confidence < 0.45 이면 제외)
#         if final_pred == 13 and avg_pred[13].item() < 0.45:
#             # 가장 높은 클래스 제외하고 2순위 선택
#             avg_pred[13] = -1  # 억제
#             final_pred = torch.argmax(avg_pred).item()

#         final_preds.append(final_pred)

#     timestamp = datetime.now().strftime('%m%d')
#     out_path = f'./submissions/{timestamp}_selective.csv'
#     submission_df['target'] = final_preds
#     submission_df.to_csv(out_path, index=False)
#     print(f"✅ 제출 파일 저장 완료: {out_path}")




import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 이미지 증강용 transform 정의
def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 클래스 민감도 기반 TTA 적용 함수
def tta_predict(model, image, img_size, device, high_risk=False):
    tta_images = []

    # 기본 회전
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(image, k=k).copy())

    # 좌우반전 후 회전
    flipped = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]:
        tta_images.append(np.rot90(flipped, k=k).copy())

    # 추가 TTA (high_risk 클래스만)
    if high_risk:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        tta_images.append(blurred)
        tta_images.append(cv2.flip(image, 0))  # 상하반전
        tta_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    transform = get_transform(img_size)
    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)

    with torch.no_grad():
        preds = model(batch)
        probs = torch.softmax(preds, dim=1)

    return torch.mean(probs, dim=0).cpu()

# 인퍼런스 앙상블 수행 함수
def ensemble_inference(image, models, img_size, device):
    # 클래스별 예외 처리 타겟 클래스
    high_risk_classes = [12, 13, 14]
    all_preds = []

    for model in models:
        pred = tta_predict(model, image, img_size, device, high_risk=True)
        all_preds.append(pred)

    # soft voting 앙상블 (weight 조절 가능)
    avg_pred = (all_preds[0] * 0.5 + all_preds[1] * 0.5)

    # 1순위 예측
    top1 = torch.argmax(avg_pred).item()
    top1_conf = avg_pred[top1].item()

    # 2순위 예측
    sorted_idx = torch.argsort(avg_pred, descending=True)
    top2 = sorted_idx[1].item()
    top2_conf = avg_pred[top2].item()

    # 보정 조건 (13번 과다 흡수 방지 등)
    if top1 == 13 and top1_conf < 0.45 and top2 in high_risk_classes:
        return top2
    elif top1 in high_risk_classes and top1_conf < 0.38:
        return top2
    else:
        return top1
