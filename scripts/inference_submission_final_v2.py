# 최고점

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


# def predict_with_tta(model, image, img_size, device):
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


# def correction_rules(pred_conv, pred_eff_x, pred_eff_y):
#     """
#     클래스별 혼동 상황에 따른 보정 규칙 적용
#     """
#     weights = {'conv': 0.1, 'eff_x': 0.6, 'eff_y': 0.3}
#     avg_pred = pred_eff_x * weights['eff_x'] + pred_eff_y * weights['eff_y'] + pred_conv * weights['conv']
#     final_pred = torch.argmax(avg_pred).item()

#     # 클래스 14 보정 예시: 상위 3개 클래스 중 13과 10이 모두 포함되면 → 14로 보정
#     top3 = torch.topk(avg_pred, 3).indices.tolist()
#     if 14 in top3 and 13 in top3 and 10 in top3:
#         return 14

#     # 클래스 12 → 13, 10 혼동시도 많았던 케이스
#     if final_pred == 13 and top3[0] == 13 and 12 in top3:
#         return 12

#     # 기타 보정 규칙 여기에 추가 가능

#     return final_pred


# if __name__ == '__main__':
#     MODEL_PATHS = {
#         'conv': './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt',
#         'eff_x': './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt',
#         'eff_y': './models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt'
#     }

#     IMG_SIZE = 512
#     TEST_DIR = './data/raw/test/'
#     SUB_FILE = './data/raw/sample_submission.csv'
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("모델 로드 중...")
#     conv_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['conv']).to(DEVICE).eval()
#     eff_x_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_x']).to(DEVICE).eval()
#     eff_y_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_y']).to(DEVICE).eval()
#     print("모든 모델 로드 완료.")

#     submission_df = pd.read_csv(SUB_FILE)
#     final_preds = []

#     for image_id in tqdm(submission_df['ID']):
#         img_path = os.path.join(TEST_DIR, image_id)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         pred_conv = predict_with_tta(conv_model, img, IMG_SIZE, DEVICE)
#         pred_eff_x = predict_with_tta(eff_x_model, img, IMG_SIZE, DEVICE)
#         pred_eff_y = predict_with_tta(eff_y_model, img, IMG_SIZE, DEVICE)

#         corrected_pred = correction_rules(pred_conv, pred_eff_x, pred_eff_y)
#         final_preds.append(corrected_pred)

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M')
#     out_path = f'./submissions/{timestamp}.csv'
#     submission_df['target'] = final_preds
#     submission_df.to_csv(out_path, index=False)
#     print(f"✅ 제출 파일 저장 완료: {out_path}")




# 결과 파일: ./submissions/20250710_1634.csv 미제출
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

# # TTA 적용 함수 (회전 + 좌우 반전)
# def predict_with_tta(model, image, img_size, device):
#     tta_images = []
#     for k in [0, 1, 2, 3]:
#         tta_images.append(np.rot90(image, k=k).copy())
#     flipped = cv2.flip(image, 1)
#     for k in [0, 1, 2, 3]:
#         tta_images.append(np.rot90(flipped, k=k).copy())

#     transform = A.Compose([
#         A.Resize(img_size, img_size),  # 이미지 크기 조정
#         A.Normalize(mean=[0.485, 0.456, 0.406],  # 정규화
#                     std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

#     batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)

#     with torch.no_grad():
#         preds = model(batch)
#         preds = torch.softmax(preds, dim=1)

#     return torch.mean(preds, dim=0).cpu()  # 평균을 내어 최종 예측 확률 벡터 반환

# # 보정 로직 함수 (클래스 혼동 방지 목적)
# def correction_rules(pred_conv, pred_eff_x, pred_eff_y):
#     weights = {'conv': 0.6, 'eff_x': 0.25, 'eff_y': 0.15}  # conv에 높은 가중치 부여
#     avg_pred = pred_conv * weights['conv'] + pred_eff_x * weights['eff_x'] + pred_eff_y * weights['eff_y']
#     final_pred = torch.argmax(avg_pred).item()

#     top3 = torch.topk(avg_pred, 3).indices.tolist()

#     if 14 in top3 and 13 in top3 and 10 in top3:
#         return 14

#     if final_pred == 13 and top3[0] == 13 and 12 in top3:
#         return 12

#     return final_pred

# # 메인 실행
# if __name__ == '__main__':
#     MODEL_PATHS = {
#         'conv': './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt',
#         'eff_x': './models/auto-tf_efficientnetv2_s-lr1e-4-sz384-epoch=22-val_f1=0.9955.ckpt',
#         'eff_y': './models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt'
#     }

#     IMG_SIZE = 512
#     TEST_DIR = './data/raw/test/'
#     SUB_FILE = './data/raw/sample_submission.csv'
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("모델 로드 중...")
#     conv_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['conv']).to(DEVICE).eval()
#     eff_x_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_x']).to(DEVICE).eval()
#     eff_y_model = CustomLightningModule.load_from_checkpoint(MODEL_PATHS['eff_y']).to(DEVICE).eval()
#     print("모든 모델 로드 완료.")

#     submission_df = pd.read_csv(SUB_FILE)
#     final_preds = []

#     for image_id in tqdm(submission_df['ID']):
#         img_path = os.path.join(TEST_DIR, image_id)
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         pred_conv = predict_with_tta(conv_model, img, IMG_SIZE, DEVICE)
#         pred_eff_x = predict_with_tta(eff_x_model, img, IMG_SIZE, DEVICE)
#         pred_eff_y = predict_with_tta(eff_y_model, img, IMG_SIZE, DEVICE)

#         corrected_pred = correction_rules(pred_conv, pred_eff_x, pred_eff_y)
#         final_preds.append(corrected_pred)

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M')
#     out_path = f'./submissions/{timestamp}.csv'
#     submission_df['target'] = final_preds
#     submission_df.to_csv(out_path, index=False)
#     print(f"✅ 제출 파일 저장 완료: {out_path}")





# scripts/inference_final_ensemble_v2.py

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
    """'공격적인 해결사'를 위한 강화된 TTA 예측 함수"""
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
    N_INFERENCES = 5
    N_TTA_ADVANCED = 20

    # 최종판 후처리 규칙
    CORRECTION_RULES = {
        13: {12: 0.1, 14: 0.15},
        14: {13: 0.1, 4: 0.05},
        12: {13: 0.05, 14: 0.05},
        10: {14: 0.1},
        3: {4: 0.1},
        4: {3: 0.1},
    }

    # --- 모델 로드 ---
    print("Loading model...")
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    
    # --- 최종 통합 추론 로직 ---
    print("\n--- Starting Final Integrated Inference ---")
    final_ensembled_softmax = []
    test_image_ids = pd.read_csv(SUBMISSION_FILE)['ID']

    for image_id in tqdm(test_image_ids, desc="Final Ensemble Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_ensembled_softmax.append(torch.ones(17) / 17)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Stage 1: Stable Ace 예측 (eval 모드)
        model.eval()
        stable_pred = simple_tta_predict(model, image, IMG_SIZE, DEVICE)

        # Stage 2: Aggressive Solver 예측 (train 모드)
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
    
    # 후처리 적용
    print("\nApplying post-processing rules...")
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