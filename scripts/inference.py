# # scripts/inference_final.py

# import torch
# import pandas as pd
# import os
# from tqdm import tqdm
# import cv2
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from src.lightning_model import CustomLightningModule 
# from skimage import io
# from deskew import determine_skew

# def predict_single_image(model, image_path, device, img_size):
#     # 1. 이미지 읽기
#     print(f"1. Processing {os.path.basename(image_path)}...")

#     image = cv2.imread(image_path)
#     if image is None: return -1

#     # 2. 전처리 파이프라인 (노이즈 제거 -> 각도 교정)
#     try:
#         denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
#         grayscale = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
#         angle = determine_skew(grayscale)
#         # cval=255는 회전 후 남는 공간을 흰색으로 채우라는 의미
#         rotated = io.rotate(denoised, angle, resize=True, cval=255) * 255
#         image = rotated.astype(np.uint8)
#     except Exception as e:
#         image = cv2.imread(image_path)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # 3. TTA를 위한 변환 준비
#     transform = A.Compose([
#         A.Resize(img_size, img_size),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])
#     tta_transform = A.Compose([
#         A.Resize(img_size, img_size),
#         A.HorizontalFlip(p=1.0),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ])

#     # 4. 원본과 TTA 버전에 대한 예측 수행
#     original_img_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
#     tta_img_tensor = tta_transform(image=image)['image'].unsqueeze(0).to(device)


#     with torch.no_grad():
#         pred_original = torch.softmax(model(original_img_tensor), dim=1)
#         pred_tta = torch.softmax(model(tta_img_tensor), dim=1)
    
#     final_pred = torch.argmax(pred_original + pred_tta, dim=1)
    
#     print("   -> 5. Done.")

#     return final_pred.item()

# if __name__ == '__main__':
#     # [수정!] 네 챔피언 모델 경로로 변경
#     CKPT_PATH = './models/tf_efficientnetv2_s-epoch=06-val_f1=0.9114.ckpt' 
#     # [수정!] 이 모델을 학습시켰던 이미지 크기로 변경
#     IMG_SIZE = 224 # 또는 384 등, 학습했을 때의 크기
    
#     TEST_DIR = './data/raw/test/'
#     SUBMISSION_FILE = './data/raw/sample_submission.csv'
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
#     model.eval()

#     submission_df = pd.read_csv(SUBMISSION_FILE)
    
#     predictions = []
#     for image_id in tqdm(submission_df['ID'], desc="Final Inference (Preproc + TTA)"):
#         image_path = os.path.join(TEST_DIR, image_id)
#         prediction = predict_single_image(model, image_path, DEVICE, IMG_SIZE)
#         predictions.append(prediction)

#     submission_df['target'] = predictions
#     submission_df.to_csv('./submission_final.csv', index=False)
#     print("\nFinal submission file `submission_final.csv` created!")

# scripts/inference.py

import torch
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lightning_model import CustomLightningModule 
from skimage import io
from deskew import determine_skew

def predict_single_image(model, image_path, device, img_size):
    """ 단일 이미지를 전처리하고 TTA 예측까지 수행하는 함수 """
    # 1. 이미지 읽기
    image = cv2.imread(image_path)
    if image is None: 
        print(f"Warning: Could not read image {os.path.basename(image_path)}, skipping.")
        return -1 # 이미지 읽기 실패 시 -1 (에러) 반환

    # 2. 전처리 (노이즈 제거 -> 각도 교정)
    try:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        grayscale = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        rotated = io.rotate(denoised, angle, resize=True, cval=255) * 255
        image = rotated.astype(np.uint8)
    except Exception:
        image = cv2.imread(image_path) # 전처리 실패 시 원본 사용

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. TTA를 위한 변환 준비
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tta_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 4. 원본과 TTA 버전에 대한 예측 수행
    original_img_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    tta_img_tensor = tta_transform(image=image)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_original = torch.softmax(model(original_img_tensor), dim=1)
        pred_tta = torch.softmax(model(tta_img_tensor), dim=1)
    
    # 5. 두 예측의 확률을 더해서 최종 클래스 결정
    final_pred = torch.argmax(pred_original + pred_tta, dim=1)
    
    return final_pred.item()

if __name__ == '__main__':
    # [수정!] 최종 학습 후, models 폴더에 저장될 가장 좋은 모델의 체크포인트 파일 경로
    CKPT_PATH = 'models/tf_efficientnetv2_s-epoch=07-val_f1=0.9383.ckpt'  # 바꾸기!!!!!
    
    # [수정!] 최종 학습시킨 모델의 이미지 크기
    IMG_SIZE = 512 
    
    # --- 경로 및 기타 설정 ---
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 모델 로드 ---
    model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(DEVICE)
    model.eval()

    # --- 추론 실행 ---
    submission_df = pd.read_csv(SUBMISSION_FILE)
    
    predictions = []
    for image_id in tqdm(submission_df['ID'], desc="Final Inference"):
        image_path = os.path.join(TEST_DIR, image_id)
        prediction = predict_single_image(model, image_path, DEVICE, IMG_SIZE)
        predictions.append(prediction)

    submission_df['target'] = predictions
    submission_df.to_csv('./submission.csv', index=False)

    print("\nInference finished! `submission.csv` has been created.")