# scripts/inference_final_push.py
# ... (이전 inference_back_to_basics.py와 모든 함수 내용은 동일) ...
import torch, pandas as pd, os
from tqdm import tqdm
# ... (필요한 모든 import 구문) ...
from src.lightning_model import CustomLightningModule
import cv2, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

def predict_with_8_way_tta(model, image, img_size, device):
    tta_images = []
    for k in [0, 1, 2, 3]: tta_images.append(np.rot90(image, k=k).copy())
    flipped_img = cv2.flip(image, 1)
    for k in [0, 1, 2, 3]: tta_images.append(np.rot90(flipped_img, k=k).copy())
    transform = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
    batch = torch.stack([transform(image=img)['image'] for img in tta_images]).to(device)
    with torch.no_grad():
        preds = model(batch)
        preds = torch.softmax(preds, dim=1)
    return torch.mean(preds, dim=0).cpu()

if __name__ == '__main__':
    # --- 설정: 최종 비밀 병기 투입 ---

    # A팀 (우리의 비밀 병기 - 가중치 손실로 학습된 모델)
    CKPT_PATH_A = './models/finetuned-tf_efficientnetv2_s-epoch=08-val_f1=0.8753.ckpt'
    IMG_SIZE_A = 512
    WEIGHT_A = 0.5

    # B팀 (ConvNeXt 챔피언)
    CKPT_PATH_B = './models/convnext_base-epoch=06-val_f1=0.9585.ckpt'
    IMG_SIZE_B = 512
    WEIGHT_B = 0.5

    # ... (이하 모든 코드는 inference_back_to_basics.py와 동일) ...
    TEST_DIR = './data/raw/test/'
    SUBMISSION_FILE = './data/raw/sample_submission.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Final Secret Weapon models...")
    model_a = CustomLightningModule.load_from_checkpoint(CKPT_PATH_A).to(DEVICE)
    model_b = CustomLightningModule.load_from_checkpoint(CKPT_PATH_B).to(DEVICE)
    model_a.eval()
    model_b.eval()
    print("Models loaded successfully.")

    submission_df = pd.read_csv(SUBMISSION_FILE)
    final_predictions = []

    for image_id in tqdm(submission_df['ID'], desc="Final Push"):
        image_path = os.path.join(TEST_DIR, image_id)
        image = cv2.imread(image_path)
        if image is None:
            final_predictions.append(0)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preds_a = predict_with_8_way_tta(model_a, image, IMG_SIZE_A, DEVICE)
        preds_b = predict_with_8_way_tta(model_b, image, IMG_SIZE_B, DEVICE)

        ensemble_preds = (preds_a * WEIGHT_A) + (preds_b * WEIGHT_B)
        final_prediction = torch.argmax(ensemble_preds, dim=0).item()
        final_predictions.append(final_prediction)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_secret_weapon_{timestamp}.csv'
    submission_df['target'] = final_predictions
    submission_df.to_csv(output_filename, index=False)
    print(f"\nSecret Weapon Inference finished! `{output_filename}` has been created.")