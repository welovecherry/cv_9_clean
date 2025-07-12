# scripts/confusion_matrix_v1.py

import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from src.lightning_model import CustomLightningModule
from src.data import CustomDataset

# ---- 사용자 정의 설정 ----
CKPT_PATH = './models/0708-convnext_base-sz384-epoch=19-val_f1=0.9950.ckpt'  # 저장된 최고 모델
VAL_CSV_PATH = './data/val.csv'  # 검증 데이터의 이미지 ID와 라벨이 담긴 CSV
IMG_DIR = './data/final_training_data/val'  # 이미지가 들어 있는 폴더
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_CLASSES = 17

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 데이터셋 정의 ----
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_dir, img_size):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['ID']
        label = self.dataframe.iloc[idx]['label']
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label

# ---- 데이터 불러오기 ----
df = pd.read_csv(VAL_CSV_PATH)
dataset = SimpleDataset(df, IMG_DIR, IMG_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- 모델 불러오기 ----
model = CustomLightningModule.load_from_checkpoint(CKPT_PATH).to(device)
model.eval()

all_preds = []
all_labels = []

# ---- 추론 실행 ----
with torch.no_grad():
    for imgs, labels in tqdm(dataloader, desc="Predicting"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ---- 혼동 행렬 계산 ----
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, output_dict=False)
print(report)

# ---- 시각화 ----
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('./confusion_matrix.png')
plt.show()
