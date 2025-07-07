# src/data.py

import torch
import pytorch_lightning as pl
import pandas as pd
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df
        self.data_root = data_root
        self.transform = transform
        self.image_paths = self.df['ID'].values
        self.labels = self.df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.image_paths[idx])
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, image_size: int, batch_size: int, num_workers: int, augmentation_level: str):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_level = augmentation_level

        # [최종 수정] Albumentations만 사용한 '궁극의' 증강 파이프라인
        ultimate_transform = [
            A.Resize(self.image_size, self.image_size),
            A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.15, rotate_limit=20, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=0.5),
            # 현실적인 노이즈 추가 (ISO 노이즈, 블러, JPEG 압축)
            A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
            ], p=0.6),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
            A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ]
        
        # 설정값에 따라 증강 레벨 결정
        if self.augmentation_level == "ultimate":
            print("Using ULTIMATE (Albumentations only) augmentation pipeline.")
            self.train_transform = A.Compose(ultimate_transform)
        else: # 기본(base) 또는 강력한(strong)
            # 필요에 따라 다른 증강 레벨을 여기에 추가할 수 있음
            self.train_transform = A.Compose([A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()])
        
        self.val_transform = A.Compose([A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()])

    def setup(self, stage=None):
        image_data_root = os.path.join(self.data_path, 'train')
        original_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        exists = [os.path.exists(os.path.join(image_data_root, fname)) for fname in original_df['ID']]
        clean_df = original_df[exists].reset_index(drop=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(iter(skf.split(clean_df, clean_df['target'])))
        train_df, val_df = clean_df.iloc[train_idx], clean_df.iloc[val_idx]
        self.train_dataset = ImageDataset(df=train_df, data_root=image_data_root, transform=self.train_transform)
        self.val_dataset = ImageDataset(df=val_df, data_root=image_data_root, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)