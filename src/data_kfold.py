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
        if image is None: return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return None, None
    return torch.utils.data.dataloader.default_collate(batch)

class CustomDataModule(pl.LightningDataModule):
    # K-Fold를 위한 n_splits와 fold_num 파라미터 추가
    def __init__(self, data_path: str, image_size: int, batch_size: int, num_workers: int, augmentation_level: str, n_splits: int, fold_num: int):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_level = augmentation_level
        self.n_splits = n_splits
        self.fold_num = fold_num

        strong_transform = [A.Resize(self.image_size, self.image_size), A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(p=0.5), A.CoarseDropout(p=0.3), A.Normalize(), ToTensorV2()]
        base_transform = [A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()]

        if self.augmentation_level == "strong":
            self.train_transform = A.Compose(strong_transform)
        else:
            self.train_transform = A.Compose(base_transform)
        
        self.val_transform = A.Compose([A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()])

    def setup(self, stage=None):
        image_data_root = os.path.join(self.data_path, 'train')
        original_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        
        exists = [os.path.exists(os.path.join(image_data_root, fname)) for fname in original_df['ID']]
        clean_df = original_df[exists].reset_index(drop=True)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        all_splits = list(skf.split(clean_df, clean_df['target']))
        train_idx, val_idx = all_splits[self.fold_num]
        
        train_df = clean_df.iloc[train_idx]
        val_df = clean_df.iloc[val_idx]
        
        self.train_dataset = ImageDataset(df=train_df, data_root=image_data_root, transform=self.train_transform)
        self.val_dataset = ImageDataset(df=val_df, data_root=image_data_root, transform=self.val_transform)
        print(f"Fold {self.fold_num}: Training on {len(self.train_dataset)}, validating on {len(self.val_dataset)}.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn)
