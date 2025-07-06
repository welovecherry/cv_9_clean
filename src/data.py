
# ''' 버전 3 최고 점수: 0.91648 / 하지만 과적합'''
# # src/data.py
# import pytorch_lightning as pl
# import numpy as np
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import datasets
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # 데이터셋에 transform을 적용하기 위한 작은 래퍼(wrapper) 클래스
# class TransformedDataset(Dataset):
#     def __init__(self, subset, transform):
#         self.subset = subset
#         self.transform = transform

#     def __getitem__(self, idx):
#         # 1. subset에서 원본 이미지와 라벨을 가져옴 (이때는 PIL 이미지)
#         x, y = self.subset[idx]
        
#         # 2. PIL 이미지를 numpy 배열로 변환
#         x = np.array(x)

#         # 3. Albumentations 변환 적용
#         if self.transform:
#             x = self.transform(image=x)['image']
        
#         return x, y

#     def __len__(self):
#         return len(self.subset)


# # src/data.py의 CustomDataModule 클래스 부분만 교체
# class CustomDataModule(pl.LightningDataModule):
#     # __init__ 함수에 augmentation_level 파라미터 추가
#     def __init__(self, data_path: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4, augmentation_level: str = "base"):
#         super().__init__()
#         self.data_path = data_path
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.augmentation_level = augmentation_level

#         # 기본(base) 증강
#         base_transform = [
#             A.Resize(self.image_size, self.image_size),
#             A.HorizontalFlip(p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ]

#         # 강력한(strong) 증강
#         strong_transform = [
#             A.Resize(self.image_size, self.image_size),
#             A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
#             A.Perspective(scale=(0.05, 0.1), p=0.3),
#             A.HorizontalFlip(p=0.5),
#             A.OneOf([
#                 A.GaussianBlur(p=1.0),
#                 A.GaussNoise(p=1.0),
#             ], p=0.4),
#             A.ColorJitter(p=0.5),
#             A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ]
        
#         # 설정값에 따라 어떤 증강을 사용할지 결정
#         if self.augmentation_level == "strong":
#             self.train_transform = A.Compose(strong_transform)
#         else:
#             self.train_transform = A.Compose(base_transform)

#         # 검증 데이터에는 증강을 적용하지 않음
#         self.val_transform = A.Compose([
#             A.Resize(self.image_size, self.image_size),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#     # ... (setup, dataloader 함수들은 이전과 동일) ...
#     def setup(self, stage=None):
#         df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         train_idx, val_idx = next(iter(skf.split(df, df['target'])))
#         train_df = df.iloc[train_idx]
#         val_df = df.iloc[val_idx]
#         image_data_root = os.path.join(self.data_path, 'train')
#         self.train_dataset = ImageDataset(df=train_df, data_root=image_data_root, transform=self.train_transform)
#         self.val_dataset = ImageDataset(df=val_df, data_root=image_data_root, transform=self.val_transform)
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)




# # 버전 4
# # src/data.py

# # import pytorch_lightning as pl
# # import numpy as np
# # from torch.utils.data import Dataset, DataLoader, random_split
# # from torchvision import datasets
# # import albumentations as A
# # from albumentations.pytorch import ToTensorV2
# # import augraphy as aug

# # # [추가] Augraphy 효과를 감싸는 중간 관리자(wrapper) 함수들
# # # kwargs를 사용해 불필요한 파라미터(예: shape)를 무시하고 image만 전달
# # def apply_inkbleed(image, **kwargs):
# #     return aug.InkBleed(p=1.0)(image, force=True)["output"]

# # def apply_dirtyrollers(image, **kwargs):
# #     return aug.DirtyRollers(p=1.0)(image, force=True)["output"]

# # def apply_letterpress(image, **kwargs):
# #     return aug.Letterpress(p=1.0)(image, force=True)["output"]

# # # TransformedDataset 클래스는 이전과 동일
# # class TransformedDataset(Dataset):
# #     def __init__(self, subset, transform):
# #         self.subset = subset
# #         self.transform = transform

# #     def __getitem__(self, idx):
# #         x, y = self.subset[idx]
# #         x = np.array(x)
# #         if self.transform:
# #             x = self.transform(image=x)['image']
# #         return x, y

# #     def __len__(self):
# #         return len(self.subset)

# # class CustomDataModule(pl.LightningDataModule):
# #     def __init__(self, data_path: str, image_size: int = 384, batch_size: int = 16, num_workers: int = 4):
# #         super().__init__()
# #         self.data_path = data_path
# #         self.image_size = image_size
# #         self.batch_size = batch_size
# #         self.num_workers = num_workers

# #         self.train_transform = A.Compose([
# #             # 1. 크기 조절
# #             A.Resize(self.image_size, self.image_size),

# #             # 2. 기하학적 변형
# #             A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
# #             A.Perspective(scale=(0.05, 0.1), p=0.3),
# #             A.HorizontalFlip(p=0.5),

# #             # 3. 품질 및 노이즈 변형
# #             A.OneOf([
# #                 A.GaussianBlur(p=1.0),
# #                 A.GaussNoise(p=1.0),
# #                 A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
# #             ], p=0.5),

# #             # 4. 색상 변형
# #             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

# #             # 5. [수정] 문제가 발생한 Augraphy 부분은 잠시 제외
            
# #             # 6. 정규화 및 텐서 변환
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2(),
# #         ])

# #         # val_transform은 그대로 둠
# #         self.val_transform = A.Compose([
# #             A.Resize(self.image_size, self.image_size),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2(),
# #         ])
    
# #     def setup(self, stage=None):
# #         # 1. ImageFolder로 전체 데이터를 불러오되, 아직 transform은 적용하지 않음
# #         full_dataset = datasets.ImageFolder(self.data_path)

# #         # 2. train/val로 8:2 분리 (결과는 Subset 객체)
# #         train_size = int(0.8 * len(full_dataset))
# #         val_size = len(full_dataset) - train_size
# #         train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# #         # 3. [수정된 부분]
# #         # 분리된 각 Subset에 우리가 만든 TransformedDataset 래퍼를 씌워
# #         # 각각 다른 transform을 적용함
# #         self.train_dataset = TransformedDataset(train_subset, self.train_transform)
# #         self.val_dataset = TransformedDataset(val_subset, self.val_transform)

# #     def train_dataloader(self):
# #         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

# #     def val_dataloader(self):
# #         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


# 0704 버전
# src/data.py
# src/data.py

# import torch
# import pytorch_lightning as pl
# import pandas as pd
# import os
# import cv2
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm

# # ImageDataset은 이제 항상 유효한 경로만 받으므로, try-except가 필요 없음
# class ImageDataset(Dataset):
#     def __init__(self, df, data_root, transform=None):
#         self.df = df
#         self.data_root = data_root
#         self.transform = transform
#         self.image_paths = self.df['ID'].values
#         self.labels = self.df['target'].values

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.data_root, self.image_paths[idx])
#         label = self.labels[idx]
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         if self.transform:
#             image = self.transform(image=image)['image']
#         return image, label

# class CustomDataModule(pl.LightningDataModule):
#     # __init__ 함수는 이전과 동일
#     def __init__(self, data_path: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4, augmentation_level: str = "base"):
#         super().__init__()
#         self.data_path = data_path
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.augmentation_level = augmentation_level

#         # ... (transform 정의 부분은 이전과 동일) ...
#         base_transform = [A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()]
#         strong_transform = [A.Resize(self.image_size, self.image_size), A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(p=0.5), A.CoarseDropout(), A.Normalize(), ToTensorV2()]
        
#         if self.augmentation_level == "strong":
#             self.train_transform = A.Compose(strong_transform)
#         else:
#             self.train_transform = A.Compose(base_transform)

#         self.val_transform = A.Compose([A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()])


#     def setup(self, stage=None):
#         # [핵심 수정] 데이터 로드 전, 실제 파일이 있는 목록만 필터링
#         image_data_root = os.path.join(self.data_path, 'train')
#         original_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        
#         print("Checking for valid image files...")
#         # apply 함수를 사용해 각 파일의 존재 여부를 확인
#         exists = [os.path.exists(os.path.join(image_data_root, fname)) for fname in tqdm(original_df['ID'])]
        
#         # 실제 존재하는 파일 정보만 담은 새로운 데이터프레임을 생성
#         clean_df = original_df[exists].reset_index(drop=True)
        
#         print(f"Found {len(clean_df)} valid images out of {len(original_df)}.")

#         # 이제 깨끗한 데이터프레임으로 train/val 분리
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         train_idx, val_idx = next(iter(skf.split(clean_df, clean_df['target'])))
        
#         train_df = clean_df.iloc[train_idx]
#         val_df = clean_df.iloc[val_idx]
        
#         self.train_dataset = ImageDataset(df=train_df, data_root=image_data_root, transform=self.train_transform)
#         self.val_dataset = ImageDataset(df=val_df, data_root=image_data_root, transform=self.val_transform)

#     # 이제 collate_fn이 필요 없으므로 원래대로 복구
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# src/data.py

import torch
import pytorch_lightning as pl
import pandas as pd
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets # ImageFolder를 위해 필요
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- CSV 파일을 읽기 위한 Dataset ---
class CsvDataset(Dataset):
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

# --- ImageFolder 데이터셋에 transform을 적용하기 위한 래퍼 ---
class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = np.array(x)
        if self.transform:
            x = self.transform(image=x)['image']
        return x, y

# --- 똑똑한 데이터 모듈 ---
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, image_size: int, batch_size: int, num_workers: int, augmentation_level: str):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_level = augmentation_level

        # --- 증강 파이프라인 정의 ---
        base_transform = [A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()]
        strong_transform = [
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.CoarseDropout(p=0.3),
            A.Normalize(),
            ToTensorV2()
        ]

        if self.augmentation_level == "strong":
            self.train_transform = A.Compose(strong_transform)
        else:
            self.train_transform = A.Compose(base_transform)
        
        self.val_transform = A.Compose([A.Resize(self.image_size, self.image_size), A.Normalize(), ToTensorV2()])

    def setup(self, stage=None):
        # [핵심] 데이터 경로에 'train.csv'가 있는지 확인
        train_csv_path = os.path.join(self.data_path, 'train.csv')
        
        if os.path.exists(train_csv_path):
            # --- 일반 학습 모드 (CSV 사용) ---
            print(f"Found train.csv in {self.data_path}. Loading data using CSV.")
            original_df = pd.read_csv(train_csv_path)
            image_data_root = os.path.join(self.data_path, 'train')
            
            exists = [os.path.exists(os.path.join(image_data_root, fname)) for fname in tqdm(original_df['ID'], desc="Validating files")]
            clean_df = original_df[exists].reset_index(drop=True)
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_idx, val_idx = next(iter(skf.split(clean_df, clean_df['target'])))
            
            train_df = clean_df.iloc[train_idx]
            val_df = clean_df.iloc[val_idx]
            
            self.train_dataset = CsvDataset(df=train_df, data_root=image_data_root, transform=self.train_transform)
            self.val_dataset = CsvDataset(df=val_df, data_root=image_data_root, transform=self.val_transform)
        else:
            # --- 파인튜닝 모드 (ImageFolder 사용) ---
            print(f"No train.csv found. Loading data using ImageFolder from {self.data_path}.")
            full_dataset = datasets.ImageFolder(self.data_path)
            
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

            self.train_dataset = TransformedDataset(train_subset, self.train_transform)
            self.val_dataset = TransformedDataset(val_subset, self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)