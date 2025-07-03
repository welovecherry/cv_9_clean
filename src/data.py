# import pytorch_lightning as pl
# import pandas as pd
# import cv2  # opencv-python 패키지
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold  # 데이터셋을 클래스 비율대로 나눠주는 도구
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# ''' 
# 문서 이미지 데이터를 불러와서, 학습용-검증용으로 나누고, 나중에 학습할 수 있도록 돕는 
# 파이토치 라이트닝 데이터 관리 클래스 
# '''

# class CustomDataset(Dataset):
#     """
#     데이터프레임과 이미지 루트 경로를 받아, 인덱스에 해당하는 이미지와 라벨을 반환하는 클래스.
#     """
#     def __init__(self, df, data_root, transform=None):
#         super().__init__()
#         self.df = df
#         self.data_root = data_root  # 'data/raw/train' 같은 이미지 폴더 경로
#         self.transform = transform  # Albumentations 이미지 변환기
        
#         # 데이터프레임에서 이미지 파일 이름과 라벨을 미리 리스트로 저장해두면 속도가 빨라짐.
#         self.image_paths = self.df['ID'].values
#         self.labels = self.df['target'].values

#     def __len__(self):
#         # 이 데이터셋의 총 아이템 개수를 반환.
#         return len(self.df)

#     def __getitem__(self, idx):
#         # DataFrame에서 특정 주소(id)에 해당하는 이미지와 라벨을 가져오는 메서드.
        
#         # 1. idx에 해당하는 이미지 경로와 라벨을 가져오기
#         image_path = f"{self.data_root}/{self.image_paths[idx]}"
#         label = self.labels[idx]

#         # 2. 이미지 파일을 실제로 읽어오기 (OpenCV 사용)
#         # 이미지를 BGR 순서로 읽어오므로, 일반적인 RGB 순서로 바꿔줘야함.
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # 3. 이미지 증강 및 전처리 적용
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']
        
#         # 4. 이미지와 라벨을 쌍으로 묶어서 반환!
#         # 이렇게 두 개를 반환해야 모델이 받아서 학습할 수 있음
#         return image, label

# ''' 모든 데이터 관리 로직(경로 설정, 데이터 분할, 로더 생성) '''
# class CustomDataModule(pl.LightningDataModule):
#     def __init__(self, path, batch_size, num_workers):
#         super().__init__()
#         self.path = path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
        
#         # 이미지 크기 조절, 정규화, 텐서 변환을 위한 기본 변환기
#         self.transform = A.Compose([
#             A.Resize(224, 224),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#     def setup(self, stage=None):
#         df = pd.read_csv(f"{self.path}/train.csv")
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
#         train_idx, val_idx = next(iter(skf.split(df, df['target'])))
        
#         train_df = df.iloc[train_idx]
#         val_df = df.iloc[val_idx]

#         # 만들어 놓은 CustomDataset 클래스를 사용.
#         image_data_root = f"{self.path}/train"
#         self.train_dataset = CustomDataset(df=train_df, data_root=image_data_root, transform=self.transform)
#         self.val_dataset = CustomDataset(df=val_df, data_root=image_data_root, transform=self.transform)

#         print(f"Train/Val 데이터셋 분리 및 생성 완료. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

#     def train_dataloader(self):
#         # DataLoader를 반환.
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers
#         )

#     def val_dataloader(self):
#         # DataLoader를 반환.
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers
#         )

''' 버전 2'''
# src/data.py

# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets
# from torchvision.transforms import v2 as T # 최신 버전의 transform 사용
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # Albumentations 변환을 PyTorch의 v2 Transform과 함께 사용하기 위한 래퍼 클래스
# class AlbumentationsTransform:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, img):
#         # PIL 이미지를 numpy 배열로 변환
#         img = np.array(img)
#         # Albumentations 변환 적용
#         return self.transform(image=img)['image']

# class CustomDataModule(pl.LightningDataModule):
#     def __init__(self, data_path: str, image_size: int = 640, batch_size: int = 32, num_workers: int = 4):
#         super().__init__()
#         # 파라미터 저장
#         self.data_path = data_path
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         # 1. 학습(Train) 데이터용 변환기 정의
#         # 완디비 기록을 보면 train_f1은 1.0인데 val_f1은 0.85라서 약간의 과적합 신호다.
#         # 이걸 해결하기 위해 학습 데이터에 변형을 주는 '데이터 증강'을 추가하기.
#         self.train_transform = A.Compose([
#             A.Resize(self.image_size, self.image_size),
#             A.HorizontalFlip(p=0.5), # 50% 확률로 좌우반전
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5), # 색상 변형
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#         # 2. 검증(Validation) 데이터용 변환기 정의 (데이터 증강 없음!)
#         self.val_transform = A.Compose([
#             A.Resize(self.image_size, self.image_size),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#     def setup(self, stage=None):
#         # 1. ImageFolder를 사용해 폴더에서 전체 데이터를 불러옴
#         # 이 한 줄이 CSV 읽고 CustomDataset 만드는 걸 다 대체해!
#         full_dataset = datasets.ImageFolder(self.data_path)

#         # 2. 전체 데이터를 train/val로 8:2 분리
#         train_size = int(0.8 * len(full_dataset))
#         val_size = len(full_dataset) - train_size
#         self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

#         # 3. 분리된 데이터셋에 각각 다른 변환기(transform)를 적용
#         # PyTorch의 v2 Transform을 사용해 이 과정을 더 깔끔하게 만듦
#         self.train_dataset = T.Compose([T.ToPILImage(), AlbumentationsTransform(self.train_transform)])
#         self.val_dataset = T.Compose([T.ToPILImage(), AlbumentationsTransform(self.val_transform)])


#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


''' 버전 3 최고 점수: 0.91648 / 하지만 과적합'''
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


# class CustomDataModule(pl.LightningDataModule):
#     def __init__(self, data_path: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 4):
#         super().__init__()
#         self.data_path = data_path
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         self.train_transform = A.Compose([
#             A.Resize(self.image_size, self.image_size),
#             A.HorizontalFlip(p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#         self.val_transform = A.Compose([
#             A.Resize(self.image_size, self.image_size),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])

#     def setup(self, stage=None):
#         # 1. ImageFolder로 전체 데이터를 불러오되, 아직 transform은 적용하지 않음
#         full_dataset = datasets.ImageFolder(self.data_path)

#         # 2. train/val로 8:2 분리 (결과는 Subset 객체)
#         train_size = int(0.8 * len(full_dataset))
#         val_size = len(full_dataset) - train_size
#         train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

#         # 3. [수정된 부분]
#         # 분리된 각 Subset에 우리가 만든 TransformedDataset 래퍼를 씌워
#         # 각각 다른 transform을 적용함
#         self.train_dataset = TransformedDataset(train_subset, self.train_transform)
#         self.val_dataset = TransformedDataset(val_subset, self.val_transform)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

# 버전 4
# src/data.py

import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import augraphy as aug

# [추가] Augraphy 효과를 감싸는 중간 관리자(wrapper) 함수들
# kwargs를 사용해 불필요한 파라미터(예: shape)를 무시하고 image만 전달
def apply_inkbleed(image, **kwargs):
    return aug.InkBleed(p=1.0)(image, force=True)["output"]

def apply_dirtyrollers(image, **kwargs):
    return aug.DirtyRollers(p=1.0)(image, force=True)["output"]

def apply_letterpress(image, **kwargs):
    return aug.Letterpress(p=1.0)(image, force=True)["output"]

# TransformedDataset 클래스는 이전과 동일
class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = np.array(x)
        if self.transform:
            x = self.transform(image=x)['image']
        return x, y

    def __len__(self):
        return len(self.subset)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, image_size: int = 384, batch_size: int = 16, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = A.Compose([
            # 1. 크기 조절
            A.Resize(self.image_size, self.image_size),

            # 2. 기하학적 변형
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.HorizontalFlip(p=0.5),

            # 3. 품질 및 노이즈 변형
            A.OneOf([
                A.GaussianBlur(p=1.0),
                A.GaussNoise(p=1.0),
                A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
            ], p=0.5),

            # 4. 색상 변형
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            # 5. [수정] 문제가 발생한 Augraphy 부분은 잠시 제외
            
            # 6. 정규화 및 텐서 변환
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # val_transform은 그대로 둠
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def setup(self, stage=None):
        # 1. ImageFolder로 전체 데이터를 불러오되, 아직 transform은 적용하지 않음
        full_dataset = datasets.ImageFolder(self.data_path)

        # 2. train/val로 8:2 분리 (결과는 Subset 객체)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # 3. [수정된 부분]
        # 분리된 각 Subset에 우리가 만든 TransformedDataset 래퍼를 씌워
        # 각각 다른 transform을 적용함
        self.train_dataset = TransformedDataset(train_subset, self.train_transform)
        self.val_dataset = TransformedDataset(val_subset, self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
