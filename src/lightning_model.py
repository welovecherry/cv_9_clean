# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# import timm
# from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
# import numpy as np  # <-- 이 줄을 추가해줘!


# class CustomLightningModule(pl.LightningModule):
#     def __init__(self, model_name, learning_rate, num_classes=17):
#         super().__init__()
#         # 파라미터를 저장하면 나중에 체크포인트에서 자동으로 불러올 수 있어 편리
#         self.save_hyperparameters()

#         # 1. 모델 로드
#         self.model = timm.create_model(
#             self.hparams.model_name,
#             pretrained=True,
#             num_classes=self.hparams.num_classes
#         )

#         # 2. 손실 함수 정의 (CrossEntropyLoss는 다중 클래스 분류에 적합)
#         self.loss_fn = nn.CrossEntropyLoss()

#         # 3. 평가지표 정의
#         self.f1_score = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro')
#         self.accuracy = MulticlassAccuracy(num_classes=self.hparams.num_classes)

#     def forward(self, x):
#         return self.model(x)

#     # 한 배치에 대한 예측 -> 손실 계산 -> 로그 기록
#     # def training_step(self, batch, batch_idx):
#     #     # 훈련 데이터에 대한 로직
#     #     images, targets = batch
#     #     preds = self(images)
#     #     loss = self.loss_fn(preds, targets)

#     #     # 훈련 과정의 손실과 정확도를 기록
#     #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#     #     self.log('train_f1', self.f1_score(preds, targets), on_step=False, on_epoch=True, prog_bar=True, logger=True)
#     #     self.log('train_acc', self.accuracy(preds, targets), on_step=False, on_epoch=True, prog_bar=True, logger=True)

#     #     return loss

#     # def validation_step(self, batch, batch_idx):
#     #     # 검증 데이터에 대한 로직
#     #     images, targets = batch
#     #     preds = self(images)
#     #     loss = self.loss_fn(preds, targets)

#     #     # 검증 과정의 손실과 정확도를 기록
#     #     self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#     #     self.log('val_f1', self.f1_score(preds, targets), on_epoch=True, prog_bar=True, logger=True)
#     #     self.log('val_acc', self.accuracy(preds, targets), on_epoch=True, prog_bar=True, logger=True)

#     #     return loss

#     # src/lightning_model.py

#     # training_step 함수만 아래 코드로 교체
#     def training_step(self, batch, batch_idx):
#         images, targets = batch
        
#         # --- Mixup 구현 부분 ---
#         # 1. 믹스업 강도를 결정 (Beta 분포에서 0~1 사이의 값 랜덤 추출)
#         alpha = 0.4 
#         lam = np.random.beta(alpha, alpha)
        
#         # 2. 배치 안에서 데이터를 무작위로 섞을 인덱스 생성
#         batch_size = images.size()[0]
#         index = torch.randperm(batch_size)
        
#         # 3. 원본 이미지와 섞을 이미지를 준비
#         images_a, images_b = images, images[index]
#         targets_a, targets_b = targets, targets[index]
        
#         # 4. 두 이미지를 lam 비율로 섞어서 새로운 '믹스업 이미지' 생성
#         mixed_images = lam * images_a + (1 - lam) * images_b
        
#         # 5. 모델이 믹스업 이미지를 보고 예측
#         preds = self(mixed_images)
        
#         # 6. 정답도 똑같은 비율로 섞어서 손실(loss) 계산
#         loss = lam * self.loss_fn(preds, targets_a) + (1 - lam) * self.loss_fn(preds, targets_b)
#         # --- Mixup 끝 ---

#         # 로그 기록 (이전과 동일)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # Mixup은 train_acc, train_f1을 직접 계산하기 어려우므로, 여기서는 생략하거나 근사치를 사용
#         # self.log('train_f1', ...) 
        
#         return loss

#     # def configure_optimizers(self):
#     #     # 4. 옵티마이저 설정 (베이스라인에서 사용한 Adam 옵티마이저)
#     #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#     #     return optimizer


#     # def configure_optimizers(self):
#     #     # 1. AdamW 옵티마이저 사용 (보통 Adam보다 안정적)
#     #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
#     #     # 2. CosineAnnealingLR 스케줄러 추가
#     #     # 학습률을 코사인 함수처럼 부드럽게 줄여나가서, 모델이 최적점에 잘 수렴하도록 도와줌
#     #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     #         optimizer,
#     #         T_max=self.trainer.max_epochs, # 전체 에폭 수
#     #         eta_min=1e-6 # 최소 학습률
#     #     )
        
#     #     return {
#     #         "optimizer": optimizer,
#     #         "lr_scheduler": {
#     #             "scheduler": scheduler,
#     #             "interval": "epoch",
#     #         },
#     #     }

#     # src/lightning_model.py # 비팀 완디비 개선 시도하기

#     def configure_optimizers(self):
#         # 1. AdamW 옵티마이저 사용
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
#         # 2. OneCycleLR 스케줄러 설정
#         # 이 스케줄러는 PyTorch Lightning이 제공하는 '학습 과정 정보'를 필요로 함
#         # 그래서 trainer.datamodule을 통해 전체 스텝 수를 계산
#         steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
#         total_steps = self.trainer.max_epochs * steps_per_epoch

#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer,
#             max_lr=self.hparams.learning_rate, # 우리가 설정한 학습률을 '최대 학습률'로 사용
#             total_steps=total_steps
#         )
        
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "step", # 매 스텝마다 학습률을 조절
#             },
#         }


# src/lightning_model.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import numpy as np

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, num_classes=17, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=True,
            num_classes=self.hparams.num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=self.hparams.class_weights)
        self.f1_score = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro')
        self.accuracy = MulticlassAccuracy(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Mixup (이전 버전에서 추가했던 것)
        alpha = 0.4 
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size = images.size()[0]
        index = torch.randperm(batch_size)
        
        images_a, images_b = images, images[index]
        targets_a, targets_b = targets, targets[index]
        
        mixed_images = lam * images_a + (1 - lam) * images_b
        
        preds = self(mixed_images)
        
        loss = lam * self.loss_fn(preds, targets_a) + (1 - lam) * self.loss_fn(preds, targets_b)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # [핵심] 빠져있던 validation_step 함수 추가
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)

        # 검증 과정의 점수들을 기록
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.f1_score(preds, targets), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(preds, targets), on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.trainer.max_epochs * steps_per_epoch

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }