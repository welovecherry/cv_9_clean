# src/lightning_model.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm        # PyTorch Image Models for pre-trained models
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import numpy as np


# LightningModule을 상속받아 사용자 정의 모델 클래스 생성
class CustomLightningModule(pl.LightningModule):
    def __init__(self, model_name, learning_rate, num_classes=17, class_weights=None):
        super().__init__() # 부모 클래스 초기화
        self.save_hyperparameters() # 넘겨받은 하이퍼파라미터 저장

        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=True, # 사전 학습된 가중치 사용
            num_classes=self.hparams.num_classes
        )
        # 손실 함수 정의
        self.loss_fn = nn.CrossEntropyLoss(weight=self.hparams.class_weights)
        self.f1_score = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro')
        self.accuracy = MulticlassAccuracy(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Mixup (두개의 이미지를 섞어서 새로운 이미지 생성해 데이터 증강)
        alpha = 0.4  # Mixup의 강도 조절값
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0 # 섞을 비율 랜덤하게
        batch_size = images.size()[0]
        index = torch.randperm(batch_size)
        
        images_a, images_b = images, images[index] 
        targets_a, targets_b = targets, targets[index]
        
        mixed_images = lam * images_a + (1 - lam) * images_b # lam 비율로 A, B 이미지 섞기
        
        preds = self(mixed_images)
        
        loss = lam * self.loss_fn(preds, targets_a) + (1 - lam) * self.loss_fn(preds, targets_b) # 믹스업에 맞게 손실 계산
        
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

    # 옵티마이저(가중치 학습시키는 도구)와 학습률 스케쥴러 설정
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate) # AdamW 옵티마이저 사용
        
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.trainer.max_epochs * steps_per_epoch

        # OneCycleLR 스케쥴러 설정 (학습률을 동적으로 조절)
        # 학습 초반에 학습률을 높였다가, 중간에 낮추고, 마지막에 다시 높이는 방식
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