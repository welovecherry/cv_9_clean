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