# src/train.py

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import torch

# 만든 모듈들을 import
from data import CustomDataModule
from lightning_model import CustomLightningModule

@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # 1. 시드 고정
    pl.seed_everything(cfg.train.seed)

    # 2. 완디비(W&B) 로거 설정
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name # config에 정의된 이름을 사용
    )

    # 3. 콜백 설정으로 학습 중 자동으로 실행되는 알람/저장 기능
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        dirpath='./models',
        filename=f"{cfg.wandb.name}-{{epoch:02d}}-{{val_f1:.4f}}"
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5 # 5번의 에폭 동안 개선이 없으면 조기 종료
    )

    # 4. 데이터 모듈과 모델 인스턴스 생성
    data_module = CustomDataModule(
        data_path=cfg.data.path,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers, # 데이터를 병렬로 읽는 CPU 스레드 수
        augmentation_level=cfg.data.augmentation_level
    )
    
    model = CustomLightningModule(
        model_name=cfg.model.name,
        learning_rate=cfg.train.learning_rate
    )
 
    # 5. Trainer 설정
    # [추가] GPU 성능 최적화를 위한 설정
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback], # 위에서 만든 자동저장, 조기종료 기능
        accelerator='gpu',
        devices=1
    )

    # 6. 학습 시작
    trainer.fit(model, datamodule=data_module)
    
    # 7. 로깅 종료
    wandb.finish()

if __name__ == "__main__":
    main()