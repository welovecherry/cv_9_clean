# train.py

import hydra # 설정 파일 관리, 실험 편하게 실행하도록
from omegaconf import DictConfig
import pytorch_lightning as pl # 학습 루프를 자동으로 관리해주는 프레임워크
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import torch
import os

# 사용자 정의 모듈 import
from data import CustomDataModule
from lightning_model import CustomLightningModule

# 하이드라로 config 파일을 불러와서 모델과 데이터셋을 초기화한 후에 Trainer.fit()로 학습 시작하는 코드
@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # 1. 시드 고정
    pl.seed_everything(cfg.train.seed)

    # 2. W&B 로깅 설정
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
    )

    # 3. 콜백 설정 (모델 저장 + 조기 종료)
    checkpoint_dir = './models'
    os.makedirs(checkpoint_dir, exist_ok=True) # 폴더 없으면 자동 생성

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 또는 'val_f1'
        mode='min',
        dirpath=checkpoint_dir,
        filename=f"{cfg.wandb.name}-{{epoch:02d}}-{{val_f1:.4f}}"
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # 또는 'val_f1'
        mode='min',
        patience=5,
        verbose=True
    )

    # 4. 클래스 가중치 로드
    class_weights_path = os.path.join(os.path.dirname(__file__), '..', 'class_weights.pth')
    class_weights = torch.load(class_weights_path)

    # 5. 데이터 파일 존재 확인
    train_csv_path = os.path.join(cfg.data.path, 'train.csv')
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"[오류] 학습 데이터 파일이 존재하지 않습니다: {train_csv_path}")

    # 6. 데이터 모듈 초기화
    data_module = CustomDataModule(
        data_path=cfg.data.path,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        augmentation_level=cfg.data.augmentation_level
    )

    # 7. 모델 초기화
    model = CustomLightningModule(
        model_name=cfg.model.name,
        learning_rate=cfg.train.learning_rate,
        num_classes=cfg.model.num_classes,
        class_weights=class_weights,
        label_smoothing=cfg.train.label_smoothing,
        max_epochs=cfg.train.max_epochs
    )

    # 8. GPU 최적화
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    # 9. Trainer 설정
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    # 10. 체크포인트에서 이어서 학습할 경우
    ckpt_path = cfg.train.get("ckpt_path", None)
    if ckpt_path and os.path.exists(ckpt_path):
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=data_module)

    # 11. W&B 종료
    wandb.finish()

if __name__ == "__main__":
    main()