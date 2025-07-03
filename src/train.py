import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import wandb

# 만든 모듈들을 import
from data import CustomDataModule
from lightning_model import CustomLightningModule

# 설정파일(config.yaml)을 불러오기 위해 hydra 사용
@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # 1. 시드 고정 및 로거 설정
    pl.seed_everything(cfg.train.seed)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        # name=cfg.wandb.name
        name=f"{cfg.model.name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    )

    # 2. 콜백 설정: 모델 저장 및 조기 종료
    # ModelCheckpoint: val_f1 기준으로 가장 좋은 모델을 저장
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        dirpath='./models',
        filename=f'{cfg.model.name}-{{epoch:02d}}-{{val_f1:.4f}}'
    )

    # EarlyStopping: val_loss 기준으로 3번 동안 성능 향상이 없으면 학습 조기 종료
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', # 검증 데이터의 손실값 모니터링
        mode='min',
        patience=3 # 3번의 에폭 동안 개선이 없으면 조기 종료
    )

    # 3. 만들어 놓은 CustomDataModule와 CustomLightningModule 인스턴스 생성
    # data_module = CustomDataModule(
    #     path=cfg.data.path,
    #     batch_size=cfg.data.batch_size,
    #     num_workers=cfg.data.num_workers
    # )
    # src/train.py 의 main 함수 안
    data_module = CustomDataModule(
        data_path=cfg.data.path,         # 'path' -> 'data_path'로 수정
        image_size=cfg.data.image_size,  # 'image_size' 파라미터 추가
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )

    model = CustomLightningModule(
        model_name=cfg.model.name,
        learning_rate=cfg.train.learning_rate
    )
 
    # 4. Trainer(총 감독) 설정 및 학습 시작
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='gpu',
        devices=1
    )

    trainer.fit(model, datamodule=data_module)
    wandb.finish()  # wandb 세션 종료

if __name__ == "__main__":
    main()