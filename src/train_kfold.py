# train_kfold.py의 import 부분
from data_kfold import CustomDataModule # data.py -> data_kfold.py
from lightning_model import CustomLightningModule

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb


@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # --- K-Fold 교차 검증을 위한 메인 루프 ---
    for fold_num in range(cfg.train.n_splits):
        print(f"========== FOLD {fold_num+1}/{cfg.train.n_splits} TRAINING START ==========")
        
        pl.seed_everything(cfg.train.seed)

        # 각 fold마다 다른 이름으로 로그가 저장되도록 설정
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"kfold-{cfg.model.name}-fold{fold_num}-sz{cfg.data.image_size}"
        )

        # 각 fold의 최고 모델이 다른 이름으로 저장되도록 filename 수정
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            dirpath='./models',
            filename=f"kfold-{cfg.model.name}-fold{fold_num}-{{epoch:02d}}-{{val_f1:.4f}}"
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=5
        )

        # 이번 fold 번호를 DataModule에 전달
        data_module = CustomDataModule(
            data_path=cfg.data.path,
            image_size=cfg.data.image_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            augmentation_level=cfg.data.augmentation_level,
            n_splits=cfg.train.n_splits,
            fold_num=fold_num
        )
        model = CustomLightningModule(
            model_name=cfg.model.name,
            learning_rate=cfg.train.learning_rate
        )
     
        trainer = pl.Trainer(
            max_epochs=cfg.train.num_epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator='gpu',
            devices=1
        )

        trainer.fit(model, datamodule=data_module)
        wandb.finish()
        
        print(f"========== FOLD {fold_num+1}/{cfg.train.n_splits} TRAINING FINISHED ==========\n")

if __name__ == "__main__":
    main()