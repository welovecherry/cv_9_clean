# # train.py
# import torch
# import hydra
# from omegaconf import DictConfig
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from datetime import datetime
# import wandb

# # 만든 모듈들을 import
# from data import CustomDataModule
# from lightning_model import CustomLightningModule

# # 설정파일(config.yaml)을 불러오기 위해 hydra 사용
# @hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
# def main(cfg: DictConfig):
#     # 1. 시드 고정 및 로거 설정
#     pl.seed_everything(cfg.train.seed)

#     wandb_logger = WandbLogger(
#         project=cfg.wandb.project,
#         entity=cfg.wandb.entity,
#     )
    
#     # [추가] 완디비 run의 이름을 우리가 원하는 형식으로 확실하게 설정
#     run_name = f"{cfg.model.name}-sz{cfg.data.image_size}-aug_{cfg.data.augmentation_level}-lr_{cfg.train.learning_rate}"
#     wandb_logger.experiment.name = run_name


#     # 2. 콜백 설정: 모델 저장 및 조기 종료
#     # ModelCheckpoint: val_f1 기준으로 가장 좋은 모델을 저장
#     checkpoint_callback = ModelCheckpoint(
#         monitor='val_f1',
#         mode='max',
#         dirpath='./models',
#         filename=f'{cfg.model.name}-{{epoch:02d}}-{{val_f1:.4f}}'
#     )

#     # # EarlyStopping: val_loss 기준으로 3번 동안 성능 향상이 없으면 학습 조기 종료
#     # early_stopping_callback = EarlyStopping(
#     #     monitor='val_loss', # 검증 데이터의 손실값 모니터링
#     #     mode='min',
#     #     patience=5 # 3번의 에폭 동안 개선이 없으면 조기 종료
#     # )
#     # src/train.py
#     early_stopping_callback = EarlyStopping(
#         monitor='val_loss', # 다시 val_loss를 감시하도록 변경
#         mode='min',
#         patience=5
#     )
#     data_module = CustomDataModule(
#         data_path=cfg.data.path,         # 'path' -> 'data_path'로 수정
#         image_size=cfg.data.image_size,  # 'image_size' 파라미터 추가
#         batch_size=cfg.data.batch_size,
#         num_workers=cfg.data.num_workers,
#         augmentation_level=cfg.data.augmentation_level # [핵심] 이 줄을 추가!
#     )

#     # FINETUNE_CKPT_PATH = './models/convnext_base-epoch=06-val_f1=0.9585.ckpt'
#     FINETUNE_CKPT_PATH = './models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt'

    
#     print(f"Loading model from {FINETUNE_CKPT_PATH} for fine-tuning...")
#     # 체크포인트로부터 모델을 불러와서 학습을 이어서 시작
#     model = CustomLightningModule.load_from_checkpoint(
#         FINETUNE_CKPT_PATH,
#         # config.yaml에 새로 정의된 아주 작은 학습률을 적용
#         learning_rate=cfg.train.learning_rate 
#     )


#     model = CustomLightningModule(
#         model_name=cfg.model.name,
#         learning_rate=cfg.train.learning_rate
#     )
 
#     # 4. Trainer(총 감독) 설정 및 학습 시작
#     trainer = pl.Trainer(
#         max_epochs=cfg.train.num_epochs,
#         logger=wandb_logger,
#         callbacks=[checkpoint_callback, early_stopping_callback],
#         accelerator='gpu',
#         devices=1
#     )

#     trainer.fit(model, datamodule=data_module)
#     wandb.finish()  # wandb 세션 종료


# src/train.py의 main 함수

# @hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
# def main(cfg: DictConfig):
#     pl.seed_everything(cfg.train.seed)

#     wandb_logger = WandbLogger(
#         project=cfg.wandb.project,
#         entity=cfg.wandb.entity,
#         name=cfg.wandb.name
#     )

#     checkpoint_callback = ModelCheckpoint(
#         monitor='val_f1',
#         mode='max',
#         dirpath='./models',
#         filename=f'finetuned-{cfg.model.name}-{{epoch:02d}}-{{val_f1:.4f}}'
#     )
#     early_stopping_callback = EarlyStopping(
#         monitor='val_loss',
#         mode='min',
#         patience=5
#     )

#     data_module = CustomDataModule(
#         data_path=cfg.data.path,
#         image_size=cfg.data.image_size,
#         batch_size=cfg.data.batch_size,
#         num_workers=cfg.data.num_workers,
#         augmentation_level=cfg.data.augmentation_level
#     )
    
#     # [핵심 수정] 1. 먼저, 새로운 '뇌'를 가진 깨끗한 모델을 만든다.
#     model = CustomLightningModule(
#         model_name=cfg.model.name,
#         learning_rate=cfg.train.learning_rate
#     )
    
#     # 2. '특별 과외'를 시킬 에이스 모델의 체크포인트 경로
#     FINETUNE_CKPT_PATH = './models/tf_efficientnetv2_s-epoch=21-val_f1=0.9565.ckpt' # 네 실제 파일 이름으로 확인/수정
    
#     print(f"Loading weights from {FINETUNE_CKPT_PATH} for fine-tuning...")
    
#     # 3. 체크포인트 파일에서 '뇌(state_dict)'만 꺼내서, 새 모델에 이식한다.
#     # strict=False는 마지막 분류층(classifier)의 모양이 달라도 에러를 내지 않게 하는 옵션
#     model.load_state_dict(torch.load(FINETUNE_CKPT_PATH)['state_dict'], strict=False)
 
#     # 4. Trainer 설정
#     trainer = pl.Trainer(
#         max_epochs=cfg.train.num_epochs,
#         logger=wandb_logger,
#         callbacks=[checkpoint_callback, early_stopping_callback],
#         accelerator='gpu',
#         devices=1
#     )

#     # 5. 학습 시작
#     trainer.fit(model, datamodule=data_module)
    
#     wandb.finish()

# if __name__ == "__main__":
#     main()


# src/train.py

# import hydra
# from omegaconf import DictConfig
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# import wandb

# # 만든 모듈들을 import
# from data import CustomDataModule
# from lightning_model import CustomLightningModule

# @hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
# def main(cfg: DictConfig):
#     # --- K-Fold 교차 검증을 위한 메인 루프 ---
#     for fold_num in range(cfg.train.n_splits):
#         print(f"========== FOLD {fold_num+1}/{cfg.train.n_splits} TRAINING START ==========")
        
#         # 1. 시드 고정 및 로거 설정
#         pl.seed_everything(cfg.train.seed)

#         # 각 fold마다 다른 이름으로 로그가 저장되도록 설정
#         wandb_logger = WandbLogger(
#             project=cfg.wandb.project,
#             entity=cfg.wandb.entity,
#             name=f"{cfg.model.name}-fold{fold_num}-sz{cfg.data.image_size}"
#         )

#         # 2. 콜백 설정
#         # 각 fold의 최고 모델이 다른 이름으로 저장되도록 filename 수정
#         checkpoint_callback = ModelCheckpoint(
#             monitor='val_f1',
#             mode='max',
#             dirpath='./models',
#             filename=f"{cfg.model.name}-fold{fold_num}-{{epoch:02d}}-{{val_f1:.4f}}"
#         )
#         early_stopping_callback = EarlyStopping(
#             monitor='val_loss',
#             mode='min',
#             patience=5
#         )

#         # 3. 데이터 모듈과 모델 인스턴스 생성
#         # [핵심] 이번 fold 번호를 DataModule에 전달
#         data_module = CustomDataModule(
#             data_path=cfg.data.path,
#             image_size=cfg.data.image_size,
#             batch_size=cfg.data.batch_size,
#             num_workers=cfg.data.num_workers,
#             augmentation_level=cfg.data.augmentation_level,
#             n_splits=cfg.train.n_splits,
#             fold_num=fold_num
#         )
#         model = CustomLightningModule(
#             model_name=cfg.model.name,
#             learning_rate=cfg.train.learning_rate
#         )
     
#         # 4. Trainer 설정
#         trainer = pl.Trainer(
#             max_epochs=cfg.train.num_epochs,
#             logger=wandb_logger,
#             callbacks=[checkpoint_callback, early_stopping_callback],
#             accelerator='gpu',
#             devices=1
#         )

#         # 5. 현재 fold 학습 시작
#         trainer.fit(model, datamodule=data_module)
        
#         # 6. 한 fold가 끝나면 완디비 세션 종료
#         wandb.finish()
        
#         print(f"========== FOLD {fold_num+1}/{cfg.train.n_splits} TRAINING FINISHED ==========\n")

# if __name__ == "__main__":
#     main()

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