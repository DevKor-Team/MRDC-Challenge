import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from env import *

from model import RiceClassificationCore, RiceClassificationModule
from data import RiceDataModule
from utils import seed_everything

if __name__ == "__main__":
    seed_everything(42)

    MODEL_ARCH = "convnext-large-384-22k-1k"
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/day5/",
        filename=MODEL_ARCH + "_{val_logloss:.2f}",
        monitor="val_logloss",
        verbose=True,
        save_last=False,
        save_top_k=1,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping("val_logloss", patience=10, verbose=True, mode="min")
    if EARLY_STOPPING == True:
        callbacks.append(early_stopping)

    wandb_logger = WandbLogger(project="MRDC_DAY5", name=MODEL_ARCH + "_4")

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1 if torch.cuda.is_available() else None,
        precision=16,
        max_epochs=EPOCHS,
        callbacks=callbacks,
    )

    core = RiceClassificationCore()
    dm = RiceDataModule(fold=4)
    model = RiceClassificationModule(
        hparams={"lr": LR, "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY},
        core=core,
    )

    trainer.use_native_amp = False
    trainer.fit(model, dm)

    print(checkpoint_callback.best_model_path, checkpoint_callback.best_model_score)
