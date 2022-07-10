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

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="model_{val_loss:.2f}",
        monitor="val_loss",
        verbose=True,
        save_last=False,
        save_top_k=1,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True, mode="min")
    if EARLY_STOPPING == True:
        callbacks.append(early_stopping)

    wandb_logger = WandbLogger(project="MRDC", name="Apply_SCE_Big_LR")

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1 if torch.cuda.is_available() else None,
        precision=16,
        max_epochs=EPOCHS,
        callbacks=callbacks,
    )

    core = RiceClassificationCore()
    dm = RiceDataModule(fold=0)
    model = RiceClassificationModule(
        hparams={"lr": LR, "batch_size": BATCH_SIZE}, core=core
    )

    trainer.use_native_amp = False
    trainer.fit(model, dm)

    print(checkpoint_callback.best_model_path, checkpoint_callback.best_model_score)
