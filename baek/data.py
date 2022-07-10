import os
import cv2
import torch

import pandas as pd
import albumentations as albu
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from env import *
from utils import get_augmentations


class RiceDataset(Dataset):
    def __init__(self, data_dir, train, train_mode=True, transforms=None):
        self.data_dir = data_dir
        self.train = train
        self.transforms = transforms
        self.train_mode = train_mode

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.train.iloc[index].Image_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if not (self.train_mode):
            return {"x": image}

        return {
            "x": image,
            "y": torch.tensor(
                int(self.train.iloc[index]["Label"]),
                dtype=torch.float64,
            ),
        }


class RiceDataModule(pl.LightningDataModule):
    def __init__(self, fold):
        super().__init__()
        self.train_aug, self.val_aug = get_augmentations()
        self.fold = fold
        self.batch_size = BATCH_SIZE

    def setup(self, stage=None):
        folds = pd.read_csv("/ssd/MRDC/train_rgb.csv")
        train_fold = folds.loc[folds["fold"] != self.fold]
        val_fold = folds.loc[folds["fold"] == self.fold]

        self.train_ds = RiceDataset(
            TRAIN_IMAGE_FOLDER, train_fold, transforms=self.train_aug
        )
        self.val_ds = RiceDataset(TRAIN_IMAGE_FOLDER, val_fold, transforms=self.val_aug)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, num_workers=4, shuffle=False)


if __name__ == "__main__":
    import pandas as pd

    train = pd.read_csv("/ssd/MRDC/train_rgb.csv")
    ricedataset = RiceDataset(train)
    print(len(ricedataset))
    data = ricedataset.__getitem__(0)
    print(data["x"].shape)
    print(data["y"], data["y"].shape)

    dm = RiceDataModule(fold=0)
    dm.setup()
    td = dm.train_dataloader()
    print(len(td))
