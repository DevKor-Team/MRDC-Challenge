import os
import random
import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

from env import *


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_augmentations():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_augmentations = albu.Compose(
        [
            albu.RandomResizedCrop(IMG_SIZE, IMG_SIZE, p=1.0),
            albu.Transpose(p=0.5),
            albu.HorizontalFlip(p=0.5),
            #         albu.VerticalFlip(0.5),
            albu.CoarseDropout(p=0.5),
            albu.Normalize(always_apply=True),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    valid_augmentations = albu.Compose(
        [
            albu.Resize(IMG_SIZE, IMG_SIZE),
            albu.Normalize(always_apply=True),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    return train_augmentations, valid_augmentations
