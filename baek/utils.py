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


# def get_augmentations():
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)

#     train_augmentations = albu.Compose(
#         [
#             albu.RandomResizedCrop(IMG_SIZE, IMG_SIZE, p=1.0),
#             albu.Transpose(p=0.5),
#             albu.HorizontalFlip(p=0.5),
#             #         albu.VerticalFlip(0.5),
#             albu.CoarseDropout(p=0.5),
#             albu.Normalize(always_apply=True),
#             ToTensorV2(p=1.0),
#         ],
#         p=1.0,
#     )

#     valid_augmentations = albu.Compose(
#         [
#             albu.Resize(IMG_SIZE, IMG_SIZE),
#             albu.Normalize(always_apply=True),
#             ToTensorV2(p=1.0),
#         ],
#         p=1.0,
#     )

#     return train_augmentations, valid_augmentations


from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    IAAPerspective,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    RandomResizedCrop,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
    Normalize,
    Cutout,
    CoarseDropout,
    ShiftScaleRotate,
    CenterCrop,
    Resize,
)

from albumentations.pytorch import ToTensorV2


def get_transforms(data):
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(IMG_SIZE, IMG_SIZE),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [
                Resize(IMG_SIZE, IMG_SIZE),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


# def get_train_transforms():
#     return Compose(
#         [
#             RandomResizedCrop(IMG_SIZE, IMG_SIZE),
#             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             ShiftScaleRotate(p=0.5),
#             HueSaturationValue(
#                 hue_shift_limit=0.2,
#                 sat_shift_limit=0.2,
#                 val_shift_limit=0.2,
#                 p=0.5,
#             ),
#             RandomBrightnessContrast(
#                 brightness_limit=(-0.1, 0.1),
#                 contrast_limit=(-0.1, 0.1),
#                 p=0.5,
#             ),
#             Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#                 max_pixel_value=255.0,
#                 p=1.0,
#             ),
#             CoarseDropout(p=0.5),
#             Cutout(p=0.5),
#             ToTensorV2(p=1.0),
#         ],
#         p=1.0,
#     )


# def get_valid_transforms():
#     return Compose(
#         [
#             CenterCrop(350, 350, p=1.0),
#             Resize(350, 350),
#             Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#                 max_pixel_value=255.0,
#                 p=1.0,
#             ),
#             ToTensorV2(p=1.0),
#         ],
#         p=1.0,
#     )
