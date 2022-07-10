import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model import RiceClassificationCore, RiceClassificationModule
from data import RiceDataset
from utils import get_augmentations, get_valid_transforms
from env import *

if __name__ == "__main__":
    core = RiceClassificationCore()
    model = RiceClassificationModule(
        hparams={"lr": LR, "batch_size": BATCH_SIZE}, core=core
    )
    model = model.to("cuda")
    _ = model.eval()
    new_state_dict = torch.load("./checkpoints/model_val_loss=0.49-v3.ckpt")[
        "state_dict"
    ]
    model.load_state_dict(new_state_dict)

    # _, test_aug = get_augmentations()
    test = pd.read_csv("/ssd/MRDC/test_rgb.csv")
    test_dataset = RiceDataset(
        TEST_IMAGE_FOLDER,
        test,
        train_mode=False,
        transforms=get_valid_transforms(),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
    )

    m = nn.Softmax(dim=1)
    result = []
    for x in test_dataloader:
        y = model(x["x"].to("cuda"))
        result.append(m(y).detach().cpu().numpy())

    y_pred = np.concatenate(result)

    submission = pd.read_csv("/ssd/MRDC/SampleSubmission.csv")
    submission.loc[:, ["blast", "brown", "healthy"]] = y_pred
    submission.to_csv("submission.csv", index=False)
