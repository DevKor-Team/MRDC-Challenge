import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ttach as tta

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
    new_state_dict = torch.load("./checkpoints/day2/model_val_logloss=0.18.ckpt")[
        "state_dict"
    ]
    model.load_state_dict(new_state_dict)

    # tta_model = tta.ClassificationTTAWrapper(
    #     model, tta.aliases.five_crop_transform(300, 300)
    # )

    test = pd.read_csv("/ssd/MRDC/test_rgb.csv")
    test_dataset = RiceDataset(
        TEST_IMAGE_FOLDER,
        test,
        train_mode=False,
        transforms=get_valid_transforms(),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=False,
    )

    m = nn.Softmax(dim=1)
    result = []
    for image in test_dataloader:
        # model_output = tta_model(image["x"].to("cuda"))
        model_output = model(image["x"].to("cuda"))
        result.append(m(model_output).detach().cpu().numpy())
    y_pred = np.concatenate(result)
    print(y_pred.shape)
    submission = pd.read_csv("/ssd/MRDC/SampleSubmission.csv")
    submission.loc[:, ["blast", "brown", "healthy"]] = y_pred
    submission.to_csv("submission.csv", index=False)
