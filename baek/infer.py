import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ttach as tta

from torch.utils.data import DataLoader

from model import RiceClassificationCore, RiceClassificationModule
from data import RiceDataset, RiceDataModule
from utils import get_transforms
from env import *

TTA = False

if __name__ == "__main__":
    core = RiceClassificationCore()
    model = RiceClassificationModule(hparams={}, core=core)
    model = model.to("cuda")
    _ = model.eval()
    new_state_dict = torch.load("./checkpoints/day3/model_val_logloss=0.10.ckpt")[
        "state_dict"
    ]
    model.load_state_dict(new_state_dict)
    m = nn.Softmax(dim=1)

    if TTA:
        model = tta.ClassificationTTAWrapper(
            model,
            tta.aliases.d4_transform(),
        )

    print("== Cross Validation ==")
    dm = RiceDataModule(fold=0)
    dm.setup()
    valid_loader = DataLoader(
        dm.val_ds,
        batch_size=2,
        num_workers=4,
        shuffle=False,
    )
    result = []
    y = []
    for image in valid_loader:
        model_output = model(image["x"].to("cuda"))
        result.append(model_output.detach().cpu().numpy())
        y.append(image["y"].numpy())
    y_pred = np.concatenate(result)
    y = np.concatenate(y)
    loss = nn.CrossEntropyLoss()

    CV = loss(torch.FloatTensor(y_pred), torch.LongTensor(y).squeeze().long()).item()
    print(f"CV Score = {CV:.4f}")
    y_pred = np.argmax(y_pred, axis=-1)
    from sklearn.metrics import classification_report

    print(classification_report(y, y_pred, target_names=["blast", "brown", "healthy"]))

    test = pd.read_csv("/ssd/MRDC/test_rgb.csv")
    test_dataset = RiceDataset(
        TEST_IMAGE_FOLDER,
        test,
        train_mode=False,
        transforms=get_transforms("valid"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=False,
    )

    result = []
    for image in test_dataloader:
        model_output = model(image["x"].to("cuda"))
        result.append(m(model_output).detach().cpu().numpy())
    y_pred = np.concatenate(result)
    print(y_pred.shape)
    submission = pd.read_csv("/ssd/MRDC/SampleSubmission.csv")
    submission.loc[:, ["blast", "brown", "healthy"]] = y_pred
    submission.to_csv("submission.csv", index=False)
