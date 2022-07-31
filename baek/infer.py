import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ttach as tta
import timm

from torch.utils.data import DataLoader
from scipy.special import softmax

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from model import RiceClassificationCore, RiceClassificationModule
from data import RiceDataset, RiceDataModule
from utils import get_transforms
from env import *

TTA = False


class InferenceCore(nn.Module):
    def __init__(self, model_arch: str):
        super().__init__()
        self.model = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-base-384-22k-1k"
        )
        # self.model = timm.create_model(model_arch, pretrained=True)
        #         self.model = base_model

        # MobileNet
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, CLASSES)

        #         Resnets
        # n_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(n_features, CLASSES)

        # ViT
        # n_features = self.model.head.in_features
        # self.model.head = nn.Linear(n_features, CLASSES)

        self._freeze_batchnorm()  # NEW NEW NEW NEW NEW NEW NEW NEW NEW

    def _freeze_batchnorm(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
                module.eval()

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model_arch_list = [""]
    ckpt_list = ["./checkpoints/day4/convnext-tiny-224_val_logloss=0.09.ckpt"]
    weight = [1.0]
    model_list = []
    for model_arch, ckpt in zip(model_arch_list, ckpt_list):
        core = InferenceCore(model_arch)
        model = RiceClassificationModule(hparams={}, core=core)
        model = model.to("cuda")
        _ = model.eval()
        new_state_dict = torch.load(ckpt)["state_dict"]
        model.load_state_dict(new_state_dict)
        model_list.append(model)

    # if TTA:
    #     model = tta.ClassificationTTAWrapper(
    #         model,
    #         tta.aliases.d4_transform(),
    #     )

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
        model_output = []
        for model, w in zip(model_list, weight):
            model_output.append(
                model(image["x"].to("cuda"))
                .logits.detach()
                .cpu()
                .numpy()
                .reshape(-1, 3, 1)
                * w
            )
        model_output = np.concatenate(model_output, axis=-1).sum(axis=-1)
        result.append(model_output)
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
        model_output = []
        for model, w in zip(model_list, weight):
            model_output.append(
                model(image["x"].to("cuda"))
                .logits.detach()
                .cpu()
                .numpy()
                .reshape(-1, 3, 1)
                * w
            )
        model_output = np.concatenate(model_output, axis=-1).sum(axis=-1)
        result.append(softmax(model_output, axis=1))

    y_pred = np.concatenate(result)
    print(y_pred.shape)
    submission = pd.read_csv("/ssd/MRDC/SampleSubmission.csv")
    submission.loc[:, ["blast", "brown", "healthy"]] = y_pred
    submission.to_csv("submission.csv", index=False)
