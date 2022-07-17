import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn.modules.loss import _WeightedLoss
from env import *


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=3):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets, reduction="mean"):
        onehot_targets = torch.eye(self.num_classes)[targets].cuda()
        ce_loss = F.cross_entropy(logits, targets, reduction=reduction)
        rce_loss = (-onehot_targets * logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == "mean":
            rce_loss = rce_loss.mean()
        elif reduction == "sum":
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def reduce_loss(self, loss):
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum()
            if self.reduction == "sum"
            else loss
        )

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


class RiceClassificationCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(MODEL_ARCH, pretrained=True)
        #         self.model = base_model

        #         Efficientnets
        # n_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(n_features, CLASSES)

        #         Resnets
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CLASSES)

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


class RiceClassificationModule(pl.LightningModule):
    def __init__(self, hparams, core):
        super().__init__()
        self.hparams.update(hparams)
        self.core = core
        self.criterion = SmoothCrossEntropyLoss(smoothing=0.1)
        self.logloss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.core(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.core.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=5,
                max_epochs=50,
            ),
            "interval": "step",
            "monitor": "train_loss",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_index):
        # One batch at a time
        features = batch["x"]
        targets = batch["y"]
        out = self(features)
        loss = self.criterion(out, targets.squeeze().long())
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        log_loss = self.logloss(out, targets.squeeze().long())
        self.log(
            "train_logloss",
            log_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_index):
        # One batch at a time
        features = batch["x"]
        targets = batch["y"]
        out = self(features)
        log_loss = self.logloss(out, targets.squeeze().long())
        self.log(
            "val_logloss",
            log_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


if __name__ == "__main__":
    core = RiceClassificationCore()
    pl_module = RiceClassificationModule(
        hparams={"lr": LR, "batch_size": BATCH_SIZE},
        core=core,
    )
