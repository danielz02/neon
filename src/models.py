from __future__ import annotations

import torch
import wandb
import os.path
import argparse
import numpy as np
from torch import nn
from typing import List, Tuple
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from dataset import IndianaDataset
from torch.nn import functional as F
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from utils import des_scatter_plot, WandbImageCallback
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support


class LSTM(pl.LightningModule):
    def __init__(
        self, label_name: str | None, seq_len: int | None, num_layers: int = 2, hidden_size: int = 64,
        in_features: int = 1, bidirectional: bool = False, lr: float = 1e-3, clf_classes: List[str] = None
    ):
        super().__init__()
        self.seq_len = seq_len
        self.learning_rate = lr
        self.label_name = label_name
        self.n_features = in_features
        self.class_names = clf_classes
        self.fig_val, self.ax_val = plt.subplots()
        self.fig_test, self.ax_test = plt.subplots()
        self.mode = "regression" if clf_classes is None else "classification"

        self.encoder = nn.LSTM(self.n_features, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        if self.mode == "regression":
            self.head = nn.Sequential(
                nn.Linear(hidden_size * seq_len, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, len(clf_classes))
            )

        self.save_hyperparameters()

    def forward(self, x):
        n, *_ = x.shape
        x, hidden = self.encoder(x)
        if self.mode == "regression":
            x = self.head(x.reshape(n, -1))
        else:
            x = self.head(x[:, -1, :])
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-2
        )
        return optimizer

    def __common_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y_true = batch
        y_pred = self.forward(x)
        if self.mode == "classification":
            loss = F.cross_entropy(y_pred, y_true)
        else:
            loss = F.mse_loss(y_pred.reshape(-1), y_true.reshape(-1))

        return loss, y_pred, y_true

    def __evaluation(self, y_pred, y_true, stage):
        assert stage == "train" or stage == "validation" or stage == "test"

        if self.mode == "regression":
            r2 = np.corrcoef(y_true.detach().cpu().reshape(-1), y_pred.detach().cpu().reshape(-1))[0, 1] ** 2
            self.log(f"{stage}/r2", r2, prog_bar=True, on_step=True, on_epoch=True)
            if stage == "validation":
                self.ax_val.clear()
                des_scatter_plot(
                    y_pred=y_pred, y_true=y_true, label_name=self.label_name, save_path=None, ax=self.ax_val
                )
            if stage == "test":
                self.ax_test.clear()
                des_scatter_plot(
                    y_pred=y_pred, y_true=y_true, label_name=self.label_name, save_path=None, ax=self.ax_test
                )
        else:
            y_prob = F.softmax(y_pred, dim=1).detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy().argmax(axis=1)
            y_true = y_true.detach().cpu().numpy()

            accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
            balanced_accuracy = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
            # clf_report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
            precision, recall, f_score, _ = precision_recall_fscore_support(
                y_pred=y_pred, y_true=y_true, average="weighted"
            )

            self.log(f"{stage}/fscore", f_score)
            # self.log(f"{stage}_report", clf_report)
            self.log(f"{stage}/precision", precision)
            self.log(f"{stage}/recall", recall)
            self.log(f"{stage}/accuracy", accuracy)
            self.log(f"{stage}/accuracy_balanced", balanced_accuracy)
            self.logger.experiment.log({
                f"{stage}/cm": wandb.plot.confusion_matrix(probs=y_prob, y_true=y_true, class_names=self.class_names),
                "global_step": self.global_step
            })

    def __evaluation_on_epoch_end(self, outputs, stage):
        y_true = torch.cat([x[1] for x in outputs]).reshape(-1)
        if self.mode == "classification":
            y_pred = torch.cat([x[0] for x in outputs]).reshape(-1, len(self.class_names))
            loss = F.cross_entropy(y_pred, y_true).item()
        else:
            y_pred = torch.cat([x[0] for x in outputs]).reshape(-1)
            loss = F.mse_loss(y_pred, y_true).item()

        self.__evaluation(y_pred, y_true, stage)
        self.log(f"{stage}/loss", loss)

    def training_step(self, train_batch, batch_idx):
        loss, y_pred, y_true = self.__common_step(train_batch)
        self.__evaluation(y_pred, y_true, "train")
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return {
            "loss": loss,
            "y_pred": y_pred,
            "y_true": y_true
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_pred = torch.cat([x["y_pred"] for x in outputs]).reshape(-1, len(self.class_names))
        y_true = torch.cat([x["y_true"] for x in outputs]).reshape(-1)

        self.__evaluation(y_pred, y_true, "train")

    def validation_step(self, val_batch, batch_idx):
        _, y_pred, y_true = self.__common_step(val_batch)

        return y_pred, y_true

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.__evaluation_on_epoch_end(outputs, "validation")

    def test_step(self, test_batch, batch_idx):
        _, y_pred, y_true = self.__common_step(test_batch)

        return y_pred, y_true

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.__evaluation_on_epoch_end(outputs, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification/Regression Models")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--attr", type=str)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()

    seed_everything(100)

    # dataset = SpectralDataset(args.data_path, 12, args.attr, 256, [0.6, 0.2, 0.2])
    dataset = IndianaDataset(
        "/home/azureuser/data/Indiana_Transet_Data", "/home/azureuser/data/combined.csv", "Residue", batch_size=512
    )

    # model
    # model = LSTM(seq_len=12, label_name=args.attr)
    model = LSTM(seq_len=None, label_name=None, in_features=6, clf_classes=dataset.labels)
    checkpoint_callback = ModelCheckpoint(
        monitor="validation/accuracy", mode="max",
        filename=f"{os.path.basename(args.data_path)}-"
                 f"{args.attr.replace(' ', '')}-{{epoch:02d}}-"
                 f"{{validation/accuracy:02f}}",
        auto_insert_metric_name=True
    )
    early_stop_callback = EarlyStopping(
        monitor="validation/accuracy", min_delta=0.005, patience=20, verbose=True, mode="max"
    )
    # img_callback = WandbImageCallback()
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # training
    logger = WandbLogger(project="indiana", log_model=True)
    trainer = pl.Trainer(
        gpus=1, precision=16, limit_train_batches=1.0, auto_lr_find=True, max_epochs=100, logger=logger,
        deterministic=True, callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]
    )
    lr_finder = trainer.tuner.lr_find(model, dataset)
    model.hparams.learning_rate = lr_finder.suggestion()
    print("Best Learning Rate:", model.hparams.learning_rate)

    trainer.fit(model, dataset)
    trainer.test(ckpt_path="best", datamodule=dataset)
    print("Best Model Saved:", checkpoint_callback.best_model_path)
