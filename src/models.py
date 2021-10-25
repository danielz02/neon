import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
import pytorch_lightning as pl
from dataset import SpectralDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils import des_scatter_plot


class LSTM(pl.LightningModule):
    def __init__(self, data_path: str, label_name: str, batch_size: int, seq_len: int, num_layers: int = 2, hidden_size: int = 64, bidirectional: bool = False, lr: float = 1e-3):
        super().__init__()
        self.seq_len = seq_len
        self.learning_rate = lr
        self.data_path = data_path
        self.batch_size = batch_size
        self.label_name = label_name

        self.dataset = SpectralDataset(
            path=self.data_path,
            seq_len=self.seq_len,
            scale=None,
            target_name=self.label_name,
            # transforms=GaussianNoise()
        )
        self.train_size, self.val_size = int(len(self.dataset) * 0.6), int(len(self.dataset) * 0.2)
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

        self.encoder = nn.LSTM(1, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        n, *_ = x.shape
        x, hidden = self.encoder(x)
        x = self.regressor(x.reshape(n, -1))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=8)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x, hidden = self.encoder(x)
        y_pred = self.regressor(x.flatten(start_dim=1))
        loss = F.mse_loss(y_pred.reshape(-1), y.reshape(-1))
        r2 = np.corrcoef(y.detach().cpu().reshape(-1), y_pred.detach().cpu().reshape(-1))[0, 1] ** 2
        self.log("train_r2", r2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x, hidden = self.encoder(x)
        y_pred = self.regressor(x.flatten(start_dim=1))
        loss = F.mse_loss(y_pred.reshape(-1), y.reshape(-1))
        r2 = np.corrcoef(y.detach().cpu().reshape(-1), y_pred.detach().cpu().reshape(-1))[0, 1] ** 2
        self.log('val_loss', loss)
        self.log("val_r2", r2, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x, hidden = self.encoder(x)
        y_pred = self.regressor(x.flatten(start_dim=1))
        loss = F.mse_loss(y_pred.reshape(-1), y.reshape(-1))
        y = y.detach().cpu().reshape(-1)
        y_pred = y_pred.detach().cpu().reshape(-1)
        r2 = np.corrcoef(y, y_pred)[0, 1] ** 2

        fig, ax = plt.subplots()
        des_scatter_plot(y_pred=y_pred, y_true=y, label_name=self.label_name, save_path=None, ax=ax)
        fig.savefig(f"../figures/{self.__class__.__name__}_{self.label_name}.png", dpi=500)
        fig.close()

        self.log('test_loss', loss)
        self.log("test_r2", r2, on_epoch=True, prog_bar=True)

        # wandb.log({"Test Scatter Plot": fig})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foliar traits prediction")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--attr", type=str)
    args = parser.parse_args()

    seed_everything(100)
    # model
    model = LSTM(seq_len=356, label_name=args.attr, data_path=args.data_path, batch_size=4096)
    checkpoint_callback = ModelCheckpoint(monitor="val_r2", mode="max", auto_insert_metric_name=True)
    early_stop_callback = EarlyStopping(monitor="val_r2", min_delta=0.05, patience=10, verbose=True, mode="max")

    # training
    logger = WandbLogger(project="neon", log_model=True)
    trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=1.0, auto_lr_find=True, max_epochs=100, logger=logger, deterministic=True, callbacks=[checkpoint_callback, early_stop_callback])
    lr_finder = trainer.tuner.lr_find(model)
    model.hparams.learning_rate = lr_finder.suggestion()
    trainer.fit(model)
    trainer.test(ckpt_path="best")

    print("Best Model Saved:", checkpoint_callback.best_model_path)
