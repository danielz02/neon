import gc

import numpy as np
import pandas as pd
import torch
import argparse
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from dataset import GaussianNoise
from dataset import SpectralDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class TinyRegNet(pl.LightningModule):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_pred = self.encoder(x)
        loss = F.mse_loss(y_pred.reshape(-1), y.reshape(-1))
        r2 = np.corrcoef(y.detach().cpu().reshape(-1), y_pred.detach().cpu().reshape(-1))[0, 1] ** 2
        self.log("train_r2", r2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_pred = self.encoder(x)
        loss = F.mse_loss(y_pred.reshape(-1), y.reshape(-1))
        r2 = np.corrcoef(y.detach().cpu().reshape(-1), y_pred.detach().cpu().reshape(-1))[0, 1] ** 2
        self.log('val_loss', loss)
        self.log("val_r2", r2, on_epoch=True, prog_bar=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foliar traits prediction")
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    wls = list(df.columns[-314:])
    del df
    gc.collect()

    # data
    dataset = SpectralDataset(
        path=args.data_path,
        scale=None,
        wls=wls,
        target_name="carbonPercent",
        # transforms=GaussianNoise()
    )
    train, val = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    train_loader = DataLoader(train, batch_size=128, num_workers=8)
    val_loader = DataLoader(val, batch_size=128, num_workers=8)

    # model
    model = TinyRegNet(seq_len=314)

    # training
    trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5, auto_lr_find=True, auto_scale_batch_size=True, max_epochs=1000)
    trainer.fit(model, train_loader, val_loader)

