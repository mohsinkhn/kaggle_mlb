import numpy as np
import pytorch_lightning as pl
from timm.optim import NovoGrad
import torch
from torch import nn
from torch.utils.data import Dataset


class RandomData(Dataset):
    def __init__(self, data, labels, scaler):
        super().__init__()
        self.data = data.copy()
        self.data[np.isnan(self.data)] = 0
        self.data = scaler.transform(self.data)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        return self.data[idx].astype(np.float32), np.sqrt(self.labels[idx]).astype(np.float32)/10.0


class FCModel(nn.Module):
    def __init__(self, in_f, out_f, d, p):
        super().__init__()
        self.fc1 = nn.Linear(in_f, d)
        self.h1 = nn.Linear(d, d)
        self.h2 = nn.Linear(d, d)
        self.classifier = nn.Linear(d, 4)
        self.bnorm1 = nn.BatchNorm1d(in_f, momentum=0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Tanh()
        self.bnorm2 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
    
    def forward(self, x):
        x = self.bnorm1(x)
        x = self.fc1(self.drop(x))
        x = self.relu(x)
        x = x + x * self.sigmoid(self.h1(x))
        x = self.relu(x)
        x = self.classifier(self.drop2(x))
        return x


class LitModel(pl.LightningModule):
    """PL Model"""
    def __init__(
        self,
        in_f=503,
        out_f=4,
        d=196,
        p=0.05,
        lr=0.003,
        wd=0.001,
        grad_avg=True,
        steps=[30],
        gamma=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FCModel(in_f, out_f, d, p)
        self.criterion = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        yhat = self.forward(x)
        loss = self.criterion(yhat, y)
        return loss, torch.square(10*yhat), torch.square(10*y)

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        loss_corr = self.criterion(y, preds)
        self.log("train/loss", loss_corr, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        loss_corr = self.criterion(y, preds)

        self.log("val/loss", loss_corr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        return {"loss": loss, "preds": preds}

    def test_epoch_end(self, outputs):
        preds = torch.cat([o["preds"] for o in outputs], 0).cpu().numpy()
        self.log("output", preds)

    def configure_optimizers(self):
        optimizer = NovoGrad(self.parameters(), grad_averaging=self.hparams.grad_avg, lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=self.hparams.gamma, milestones=self.hparams.steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
