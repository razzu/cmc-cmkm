import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

from models.mlp import MLP, MLPMM

class SupervisedUnimodalHAR(LightningModule):
    """
    Takes a pre-trained encoder, freezes it and adds an MLP on top.
    """
    def __init__(
        self,
        modality,
        encoder,
        out_size,
        hidden = [256, 128],
        optimizer_name = 'adam',
        lr = 1e-3,
        metric_scheduler='accuracy',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters('modality', 'out_size', 'hidden', 'optimizer_name', 'lr')
        self.modality = modality
        self.encoder = encoder
        self.MLP = MLP(in_size=encoder.out_size, out_size=out_size, hidden=hidden)
        self.flatten = nn.Flatten()
        self.loss = nn.CrossEntropyLoss()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.metric_scheduler = metric_scheduler

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.MLP(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.modality]
        y = batch['label'] - 1
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch[self.modality]
        y = batch['label'] - 1
        out = self(x)
        preds = torch.argmax(out, dim=1)

        loss = self.loss(out, y)
        self.log(f"{prefix}_loss", loss)
        return {f"{prefix}_loss": loss, "preds": preds}

    def configure_optimizers(self):
      return self._initialize_optimizer()

    def _initialize_optimizer(self):
        ### Add LR Schedulers
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": '_'.join(['val', self.metric_scheduler])
            }
        }