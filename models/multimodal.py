import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

class MultiModalClassifier(LightningModule):
    """
    Generous Multimodal Classifier

    models_dict : dict
        dictionary of initialized encoders
    mlp_out : int
        output size for the MLP model (num_classes in case of supervised learning)
    hidden : list
        hidden sizes for mapping MLP models per modality
    modalities : list
        list of the used modalities
    optimizer_name : str
        name of the optimizer
    metric_scheduler : str
        name of the metric used for scheduling
    lr : float
        learning rate
    """
    def __init__(self, 
        models_dict,  
        out_size, 
        hidden=[256, 128],
        modalities = ['inertial', 'skeleton'],
        optimizer_name='adam',
        metric_scheduler='accuracy',
        lr=0.001,
        freeze_encoders=False) -> None:

        super().__init__()
        self.save_hyperparameters("out_size", "hidden", "modalities", "optimizer_name", "metric_scheduler", "lr")

        self.modalities = modalities
        self.models_dict = nn.ModuleDict(models_dict)

        self.encoders_out_size = 0
        self.projections = {}
        for modality in self.modalities:
            self.encoders_out_size += hidden[0]
            self.projections[modality] = nn.Sequential(
                nn.Linear(self.models_dict[modality].out_size, hidden[0]),
                nn.BatchNorm1d(hidden[0]),
                nn.ReLU(inplace=True)
            )
            if freeze_encoders:
                self.models_dict[modality].freeze()
        self.projections = nn.ModuleDict(self.projections)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self.encoders_out_size, out_size)

        # training and optimization parameters
        self.optimizer_name = optimizer_name
        self.metric_scheduler = metric_scheduler
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def on_fit_start(self):
        for modality in self.models_dict.keys():
            self.models_dict[modality].to(self.device)

    def forward(self, x):
        outs = []
        for modality in self.modalities:
            out = self.models_dict[modality](x[modality])
            out = self.flatten(out)
            out = self.projections[modality](out)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        return self.classifier(outs)

    def training_step(self, batch, batch_idx):
        x = {modality: batch[modality] for modality in batch.keys() if modality != 'label'}
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
        x = {modality: batch[modality] for modality in batch.keys() if modality != 'label'}
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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": '_'.join(['val', self.metric_scheduler])
            }
        }
