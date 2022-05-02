import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

from models.mlp import ProjectionMLP

class MM_NTXent(LightningModule):
    """
    Multimodal adaptation of NTXent, according to the original CMC paper.
    """
    def __init__(self, batch_size, modalities, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = modalities
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, self.batch_size, self.modalities, self.temperature)
        return self.criterion(logits, labels), pos, neg
    
    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2):
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        similarity_matrix = torch.matmul(features_1, features_2.T)
        return similarity_matrix

    def get_infoNCE_logits_labels(self, features, batch_size, modalities=2, temperature=0.1):
        # Let M1 and M2 be abbreviations for the first and the second modality, respectively.

        # Computes similarity matrix by multiplication, shape: (batch_size, batch_size).
        # This computes the similarity between each sample in M1 with each sample in M2.
        features_1 = features[modalities[0]]
        features_2 = features[modalities[1]]
        similarity_matrix = MM_NTXent.get_cosine_sim_matrix(features_1, features_2)

        # We need to formulate (2 * batch_size) instance discrimination problems:
        # -> each instance from M1 with each instance from M2
        # -> each instance from M2 with each instance from M1

        # Similarities on the main diagonal are from positive pairs, and are the same in both directions.
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives_m1_m2 = similarity_matrix[mask].view(batch_size, -1)
        positives_m2_m1 = similarity_matrix[mask].view(batch_size, -1)
        positives = torch.cat([positives_m1_m2, positives_m2_m1], dim=0)

        # The rest of the similarities are from negative pairs. Row-wise for the loss from M1 to M2, and column-wise for the loss from M2 to M1.
        negatives_m1_m2 = similarity_matrix[~mask].view(batch_size, -1)
        negatives_m2_m1 = similarity_matrix.T[~mask].view(batch_size, -1)
        negatives = torch.cat([negatives_m1_m2, negatives_m2_m1])
        
        # Reshuffle the values in each row so that positive similarities are in the first column.
        logits = torch.cat([positives, negatives], dim=1)

        # Labels are a zero vector because all positive logits are in the 0th column.
        labels = torch.zeros(2 * batch_size)

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()

class ContrastiveMultiviewCoding(LightningModule):
    """
    Implementation of CMC (contrastive multiview coding), currently supporting exactly 2 views.
    """
    def __init__(self, modalities, encoders, hidden=[256, 128], batch_size=64, temperature=0.1, optimizer_name_ssl='adam', lr=0.001, **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'temperature', 'optimizer_name_ssl', 'lr')
        
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)

        self.projections = {}
        for m in modalities:
            self.projections[m] = ProjectionMLP(in_size=encoders[m].out_size, hidden=hidden)
        self.projections = nn.ModuleDict(self.projections)

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.loss = MM_NTXent(batch_size, modalities, temperature)

    def _forward_one_modality(self, modality, inputs):
        x = inputs[modality]
        x = self.encoders[modality](x)
        x = nn.Flatten()(x)
        x = self.projections[modality](x)
        return x

    def forward(self, x):
        outs = {}
        for m in self.modalities:
            outs[m] = self._forward_one_modality(m, x)
        return outs

    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        loss, pos, neg = self.loss(outs)
        self.log("ssl_train_loss", loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        loss, _, _ = self.loss(outs)
        self.log("ssl_val_loss", loss)

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }