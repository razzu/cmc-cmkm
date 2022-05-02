import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

from models.mlp import ProjectionMLP


class NTXent(LightningModule):
    def __init__(self, batch_size, n_views=2, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, self.batch_size, self.n_views, self.temperature)
        return self.criterion(logits, labels), pos, neg
    
    def get_infoNCE_logits_labels(self, features, batch_size, n_views=2, temperature=0.1):
        """
            Implementation from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        # creates a vector with labels [0, 1, 2, 0, 1, 2] 
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        # creates matrix where 1 is on the main diagonal and where indexes of the same intances match (e.g. [0, 4][1, 5] for batch_size=3 and n_views=2) 
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # computes similarity matrix by multiplication, shape: (batch_size * n_views, batch_size * n_views)
        similarity_matrix = get_cosine_sim_matrix(features)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.args.device)
        # mask out the main diagonal - output has one column less 
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix_wo_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # select and combine multiple positives
        positives = similarity_matrix_wo_diag[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives 
        negatives = similarity_matrix_wo_diag[~labels.bool()].view(similarity_matrix_wo_diag.shape[0], -1)

        # reshuffles values in each row so that positive similarity value for each row is in the first column
        logits = torch.cat([positives, negatives], dim=1)
        # labels is a zero vector because all positive logits are in the 0th column
        labels = torch.zeros(logits.shape[0])

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()


def get_cosine_sim_matrix(features):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    return similarity_matrix


class SimCLRUnimodal(LightningModule):
    def __init__(self, modality, encoder, mlp_in_size, hidden=[256, 128], batch_size=64, temperature=0.1, n_views=2, optimizer_name_ssl='lars', lr=0.001, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters('modality', 'hidden', 'batch_size', 'temperature', 'n_views', 'optimizer_name_ssl', 'lr')
        self.encoder = encoder
        self.projection = ProjectionMLP(mlp_in_size, hidden)
        self.modality = modality

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.loss = NTXent(batch_size, n_views, temperature)

    def forward(self, x):
        x = self.encoder(x)
        x = nn.Flatten()(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        batch = torch.cat(batch[self.modality], dim=0).float()
        out = self(batch)
        loss, pos, neg = self.loss(out)
        self.log("ssl_train_loss", loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = torch.cat(batch[self.modality], dim=0).float()
        out = self(batch)
        loss, _, _ = self.loss(out)
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

        elif self.optimizer_name_ssl.lower() == 'lars':
            optimizer = LARS(
                self.parameters(),
                self.lr,
                momentum=0.9,
                weight_decay=1e-6,
                trust_coefficient=0.001
            )

            return {
                "optimizer": optimizer
            }

