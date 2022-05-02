import math
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

class ConvolutionalBlocks(nn.Module):
    def __init__(self, 
                in_channels, 
                sample_length,
                out_channels=[32, 64, 128],  
                kernel_size=3, 
                stride=1, 
                padding=1,
                pool_padding=0, 
                pool_size=2):
        """
        1D-Convolutional Network
        """
        super(ConvolutionalBlocks, self).__init__()
        self.name = 'conv_layers_1d'
        self.num_layers = len(out_channels)
        self.out_size = self._compute_out_size(out_channels[-1], sample_length, self.num_layers, padding, kernel_size, stride, pool_padding, pool_size=pool_size)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=pool_size, stride=None)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[1]),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=pool_size, stride=None)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[2]),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=pool_size, stride=None)
        )

    @staticmethod
    def _compute_out_size(num_channels, sample_length, num_layers, padding, kernel_size, stride, pool_padding, pool_size):
        conv_out_size = sample_length
        for _ in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            # conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class ConvolutionalBlocksTransformer(nn.Module):
    def __init__(self, 
                in_channels, 
                sample_length,
                out_channels=[32, 64, 128],  
                kernel_size=3, 
                stride=1, 
                padding=1):
        """
        1D-Convolutional Network without pooling for Transformer
        """
        super(ConvolutionalBlocksTransformer, self).__init__()
        self.name = 'conv_layers_1d'
        self.num_layers = len(out_channels)
        self.out_size = self._compute_out_size(out_channels[-1], sample_length, self.num_layers, padding, kernel_size, stride)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[1]),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels[2]),
            nn.ReLU()
        )

    @staticmethod
    def _compute_out_size(num_channels, sample_length, num_layers, padding, kernel_size, stride):
        conv_out_size = sample_length
        for _ in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class SupervisedCNN1D(LightningModule):
    def __init__(
        self,
        in_channels, 
        sample_length, 
        out_size,
        lr = 1e-3,
        optimizer_name='adam',
        out_channels=[32, 64, 128],  
        kernel_size=3, 
        stride=1, 
        padding=1,
        pool_padding=0, 
        pool_size=2,
        metric_scheduler='accuracy',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv_layers = ConvolutionalBlocks(in_channels, sample_length, out_channels=out_channels, kernel_size=kernel_size, \
            stride=stride, padding=padding, pool_padding=pool_padding, pool_size=pool_size)
        self.classifier = nn.Linear(self.conv_layers.out_size, out_size)
        self.loss = nn.CrossEntropyLoss()
        self.out_size = out_size
        self.metric_scheduler = metric_scheduler
        self.lr = lr
        self.optimizer_name = optimizer_name

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x = batch['inertial']
        y = batch['label'] - 1
        out = self(x)
        loss = nn.CrossEntropyLoss()(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch['inertial']
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


class PositionalEncoding(nn.Module):
    """
    Implementation of positional encoding from https://github.com/pytorch/examples/tree/master/word_language_model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNNTransformer(LightningModule):
    def __init__(self,  
        in_channels, 
        sample_length, 
        out_channels=[32, 64, 128],  
        kernel_size=3, 
        stride=1, 
        padding=1,
        dropout=0.1,
        num_head=2,
        num_attn_layers=2,
		**kwargs):

        super().__init__()
        self.conv_layers = ConvolutionalBlocksTransformer(in_channels, sample_length, out_channels=out_channels, kernel_size=kernel_size, \
            stride=stride, padding=padding)
        self.positional_encoding = PositionalEncoding(d_model=out_channels[-1], dropout=dropout, max_len=sample_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels[-1], nhead=num_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_attn_layers)
        self.out_size = self.conv_layers.out_size
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(2, 0, 1)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        return x


class SupervisedTransformer(LightningModule):
    def __init__(
        self,
        in_channels, 
        sample_length, 
        out_size,
        lr = 1e-3,
        optimizer_name='adam',
        out_channels=[32, 64, 128],  
        kernel_size=3, 
        stride=1, 
        padding=1,
        dropout=0.1,
        num_head=2,
        num_attn_layers=2,
        metric_scheduler='accuracy',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.cnn_transformer = CNNTransformer(in_channels, sample_length, out_channels, kernel_size, stride, padding,
            dropout, num_head, num_attn_layers)
        self.classifier = nn.Linear(self.cnn_transformer.out_size, out_size)
        self.loss = nn.CrossEntropyLoss()
        self.out_size = out_size
        self.metric_scheduler = metric_scheduler
        self.lr = lr
        self.optimizer_name = optimizer_name

    def forward(self, x):
        x = self.cnn_transformer(x)
        x = nn.Flatten()(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x = batch['inertial']
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
        x = batch['inertial']
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
