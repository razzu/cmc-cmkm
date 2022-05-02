import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from functools import reduce

from models.mlp import MLP

class SkeletonCooccurenceBlocks(LightningModule):
    """
    Shape:
        - Input: (N, C, F, J).
        - Output: (N, num_features, 1, 1) where
            N is a batch size,
            C is the number of channels,
            F is a length of input sequence,
            J is the number of joints,
            num_features is the number of output channels of the last convolutional block. 
    """
    def __init__(self,
                 input_channels = 3,
                 n_joints       = 20,
                 sample_length  = 50,
                 out_channels   = [64, 32, 32, 64, 128, 256],
                 kernel_sizes   = [(1, 1), (3, 1), (3, 3), (3, 3), (3, 3), (3, 3)],
                 max_pool_sizes = [None, None, 2, 2, 2, 2],
                 **kwargs):

        super(SkeletonCooccurenceBlocks, self).__init__()
        self.name = 'skeleton_cooccurence'
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.max_pool_sizes = max_pool_sizes

        self.conv1_joints = nn.Sequential(
            nn.Conv2d(input_channels, self.out_channels[0], self.kernel_sizes[0]),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU()
        )
        self.conv1_motions = nn.Sequential(
            nn.Conv2d(input_channels, self.out_channels[0], self.kernel_sizes[0]),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU()
        )
        self.conv2_joints = nn.Sequential(
            nn.Conv2d(self.out_channels[0], self.out_channels[1], self.kernel_sizes[1]),
            nn.BatchNorm2d(self.out_channels[1]),
        )
        self.conv2_motions = nn.Sequential(
            nn.Conv2d(self.out_channels[0], self.out_channels[1], self.kernel_sizes[1]),
            nn.BatchNorm2d(self.out_channels[1]),
        )

        self.conv3_joints = nn.Sequential(
            nn.Conv2d(n_joints, self.out_channels[2], self.kernel_sizes[2]),
            nn.BatchNorm2d(self.out_channels[2]),
            nn.MaxPool2d(self.max_pool_sizes[2]),
        )
        self.conv3_motions = nn.Sequential(
            nn.Conv2d(n_joints, self.out_channels[2], self.kernel_sizes[2]),
            nn.BatchNorm2d(self.out_channels[2]),
            nn.MaxPool2d(self.max_pool_sizes[2]),
        )

        self.conv4_joints = nn.Sequential(
            nn.Conv2d(self.out_channels[2], self.out_channels[3], self.kernel_sizes[3]),
            nn.BatchNorm2d(self.out_channels[3]),
            nn.MaxPool2d(self.max_pool_sizes[3]),
        )
        self.conv4_motions = nn.Sequential(
            nn.Conv2d(self.out_channels[2], self.out_channels[3], self.kernel_sizes[3]),
            nn.BatchNorm2d(self.out_channels[3]),
            nn.MaxPool2d(self.max_pool_sizes[3]),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.out_channels[3], self.out_channels[4], self.kernel_sizes[4]),
            nn.BatchNorm2d(self.out_channels[4]),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_sizes[4]),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.out_channels[4], self.out_channels[5], self.kernel_sizes[5]),
            nn.BatchNorm2d(self.out_channels[5]),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_sizes[5]),
        )

        self.output_shape = self.get_output_shape((1, input_channels, sample_length, n_joints))
        self.out_size = reduce(lambda x, y: x * y, self.output_shape)


    def forward(self, x):
        joints = x
        motions = torch.zeros_like(joints)
        motions[:, :, 1:, :] = joints[:, :, 1:, :] - joints[:, :, :-1, :]

        joints = self.conv1_joints(joints)
        motions = self.conv1_motions(motions)

        joints = self.conv2_joints(joints)
        motions = self.conv2_motions(motions)

        joints = joints.permute(0, 3, 1, 2)
        motions = motions.permute(0, 3, 1, 2)

        joints = self.conv3_joints(joints)
        motions = self.conv3_motions(motions)

        joints = self.conv4_joints(joints)
        motions = self.conv4_motions(motions)

        fused = torch.cat((joints, motions), dim=2)
        fused = self.conv5(fused)
        fused = self.conv6(fused)
        return fused

    def get_output_shape(self, input_shape):
        return self(torch.rand(*(input_shape))).data.shape

class SupervisedSkeletonCooccurenceModel(LightningModule):
    """
    Shape:
        - Input: (N, C, F, J).
        - Output: (N, num_class) where
            N is a batch size,
            C is the number of channels,
            F is a length of input sequence,
            J is the number of joints.
    """
    def __init__(
            self,
            # Co-occurence model params
            input_channels = 3,
            n_joints       = 20,
            out_channels   = [64, 32, 32, 64, 128, 256],
            kernel_sizes   = [[1, 1], [3, 1], [3, 3], [3, 3], [3, 3], [3, 3]],
            max_pool_sizes = [None, None, 2, 2, 2, 2],
            # MLP params
            output_size       = 27,
            # Training/dataset related params
            sample_length    = 50,
            lr               = 1e-3,
            optimizer_name   ='adam',
            metric_scheduler ='accuracy',
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Model definition
        self.blocks = SkeletonCooccurenceBlocks(input_channels=input_channels, n_joints=n_joints,
                                                out_channels=out_channels, kernel_sizes=kernel_sizes,
                                                max_pool_sizes=max_pool_sizes, sample_length=sample_length)
        blocks_output_shape = self.blocks.output_shape
        self.encoder_out_size = reduce(lambda x, y: x * y, blocks_output_shape)
        self.classifier = nn.Linear(self.encoder_out_size, output_size)

        # Loss function.
        self.loss = nn.CrossEntropyLoss()
        
        # Validation/test metrics.
        self.metric_scheduler = metric_scheduler
        
        # Optimizer params.
        self.lr = lr
        self.optimizer_name = optimizer_name

    def forward(self, x):
        x = self.blocks(x)
        x = nn.Flatten()(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x = batch['skeleton']
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
        x = batch['skeleton']
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
