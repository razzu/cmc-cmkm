from pytorch_lightning.core.lightning import LightningModule
from torch import nn
import torch

class UnimodalLinearEvaluator(LightningModule):
	def __init__(self, modality, encoder, in_size, out_size, metric_scheduler="accuracy", lr=0.001, optimizer_name="adam"):
		super().__init__()
		self.modality = modality
		self.encoder = encoder
		self.encoder.freeze()
		self.out_size = out_size
		self.save_hyperparameters('in_size', 'out_size', 'metric_scheduler', 'lr', 'optimizer_name')

		self.flatten = nn.Flatten()
		self.linear = nn.Linear(in_size, out_size)
		self.loss = nn.CrossEntropyLoss()
		self.metric_scheduler = metric_scheduler
		self.lr = lr
		self.optimizer_name = optimizer_name

	def forward(self, x):
		x = self.encoder(x)
		x = self.flatten(x)
		return self.linear(x)

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
		if self.optimizer_name.lower() == 'adam':
			optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		return optimizer

		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
		# return {
		# 	"optimizer": optimizer,
		# 	"lr_scheduler": {
		# 		"scheduler": scheduler,
		# 		"monitor": '_'.join(['val', self.metric_scheduler])
		# 	}
		# }


class MLP(nn.Module):
	def __init__(self, in_size, out_size, hidden=[256, 128]):
		super(MLP, self).__init__()
		self.name = 'MLP'
		self.relu = nn.ReLU()
		self.linear1 = nn.Sequential(
			nn.Linear(in_size, hidden[0]),
			nn.ReLU(inplace=True)
		)
		self.linear2 = nn.Sequential(
			nn.Linear(hidden[0], hidden[1]),
			nn.ReLU(inplace=True)
		)
		self.output = nn.Linear(hidden[1], out_size)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.output(x)
		return x


class MLPMM(nn.Module):
	def __init__(self, in_size, hidden=[256, 128]):
		super().__init__()
		self.name = 'MLP'
		self.relu = nn.ReLU()
		self.linear1 = nn.Sequential(
			nn.Linear(in_size, hidden[0]),
			nn.BatchNorm1d(hidden[0]),
			nn.ReLU(inplace=True)
		)
		self.linear2 = nn.Sequential(
			nn.Linear(hidden[0], hidden[1]),
			nn.BatchNorm1d(hidden[1]),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		return x


class ProjectionMLP(nn.Module):
	def __init__(self, in_size, hidden=[256, 128]):
		super().__init__()
		self.name = 'MLP'
		self.relu = nn.ReLU()
		self.linear1 = nn.Sequential(
			nn.Linear(in_size, hidden[0]),
			nn.BatchNorm1d(hidden[0]),
			nn.ReLU(inplace=True)
		)
		self.linear2 = nn.Sequential(
			nn.Linear(hidden[0], hidden[1])
		)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		return x