import torch
import torch.nn.functional as F
from torch import nn

from models.similarity_metrics.similarity_metric import SimilarityMetric

class LatentSpaceSimilarity(SimilarityMetric):
    """
    Uses a pre-trained feature encoder to compute the cosine similarities between all
    latent representations in the batch.
    """
    def __init__(self, modality, encoder):
        super(LatentSpaceSimilarity, self).__init__(modality)
        self.encoder = encoder
        self.flatten = nn.Flatten()

    def move_to_device(self, device):
        self.encoder = self.encoder.to(device)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def compute_similarity_matrix(self, batch, batch_idx):
        inputs = batch[self.modality]
        features = self.encoder(inputs)
        features = self.flatten(features)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        return similarity_matrix