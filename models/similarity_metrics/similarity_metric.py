import abc

class SimilarityMetric(abc.ABC):

    @abc.abstractmethod
    def compute_similarity_matrix(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def move_to_device(self, device):
        """
        Required by some similarity metrics which use torch networks.
        """
        pass

    @abc.abstractmethod
    def set_dataset(self, dataset):
        """
        Required by some similarity metrics which use precomputed values.
        """
        pass

    def __init__(self, modality):
        self.modality = modality