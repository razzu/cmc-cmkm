import unittest
import torch
import random

from datasets.utd_mhad import UTDDatasetManager
from data_modules.utd_mhad_data_module import UTDDataset, UTDDataModule
from transforms.inertial_transforms import InertialSampler
from transforms.skeleton_transforms import SkeletonSampler

DATA_PATH = '/home/data/multimodal_har_datasets/utd_mhad'

class TestUTDDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DATA_PATH
        cls.all_subjects = {"subject": list(range(1,9))}
        cls.all_modalities = UTDDataset._supported_modalities()
        cls.dataset_manager = UTDDatasetManager(cls.path)
        cls.default_dataset = UTDDataset(cls.all_modalities, cls.dataset_manager, cls.all_subjects)

    def test_same_sample_for_index(self):
        dataset = self.default_dataset
        for idx in random.sample(range(len(dataset)), 5):
            tuples = []
            for modality in self.all_modalities:
                (label, subject, trial) = tuple(dataset.data_tables[modality].iloc[idx])[-3:]
                tuples.append((label, subject, trial))
            self.assertEqual(len(set(tuples)), 1)

    def test_valid_modalities(self):
        self._test_valid_modalities(["inertial"])
        self._test_valid_modalities(["inertial", "skeleton"])

    def test_invalid_modality(self):
        self._test_invalid_modalities(["skeleton", "temperature"])
        self._test_invalid_modalities([])

    def test_all_subjects(self):
        dataset = self.default_dataset
        self.assertEqual(len(dataset), 861)

    def test_some_subjects(self):
        split = {"subject": [1, 2, 3, 4]}
        dataset = UTDDataset(self.all_modalities, self.dataset_manager, split)
        self.assertEqual(len(dataset), 431)

    def test_no_transforms(self):
        dataset = self.default_dataset
        self.assertEqual(dataset[5]["inertial"].shape, (156, 6))

    def test_some_transforms(self):
        transforms = {"inertial": InertialSampler(100)}
        dataset = UTDDataset(self.all_modalities, self.dataset_manager, self.all_subjects, transforms)
        self.assertEqual(dataset[5]["inertial"].shape, (100, 6))

    def _test_valid_modalities(self, modalities):
        dataset = UTDDataset(modalities, self.dataset_manager, self.all_subjects)
        instance = dataset[5]
        for modality in modalities:
            self.assertIn(modality, instance) 

    def _test_invalid_modalities(self, modalities):
        with self.assertRaises(AssertionError):
            UTDDataset(modalities, self.dataset_manager, self.all_subjects)

class TestUTDDataModule(unittest.TestCase):

    def test_batch_size(self):
        modalities = ["inertial", "skeleton"]
        batch_size = 64
        transforms = {
            "inertial": InertialSampler(100),
            "skeleton": SkeletonSampler(125)
        }

        data_module = UTDDataModule(modalities=modalities, batch_size=batch_size, train_transforms=transforms)
        data_module.setup()

        batch = next(iter(data_module.train_dataloader()))
        self.assertEqual(batch["inertial"].shape, torch.Size([batch_size, 100, 6]))
        self.assertEqual(batch["skeleton"].shape, torch.Size([batch_size, 20, 3, 125]))

    def test_data_split(self):
        data_module = UTDDataModule()
        data_module.setup()
        
        self.assertEqual(len(data_module.train_dataloader().dataset), 323)
        self.assertEqual(len(data_module.val_dataloader().dataset), 108)
        self.assertEqual(len(data_module.test_dataloader().dataset), 430)

    def test_train_test_transforms(self):
        modality = "inertial"
        batch_size = 32
        train_sample_size = 100
        test_sample_size = 200

        train_transforms = {
            modality: InertialSampler(train_sample_size)
        }
        test_transforms = {
            modality: InertialSampler(test_sample_size)
        }
        data_module = UTDDataModule(modalities=[modality], batch_size=batch_size,
                                    train_transforms=train_transforms, test_transforms=test_transforms)
        data_module.setup()

        train_batch = next(iter(data_module.train_dataloader()))
        self.assertEqual(train_batch[modality].shape, torch.Size([batch_size, train_sample_size, 6]))

        test_batch = next(iter(data_module.test_dataloader()))
        self.assertEqual(test_batch[modality].shape, torch.Size([batch_size, test_sample_size, 6]))

if __name__ == '__main__':
    unittest.main()