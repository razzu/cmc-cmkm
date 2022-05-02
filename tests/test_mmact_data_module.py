import unittest
import torch

from datasets.mmact import MMActDatasetManager
from data_modules.mmact_data_module import MMActDataset, MMActDataModule
from transforms.inertial_transforms import InertialSampler
from transforms.skeleton_transforms import SkeletonSampler
from utils.experiment_utils import load_yaml_to_dict

DATA_PATH = '/home/data/multimodal_har_datasets/mmact_new'

class TestMMActDataset(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DATA_PATH
        cls.all_subjects = {"subject": list(range(1, 21))}
        cls.all_modalities = MMActDataset._supported_modalities()
        cls.dataset_manager = MMActDatasetManager(cls.path)
        cls.default_dataset = MMActDataset(cls.all_modalities, cls.dataset_manager, cls.all_subjects)
        cls.cross_subject_split = load_yaml_to_dict("configs/dataset_configs.yaml")["datasets"]["mmact"]["protocols"]["cross_subject"]["train"]
        cls.cross_scene_split = load_yaml_to_dict("configs/dataset_configs.yaml")["datasets"]["mmact"]["protocols"]["cross_scene"]["train"]

    def test_same_sample_for_index_all(self):
        self._test_same_sample_for_index(self.default_dataset)

    def test_same_sample_for_index_cross_subject(self):
        dataset = MMActDataset(self.all_modalities, self.dataset_manager, self.cross_subject_split)
        self._test_same_sample_for_index(dataset)

    def test_same_sample_for_index_cross_scene(self):
        dataset = MMActDataset(self.all_modalities, self.dataset_manager, self.cross_scene_split)
        self._test_same_sample_for_index(dataset)

    def _test_same_sample_for_index(self, dataset):
        for idx in range(len(dataset)):
            tuples = []
            for modality in self.all_modalities:
                (label, subject, scene, session) = tuple(dataset.data_tables[modality].iloc[idx][["label", "subject", "scene", "session"]])
                tuples.append((label, subject, scene, session))
            self.assertEqual(len(set(tuples)), 1)

    def test_valid_modalities(self):
        self._test_valid_modalities(["inertial"])
        self._test_valid_modalities(["inertial", "skeleton"])

    def test_invalid_modality(self):
        self._test_invalid_modalities(["rgb", "temperature"])
        self._test_invalid_modalities([])

    def test_all_subjects(self):
        dataset = self.default_dataset
        self.assertEqual(len(dataset), 8360)

    def test_some_subjects(self):
        split = {"subject": [1, 2, 3, 4]}
        dataset = MMActDataset(self.all_modalities, self.dataset_manager, split)
        self.assertEqual(len(dataset), 1558)

    def test_no_transforms(self):
        dataset = self.default_dataset
        self.assertEqual(dataset[5]["inertial"].shape, (473, 12))

    def test_some_transforms(self):
        transforms = {"inertial": InertialSampler(100)}
        dataset = MMActDataset(self.all_modalities, self.dataset_manager, self.all_subjects, transforms)
        self.assertEqual(dataset[5]["inertial"].shape, (100, 12))

    def _test_valid_modalities(self, modalities):
        dataset = MMActDataset(modalities, self.dataset_manager, self.all_subjects)
        instance = dataset[5]
        for modality in modalities:
            self.assertIn(modality, instance) 

    def _test_invalid_modalities(self, modalities):
        with self.assertRaises(AssertionError):
            MMActDataset(modalities, self.dataset_manager, self.all_subjects)

class TestMMActDataModule(unittest.TestCase):

    def test_batch_size(self):
        modalities = ["inertial", "skeleton"]
        batch_size = 8
        transforms = {
            "inertial": InertialSampler(100),
            "skeleton": SkeletonSampler(50)
        }

        data_module = MMActDataModule(modalities=modalities, batch_size=batch_size, train_transforms=transforms)
        data_module.setup()

        batch = next(iter(data_module.train_dataloader()))
        self.assertEqual(batch["inertial"].shape, torch.Size([batch_size, 100, 12]))
        self.assertEqual(batch["skeleton"].shape, torch.Size([batch_size, 17, 2, 50]))

    def test_data_split(self):
        data_module = MMActDataModule()
        data_module.setup()
        
        self.assertEqual(len(data_module.train_dataloader().dataset), 4847)
        self.assertEqual(len(data_module.val_dataloader().dataset), 1315)
        self.assertEqual(len(data_module.test_dataloader().dataset), 1795)

    def test_cross_scene_data_split(self):
        dataset_cfg = load_yaml_to_dict("configs/dataset_configs.yaml")['datasets']['mmact']
        protocol = "cross_scene"
        split = dataset_cfg["protocols"][protocol]
        modalities = ["inertial", "skeleton"]
        data_module = MMActDataModule(split=split, modalities=modalities)
        data_module.setup()

        self.assertEqual(len(data_module.train_dataloader().dataset), 4732)
        self.assertEqual(len(data_module.val_dataloader().dataset), 1264)
        self.assertEqual(len(data_module.test_dataloader().dataset), 2364)

        for m in modalities:
            self.assertSetEqual(set(data_module.train_dataloader().dataset.data_tables[m].scene.unique()), set(split["train"]["scene"]))
            self.assertSetEqual(set(data_module.val_dataloader().dataset.data_tables[m].scene.unique()), set(split["val"]["scene"]))
            self.assertSetEqual(set(data_module.test_dataloader().dataset.data_tables[m].scene.unique()), set(split["test"]["scene"]))

            self.assertSetEqual(set(data_module.train_dataloader().dataset.data_tables[m].subject.unique()), set(split["train"]["subject"]))
            self.assertSetEqual(set(data_module.val_dataloader().dataset.data_tables[m].subject.unique()), set(split["val"]["subject"]))

        train_lengths = []
        val_lengths = []
        test_lengths = []
        for m in modalities:
            train_lengths.append(len(data_module.train_dataloader().dataset.data_tables[m]))
            val_lengths.append(len(data_module.val_dataloader().dataset.data_tables[m]))
            test_lengths.append(len(data_module.test_dataloader().dataset.data_tables[m]))
        self.assertEqual(len(set(train_lengths)), 1)
        self.assertEqual(len(set(val_lengths)), 1)
        self.assertEqual(len(set(test_lengths)), 1)

    def test_train_test_transforms(self):
        modality = "inertial"
        batch_size = 8
        train_sample_size = 100
        test_sample_size = 200

        train_transforms = {
            modality: InertialSampler(train_sample_size)
        }
        test_transforms = {
            modality: InertialSampler(test_sample_size)
        }
        data_module = MMActDataModule(modalities=[modality], batch_size=batch_size,
                                    train_transforms=train_transforms, test_transforms=test_transforms)
        data_module.setup()

        train_batch = next(iter(data_module.train_dataloader()))
        self.assertEqual(train_batch[modality].shape, torch.Size([batch_size, train_sample_size, 12]))

        test_batch = next(iter(data_module.test_dataloader()))
        self.assertEqual(test_batch[modality].shape, torch.Size([batch_size, test_sample_size, 12]))

if __name__ == '__main__':
    unittest.main()