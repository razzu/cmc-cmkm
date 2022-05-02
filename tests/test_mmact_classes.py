import unittest

import datasets.mmact as mmact

DATA_PATH = '/home/data/multimodal_har_datasets/mmact_new'

class TestMMActDatasetManager(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_manager = mmact.MMActDatasetManager(DATA_PATH)

    def test_all_modalities_present(self):
        self.assertEqual(len(self.dataset_manager.data_files_df), 17354)
        self.assertEqual(self._count_by_modality("inertial"), 8732)
        self.assertEqual(self._count_by_modality("skeleton"), 8622)

    def _count_by_modality(self, modality):
        return len(self.dataset_manager.data_files_df[self.dataset_manager.data_files_df["modality"] == modality])

class TestMMActInertial(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = mmact.MMActInertialInstance(f'{DATA_PATH}/Inertial/a10_s13_t13_ses3_sc4.csv')

    def test_shape(self):
        self.assertEqual(self.instance.signal.shape, (412, 12))

    def test_label_subject_trial(self):
        self.assertEqual([self.instance.label, self.instance.subject, self.instance.trial], [10, 13, 13])

class TestMMActSkeleton(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = mmact.MMActSkeletonInstance(f'{DATA_PATH}/Skeleton/a1_s16_t10_ses5_sc2.npy')

    def test_shape(self):
        self.assertTrue(self.instance.joints.shape == (17, 2, 404))

    def test_label_subject_trial(self):
        self.assertEqual([self.instance.label, self.instance.subject, self.instance.trial], [1, 16, 10])

if __name__ == '__main__':
    unittest.main()