import unittest

import datasets.utd_mhad as utd_mhad

DATA_PATH = '/home/data/multimodal_har_datasets/utd_mhad'

class TestUTDDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = utd_mhad.UTDDatasetManager(DATA_PATH)

    def test_all_modalities_present(self):
        self.assertEqual(len(self.dataset.data_files_df), 1722)
        self.assertEqual(self._count_by_modality("inertial"), 861)
        self.assertEqual(self._count_by_modality("skeleton"), 861)

    def _count_by_modality(self, modality):
        return len(self.dataset.data_files_df[self.dataset.data_files_df["modality"] == modality])

class TestUTDInertial(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = utd_mhad.UTDInertialInstance(f'{DATA_PATH}/Inertial/a23_s5_t4_inertial.mat')

    def test_shape(self):
        self.assertEqual(self.instance.signal.shape, (184, 6))

    def test_label_subject_trial(self):
        self.assertEqual([self.instance.label, self.instance.subject, self.instance.trial], [23, 5, 4])


class TestUTDSkeleton(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = utd_mhad.UTDSkeletonInstance(f'{DATA_PATH}/Skeleton/a23_s5_t4_skeleton.mat')

    def test_shape(self):
        self.assertEqual(self.instance.joints.shape, (20, 3, 72))

    def test_label_subject_trial(self):
        self.assertEqual([self.instance.label, self.instance.subject, self.instance.trial], [23, 5, 4])

    
if __name__ == '__main__':
    unittest.main()