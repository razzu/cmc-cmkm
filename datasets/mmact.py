import os
import string

import numpy as np
import pandas as pd
import tqdm

DATA_EXTENSIONS = {'.csv', '.npy'}

class MMActDatasetManager:
    def __init__(self, path):
        self.path = path
        self.data_files_df = self.get_table()

    def get_table(self):
        out_table = []
        all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(self.path) for f in filenames]
        for file_path in tqdm.tqdm(all_files):
            tmp_instance = MMActInstance(file_path)
            modality = tmp_instance.parse_modality().lower()
            ext = tmp_instance.parse_extension()

            a, s, t, ses, sc = tmp_instance.label, tmp_instance.subject, tmp_instance.trial, tmp_instance.session, tmp_instance.scene
            if np.nan not in [modality, ext, a, s, t, ses, sc] and ext in DATA_EXTENSIONS:
                out_table.append((file_path, modality, ext, a, s, t, ses, sc))

        columns = ['path', 'modality', 'extension', 'label', 'subject', 'trial', 'session', 'scene']
        return pd.DataFrame(out_table, columns=columns)

    def get_data_dict(self):
        ### We might need to store data in a JSON-like format as well
        pass

class MMActInstance:
    def __init__(self, file_):
        self._file = file_
        self.label, self.subject, self.trial, self.session, self.scene = self.parse_subject_label()

    def parse_subject_label(self):
        filename = os.path.split(self._file)[1]
        return [int(file_.strip(string.ascii_letters)) for file_ in filename[:-4].split('_')[:5]]

    def parse_modality(self):
        folder = os.path.split(self._file)[0].replace("\\", "/")
        return folder.split('/')[-1]

    def parse_extension(self):
        return os.path.splitext(self._file)[1]

class MMActInertialInstance(MMActInstance):
    def __init__(self, file_):
        super(MMActInertialInstance, self).__init__(file_)
        self.signal = self.read_inertial()

    def read_inertial(self):
        signal = pd.read_csv(self._file)
        return np.array(signal)

class MMActSkeletonInstance(MMActInstance):
    def __init__(self, file_):
        super(MMActSkeletonInstance, self).__init__(file_)
        self.joints = self.read_joints_npy()

    def read_joints_npy(self):
        return np.load(self._file)

if __name__ == '__main__':
    DATA_PATH = '/home/data/multimodal_har_datasets/mmact_new'
    inertial_instance_path = f'{DATA_PATH}/Inertial/a1_s1_t1_ses1_sc1.csv'
    skeleton_instance_path = f'{DATA_PATH}/Skeleton/a1_s16_t10_ses5_sc2.npy'

    dataset_manager = MMActDatasetManager(DATA_PATH)
    inertial_instance = MMActInertialInstance(inertial_instance_path)
    skeleton_instance = MMActSkeletonInstance(skeleton_instance_path)
