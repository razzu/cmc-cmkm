import os
import scipy.io

import numpy as np
import pandas as pd


DATA_EXTENSIONS = {'.mat'}


class UTDDatasetManager:
    def __init__(self, path): 
        self.path = path
        self.data_files_df = self.get_table()

    def get_table(self):
        out_table = []
        all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(self.path) for f in filenames]
        for file_path in all_files:
            tmp_instance = UTDInstance(file_path)
            modality = tmp_instance.parse_modality().lower() 
            ext = tmp_instance.parse_extension()
            if np.nan not in [modality, ext, tmp_instance.label, tmp_instance.subject, tmp_instance.trial] and ext in DATA_EXTENSIONS:
                out_table.append((file_path, modality, ext, tmp_instance.label, tmp_instance.subject, tmp_instance.trial))
        return pd.DataFrame(out_table, columns=['path', 'modality', 'extension', 'label', 'subject', 'trial'])
    
    def get_data_dict(self):
        ### We might need to store data in a JSON-like format as well
        pass


class UTDInstance:
    def __init__(self, file_):
        self._file = file_
        self.label, self.subject, self.trial = self.parse_subject_label()

    def parse_subject_label(self):
        filename = os.path.split(self._file)[1]
        try:
            return [int(file_[1:]) for file_ in filename.split('_')[:3]]
        except ValueError:
            print('Wrong input file: {}'.format(filename))
            return np.nan, np.nan, np.nan
    
    def parse_modality(self):
        folder = os.path.split(self._file)[0].replace("\\","/")
        return folder.split('/')[-1]

    def parse_extension(self):
        return os.path.splitext(self._file)[1]


class UTDInertialInstance(UTDInstance):
    def __init__(self, file_):
        super(UTDInertialInstance, self).__init__(file_)
        self.signal = self.read_inertial()
        
    def read_inertial(self):
        signal = scipy.io.loadmat(self._file)
        return signal['d_iner']


class UTDSkeletonInstance(UTDInstance):
    def __init__(self, file_):
        super(UTDSkeletonInstance, self).__init__(file_)
        self.joints = self.read_skeletons()

    def read_skeletons(self):
        skeletons = scipy.io.loadmat(self._file)
        return skeletons['d_skel']


if __name__ == '__main__':
    DATA_PATH = '/home/data/multimodal_har_datasets/utd_mhad/'
    instance_path = f'{DATA_PATH}/Skeleton/a1_s1_t1_skeleton.mat'
    skeleton_instance = UTDSkeletonInstance(instance_path)
    print(skeleton_instance.joints[0])
