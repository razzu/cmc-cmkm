import argparse
import json
import os

from shutil import unpack_archive, rmtree
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

ACTIVITY_DICT = {
    'carrying': 1,
    'checking_time': 2,
    'closing': 3,
    'crouching': 4,
    'entering': 5,
    'exiting': 6,
    'fall': 7,
    'jumping': 8,
    'kicking': 9,
    'loitering': 10,
    'looking_around': 11,
    'opening': 12,
    'picking_up': 13,
    'pointing': 14,
    'pulling': 15,
    'pushing': 16,
    'running': 17,
    'setting_down': 18,
    'standing': 19,
    'talking': 20,
    'talking_on_phone': 21,
    'throwing': 22,
    'transferring_object': 23,
    'using_phone': 24,
    'walking': 25,
    'waving_hand': 26,
    'drinking': 27,
    'pocket_in': 28,
    'pocket_out': 29,
    'sitting': 30,
    'sitting_down': 31,
    'standing_up': 32,
    'using_pc': 33,
    'using_phone': 34,
    'carrying_heavy': 35,
    'carrying_light': 36
}

class MmactRaw():
    """
    Unzips MMAct data and then pre-processes it into the UTD-MHAD format in the given destination folder.
    Expects the following files and folders under data_path:
        acc_phone_clip.tar.gz
        acc_watch_clip.tar.gz
        gyro_clip.tar.gz
        orientation_clip.tar.gz
        trimmed_pose.zip (from challenge data)
    """
    
    def __init__(self, data_path, destination) -> None:
        self.data_path = data_path
        self.destination = destination

    def process_dataset(self):
        print('Processing inertial data...')
        self.process_inertial_data()
        print('Processing pose data...')
        self.process_pose_data()

    def process_inertial_data(self):
        inertial_sources = ["acc_phone_clip", "acc_watch_clip", "gyro_clip", "orientation_clip"]
        tmp_root = os.path.join(self.data_path, "inertial_tmp")
        
        # Unpack inertial zip files.
        print("Unzipping inertial data archives...")
        for source in tqdm(inertial_sources):
            zip_path = os.path.join(self.data_path, f"{source}.tar.gz")
            unpack_archive(zip_path, tmp_root)

        # List all inertial data files.
        all_source_paths = []
        for source in inertial_sources:
            source_root_path = os.path.join(tmp_root, f"{source}")
            file_paths = sorted([os.path.join(dp, f) for dp, _, filenames in os.walk(source_root_path) for f in filenames if os.path.splitext(f)[1] == '.csv'])
            file_paths = [f.replace("\\", "/") for f in file_paths] # Windows compatibility
            all_source_paths += file_paths

        # Create a temporary dataframe with paths and extracted metadata from inertial data.
        print("Creating temporary dataframe...")
        temp_df = pd.DataFrame(columns=["subject", "action", "scene", "session", "source", "source_path"])
        for i, source_path in enumerate(tqdm(all_source_paths)):
            source, subject, scene, session, video_name = source_path.split("/")[-5:]
            dest_subject = int(subject[7:])
            dest_action = ACTIVITY_DICT[video_name.split('.')[0].lower()]
            dest_scene = int(scene[5:]) # might be needed for cross-scene split
            dest_session = int(session[7:]) # might be needed for cross-session split
            temp_df.loc[i] = [dest_subject, dest_action, dest_scene, dest_session, source, source_path]

        inertial_destination = os.path.join(self.destination, "Inertial")
        os.makedirs(inertial_destination, exist_ok=True)

        # A little bit of voodoo:
        #   -> group all the files by subject and action, resulting in a list of trials for each subject-action pair
        #   -> then group by scene and session, resulting in a list of data files for each trial
        #   -> read the csv files for that particular trial
        #   -> clip and resample the signals to make them all a fixed length and sampling rate
        #   -> write the resulting data in a csv file
        print("Reading data, writing csv files...")
        subject_action_group_by = temp_df.groupby(["subject", "action"])
        for subject_action_key in tqdm(subject_action_group_by.groups.keys()):
            trial_group_by = subject_action_group_by.get_group(subject_action_key).groupby(["scene", "session"])

            for trial, scene_session_key in enumerate(trial_group_by.groups.keys(), start=1):
                subject, action = subject_action_key
                scene, session = scene_session_key
                trial_files = trial_group_by.get_group(scene_session_key)

                # Discard if not all sensor sources are present.
                if len(trial_files.index) != len(inertial_sources):
                    print(f'Skipping subject {subject}, action {action}, scene {scene}, session {session}, not all sensor sources present.')
                    trial = trial - 1
                    continue

                # Read sensor data from each csv file.
                trial_data = {}
                for source in inertial_sources:
                    r = trial_files[trial_files["source"] == source].iloc[0]
                    trial_data[source] = pd.read_csv(r["source_path"], names=["ts", "x", "y", "z"])[["x", "y", "z"]].to_numpy()

                # Discard if any of the sensor files is empty.
                if any(len(trial_data[k]) == 0 for k in trial_data):
                    print(f'Skipping subject {subject}, action {action}, scene {scene}, session {session}, empty time series found.')
                    trial = trial - 1
                    continue

                # Resample acc_phone and acc_watch from 100Hz to 50Hz
                acc_phone_length = trial_data["acc_phone_clip"].shape[0]
                trial_data["acc_phone_clip"] = resample(trial_data["acc_phone_clip"], acc_phone_length // 2)
                acc_watch_length = trial_data["acc_watch_clip"].shape[0]
                trial_data["acc_watch_clip"] = resample(trial_data["acc_watch_clip"], acc_watch_length // 2)

                # Drop the last timestamps to make all signals equal in length.
                shortest_length = min([d.shape[0] for d in trial_data.values()])
                for source in inertial_sources:
                    trial_data[source] = trial_data[source][:shortest_length]
                
                # Concatenate the values and write to a new file.
                final_sample = np.concatenate(list(trial_data.values()), axis=1)
                dest_filename = f'a{action}_s{subject}_t{trial}_ses{session}_sc{scene}.csv'
                dest_path = os.path.join(inertial_destination, dest_filename)
                pd.DataFrame(final_sample).to_csv(dest_path, header=None, index=None, float_format="%.6f")

        # Cleanup extracted data.
        rmtree(tmp_root)


    def process_pose_data(self):
        """
        Currently only extracts one view (cam1).
        """

        zip_path = os.path.join(self.data_path, "trimmed_pose.zip")
        tmp_destination = os.path.join(self.data_path, "pose_tmp")

        # Unpack zip file.
        print("Unzipping pose data archive...")
        unpack_archive(zip_path, tmp_destination)

        # List all pose data paths.
        data_path = os.path.join(tmp_destination, "pose", "cross_view", "trainval")
        file_paths = sorted([os.path.join(dp, f) for dp, _, filenames in os.walk(data_path) for f in filenames if os.path.splitext(f)[1] == '.json'])
        file_paths = [f.replace("\\", "/") for f in file_paths] # Windows compatibility
        file_paths = list(filter(lambda f: "cam1" in f, file_paths)) # Filter view.

        # Create a temporary dataframe with paths and extracted metadata from pose data.
        print("Creating temporary dataframe...")
        temp_df = pd.DataFrame(columns=["subject", "action", "scene", "session", "source_path"])
        for i, source_path in enumerate(tqdm(file_paths)):
            subject, _, scene, session, video_name = source_path.split("/")[-5:]
            dest_subject = int(subject[7:])
            dest_action = ACTIVITY_DICT[video_name.split('.')[0].lower()]
            dest_scene = int(scene[5:]) # might be needed for cross-scene split
            dest_session = int(session[7:]) # might be needed for cross-session split
            temp_df.loc[i] = [dest_subject, dest_action, dest_scene, dest_session, source_path]

        # Sort the dataframe and add a trial counter.
        temp_df.sort_values(by=["subject", "action", "scene", "session"], inplace=True)
        temp_df["trial"] = temp_df.groupby(["subject", "action"]).cumcount() + 1

        pose_destination = os.path.join(self.destination, "Skeleton")
        os.makedirs(pose_destination, exist_ok=True)
        
        # Process each data file.
        print("Processing and saving pose files...")
        N_JOINTS = 17 # COCO keypoints
        N_CHANNELS = 2 # 2D keypoints
        n_rows = temp_df.shape[0]
        for i, row in tqdm(temp_df.iterrows(), total=n_rows):
            # Read the original data file.
            with open(row["source_path"], 'r') as f:
                raw_data = json.load(f)
            
            # Read and reshape the data, discarding confidence values and keeping only the second subject if there are two subjects.
            n_frames = len(raw_data.keys())
            np_data = np.zeros((N_JOINTS, N_CHANNELS, n_frames))
            for fi, frame in enumerate(raw_data):
                frame_data = raw_data[frame]
                if "person2" not in frame_data or frame_data["person2"] == []:
                    person_key = "person1"
                else:
                    person_key = "person2"
                person_data = frame_data[person_key]
                keypoint_data = np.array(person_data).reshape((N_JOINTS, 3))[:, :2]
                np_data[:, :, fi] = keypoint_data

            # If a sample is completely empty, skip it.
            if (np.sum(np_data)) == 0:
                continue
            
            # Save the data as a Numpy file.
            dest_filename = f'a{row["action"]}_s{row["subject"]}_t{row["trial"]}_ses{row["session"]}_sc{row["scene"]}'
            dest_path = os.path.join(pose_destination, dest_filename)
            np.save(dest_path, np_data)

        # Cleanup extracted data.
        rmtree(tmp_destination)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='initial data path', required=True)
    parser.add_argument('--destination_path', type=str, help='destination path', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    mmact = MmactRaw(args.data_path, args.destination_path)
    mmact.process_dataset()
