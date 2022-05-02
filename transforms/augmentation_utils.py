from torchvision import transforms

from datasets.utd_mhad import UTDInertialInstance, UTDSkeletonInstance
from transforms.inertial_augmentations import ChannelShuffle, Jittering, Permutation, Rotation, Scaling
from transforms.skeleton_transforms import *
from utils.experiment_utils import load_yaml_to_dict

def compose_random_augmentations(modality, config_dict):
    inertial_augmentations = {
        'jittering': Jittering,
        'scaling': Scaling,
        'rotation': Rotation,
        'permutation': Permutation,
        'channel_shuffle': ChannelShuffle
    }
    skeleton_augmentations = {
        'jittering': Jittering,
        'crop_and_resize': CropAndResize,
        'scaling': RandomScale,
        'rotation': RandomRotation,
        'shear': RandomShear
    }

    all_augmentations = {
        "inertial": inertial_augmentations,
        "skeleton": skeleton_augmentations
    }
    transforms_list = []
    augmentations_for_modality = all_augmentations[modality]
    for key in config_dict:
        if config_dict[key]['apply']:
            if 'parameters' not in config_dict[key]:
                config_dict[key]['parameters'] = {}
            augmentation = augmentations_for_modality[key](**config_dict[key]['parameters'])
            probability = config_dict[key]['probability']
            transforms_list.append(transforms.RandomApply([augmentation], p=probability))
    return transforms_list 

def test_inertial_augmentations():
    augmentations_dict = load_yaml_to_dict('./configs/inertial_augmentations/augmentations_utd.yaml')
    composed_transform = transforms.Compose(compose_random_augmentations("inertial", augmentations_dict))
    test_signal = UTDInertialInstance('/home/data/multimodal_har_datasets/utd_mhad/Inertial/a23_s5_t4_inertial.mat').signal
    print("Original (inertial): ", test_signal[0])
    augmented_signal = composed_transform(test_signal)
    print("Augmented (inertial): ", augmented_signal[0])

def test_skeleton_augmentations():
    augmentations_dict = load_yaml_to_dict('./configs/skeleton_augmentations/augmentations.yaml')
    composed_transform = transforms.Compose(compose_random_augmentations("skeleton", augmentations_dict))
    test_joints = UTDSkeletonInstance('/home/data/multimodal_har_datasets/utd_mhad/Skeleton/a23_s5_t4_skeleton.mat').joints
    print("Original (skeleton): ", test_joints[:, :, 0])
    augmented_joints = composed_transform(test_joints)
    print("Augmented (skeleton): ", augmented_joints[:, :, 0])

if __name__ == '__main__':
    test_inertial_augmentations()
    test_skeleton_augmentations()
    
