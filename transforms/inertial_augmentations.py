import numpy as np
import torch

from datasets.utd_mhad import UTDInertialInstance

class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
        x = x + torch.tensor(noise).float()
        return x

class Scaling():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=(x.shape))
        x = x * factor
        return x

class Rotation():
    def __init__(self):
        pass

    def __call__(self, x):
        flip = torch.tensor(np.random.choice([-1, 1], size=(x.shape)))
        return flip * x

class ChannelShuffle():
    def __init__(self):
        pass

    def __call__(self, x):
        rotate_axis = np.arange(x.shape[1])
        np.random.shuffle(rotate_axis)
        return x[:, rotate_axis]

class Permutation():
    def __init__(self, max_segments=5):
        self.max_segments = max_segments

    def __call__(self, x):
        orig_steps = np.arange(x.shape[0])
        num_segs = np.random.randint(1, self.max_segments)
        
        ret = np.zeros_like(x)
        if num_segs > 1:
            splits = np.array_split(orig_steps, num_segs)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret = x[warp]
        else:
            ret = x
        return ret


if __name__ == '__main__':
    test_signal = UTDInertialInstance('/home/data/multimodal_har_datasets/utd_mhad/Inertial/a23_s5_t4_inertial.mat').signal
    print("Original: ", test_signal[0])

    jittered = Jittering(0.05)(test_signal)
    print("Jittered: ", jittered[0])

    scaled = Scaling(0.9)(test_signal)
    print("Scaled:   ", scaled[0])

    rotated = Rotation()(test_signal)
    print("Rotated:  ", rotated[0])

    permuted = Permutation()(test_signal)
    print("Permuted: ", permuted[0])

    shuffled = ChannelShuffle()(test_signal)
    print("Shuffled: ", shuffled[0])
