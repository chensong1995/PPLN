import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

import pdb

class DHP19Dataset(Dataset):
    def __init__(self,
                 hdf5_dir='data/processed',
                 cameras=[2, 3],
                 split='train'):
        self.hdf5_dir = hdf5_dir
        self.cameras = cameras
        self.split = split
        self.img_size = (260, 344)
        self.subject = []
        self.hdf5_idx = []
        self.hdf5 = {}
        if split == 'train':
            rng = range(1, 13)
        else:
            rng = range(13, 18)
        for subject in rng:
            hdf5_name = os.path.join(hdf5_dir, '{}.hdf5'.format(subject))
            self.hdf5[subject] = h5py.File(hdf5_name, 'r')
            self.subject += [subject] * len(self.hdf5[subject]['event'])
            self.hdf5_idx += list(range(len(self.hdf5[subject]['event'])))

    def get_heatmap(self, keypoint):
        # https://github.com/SensorsINI/DHP19/blob/master/heatmap_generation_example.ipynb
        def decay_heatmap(heatmap, sigma2=4):
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
            heatmap /= np.max(heatmap) # to keep the max to 1
            return heatmap
        heatmap = np.zeros(self.img_size, np.float32)
        x, y = keypoint
        if np.isnan(x) or np.isnan(y):
            return heatmap
        x = int(round(x))
        y = int(round(y))
        if x >= 0 and x < self.img_size[1] and y >= 0 and y < self.img_size[0]:
            heatmap[y, x] = 1
            heatmap = decay_heatmap(heatmap)
        return heatmap

    def __len__(self):
        return len(self.subject) * len(self.cameras)

    def __getitem__(self, idx):
        channel_idx = idx % len(self.cameras)
        example_idx = idx // len(self.cameras)
        camera_idx = self.cameras[channel_idx]
        subject = self.subject[example_idx]
        hdf5 = self.hdf5[subject]
        hdf5_idx = self.hdf5_idx[example_idx]
        session = hdf5['session'][hdf5_idx]
        move = hdf5['move'][hdf5_idx]
        frame = hdf5['frame'][hdf5_idx]
        key3d = hdf5['key3d'][hdf5_idx]
        key2d = hdf5['key2d'][hdf5_idx, channel_idx]
        event = hdf5['event'][hdf5_idx, channel_idx][None, :, :]
        heatmap = [self.get_heatmap(key) for key in key2d]
        heatmap = np.stack(heatmap)
        heatmap = np.float32(heatmap)
        return {
                'camera_idx': camera_idx,
                'subject': subject,
                'session': session,
                'frame_idx': frame,
                'move': move,
                'key3d': key3d,
                'key2d': key2d,
                'event': np.float32(event),
                'heatmap': heatmap
                }

if __name__ == '__main__':
    dataset = DHP19Dataset()
    element = dataset[333]
    pdb.set_trace()
    for i in range(13):
        cv2.imwrite('heatmap{}.png'.format(i), element['heatmap'][i] * 255)
    cv2.imwrite('event.png', element['event'][0])
    pdb.set_trace()
