import itertools
import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

import pdb

video_map = {
        'bike_bay_hdr': 0,
        'boxes': 1,
        'desk': 2,
        'desk_fast': 3,
        'desk_hand_only': 4,
        'desk_slow': 5,
        'engineering_posters': 6,
        'high_texture_plants': 7,
        'poster_pillar_1': 8,
        'poster_pillar_2': 9,
        'reflective_materials': 10,
        'slow_and_fast_desk': 11,
        'slow_hand': 12,
        'still_life': 13
        }

class HQFDataset(Dataset):
    def __init__(self,
                 hdf5_name='data/HQF/bike_bay_hdr.hdf5'):
        super(HQFDataset, self).__init__()
        self.data = h5py.File(hdf5_name, 'r')
        self.video_idx = video_map[os.path.basename(hdf5_name[:-5])]

    def __len__(self):
        return len(self.data['blurry_frame']) * 14

    def __getitem__(self, idx):
        frame_idx = idx // 14
        time_idx = idx % 14
        blurry_frame = self.data['blurry_frame'][frame_idx]
        event_map = self.data['event_map'][frame_idx]
        keypoints = self.data['keypoints11'][frame_idx]
        sharp_frame = self.data['sharp_frame'][frame_idx][time_idx][None, :, :]
        timestamps = np.linspace(-1, 1, 14, dtype=np.float32)[time_idx]
        item = {
                'video_idx': self.video_idx, # scalar
                'frame_idx': frame_idx, # scalar
                'time_idx': time_idx, # scalar
                'blurry_frame': blurry_frame, # (1, 180, 240)
                'event_map': event_map, # (26, 180, 240)
                'keypoints': keypoints, # (10, 180, 240)
                'sharp_frame': sharp_frame, # (1, 180, 240)
                'timestamps': timestamps # scalar
                }
        return item

if __name__ == '__main__':
    dataset = HQFDataset()
    item = dataset[0]
    pdb.set_trace()
    print(123)
