import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import pdb

class DDD20Dataset(Dataset):
    def __init__(self, data_dir='data', split='train'):
        self.img_size = (172, 128)
        self.night_files = set([1499656391, 1499657850, 1501649676, 1501650719,
                                1501994881, 1502336427, 1502337436, 1498946027,
                                1501651162, 1499025222, 1502338023, 1502338983,
                                1502339743, 1498949617, 1502599151])
        meta_name = os.path.join(data_dir, 'meta.pkl')
        meta = pd.read_pickle(meta_name)
        self.hdf5 = {}
        for idx in np.unique(meta['file_idx']):
            hdf5_name = os.path.join(data_dir, 'hdf5', 'rec{}_export.hdf5'.format(idx))
            self.hdf5[idx] = h5py.File(hdf5_name, 'r')
        if split == 'train':
            validity = ~meta['test'] & meta['valid']
        else:
            validity = meta['test'] & meta['valid']
        self.file_idx = meta['file_idx'][validity].to_numpy()
        self.example_idx = meta['example_idx'][validity].to_numpy()
        self.angle = meta['angle'][validity].to_numpy()
        self.speed = meta['speed'][validity].to_numpy()

    def __len__(self):
        return len(self.file_idx)

    def normalize_img(self, img):
        if img.max() != img.min():
            img = (img - img.min()) / (img.max() - img.min())
        elif img.max() != 0:
            img /= img.max()
        return img

    def __getitem__(self, idx):
        file_idx = self.file_idx[idx]
        example_idx = self.example_idx[idx]
        aps = self.hdf5[file_idx]['aps_frame'][example_idx]
        dvs = self.hdf5[file_idx]['dvs_frame'][example_idx]
        aps = self.normalize_img(np.float32(aps))
        dvs = self.normalize_img(np.float32(dvs))
        aps = cv2.resize(aps, self.img_size)
        dvs = cv2.resize(dvs, self.img_size)
        angle = np.float32(self.angle[idx])
        night = file_idx in self.night_files
        return {
                'aps': aps,
                'dvs': dvs,
                'angle': angle,
                'night': night
                }

if __name__ == '__main__':
    dataset = DDD20Dataset(split='train')
    item = dataset[0]
    # sanity check
    meta_name = os.path.join('data', 'meta.pkl')
    meta = pd.read_pickle(meta_name)
    targets = meta['angle'][~meta['test'] & meta['valid']].to_numpy(dtype=np.float32)
    predictions = np.zeros((len(targets),), dtype=np.float32)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    print('train rmse: {}'.format(rmse))
    targets = meta['angle'][meta['test'] & meta['valid']].to_numpy(dtype=np.float32)
    predictions = np.zeros((len(targets),), dtype=np.float32)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    print('test rmse: {}'.format(rmse))
    targets = meta['angle'][meta['valid']].to_numpy(dtype=np.float32)
    predictions = np.zeros((len(targets),), dtype=np.float32)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    print('all rmse: {}'.format(rmse))
    pdb.set_trace()
