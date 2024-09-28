import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import pdb

data_list = [
    "rec1499656391_export.hdf5",
    "rec1499657850_export.hdf5",
    "rec1501649676_export.hdf5",
    "rec1501650719_export.hdf5",
    "rec1501994881_export.hdf5",
    "rec1502336427_export.hdf5",
    "rec1502337436_export.hdf5",
    "rec1498946027_export.hdf5",
    "rec1501651162_export.hdf5",
    "rec1499025222_export.hdf5",
    "rec1502338023_export.hdf5",
    "rec1502338983_export.hdf5",
    "rec1502339743_export.hdf5",
    "rec1498949617_export.hdf5",
    "rec1502599151_export.hdf5",
    "rec1500220388_export.hdf5",
    "rec1500383971_export.hdf5",
    "rec1500402142_export.hdf5",
    "rec1501288723_export.hdf5",
    "rec1501349894_export.hdf5",
    "rec1501614399_export.hdf5",
    "rec1502241196_export.hdf5",
    "rec1502825681_export.hdf5",
    "rec1499023756_export.hdf5",
    "rec1499275182_export.hdf5",
    "rec1499533882_export.hdf5",
    "rec1500215505_export.hdf5",
    "rec1500314184_export.hdf5",
    "rec1500329649_export.hdf5",
    "rec1501953155_export.hdf5"]

frame_cuts = [
    [2000, 4000],
    [2600, 1200],
    [500, 500],
    [500, 500],
    [200, 800],
    [100, 400],
    [100, 400],
    [3000, 1000],
    [850, 4500],
    [200, 1500],
    [200, 1500],
    [200, 2500],
    [200, 1500],
    [1000, 2200],
    [1500, 3000],
    [500, 200],
    [500, 1000],
    [200, 2000],
    [200, 1000],
    [200, 1500],
    [200, 1500],
    [500, 1000],
    [500, 1700],
    [800, 2000],
    [200, 1000],
    [500, 800],
    [200, 2200],
    [500, 500],
    [200, 600],
    [500, 1500]]

np.random.seed(0)
data = {
        'file_idx': [],
        'example_idx': [],
        'angle': [],
        'speed': [],
        'test': []
        }

for hdf5_name, (start, end) in zip(data_list, frame_cuts):
    print('reading {}'.format(hdf5_name))
    with h5py.File(os.path.join('hdf5', hdf5_name), 'r') as f:
        file_idx = int(hdf5_name[3:13])
        length = len(f['steering_wheel_angle'])
        first_f, last_f = start * 2, end * 2
        num_train = int((length - first_f - last_f) * 0.7)
        for example_idx in tqdm(range(first_f, length - last_f)):
            data['file_idx'].append(file_idx)
            data['example_idx'].append(example_idx)
            angle = f['steering_wheel_angle'][example_idx]
            speed = f['vehicle_speed'][example_idx]
            data['angle'].append(angle)
            data['speed'].append(speed)
            if example_idx < first_f + num_train:
                data['test'].append(False)
            else:
                data['test'].append(True)
data['file_idx'] = np.array(data['file_idx'])
data['example_idx'] = np.array(data['example_idx'])
data['angle'] = np.array(data['angle'])
data['speed'] = np.array(data['speed'])
data['test'] = np.array(data['test'])
df = pd.DataFrame(data=data)
print('raw frame count: {}'.format(len(df)))
print('raw train frame count: {}'.format(len(df[~df['test']])))
print('raw test frame count: {}'.format(len(df[df['test']])))
mean_angle = df['angle'].mean()
std_angle = df['angle'].std()
print('raw angle mean: {} std: {}'.format(mean_angle, std_angle))
# speeds below 15km/h are eliminated
df['valid'] = df['speed'] >= 15
# filter out frames where the steering angles are larger than three times the
# standard deviation of all angles
df['valid'] &= (df['angle'] <= mean_angle + 3 * std_angle) & (df['angle'] >= mean_angle - 3 * std_angle)
# randomly prune 70% of frames that have steering angles between +-5 degrees
small_degree = 5.
small_angle_idxs = np.where((df['angle'] <= small_degree) & (df['angle'] >= -small_degree) & ~df['test'])[0]
prune_idxs = np.random.choice(small_angle_idxs, int(len(small_angle_idxs) * 0.7))
new_valid = df['valid'].copy()
new_valid[prune_idxs] = False
df['valid'] = new_valid
print('filtered train frame count: {}'.format(len(df[~df['test'] & df['valid']])))
print('filtered test frame count: {}'.format(len(df[df['test'] & df['valid']])))
df.to_pickle('meta.pkl')
