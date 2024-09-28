import argparse
import os

import h5py
import numpy as np

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='h5_dataset_7500_events/346x260')
    parser.add_argument('--cam_dir', type=str, default='P_matrices')
    parser.add_argument('--out_dir', type=str, default='processed')
    args = parser.parse_args()
    return args

def check_length(subject, raw_dir):
    length = 0
    for session in range(1, 6):
        for move in range(1, 9):
            data_name = 'S{}_session{}_mov{}_7500events.h5'.format(subject,
                                                                   session,
                                                                   move)
            if not os.path.exists(os.path.join(raw_dir, data_name)):
                continue
            with h5py.File(os.path.join(raw_dir, data_name), 'r') as f_data:
                length += len(f_data['DVS'])
    return length

def project(key3d, P):
    key3dh = np.concatenate([key3d, np.ones((13, 1))], axis=-1)
    key2dh = np.matmul(P, key3dh.transpose()).transpose()
    key2dx, key2dy = key2dh[:, 0] / key2dh[:, 2], key2dh[:, 1] / key2dh[:, 2]
    key2dy = 260 - key2dy
    key2d = np.float32(np.stack([key2dx, key2dy], axis=-1))
    return key2d

def write_move(f_out, f_label, f_data, i_write, session, move, raw_dir, cam_dir,
               out_dir):
    P2 = np.load(os.path.join(cam_dir, 'P2.npy'))
    P3 = np.load(os.path.join(cam_dir, 'P3.npy'))
    for i_read in range(len(f_data['DVS'])):
        # cam2: channel=3
        # cam3: channel=2
        event_cam2 = f_data['DVS'][i_read, :, :344, 3]
        event_cam3 = f_data['DVS'][i_read, :, :344, 2]
        key3d = np.transpose(f_label['XYZ'][i_read])
        key2d_cam2 = project(key3d, P2)
        key2d_cam3 = project(key3d, P3)
        f_out['event'][i_write, 0, :, :] = event_cam2
        f_out['event'][i_write, 1, :, :] = event_cam3
        f_out['key3d'][i_write, :, :] = key3d
        f_out['key2d'][i_write, 0, :, :] = key2d_cam2
        f_out['key2d'][i_write, 1, :, :] = key2d_cam3
        f_out['session'][i_write] = session
        f_out['move'][i_write] = move
        f_out['frame'][i_write] = i_read
        i_write += 1
    return i_write

def write_subject(f_out, raw_dir, cam_dir, out_dir):
    i_write = 0
    for session in range(1, 6):
        for move in range(1, 9):
            print(session, move)
            data_name = 'S{}_session{}_mov{}_7500events.h5'.format(subject,
                                                                   session,
                                                                   move)
            if not os.path.exists(os.path.join(raw_dir, data_name)):
                continue
            label_name = data_name[:-3] + '_label.h5'
            with h5py.File(os.path.join(raw_dir, label_name), 'r') as f_label:
                with h5py.File(os.path.join(raw_dir, data_name), 'r') as f_data:
                    i_write = write_move(f_out, f_label, f_data, i_write, session,
                                         move, raw_dir, cam_dir, out_dir)

def process(subject, raw_dir, cam_dir, out_dir):
    length = check_length(subject, raw_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, '{}.hdf5'.format(subject))
    with h5py.File(out_name, 'w', libver='latest') as f_out:
        f_out.create_dataset('event', (length, 2, 260, 344), dtype='u8',
                             chunks=(1, 1, 260, 344), compression='gzip',
                             compression_opts=9)
        f_out.create_dataset('key3d', (length, 13, 3), dtype='f')
        f_out.create_dataset('key2d', (length, 2, 13, 2), dtype='f')
        f_out.create_dataset('session', (length,), dtype='i')
        f_out.create_dataset('move', (length,), dtype='i')
        f_out.create_dataset('frame', (length,), dtype='i')
        write_subject(f_out, raw_dir, cam_dir, out_dir)

if __name__ == '__main__':
    args = parse_args()
    for subject in range(1, 18):
        process(subject, args.raw_dir, args.cam_dir, args.out_dir)
