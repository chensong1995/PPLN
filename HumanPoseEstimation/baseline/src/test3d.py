import argparse
import os

import h5py
import numpy as np

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--pred_name', type=str, default='saved_weights/debug/keypoints.npy')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--camera_dir', type=str, default='data/P_matrices')
    args = parser.parse_args()
    return args

def read_prediction(pred_name):
    pred = np.load(pred_name, allow_pickle=True).item()
    return pred

def read_cameras(camera_dir):
    cameras_pos = np.load(os.path.join(camera_dir, 'camera_positions.npy'))
    P_mat = {}
    for camera in [2, 3]:
        P_mat[camera] = np.load(os.path.join(camera_dir, 'P{}.npy'.format(camera)))
    return cameras_pos, P_mat
        
def project_uv_xyz_cam(uv, M):
    # adapted from: https://www.cc.gatech.edu/~hays/compvision/proj3/
    N = len(uv)
    uv_homog = np.hstack((uv, np.ones((N, 1))))
    M_inv= np.linalg.pinv(M)
    xyz = np.dot(M_inv, uv_homog.T).T
    x = xyz[:, 0] / xyz[:, 3]; y = xyz[:, 1] / xyz[:, 3]; z = xyz[:, 2] / xyz[:, 3]
    return x, y, z

def find_intersection(P0, P1):
    # from: https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python/52089867
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function 
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf."""
    # generate all line direction vectors 
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis] # normalized
    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis] * n[:, np.newaxis]  # I - n*n.T
    # see fig. 1 
    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (np.matmul(projs,P0[:, :, np.newaxis])).sum(axis=0)
    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    return p.T

def triangulate(pred_2d_cam2, pred_2d_cam3):
    # https://github.com/SensorsINI/DHP19/blob/master/Eval_2D_triangulation_and_3D_tutorial.ipynb
    # centers of the 2 used cameras
    Point0 = (np.stack((cameras_pos[1], cameras_pos[2])))
    # initialize empty sample of 3D prediction
    pred_3d = np.zeros((13, 3))
    # adapt the 2D prediction to match the same format of 3D label after back projection
    pred_2d_cam2_ = np.zeros(pred_2d_cam2.shape)
    pred_2d_cam3_ = np.zeros(pred_2d_cam3.shape)
    pred_2d_cam2_[:, 0] = pred_2d_cam2[:, 0]
    pred_2d_cam2_[:, 1] = 260 - pred_2d_cam2[:, 1]
    pred_2d_cam3_[:, 0] = pred_2d_cam3[:, 0]
    pred_2d_cam3_[:, 1] = 260 - pred_2d_cam3[:, 1]
    # back project each 2D point to 3D space, and find intersection of rays using least squares
    x_cam2_pred, y_cam2_pred, z_cam2_pred = project_uv_xyz_cam(pred_2d_cam2_, P_mat[2])
    x_cam3_pred, y_cam3_pred, z_cam3_pred = project_uv_xyz_cam(pred_2d_cam3_, P_mat[3])
    xyz_cam2 = np.stack((x_cam2_pred, y_cam2_pred, z_cam2_pred), axis=1)
    xyz_cam3 = np.stack((x_cam3_pred, y_cam3_pred, z_cam3_pred), axis=1)
    for joint_idx in range(13):
        # coordinates for both cameras of 2nd point of triangulation line.
        Point1 = np.stack((xyz_cam2[joint_idx,:], xyz_cam3[joint_idx,:]), axis=1).T
        if np.isnan(Point1).sum() > 0:
            pred_3d[joint_idx] = np.nan
        else:
            intersection = find_intersection(Point0, Point1)
            pred_3d[joint_idx] = intersection[0]
    return pred_3d

def project_to_2d(key3d, camera_idx):
    key3dh = np.concatenate([key3d, np.ones((13, 1))], axis=-1)
    key2dh = np.matmul(P_mat[camera_idx], key3dh.transpose()).transpose()
    key2dx, key2dy = key2dh[:, 0] / key2dh[:, 2], key2dh[:, 1] / key2dh[:, 2]
    key2dy = 260 - key2dy
    key2d = np.stack([key2dx, key2dy], axis=-1)
    return key2d


def test(pred, cameras_pos, P_mat, data_dir, thres=0.3):
    all_sum2, all_sum3, all_sum3d, all_count = 0., 0., 0., 0
    for subj in range(13, 18):
        hdf5_name = os.path.join(data_dir, '{}.hdf5'.format(subj))
        with h5py.File(hdf5_name, 'r') as f:
            read_i = 0
            subj_sum2, subj_sum3, subj_sum3d, subj_count = 0., 0., 0., 0
            for sess in range(1, 6):
                sess_sum2, sess_sum3, sess_sum3d, sess_count = 0., 0., 0., 0
                for move in range(1, 9):
                    move_sum2, move_sum3, move_sum3d, move_count = 0., 0., 0., 0
                    if (2, subj, sess, move, 0) not in pred:
                        continue
                    frame = 0
                    key2_history = np.full((13, 2), np.nan)
                    key3_history = np.full((13, 2), np.nan)
                    while (2, subj, sess, move, frame) in pred:
                        key2, con2 = pred[2, subj, sess, move, frame]
                        key3, con3 = pred[3, subj, sess, move, frame]
                        if frame != 0:
                            key2[con2 < thres] = np.nan
                            key3[con3 < thres] = np.nan
                        key2[np.isnan(key2)] = key2_history[np.isnan(key2)]
                        key3[np.isnan(key3)] = key3_history[np.isnan(key3)]
                        key2_history = key2
                        key3_history = key3
                        # compute 3D loss
                        key3d_pred = triangulate(key2, key3)
                        key3d_gt = f['key3d'][read_i]
                        mpjpe_3d_joints = np.linalg.norm((key3d_gt - key3d_pred), axis=-1)
                        # compute 2D loss
                        key2_gt = project_to_2d(key3d_gt, 2)
                        key3_gt = project_to_2d(key3d_gt, 3)
                        mpjpe_2_joints = np.linalg.norm((key2_gt - key2), axis=-1)
                        mpjpe_3_joints = np.linalg.norm((key3_gt - key3), axis=-1)
                        for i in range(13):
                            if not np.isnan(mpjpe_3d_joints[i]):
                                move_sum2 += mpjpe_2_joints[i]
                                move_sum3 += mpjpe_3_joints[i]
                                move_sum3d += mpjpe_3d_joints[i]
                                move_count += 1
                        frame += 1
                        read_i += 1
                    print('subj {} sess {} mov {} MPJPE 2D-2: {:.2f} 2D-3: {:.2f} 3D: {:.2f}'.format(\
                            subj, sess, move,
                            move_sum2 / move_count,
                            move_sum3 / move_count,
                            move_sum3d / move_count))
                    sess_sum2 += move_sum2
                    sess_sum3 += move_sum3
                    sess_sum3d += move_sum3d
                    sess_count += move_count
                print('subj {} sess {} MPJPE 2D-2: {:.2f} 2D-3: {:.2f} 3D: {:.2f}'.format(\
                        subj, sess,
                        sess_sum2 / sess_count,
                        sess_sum3 / sess_count,
                        sess_sum3d / sess_count))
                subj_sum2 += sess_sum2
                subj_sum3 += sess_sum3
                subj_sum3d += sess_sum3d
                subj_count += sess_count
        print('subj {} MPJPE 2D-2: {:.2f} 2D-3: {:.2f} 3D: {:.2f}'.format(\
                subj,
                subj_sum2 / subj_count,
                subj_sum3 / subj_count,
                subj_sum3d / subj_count))
        all_sum2 += subj_sum2
        all_sum3 += subj_sum3
        all_sum3d += subj_sum3d
        all_count += subj_count
    print('MPJPE 2D-2: {} 2D-3: {} 3D: {}'.format(all_sum2 / all_count,
                                                  all_sum3 / all_count,
                                                  all_sum3d / all_count))

if __name__ == '__main__':
    args = parse_args()
    pred = read_prediction(args.pred_name)
    cameras_pos, P_mat = read_cameras(args.camera_dir)
    test(pred, cameras_pos, P_mat, args.data_dir)


