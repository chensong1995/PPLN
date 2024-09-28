import cv2
from lib.utils import save_session, AverageMeter
import numpy as np
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb

cuda = torch.cuda.is_available()

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, test_loader, args):
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.args = args

    def train(self, epoch):
        self.model.train()
        time_record = AverageMeter()
        loss_record = AverageMeter()
        err_record = AverageMeter()
        for i_batch, batch in enumerate(self.train_loader):
            start_time = time.time()
            heatmap_pred, loss = self.model(batch)
            loss = loss.mean()
            # step optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # compute 2D keypoint error
            if i_batch % 100 == 0:
                key2d_pred, _ = self.extract_keypoints(heatmap_pred)
                key2d_gt = batch['key2d'].detach().numpy()
                err = np.linalg.norm(key2d_pred - key2d_gt, axis=-1)
                valid_count = (~np.isnan(err)).sum()
                err = np.nanmean(err)
                err_record.update(err, valid_count)
            # print information during training
            time_record.update(time.time() - start_time)
            loss_record.update(loss.detach().cpu().numpy(), len(batch['event']))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {time.val:.3f} ({time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Err.: {err.val:.4f} ({err.avg:.4f})\t'.format(epoch, i_batch,
                                                                 len(self.train_loader),
                                                                 time=time_record,
                                                                 loss=loss_record,
                                                                 err=err_record),
                  end='\n' if i_batch == len(self.train_loader) - 1 else '\r')

    def extract_keypoints(self, heatmap):
        heatmap = heatmap.detach().cpu().numpy()
        batch_size = heatmap.shape[0]
        num_markers = heatmap.shape[1]
        keypoints = np.zeros((batch_size, num_markers, 2))
        confidence = np.zeros((batch_size, num_markers))
        for i_batch in range(batch_size):
            for i_marker in range(num_markers):
                heatmap_ = heatmap[i_batch, i_marker]
                y, x = np.unravel_index(np.argmax(heatmap_), heatmap_.shape)
                confidence[i_batch, i_marker] = heatmap_[y, x]
                keypoints[i_batch, i_marker, 0] = x
                keypoints[i_batch, i_marker, 1] = y
        return keypoints, confidence

    def visualize(self, y, name_prefix='debug'):
        base_dir = os.path.join(self.args.save_dir, 'visualization')
        os.makedirs(base_dir, exist_ok=True)
        y = y.detach().cpu().numpy()[0] # only extract the first example in batch
        for i in range(len(y)): # loop through all keypoints
            heatmap = np.uint8(y[i] * 255)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            save_name = os.path.join(base_dir, '{}_{}.png'.format(name_prefix, i))
            cv2.imwrite(save_name, heatmap)

    def test(self, epoch):
        print('Testing...')
        self.model.eval()
        loss_record = AverageMeter()
        err_record = AverageMeter()
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.test_loader)):
                heatmap_pred, loss = self.model(batch)
                loss = loss.mean()
                # compute 2D keypoint error
                key2d_pred, _ = self.extract_keypoints(heatmap_pred)
                key2d_gt = batch['key2d'].detach().numpy()
                err = np.linalg.norm(key2d_pred - key2d_gt, axis=-1)
                valid_count = (~np.isnan(err)).sum()
                err = np.nanmean(err)
                # update loss record
                loss_record.update(loss.detach().cpu().numpy(), len(batch['event']))
                err_record.update(err, valid_count)
                # visualization
                if i_batch > 0:
                    continue
                self.visualize(batch['heatmap'], name_prefix='epoch{}_gt'.format(epoch))
                self.visualize(heatmap_pred, name_prefix='epoch{}_pred'.format(epoch))
        print('Loss: {:.4f}, Err: {:.4f}'.format(loss_record.avg, err_record.avg))


    def save_keypoints(self):
        self.model.eval()
        save = {}
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.test_loader)):
                heatmap_pred, _ = self.model(batch)
                key2d_pred, confidence = self.extract_keypoints(heatmap_pred)
                for i_example in range(key2d_pred.shape[0]):
                    camera_idx = int(batch['camera_idx'][i_example].detach().numpy())
                    subject = int(batch['subject'][i_example].detach().numpy())
                    session = int(batch['session'][i_example].detach().numpy())
                    move = int(batch['move'][i_example].detach().numpy())
                    frame_idx = int(batch['frame_idx'][i_example].detach().numpy())
                    save[camera_idx, subject, session, move, frame_idx] = \
                            (key2d_pred[i_example], confidence[i_example])
        save_name = os.path.join(self.args.save_dir, 'keypoints.npy')
        np.save(save_name, save)

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.save_dir, 'checkpoints')
        note = str(self.args.lr)
        save_session(self.model, self.optimizer, ckpt_dir, note, epoch)
