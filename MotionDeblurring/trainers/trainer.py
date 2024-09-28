import os
import time

import cv2
import numpy as np
import skimage.metrics
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.utils import save_session, AverageMeter

import pdb

cuda = torch.cuda.is_available()

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, test_loader, args):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.criteria = self.setup_criteria()
        if cuda:
            for name in self.criteria:
                self.criteria[name] = nn.DataParallel(self.criteria[name]).cuda()
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'log'))
        self.step = {
                'train': 0,
                'validate': 0
                }

    def setup_criteria(self):
        criteria = {}
        criteria['fra'] = torch.nn.L1Loss()
        return criteria

    def setup_records(self):
        records = {}
        records['time'] = AverageMeter()
        records['total'] = AverageMeter()
        return records

    def compute_fra_loss(self, pred, batch, stage='train'):
        step = self.step[stage]
        loss_fra = self.criteria['fra'](pred['sharp_frame'],
                                        batch['sharp_frame']).mean()
        self.writer.add_scalar(stage + '/frame', loss_fra, step)
        loss = self.args.lambda_fra * loss_fra
        return loss

    def compute_losses(self, pred, batch, records, stage='train'):
        loss = 0.
        batch_size = batch['blurry_frame'].shape[0]
        info = []
        if self.args.lambda_fra > 0:
            loss = loss + self.compute_fra_loss(pred, batch, stage)
        records['total'].update(loss.detach().cpu().numpy(), batch_size)
        info.append('Total: {:.3f} ({:.3f})'.format(records['total'].val,
                                                    records['total'].avg))
        info = '\t'.join(info)
        step = self.step[stage]
        self.writer.add_scalar(stage + '/total', loss, step)
        self.writer.flush()
        self.step[stage] += 1
        return loss, info

    def train(self, epoch):
        for key in self.model.keys():
            self.model[key].train()
        records = self.setup_records()
        num_iters = min(len(self.train_loader), self.args.iters_per_epoch)
        for i_batch, batch in enumerate(self.train_loader):
            start_time = time.time()
            if i_batch >= self.args.iters_per_epoch:
                break
            pred = self.model['pred'](batch)
            loss, loss_info = self.compute_losses(pred,
                                                  batch,
                                                  records,
                                                  stage='train')

            self.optimizer['pred'].zero_grad()
            loss.backward()
            self.optimizer['pred'].step()

            # print information during training
            records['time'].update(time.time() - start_time)
            info = 'Epoch: [{}][{}/{}]\t' \
                   'Time: {:.3f} ({:.3f})\t{}'.format(epoch,
                                                      i_batch,
                                                      num_iters,
                                                      records['time'].val,
                                                      records['time'].avg,
                                                      loss_info)
            print(info)

    def test(self):
        for key in self.model.keys():
            self.model[key].eval()
        metrics = {}
        for metric_name in ['MSE', 'PSNR', 'SSIM']:
            metrics[metric_name] = AverageMeter()
        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(self.test_loader)):
                pred = self.model['pred'](batch)
                pred = np.clip(pred['sharp_frame'].detach().cpu().numpy(), 0, 1)
                gt = batch['sharp_frame'].detach().cpu().numpy()
                video_idx = batch['video_idx'].detach().cpu().numpy()
                frame_idx = batch['frame_idx'].detach().cpu().numpy()
                time_idx = batch['time_idx'].detach().cpu().numpy()
                for i_example in range(pred.shape[0]):
                    save_dir = os.path.join(self.args.save_dir,
                                            'output',
                                            '{:03d}'.format(video_idx[i_example]))
                    os.makedirs(save_dir, exist_ok=True)
                    i_time = time_idx[i_example]
                    save_name = os.path.join(save_dir,
                                            '{:06d}_{}.png'.format(frame_idx[i_example],
                                                                   i_time))
                    cv2.imwrite(save_name, pred[i_example, 0] * 255)
                    gt_ = np.uint8(gt[i_example, 0] * 255)
                    pred_ = np.uint8(pred[i_example, 0] * 255)
                    for metric_name, metric in zip(['MSE', 'PSNR', 'SSIM'],
                                                   [skimage.metrics.normalized_root_mse,
                                                    skimage.metrics.peak_signal_noise_ratio,
                                                    skimage.metrics.structural_similarity]):
                        metrics[metric_name].update(metric(gt_, pred_))

        info = 'MSE: {:.3f}\tPSNR: {:.3f}\tSSIM: {:.3f}'.format(metrics['MSE'].avg,
                                                                metrics['PSNR'].avg,
                                                                metrics['SSIM'].avg)
        print('Results:')
        print(info)

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.save_dir, 'checkpoints')
        save_session(self.model, self.optimizer, ckpt_dir, epoch)
