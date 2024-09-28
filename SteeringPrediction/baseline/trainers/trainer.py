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
        for i_batch, batch in enumerate(self.train_loader):
            start_time = time.time()
            angle_pred, loss = self.model(batch)
            loss = loss.mean()
            # Step optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_record.update(time.time() - start_time)
            loss_record.update(loss.detach().cpu().numpy(), len(batch['aps']))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {time.val:.3f} ({time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch,
                                                                   len(self.train_loader),
                                                                   time=time_record,
                                                                   loss=loss_record),
                  end='\n' if i_batch == len(self.train_loader) - 1 else '\r')

    def test(self, epoch):
        print('Testing...')
        self.model.eval()
        predictions = { 'day': [], 'night': [], 'all': [] }
        targets = { 'day': [], 'night': [], 'all': [] }
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.test_loader)):
                angle_pred, _ = self.model(batch)
                angle_pred = angle_pred.detach().cpu().numpy().tolist()
                angle_gt = batch['angle'].detach().cpu().numpy().tolist()
                night = batch['night'].detach().cpu().numpy().tolist()
                for i_ex in range(len(angle_pred)):
                    times = ['all']
                    times.append('night' if night[i_ex] else 'day')
                    for key in times:
                        predictions[key].append(angle_pred[i_ex])
                        targets[key].append(angle_gt[i_ex])
        for key in ['day', 'night', 'all']:
            predictions[key] = np.array(predictions[key])
            targets[key] = np.array(targets[key])
            rmse = np.sqrt(np.mean((predictions[key] - targets[key]) ** 2))
            eva = 1 - np.var(predictions[key] - targets[key]) / np.var(targets[key])
            print('{} RMSE: {:.4f} EVA: {:.4f}'.format(key,
                                                       rmse,
                                                       eva))

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.save_dir, 'checkpoints')
        note = str(self.args.lr)
        save_session(self.model, self.optimizer, ckpt_dir, note, epoch)
