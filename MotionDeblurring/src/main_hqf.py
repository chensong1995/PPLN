import _init_paths
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import lib
from lib.datasets import HQFDataset
from lib.models import PredNet
from lib.utils import *
from trainers.trainer import Trainer

import pdb

cuda = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--iters_per_epoch', type=int, default=10000)
    parser.add_argument('--hqf_data_dir', type=str, default='data/HQF')
    parser.add_argument('--save_dir', type=str, default='saved_weights/hqf')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_fra', type=float, default=1)
    parser.add_argument('--constant_norm', type=int, default=1)
    parser.add_argument('--sconv', type=int, default=1)
    parser.add_argument('--use_temp', type=int, default=1)
    parser.add_argument('--temp', type=float, default=100)
    args = parser.parse_args()
    return args

def initialize(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_loaders(args):
    train_set, test_set = [], []
    for hdf5_name in os.listdir(args.hqf_data_dir):
        dataset = HQFDataset(hdf5_name=os.path.join(args.hqf_data_dir, hdf5_name))
        if hdf5_name[:-5] in ['bike_bay_hdr',
                              'boxes',
                              'desk',
                              'engineering_posters',
                              'reflective_materials']:
            test_set.append(dataset)
        else:
            train_set.append(dataset)
    train_set = torch.utils.data.ConcatDataset(train_set)
    test_set = torch.utils.data.ConcatDataset(test_set)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               shuffle=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    return train_loader, test_loader

def setup_model(args):
    model, optimizer = {}, {}
    if args.use_temp:
        temp = args.temp
    else:
        temp = None
    model['pred'] = PredNet(constant_norm=bool(args.constant_norm),
                            sconv=bool(args.sconv),
                            temp=temp)
    # move models to cuda devices
    if cuda:
        for key in model.keys():
            model[key] = nn.DataParallel(model[key]).cuda()
    # build optimizators
    optimizer['pred'] = optim.Adam(model['pred'].parameters(), lr=args.lr)
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_session(model, optimizer, args)
    else:
        start_epoch = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer['pred'],
                                               milestones=[20, 40],
                                               gamma=0.5)
    return model, optimizer, start_epoch, scheduler

# main function
if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    train_loader, test_loader = setup_loaders(args)
    model, optimizer, start_epoch, scheduler = setup_model(args)
    trainer = Trainer(model, optimizer, train_loader, test_loader, args)
    for epoch in range(start_epoch, args.n_epochs):
        trainer.train(epoch)
        if (epoch + 1) % args.save_every == 0:
            trainer.save_model(epoch)
        scheduler.step()
    trainer.test()
