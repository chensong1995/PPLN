import _init_paths
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from lib.datasets import DHP19Dataset
from lib.models.model_repository import PredictNet
from lib.utils import *
from trainers.trainer import Trainer
import pdb

cuda = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='saved_weights/debug')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--test_every', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--group_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--norm', type=str, default='constant')
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
    train_set = DHP19Dataset(split='train', group_size=args.group_size)
    test_set = DHP19Dataset(split='test', group_size=args.group_size)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              num_workers=8,
                                              shuffle=False)
    return train_loader, test_loader

def setup_model(args):
    if args.use_temp:
        temp = args.temp
    else:
        temp = None
    model = PredictNet(group_size=args.group_size, norm=args.norm, temp=temp)
    if cuda:
        model = nn.DataParallel(model).cuda()
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-7)
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_session(model, optimizer, args)
    else:
        start_epoch = 0
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda x: 0.1)
    return model, optimizer, start_epoch, scheduler

# main function
if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    train_loader, test_loader = setup_loaders(args)
    model, optimizer, start_epoch, scheduler = setup_model(args)
    trainer = Trainer(model,
                      optimizer,
                      train_loader,
                      test_loader,
                      args)
    for epoch in range(start_epoch, args.n_epochs):
        if epoch in [10, 15]:
            scheduler.step()
        trainer.train(epoch)
        if (epoch + 1) % args.save_every == 0:
            trainer.save_model(epoch)
        if (epoch + 1) % args.test_every == 0:
            trainer.test(epoch)
    trainer.save_keypoints()
