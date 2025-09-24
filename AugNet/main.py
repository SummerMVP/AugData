#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

from train import *
from tools.funcs import *
from tools.TES import *
from AugNet import UNet
from MyDataloader import MyDatasetSingle, MyDatasetSingleADS

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.cuda.set_device(0)
EVAL_PRINT_FREQUENCY = 10


def main(args):
    device = torch.device("cuda")

    kwargs = {'num_workers': 2, 'pin_memory': True}

    PARAMS_NAME = 'model_params.pt'
    LOG_NAME = 'model_log'
    if not os.path.exists(args.ckt):
        os.mkdir(args.ckt)
    PARAMS_PATH = os.path.join(args.ckt, PARAMS_NAME)  # Path to save models
    LOG_PATH = os.path.join(args.ckt, LOG_NAME)  # Path to save logs

    setLogger(LOG_PATH, mode='a')

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    train_dataset = MyDatasetSingleADS(args.train_txt, Type='train')
    valid_dataset = MyDatasetSingleADS(args.val_txt, Type='val')
    test_dataset = MyDatasetSingle(args.cover_dir, 2, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    AugNet = UNet().cuda()
    AugNet = nn.DataParallel(AugNet)
    tes = TES().cuda()

    optimizer_Aug = optim.Adam(AugNet.parameters(), 0.0001)

    scheduler_Aug = optim.lr_scheduler.MultiStepLR(optimizer_Aug, milestones=[100, 200], gamma=0.1)
    best_loss = 9999999999.0

    if args.load:
        # Load AugNet
        all_state = torch.load(args.load)
        original_state = all_state['AugNet_state']
        AugNet.load_state_dict(original_state)
        optimizer_Aug.load_state_dict(all_state['optimizer_state'])

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        Test(device, test_loader, AugNet, tes, args.A, args.num, args.output_dir)
        return 0

    best_loss = evaluate(device, valid_loader, 0, AugNet, tes, optimizer_Aug, args.A, args.num, best_loss,
                          PARAMS_PATH)

    for epoch in range(1, args.epoch + 1):
        scheduler_Aug.step()

        train(device, train_loader, epoch, AugNet, tes, optimizer_Aug, args.A, args.num)

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            best_loss = evaluate(device, valid_loader, epoch, AugNet, tes, optimizer_Aug, args.A, args.num, best_loss,
                                  PARAMS_PATH)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cover_dir', dest='cover_dir', type=str, default=r'/data2/mingzhihu/dataset/Bossbase',
        help='the path of cover images'
    )
    parser.add_argument(
        '--train-txt', dest='train_txt', type=str, required=False,
        default=r"dataset/txt/BB-suni04-train.txt",
    )
    parser.add_argument(
        '--val-txt', dest='val_txt', type=str, required=False,
        default=r"dataset/txt/BB-suni04-val.txt",
    )

    parser.add_argument('--output_dir', dest='output_dir', type=str, default=None)
    parser.add_argument('--load', dest='load', type=str, default=None, help='Loading pre-models for the AugNet')
    parser.add_argument(
        '--lr', dest='lr', type=float, default=0.0001, help='learning rate'
    )
    parser.add_argument(
        '--A', dest='A', type=float, default=8, help='the amplitude of the noises'
    )
    parser.add_argument(
        '--num', dest='num', type=float, default=400, help='the number of the noises'
    )
    parser.add_argument(
        '--epoch', dest='epoch', type=float, default=200
    )
    parser.add_argument(
        '--batchsize', dest='batchsize', type=float, default=24  # test时自动设置为1
    )
    parser.add_argument(
        '--ckt', dest='ckt', type=str, default=None,
        help='Path to save models and logs'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)