#!/usr/bin/env python3
import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PDDataset
from models import Net
from loss import YOLOLoss, YOLOFocalLoss


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--annotations-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val-size', type=float, default=0.2)

    parser.add_argument('--spatial-coef', type=float, default=5.0)
    parser.add_argument('--positive-coef', type=float, default=2.0)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-ddr', type=float, default=0.0001)
    parser.add_argument('--lr-steps', type=int, default=20)
    parser.add_argument('--lr-gamma', type=float, default=0.1)

    parser.add_argument('--checkpoints-dir', default='checkpoints')
    parser.add_argument('--ddr-weights', default='extra/DDRNet23s.pth')
    parser.add_argument('--checkpoints-freq', type=int, default=5)
    parser.add_argument('--start-epoch', type=int, default=0)

    parser.add_argument('--verbose', action='store_true')
    return parser


def create_dataloaders(args):
    with open(args.annotations_path, 'rb') as file:
        targets = pickle.load(file)
    dataset = PDDataset((args.data_path,), targets, device=args.device)

    test_size = round(len(dataset) * args.val_size)
    train_size = len(dataset) - test_size
    data_train, data_test = random_split(dataset, (train_size, test_size))

    dl_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)
    return dl_train, dl_test


def create_model(args):
    model = Net()
    if args.start_epoch:
        path = os.path.join(args.checkpoints_dir, f'checkpoint-{args.start_epoch}.pth')
        model.load_state_dict(torch.load(path))
    else:
        model.load_pretrained(args.ddr_weights)
    return model.to(args.device)


def create_optimizer(args, model: Net):
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR

    opt1 = Adam(model.partial_parameters(True), lr=args.lr_ddr)
    opt2 = Adam(model.partial_parameters(False), lr=args.lr)

    lrs1 = StepLR(opt1, args.lr_steps, args.lr_gamma)
    lrs1.last_epoch = args.start_epoch - 1
    lrs2 = StepLR(opt2, args.lr_steps, args.lr_gamma)
    lrs2.last_epoch = args.start_epoch - 1
    return opt1, opt2, lrs1, lrs2


def main(args):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file:///tmp/init', rank=0, world_size=1)

    dl_train, dl_test = create_dataloaders(args)
    model = create_model(args).to(args.device)
    opt1, opt2, lrs1, lrs2 = create_optimizer(args, model)
    loss_layer = YOLOFocalLoss(args.spatial_coef, args.positive_coef)

    for epoch in range(args.start_epoch, args.epochs):
        s_losses = []
        c_losses = []

        model.train(True)
        for input, targets in tqdm(dl_train) if args.verbose else dl_train:
            output = model(input)
            loss = loss_layer(output, targets)
            loss.backward()
            opt1.step()
            opt2.step()
            lrs1.step()
            lrs2.step()
            opt1.zero_grad()
            opt2.zero_grad()
            s_losses.append(loss_layer.last_losses[0])
            c_losses.append(loss_layer.last_losses[1])

        s_loss_train = np.mean(s_losses)
        c_loss_train = np.mean(c_losses)
        s_losses.clear()
        c_losses.clear()

        model.train(False)
        with torch.no_grad():
            for input, targets in tqdm(dl_test) if args.verbose else dl_test:
                output = model(input)
                loss = loss_layer(output, targets)
                s_losses.append(loss_layer.last_losses[0])
                c_losses.append(loss_layer.last_losses[1])

        s_loss_test = np.mean(s_losses)
        c_loss_test = np.mean(c_losses)

        print("Epoch {:2d}/{:2d}\tTrain losses ({:.6f},{:.6f})\tTest losses ({:.6f},{:.6f})".format(
            epoch, args.epochs,
            s_loss_train, c_loss_train,
            s_loss_test, c_loss_test
        ))

        if epoch and epoch % args.checkpoints_freq == 0:
            path = os.path.join(args.checkpoints_dir, f'checkpoint-{epoch}.pth')
            with open(path, 'wb') as file:
                torch.save(model.state_dict(), file)


if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args)

